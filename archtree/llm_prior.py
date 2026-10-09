# archtree/llm_prior.py
"""
LLM warm-start prior for trie traversal (C7).

Backends:
  - ollama:    local server
  - groq:      cloud. Single key (GROQ_API_KEY) or key pool (use_key_pool=True).
  - heuristic: NO LLM (Oct 3, 2026 control). Knowledge-free "size prior": an op scores in
               proportion to its parameter count (conv_3x3 = 9, conv_1x1 = 1, others 0, /9), the
               same 0-1 scale as the model scores. If the LLM prior is no better than this, the LLM
               adds nothing beyond "bigger operations are better".

Prompt modes (Oct 3, 2026, contamination test; groq / ollama only):
  - standard:  the original prompt (names NAS-Bench-201 and its op names)
  - described: no benchmark name, no NB201 op names: the ops are labels op_A..op_E (fixed,
               scrambled order) with a plain functional description. If the LLM's help survives,
               it does not come from recalling NAS-Bench-201 by name.
  - anonymous: labels only, no descriptions - negative control; the model has nothing to go on,
               so its ranking quality should be ~chance (validates the probe).

Pool mode: when every pooled key is exhausted, scores fall back to uniform
(no blocking sleep), so the search keeps running without LLM guidance.

Oct 2026: fallbacks are no longer silent. Any call that does not return parsed model
scores (pool exhausted, API error, unparseable reply) increments `fallback_count`, is
NOT written to the cache, and is reported in the run log / cache-stats CSV. Once the pool
is exhausted no further calls are attempted in that run. Studies of the LLM effect must
discard runs with fallback_count > 0 (run_convergence_study.py / run_llm_size_sensitivity.py
do this automatically).

Oct 2026 (token efficiency, see utils/groq_client.py):
- Every Groq call goes through GroqChat: key pool, per-key daily + per-minute (req AND token) pacing,
  per-day 429 -> next key, strict json_schema output (no unparseable answers).
- The prompt no longer contains example scores. The old example {"none":0.7, ...} was copied by the
  model: the Sep-2026 caches show the SAME scores for every prefix, i.e. "none" (the worst op)
  always got the highest prior. The model now returns an integer 0-9 per op, schema-enforced.
- The model scores ALL ops of the next position (one call per trie prefix, as before); every later
  subset of unvisited ops is answered from that one answer (softmax over the subset).
- No call when only one op is unvisited (the prior cannot change the choice).
- Shared answer cache across seeds / runs / alphas (config llm.shared_cache, default on), keyed by
  (model, prompt): the question does not depend on the seed, so it is asked once.
"""

import json
import math
import time
from typing import Dict, List, Tuple

import archtree.storage as storage
from utils.checkpoint import Checkpoint


DEFAULT_MODEL = "llama3.1:8b"
MAX_RETRIES   = 3
CKPT_EVERY    = 10   # checkpoint every N real LLM calls
NB201_OPS     = ["none", "skip_connect", "conv_1x1", "conv_3x3", "avg_pool_3x3"]
SCORE_MAX     = 9    # model answers integers 0..9; divided by 9 -> same 0-1 scale as before
PROMPT_MODES  = ("standard", "described", "anonymous")

# Knowledge-free control prior: parameter count of the op (per C^2), divided by the max (9).
SIZE_PRIOR = {"none": 0.0, "skip_connect": 0.0, "avg_pool_3x3": 0.0, "conv_1x1": 1 / 9, "conv_3x3": 1.0}

# described / anonymous modes: fixed, scrambled label assignment (label order != NB201 op order)
_LABEL_OF = {"avg_pool_3x3": "op_A", "conv_3x3": "op_B", "none": "op_C",
             "conv_1x1": "op_D", "skip_connect": "op_E"}
_OP_OF = {v: k for k, v in _LABEL_OF.items()}
_DESCRIPTION = {
    "op_A": "3x3 average pooling (stride 1, no weights)",
    "op_B": "3x3 convolution with batch norm and ReLU",
    "op_C": "zero - the edge is removed and passes nothing",
    "op_D": "1x1 convolution with batch norm and ReLU",
    "op_E": "identity - the input is passed through unchanged",
}
_PROMPT_HEAD_DESCRIBED = (
    "You design a small convolutional cell for image classification. The cell is a directed acyclic "
    "graph whose 6 edges each apply one operation. The available operations are:\n"
    + "\n".join(f"{k}: {v}" for k, v in sorted(_DESCRIPTION.items())) +
    "\nGiven the operations already chosen for the first edges (in a fixed edge order), rate EVERY "
    "operation for the next edge by how likely it leads to a high-accuracy cell: integer 0 (bad) to 9 "
    "(best). Answer with a JSON object with one key per operation label and no other text.")
_PROMPT_HEAD_ANONYMOUS = (
    "You design a small convolutional cell for image classification. The cell is a directed acyclic "
    "graph whose 6 edges each apply one operation, chosen from five operations labelled op_A, op_B, "
    "op_C, op_D, op_E (no further information is given about them). Given the operations already "
    "chosen for the first edges (in a fixed edge order), rate EVERY operation for the next edge by how "
    "likely it leads to a high-accuracy cell: integer 0 (bad) to 9 (best). Answer with a JSON object "
    "with one key per operation label and no other text.")

# Fixed instructions first (stable prefix for Groq prompt caching), the changing part last.
_PROMPT_HEAD = (
    "You rate operations for a NAS-Bench-201 cell (image classification). The cell has 6 edges; "
    "each edge gets one operation from: none, skip_connect, conv_1x1, conv_3x3, avg_pool_3x3. "
    "Given the operations already chosen for the first edges, rate EVERY operation for the next "
    "edge by how likely it leads to a high-accuracy cell: integer 0 (bad) to 9 (best). "
    "Answer with a JSON object with one key per operation and no other text.")

# ── shared (model, prompt) -> raw scores cache, process-wide, persisted ──────
_SHARED = None
_SHARED_PATH = storage.PROJECT_ROOT / "outputs" / "_llm_shared_cache.json"


def _shared_cache() -> dict:
    global _SHARED
    if _SHARED is None:
        try:
            with open(_SHARED_PATH) as f:
                _SHARED = json.load(f)
        except (OSError, json.JSONDecodeError):
            _SHARED = {}
    return _SHARED


def _shared_put(model: str, prompt: str, raw: dict) -> None:
    c = _shared_cache()
    c.setdefault(model, {})[prompt] = raw
    try:
        storage._atomic_write_json(_SHARED_PATH, c)
    except Exception as e:      # noqa: BLE001  (cache is an optimisation only)
        print(f"[llm_prior] could not write shared cache ({e})")


class LLMPrior:
    def __init__(self,
                 phase: str = "phase3",
                 bench: str = "nb201",
                 dataset: str = "cifar10",
                 seed: int = 42,
                 model: str = None,
                 base_url: str = "http://localhost:11434",
                 host: str = None,
                 max_calls: int = 300,
                 timeout: int = 30,
                 namespace: str = "",
                 backend: str = "ollama",
                 groq_model: str = "openai/gpt-oss-20b",
                 groq_daily_token_budget: int = 190_000,    # free tier cap: 200K TPD
                 groq_daily_request_budget: int = 950,      # free tier cap: 1,000 RPD
                 groq_reasoning_effort: str = "low",        # keeps hidden CoT from eating max_tokens
                 use_key_pool: bool = False,                # rotate across GROQ_API_KEYS / _FILE
                 alpha_floor: float = 0.02,
                 depth_cache: bool = True,
                 resume_counters: bool = False,
                 shared_cache: bool = None,                 # None -> config llm.shared_cache
                 groq_rpm_per_key: int = None,              # None -> config llm.groq.rpm_per_key
                 groq_tpm_per_key: int = None,              # None -> config llm.groq.tpm_per_key
                 prompt_mode: str = "standard"):            # standard | described | anonymous

        self.phase     = phase
        self.bench     = bench
        self.dataset   = dataset
        self.seed      = seed
        self.backend   = backend
        if prompt_mode not in PROMPT_MODES:
            raise ValueError(f"[llm_prior] prompt_mode must be one of {PROMPT_MODES}, got {prompt_mode!r}")
        if prompt_mode != "standard" and bench != "nb201":
            raise ValueError("[llm_prior] described/anonymous prompts are defined for NB201 ops only")
        self.prompt_mode = prompt_mode
        self.max_calls = max_calls
        self.timeout   = timeout
        self.namespace = namespace
        self.alpha_floor = alpha_floor
        self.depth_cache_enabled = depth_cache
        from archtree import settings
        self.shared_enabled = settings.LLM_SHARED_CACHE if shared_cache is None else shared_cache
        self.shared_hit_count = 0
        self._chat = None

        # Exact-path cache (persisted) and depth cache (in-memory fallback only)
        self.cache: Dict[tuple, dict] = storage.load_llm_cache(phase, bench, dataset, seed, namespace)
        self.depth_cache: Dict[Tuple[int, frozenset], dict] = {}

        self._ckpt = Checkpoint(
            f"outputs/{phase}/logs/llm_prior_ckpt__{bench}__{dataset}__seed{seed}"
            f"{'__' + namespace if namespace else ''}.json",
            save_every=CKPT_EVERY,
        )
        # Counters start at zero for every new search (Oct 2026). search_phase3 always restarts a
        # run from iteration 0, so restoring call_count from an earlier run's checkpoint used to
        # shrink the new run's call budget (cached answers are still reused via self.cache).
        _ck = self._ckpt.load() if resume_counters else {}
        _st = _ck.get("state", {})

        # Restore counters from checkpoint so resumed runs keep correct totals
        self.call_count       = _st.get("call_count", 0)
        self.hit_count        = _st.get("hit_count", 0)
        self.depth_hit_count  = _st.get("depth_hit_count", 0)
        self.groq_call_count  = _st.get(
            "groq_call_count",
            storage.load_groq_counter(phase, bench, dataset, seed, namespace) if resume_counters else 0,
        )
        self._resume_iter = _ck.get("last_iter", -1)

        # Log "pool exhausted" only once per run
        self._budget_exhausted_logged = False

        # Diagnostic: model responses missing an expected op key.
        # Should stay near zero; a high rate means the model isn't echoing op names.
        self.name_mismatch_count = 0
        self.name_mismatch_calls = 0
        self.fallback_count = _st.get("fallback_count", 0)   # calls that returned no model scores
        self._exhausted = False                              # pool exhausted -> stop calling

        if backend == "heuristic":
            self.model = "size_prior"           # no LLM, no network, never a fallback
            self._client, self._limiter, self._key_pool = None, None, None
        elif backend == "groq":
            from utils.groq_client import GroqChat
            self.model = groq_model
            self._reasoning_effort = groq_reasoning_effort
            # Single key = a 1-key pool: same per-key budget files, never sleeps for hours on the
            # daily cap (an exhausted budget is a counted fallback, the run is redone later).
            self._chat = GroqChat(
                groq_model, use_key_pool=use_key_pool,
                daily_token_budget=groq_daily_token_budget,
                daily_request_budget=groq_daily_request_budget,
                rpm_per_key=groq_rpm_per_key or settings.GROQ_RPM_PER_KEY,
                tpm_per_key=groq_tpm_per_key or settings.GROQ_TPM_PER_KEY,
                reasoning_effort=groq_reasoning_effort, tag="llm_prior")
            self._key_pool = self._chat.pool
            print(f"[llm_prior] Groq: {len(self._key_pool.slots)} key(s) - "
                  f"{self._key_pool.remaining_summary()}")
            self._client, self._limiter = None, None
        else:
            self.model     = model or DEFAULT_MODEL
            self.base_url  = host if host is not None else base_url
            self._client    = _ollama_client(self.base_url)
            self._limiter   = None
            self._key_pool  = None

    # ── Public API ────────────────────────────────────────────────────────

    def score_branches(self,
                        path_so_far: List[str],
                        unvisited_ops: List[str],
                        alpha_t: float = 1.0) -> Dict[str, float]:
        if not unvisited_ops:
            return {}

        # Prior influence has decayed to ~0: skip the LLM entirely
        if alpha_t < self.alpha_floor:
            return self._uniform(unvisited_ops)

        # One unvisited op: the prior cannot change which child is taken -> no call
        if len(unvisited_ops) == 1:
            return {unvisited_ops[0]: 1.0}

        # Knowledge-free control: fixed size scores, no model, no cache, no budget
        if self.backend == "heuristic":
            self.hit_count += 1
            return self._from_raw({op: SIZE_PRIOR.get(op, 0.5) for op in unvisited_ops}, unvisited_ops)

        key = tuple(path_so_far)

        # 1) Exact-path cache hit (raw 0-1 scores for every op of the position)
        if key in self.cache:
            self.hit_count += 1
            return self._from_raw(self.cache[key], unvisited_ops)

        # 1b) Shared cache: the same question was answered in another seed / run
        ask_ops = self._ask_ops(unvisited_ops)
        prompt = self._build_prompt(path_so_far, ask_ops)
        if self.shared_enabled:
            raw = _shared_cache().get(self.model, {}).get(prompt)
            if raw is not None:
                self.hit_count += 1
                self.shared_hit_count += 1
                self.cache[key] = raw
                return self._from_raw(raw, unvisited_ops)

        # 2) Real call, if budget allows. Must come BEFORE the depth cache:
        #    the depth cache has no path context, so checking it first let
        #    unrelated architectures share scores and starved max_calls.
        if self.call_count < self.max_calls and not self._exhausted:
            raw, ok = self._call_with_retry(prompt, ask_ops)
            if not ok:
                # Not a model answer: count it, never cache it, return neutral scores.
                self.fallback_count += 1
                self.call_count += 1
                return self._uniform(unvisited_ops)

            self.cache[key] = raw
            if self.shared_enabled:
                _shared_put(self.model, prompt, raw)
            if self.depth_cache_enabled:
                self.depth_cache[len(path_so_far)] = raw

            self.call_count += 1
            if self.backend == "groq":
                self.groq_call_count += 1
                storage.save_groq_counter(self.phase, self.bench, self.dataset, self.seed,
                                           self.groq_call_count, self.namespace)

            storage.save_llm_cache(self.phase, self.bench, self.dataset, self.seed,
                                    self.cache, self.namespace)

            self._ckpt.maybe_save(self.call_count, self.max_calls, state=self._ckpt_state())

            return self._from_raw(raw, unvisited_ops)

        # 3) Call budget exhausted: reuse the latest same-depth scores (better than
        #    uniform). Deliberately NOT written to the exact-path cache, so it
        #    can't poison later lookups for other architectures.
        if self.depth_cache_enabled and len(path_so_far) in self.depth_cache:
            self.depth_hit_count += 1
            return self._from_raw(self.depth_cache[len(path_so_far)], unvisited_ops)

        # 4) Nothing available: neutral scores
        return self._uniform(unvisited_ops)

    def log_cache_stats(self, dataset: str = None, filename: str = "phase3_llm_cache_stats.csv") -> None:
        # Guard against logging stats under the wrong dataset
        if dataset is not None and dataset != self.dataset:
            raise ValueError(
                f"[llm_prior] dataset mismatch: got '{dataset}', instance is '{self.dataset}'"
            )
        total = self.call_count + self.hit_count + self.depth_hit_count
        storage.append_result_row(self.phase, filename, {
            "benchmark":               self.bench,
            "dataset":                 self.dataset,
            "seed":                    self.seed,
            "backend":                 self.backend,
            "total_llm_calls":         total,
            "cache_hits":              self.hit_count,
            "depth_cache_hits":        self.depth_hit_count,
            "cache_hit_rate":          (self.hit_count + self.depth_hit_count) / max(total, 1),
            "groq_call_count":         self.groq_call_count,
            "unique_prefixes_queried": len(self.cache),
            "name_mismatch_calls":     self.name_mismatch_calls,
            "name_mismatch_ops":       self.name_mismatch_count,
            "fallback_count":          self.fallback_count,
            "shared_cache_hits":       self.shared_hit_count,
            "llm_tokens":              (self._chat.stats["total_tokens"] if self._chat else 0),
            "truncated_replies":       (self._chat.stats["truncated"] if self._chat else 0),
        })
        self._ckpt.save(self.call_count, state=self._ckpt_state())

    def verify_server(self) -> bool:
        if self.backend == "heuristic":
            print("[llm_prior] heuristic size prior - no model needed.")
            return True
        if self.backend == "groq":
            if self._key_pool is not None:
                print(f"[llm_prior] Groq key pool selected, model='{self.model}', "
                      f"{len(self._key_pool.slots)} account(s) — no local server check needed.")
            else:
                print(f"[llm_prior] Groq backend selected, model='{self.model}' — "
                      f"no local server check needed.")
            return True
        try:
            models = self._client.list()
            available = [m.model for m in models.models]
            if not any(self.model in m for m in available):
                print(f"[llm_prior] WARNING: model '{self.model}' not found.")
                print(f"[llm_prior] Available: {available}")
                print(f"[llm_prior] Run: ollama pull {self.model}")
                return False
            print(f"[llm_prior] OK Ollama server OK, model '{self.model}' ready.")
            return True
        except Exception as e:
            print(f"[llm_prior] ERROR: Ollama server unreachable at {self.base_url}")
            print(f"[llm_prior] Start it with: ollama serve &")
            print(f"[llm_prior] Detail: {e}")
            return False

    # ── Internal ──────────────────────────────────────────────────────────

    def _ckpt_state(self) -> dict:
        return {
            "call_count":      self.call_count,
            "hit_count":       self.hit_count,
            "depth_hit_count": self.depth_hit_count,
            "groq_call_count": self.groq_call_count,
            "fallback_count":  self.fallback_count,
        }

    def _ask_ops(self, unvisited_ops):
        """Ops the model is asked about: all ops of the position when they are known (NB201), so
        one answer serves every later subset of unvisited children of this prefix."""
        if self.bench == "nb201" and set(unvisited_ops) <= set(NB201_OPS):
            return list(NB201_OPS)
        return list(unvisited_ops)

    def _from_raw(self, raw, unvisited_ops):
        """Raw 0-1 scores -> softmax over the unvisited subset (same scale as the Sep-2026 prior)."""
        neutral = 0.5
        return self._softmax({op: float(raw.get(op, neutral)) for op in unvisited_ops})

    @staticmethod
    def _schema(ops):
        return {"type": "object", "additionalProperties": False, "required": list(ops),
                "properties": {op: {"type": "integer", "enum": list(range(SCORE_MAX + 1))}
                               for op in ops}}

    def _call_with_retry(self, prompt, ask_ops):
        if self.backend == "groq":
            return self._call_groq(prompt, ask_ops)
        return self._call_ollama(prompt, ask_ops)

    def _call_groq(self, prompt, ask_ops):
        """-> (raw 0-1 scores for ask_ops, ok)."""
        res = self._chat.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=300,             # reasoning (effort=low) + ~40-token JSON; unused tokens are not billed
            temperature=0.1, schema=self._schema(self._keys(ask_ops)), schema_name="op_scores")
        if res is None:
            if self._chat.exhausted:
                self._exhausted = True
                if not self._budget_exhausted_logged:
                    print(f"[llm_prior:groq] all keys exhausted after {self.call_count} LLM calls this "
                          f"run - no more calls; this run is NOT valid LLM-on data.")
                    self._budget_exhausted_logged = True
            return self._uniform(ask_ops), False
        return self._parse_scores(res["text"], ask_ops)

    def _call_ollama(self, prompt, ask_ops):
        for attempt in range(MAX_RETRIES):
            try:
                try:        # Ollama >= 0.5: JSON-schema constrained output
                    response = self._client.chat(model=self.model, format=self._schema(self._keys(ask_ops)),
                                                 messages=[{"role": "user", "content": prompt}],
                                                 options={"temperature": 0.1})
                except TypeError:
                    response = self._client.chat(model=self.model, format="json",
                                                 messages=[{"role": "user", "content": prompt}],
                                                 options={"temperature": 0.1})
                return self._parse_scores(response.message.content.strip(), ask_ops)
            except Exception as e:      # noqa: BLE001
                if attempt == MAX_RETRIES - 1:
                    print(f"[llm_prior:ollama] failed after {MAX_RETRIES} tries: {e}")
                    return self._uniform(ask_ops), False
                time.sleep(1)
        return self._uniform(ask_ops), False

    def _keys(self, ops):
        """JSON keys the model answers with: op names (standard) or op labels (described/anonymous)."""
        if self.prompt_mode == "standard":
            return list(ops)
        return [_LABEL_OF[o] for o in ops]

    def _build_prompt(self, path_so_far, ask_ops):
        depth = len(path_so_far)
        if self.prompt_mode == "standard":
            chosen = ", ".join(path_so_far) if path_so_far else "(none yet)"
            tail = f"\nEdge {depth + 1} of 6. Already chosen: {chosen}."
            if ask_ops != NB201_OPS:
                tail += f" Rate only: {', '.join(ask_ops)}."
            return _PROMPT_HEAD + tail
        head = _PROMPT_HEAD_DESCRIBED if self.prompt_mode == "described" else _PROMPT_HEAD_ANONYMOUS
        chosen = ", ".join(_LABEL_OF[t] for t in path_so_far) if path_so_far else "(none yet)"
        tail = f"\nEdge {depth + 1} of 6. Already chosen: {chosen}."
        if ask_ops != NB201_OPS:
            tail += f" Rate only: {', '.join(self._keys(ask_ops))}."
        return head + tail

    def _parse_scores(self, text, ask_ops):
        """-> (raw scores in 0-1 for ask_ops, ok). Integers 0-9 are divided by 9; values already in
        0-1 (non-schema backends) are kept. ok=False when the reply could not be parsed."""
        try:
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()
            if not text.startswith("{"):
                start, end = text.rfind("{"), text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    text = text[start:end + 1]
            raw = json.loads(text)
            if self.prompt_mode != "standard" and isinstance(raw, dict):
                raw = {_OP_OF.get(k, k): v for k, v in raw.items()}     # labels -> op names
            missing = [op for op in ask_ops if op not in raw]
            if missing:
                self.name_mismatch_count += len(missing)
                self.name_mismatch_calls += 1
            if len(missing) == len(ask_ops):
                return self._uniform(ask_ops), False
            vals = {op: float(raw[op]) for op in ask_ops if op in raw}
            scale = float(SCORE_MAX) if any(v > 1.0 for v in vals.values()) else 1.0
            out = {op: min(max(vals.get(op, 0.5 * scale) / scale, 0.0), 1.0) for op in ask_ops}
            return out, True
        except (json.JSONDecodeError, ValueError, KeyError, TypeError, AttributeError):
            return self._uniform(ask_ops), False

    @staticmethod
    def _softmax(scores):
        vals   = list(scores.values())
        max_v  = max(vals)   # subtract max for numerical stability
        exps   = {k: math.exp(v - max_v) for k, v in scores.items()}
        total  = sum(exps.values())
        return {k: v / total for k, v in exps.items()}

    @staticmethod
    def _uniform(ops):
        p = 1.0 / len(ops)
        return {op: p for op in ops}


def _ollama_client(base_url):
    import ollama as _ollama
    return _ollama.Client(host=base_url)