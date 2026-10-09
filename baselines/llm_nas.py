# baselines/llm_nas.py
"""
LLM-NAS baseline: an LLM proposes candidate architectures directly, with no
structural memory, no UCB and no zero-cost proxy.

How it works:
  - Batched generation: each API call asks for `batch_size` architectures, and
    the proposals are consumed one per iteration (fewer calls per eval).
  - Context: the prompt shows the top-K best architectures seen so far.
  - Budget-safe fallback chain when Groq is unavailable:
        Groq (key pool or single key) -> local Ollama model (optional)
        -> local mutation of the best-known architecture (no API calls).
  - After each real Groq call, the estimated token usage is replaced with the
    actual usage so the daily-budget tracker doesn't drift.
  - Diagnostics (calls, parse failures, fallback counts) are returned in the metrics.

Oct 2026 token efficiency (utils/groq_client.py): strict json_schema output (ops are an enum, so
no invalid / unparseable archs), per-key per-minute token pacing, per-day 429 -> next key, fixed
instructions first and the changing context last (Groq prompt caching), max_tokens sized to the
answer, tokens / truncations / 429s reported in the metrics.
"""
import json
import random
import time
from typing import Dict, List, Optional, Tuple

from baselines.base import NASMethod

OPS_NB201 = ["none", "skip_connect", "conv_1x1", "conv_3x3", "avg_pool_3x3"]
MAX_RETRIES = 3
ARCH_LEN = 6  # NB201 cells have 6 edges


class LLMNASBaseline(NASMethod):
    def __init__(self, bench, dataset: str, seed: int,
                 model: str = "llama3.1:8b", host: str = "http://localhost:11434",
                 backend: str = "ollama",
                 groq_model: str = "openai/gpt-oss-20b",
                 groq_daily_token_budget: int = 190_000,
                 groq_daily_request_budget: int = 950,
                 groq_reasoning_effort: str = "low",
                 batch_size: int = 5,                 # architectures requested per call
                 top_k_context: int = 8,              # best-so-far architectures shown to the LLM
                 use_key_pool: bool = True,           # round-robin across multiple Groq accounts
                 ollama_fallback_model: Optional[str] = None,   # e.g. "llama3.1:8b"
                 ollama_fallback_host: str = "http://localhost:11434"):
        self.bench, self.dataset, self.seed = bench, dataset, seed
        self.backend = backend
        self.batch_size = max(1, batch_size)
        self.top_k_context = top_k_context
        random.seed(seed)  # NOTE: seeds the global `random` module
        self._metrics = {}

        # Top-K (accuracy, tokens) pairs seen so far, sorted best-first
        self._best_seen: List[Tuple[float, List[str]]] = []
        self._pending: List[List[str]] = []  # proposals generated but not yet evaluated

        # Diagnostics
        self._n_calls = 0
        self._n_parse_failures = 0
        self._n_fallback_archs = 0          # archs from local mutation (no LLM)
        self._n_ollama_fallback_archs = 0   # archs from the Ollama fallback tier
        self._budget_exhausted_logged = False

        # Optional local-Ollama tier, used only once Groq is unavailable
        self._ollama_fallback = None
        if ollama_fallback_model:
            try:
                import ollama as _ollama
                self._ollama_fallback = _ollama.Client(host=ollama_fallback_host)
                self._ollama_fallback_model = ollama_fallback_model
            except Exception as e:
                print(f"[llm_nas] ollama fallback requested but unavailable ({e}) — "
                      f"will degrade to local mutation instead once Groq keys run out.")

        self._chat = None
        if backend == "groq":
            from archtree import settings
            from utils.groq_client import GroqChat
            self._reasoning_effort = groq_reasoning_effort
            self.model = groq_model
            self._chat = GroqChat(groq_model, use_key_pool=use_key_pool,
                                  daily_token_budget=groq_daily_token_budget,
                                  daily_request_budget=groq_daily_request_budget,
                                  rpm_per_key=settings.GROQ_RPM_PER_KEY,
                                  tpm_per_key=settings.GROQ_TPM_PER_KEY,
                                  reasoning_effort=groq_reasoning_effort, tag="llm_nas")
            self._key_pool = self._chat.pool
            print(f"[llm_nas] Groq: {len(self._key_pool.slots)} key(s) - "
                  f"{self._key_pool.remaining_summary()}")
            self._client, self._limiter = None, None
        else:
            import ollama as _ollama
            self.model = model
            self._client = _ollama.Client(host=host)
            self._limiter = None
            self._key_pool = None

    # ── public API ────────────────────────────────────────────────────────

    def search(self, budget: int) -> Dict:
        from archtree.encoder import index_from_tokens_nb201

        history, best_acc = [], 0.0
        visited = set()  # unique architecture indices evaluated so far

        for i in range(budget):
            tokens = self._next_arch()
            idx = index_from_tokens_nb201(tokens) % self.bench.total_archs

            acc = self.bench.query(idx, self.dataset)
            visited.add(idx)
            best_acc = max(best_acc, acc)

            self._record_result(tokens, acc)

            history.append({"iter": i + 1, "arch": "|".join(tokens), "acc": acc,
                            "tcr": len(visited) / self.bench.total_archs})  # fraction of space covered

        self._metrics = {
            "eval_history": history, "best_acc": best_acc, "eval_count": budget,
            "llm_calls": self._n_calls, "parse_failures": self._n_parse_failures,
            "fallback_archs": self._n_fallback_archs,
            "ollama_fallback_archs": self._n_ollama_fallback_archs,
            "llm_tokens": self._chat.stats["total_tokens"] if self._chat else 0,
            "truncated_replies": self._chat.stats["truncated"] if self._chat else 0,
        }
        return self._metrics

    def get_metrics(self) -> Dict:
        return self._metrics

    # ── batching layer ────────────────────────────────────────────────────

    def _next_arch(self) -> List[str]:
        """Pop one proposal, generating a fresh batch first if none are pending."""
        if not self._pending:
            self._pending = self._generate_batch()
        return self._pending.pop()

    def _record_result(self, tokens: List[str], acc: float) -> None:
        """Keep only the top-K best (accuracy, tokens) pairs for the prompt context."""
        self._best_seen.append((acc, tokens))
        self._best_seen.sort(key=lambda x: -x[0])
        del self._best_seen[self.top_k_context:]

    def _generate_batch(self) -> List[List[str]]:
        prompt = self._build_prompt()

        if self.backend == "groq":
            archs = self._generate_batch_groq(prompt)
        else:
            archs = self._generate_batch_ollama(prompt, self._client, self.model)

        # Pad a short batch (partial parse or partial validity) with mutations
        while len(archs) < self.batch_size:
            archs.append(self._mutate_one())
            self._n_fallback_archs += 1

        return archs

    # ── local fallback (no API calls) ─────────────────────────────────────

    def _mutate_one(self) -> List[str]:
        """Change one random position of the best-known arch (random arch if none seen yet)."""
        if self._best_seen:
            base = list(self._best_seen[0][1])
        else:
            base = [random.choice(OPS_NB201) for _ in range(ARCH_LEN)]
        pos = random.randrange(len(base))
        base[pos] = random.choice(OPS_NB201)  # may pick the same op, i.e. no change
        return base

    def _mutate_fallback_batch(self) -> List[List[str]]:
        batch = [self._mutate_one() for _ in range(self.batch_size)]
        self._n_fallback_archs += len(batch)
        return batch

    # ── Groq / Ollama batched calls ───────────────────────────────────────

    def _schema(self):
        return {"type": "object", "additionalProperties": False, "required": ["archs"],
                "properties": {"archs": {"type": "array", "items": {
                    "type": "array", "items": {"type": "string", "enum": list(OPS_NB201)}}}}}

    def _generate_batch_groq(self, prompt: str) -> List[List[str]]:
        res = self._chat.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=45 * self.batch_size + 300,   # ~35 tok per arch + low-effort reasoning
            temperature=0.6, schema=self._schema(), schema_name="architectures")
        if res is None:
            if self._chat.exhausted and not self._budget_exhausted_logged:
                print(f"[llm_nas] all Groq keys exhausted after {self._n_calls} calls. "
                      f"{self._chat.remaining_summary()}")
                self._budget_exhausted_logged = True
            return self._next_tier_fallback(prompt)
        self._n_calls += 1
        archs, n_bad = self._parse_archs(res["text"])
        self._n_parse_failures += n_bad
        return archs

    def _next_tier_fallback(self, prompt: str) -> List[List[str]]:
        """Groq unavailable: try the local Ollama model, then fall back to pure mutation."""
        if self._ollama_fallback is not None:
            archs = self._generate_batch_ollama(
                prompt, self._ollama_fallback, self._ollama_fallback_model)
            if archs:
                self._n_ollama_fallback_archs += len(archs)
                return archs
        return self._mutate_fallback_batch()

    def _generate_batch_ollama(self, prompt: str, client, model: str) -> List[List[str]]:
        """Returns parsed archs, or [] if all retries fail (the caller picks the next fallback)."""
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.6},
                )
                self._n_calls += 1
                text = resp.message.content.strip()
                archs, n_bad = self._parse_archs(text)
                self._n_parse_failures += n_bad
                return archs
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    return []
                time.sleep(1)
        return []

    # ── prompt / parsing ──────────────────────────────────────────────────

    def _build_prompt(self) -> str:
        if self._best_seen:
            ctx_lines = "\n".join(
                f"{'|'.join(toks)} (acc={acc:.1f}%)" for acc, toks in self._best_seen
            )
        else:
            ctx_lines = "(none yet)"
        # Fixed instructions first (stable prefix -> Groq prompt caching), changing context last.
        return f"""You design 6-operation CNN cell architectures for NAS-Bench-201.
Valid operations: {OPS_NB201}
Each architecture is exactly 6 operations. Propose NEW architectures, different from each other and
from the ones listed below, that you expect to reach higher accuracy than the best ones so far.
Respond ONLY with JSON: {{"archs": [[6 operations], ...]}}

Number of architectures to propose: {self.batch_size}
Best architectures so far (by accuracy):
{ctx_lines}"""

    def _parse_archs(self, text: str) -> Tuple[List[List[str]], int]:
        """
        Parse the LLM reply into valid architectures.
        Returns (valid_archs, number_of_invalid_or_missing). Never raises: on a
        total parse failure it returns ([], batch_size) and the caller pads.
        """
        try:
            text = text.strip()
            # Strip a markdown code fence if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()
            # If there is extra text around the JSON, cut out the outermost {...}
            if not text.startswith("{"):
                start, end = text.rfind("{"), text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    text = text[start:end + 1]
            candidates = json.loads(text).get("archs", [])
        except Exception:
            return [], self.batch_size

        valid, n_bad = [], 0
        seen = set()  # drop duplicates within this batch
        for ops in candidates:
            key = tuple(ops) if isinstance(ops, list) else None
            if (key is not None and len(key) == ARCH_LEN and all(o in OPS_NB201 for o in key)
                    and key not in seen):
                valid.append(list(key))
                seen.add(key)
            else:
                n_bad += 1
        n_bad += max(0, self.batch_size - len(candidates))  # LLM returned fewer than requested
        return valid[:self.batch_size], n_bad