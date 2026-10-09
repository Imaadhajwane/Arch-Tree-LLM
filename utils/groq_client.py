# utils/groq_client.py
"""
One place for every Groq chat call (Oct 2026). Used by archtree/llm_prior.py and baselines/llm_nas.py.

What it does to get the most out of the free-tier token budget:
  1. Key pool for everything. A single GROQ_API_KEY is just a 1-key pool, so single- and multi-key
     setups share the same per-key daily budget files and logic.
  2. Daily budget (TPD/RPD) per key, checked BEFORE the call (no call is sent that would be refused).
  3. Per-minute pacing per key on requests AND tokens (MinuteLimiter). The 8K TPM cap binds long
     before 30 RPM for batched calls; pacing avoids 429s instead of retrying after them.
  4. 429 handling by kind: per-minute -> wait `retry-after`, retry the same key; per-DAY -> mark the
     key exhausted until its reset and move to the next key at once (the old code retried the same
     dead key three times and then gave up, although other keys still had budget).
  5. Structured outputs: strict `json_schema` (constrained decoding, supported by gpt-oss-20b on
     Groq) so every answer parses - a failed parse is a whole call's tokens wasted. If the API
     rejects the schema the client downgrades once to `json_object` for the rest of the process.
  6. Exact accounting: the reserved estimate is replaced by `usage.total_tokens`; cached prompt
     tokens, truncated answers (finish_reason == "length") and 429s are counted in `stats`.
  7. SDK retries off (max_retries=0) so the SDK does not silently retry a dead key.
  8. Groq validates the schema AFTER generation: a rare off-enum value (e.g. "conv_3x1") gives a 400
     `json_validate_failed` with the answer in `failed_generation`. That answer is returned to the
     caller (finish_reason "schema_invalid"), whose parser keeps the valid items - the batch is not
     thrown away and replaced by fallback mutations (Oct 3, 2026).
Prompt caching on Groq (gpt-oss) needs a stable prefix: callers put the fixed instructions first
and the changing part last.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from loguru import logger

from utils.groq_key_pool import GroqKeyPool
from utils.rate_limiter import call_with_backoff, classify_rate_limit

HARMONY_OVERHEAD_TOKENS = 80      # chat template / system tokens Groq adds to every gpt-oss request


def _failed_generation(exc: Exception) -> Optional[str]:
    """Groq 400 `json_validate_failed`: the model DID answer (tokens spent), but one value broke the
    schema (e.g. an op name outside the enum). Groq returns that answer as `failed_generation`; the
    caller's parser keeps the valid parts. -> the text, or None if this is not such an error."""
    if "json_validate_failed" not in str(exc) and "failed_generation" not in str(exc):
        return None
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and isinstance(err.get("failed_generation"), str):
            return err["failed_generation"]
    m = str(exc)
    i = m.find("{")
    if i != -1:
        import ast
        try:
            d = ast.literal_eval(m[i:])
            err = d.get("error", d) if isinstance(d, dict) else {}
            if isinstance(err.get("failed_generation"), str):
                return err["failed_generation"]
        except (ValueError, SyntaxError):
            pass
    return None


def _supports_reasoning_effort(model: str) -> bool:
    """`reasoning_effort` is only accepted by Groq's reasoning models (gpt-oss); others 400 on it."""
    return "gpt-oss" in model.lower()


def _is_reasoning_rejection(exc: Exception) -> bool:
    m = str(exc).lower()
    return (getattr(exc, "status_code", None) == 400 or "400" in m) and "reasoning" in m


def _is_schema_rejection(exc: Exception) -> bool:
    m = str(exc).lower()
    return (getattr(exc, "status_code", None) == 400 or "400" in m) and (
        "response_format" in m or "json_schema" in m or "structured" in m or "not supported" in m)


class GroqChat:
    def __init__(self, model: str, use_key_pool: bool = True,
                 daily_token_budget: int = 190_000, daily_request_budget: int = 950,
                 rpm_per_key: int = 25, tpm_per_key: int = 7000,
                 reasoning_effort: Optional[str] = "low", tag: str = "groq"):
        self.model, self.tag = model, tag
        self.reasoning_effort = reasoning_effort if _supports_reasoning_effort(model) else None
        kw = dict(daily_token_budget=daily_token_budget, daily_request_budget=daily_request_budget,
                  rpm_per_key=rpm_per_key, tpm_per_key=tpm_per_key)
        if use_key_pool:
            self.pool = GroqKeyPool.from_env(**kw)
        else:
            key = os.environ.get("GROQ_API_KEY", "").strip()
            if not key:
                raise RuntimeError(f"[{tag}] GROQ_API_KEY not set in environment.")
            self.pool = GroqKeyPool([key], **kw)
        self.schema_mode = "json_schema"          # -> "json_object" after one rejection
        self.exhausted = False
        self.stats: Dict[str, int] = {k: 0 for k in (
            "calls", "prompt_tokens", "completion_tokens", "cached_tokens", "total_tokens",
            "truncated", "minute_429", "day_429", "errors", "schema_downgrades", "schema_invalid")}

    def estimate(self, messages: List[dict], max_tokens: int) -> int:
        chars = sum(len(m.get("content", "")) for m in messages)
        return chars // 4 + HARMONY_OVERHEAD_TOKENS + max_tokens

    def complete(self, messages: List[dict], max_tokens: int, temperature: float = 0.1,
                 schema: Optional[dict] = None, schema_name: str = "answer") -> Optional[dict]:
        """-> {"text", "finish_reason", "total_tokens"} or None (every key exhausted / call failed).
        Never raises for API problems; the caller treats None as a fallback."""
        est = self.estimate(messages, max_tokens)
        for _ in range(len(self.pool.slots) + 2):          # bounded: each pass rotates or returns
            slot = self.pool.acquire(est)
            if slot is None:
                self.exhausted = True
                return None
            kwargs = dict(model=self.model, messages=messages, temperature=temperature,
                          max_tokens=max_tokens)
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
            if schema is not None:
                kwargs["response_format"] = (
                    {"type": "json_schema",
                     "json_schema": {"name": schema_name, "strict": True, "schema": schema}}
                    if self.schema_mode == "json_schema" else {"type": "json_object"})
            try:
                resp = call_with_backoff(lambda: slot.client.chat.completions.create(**kwargs),
                                         max_retries=4)
            except Exception as e:                        # noqa: BLE001
                gen = _failed_generation(e) if schema is not None else None
                if gen is not None:                       # answered, one value off-schema
                    slot.record_actual(est)               # generation tokens were spent
                    self.stats["calls"] += 1
                    self.stats["schema_invalid"] += 1
                    logger.info(f"[{self.tag}] answer broke the schema in one value - "
                                f"keeping the valid part (failed_generation).")
                    return {"text": gen, "finish_reason": "schema_invalid", "total_tokens": 0}
                slot.record_actual(0)                     # refused calls consume no tokens
                kind, retry_after = classify_rate_limit(e)
                if kind == "day":
                    self.stats["day_429"] += 1
                    slot.mark_day_exhausted(retry_after)
                    logger.warning(f"[{self.tag}] a key hit Groq's DAILY limit - skipping it until reset "
                                   f"(~{(retry_after or 3600) / 60:.0f} min), trying the next key.")
                    continue
                if kind == "minute":
                    self.stats["minute_429"] += 1
                    logger.warning(f"[{self.tag}] per-minute limit persisted after retries: {e}")
                    return None
                if self.reasoning_effort and _is_reasoning_rejection(e):
                    self.reasoning_effort = None          # model has no reasoning setting
                    logger.warning(f"[{self.tag}] {self.model} rejects reasoning_effort; sending without it.")
                    continue
                if schema is not None and self.schema_mode == "json_schema" and _is_schema_rejection(e):
                    self.schema_mode = "json_object"
                    self.stats["schema_downgrades"] += 1
                    logger.warning(f"[{self.tag}] strict json_schema rejected ({e}); using json_object.")
                    continue
                self.stats["errors"] += 1
                logger.warning(f"[{self.tag}] call failed: {e}")
                return None

            u = getattr(resp, "usage", None)
            total = int(getattr(u, "total_tokens", 0) or 0)
            slot.record_actual(total if total else est)
            self.stats["calls"] += 1
            self.stats["total_tokens"] += total
            self.stats["prompt_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
            self.stats["completion_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)
            det = getattr(u, "prompt_tokens_details", None)
            self.stats["cached_tokens"] += int(getattr(det, "cached_tokens", 0) or 0) if det else 0
            choice = resp.choices[0]
            fr = getattr(choice, "finish_reason", None)
            if fr == "length":
                self.stats["truncated"] += 1
            return {"text": (choice.message.content or "").strip(), "finish_reason": fr,
                    "total_tokens": total}
        return None

    def remaining_summary(self) -> str:
        return self.pool.remaining_summary()
