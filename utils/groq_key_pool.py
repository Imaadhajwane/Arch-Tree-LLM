# utils/groq_key_pool.py
"""
Round-robins Groq API calls across several keys, each tracked against its OWN
daily token/request budget (separate usage file per key hash, see GroqRateLimiter).
N accounts give N independent budgets.

Usage:
    from utils.groq_key_pool import GroqKeyPool
    pool = GroqKeyPool.from_env()          # GROQ_API_KEYS (comma-separated), GROQ_API_KEYS_FILE, or GROQ_API_KEY
    slot = pool.acquire(estimated_tokens=900)
    if slot is None:
        ...                                # every key exhausted -> caller falls back (e.g. Ollama)
    else:
        resp = slot.client.chat.completions.create(...)
        slot.record_actual(resp.usage.total_tokens)

Logging: exhaustion is reported once per key state change (OK <-> EXHAUSTED), plus a
periodic status table every `status_every` successful acquires. Not thread-safe
(round-robin index is unguarded); use one pool per thread if needed.
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from loguru import logger
import time

from utils.rate_limiter import GroqRateLimiter, MinuteLimiter


@dataclass
class GroqSlot:
    key: str
    client: "object"                  # groq.Client — typed loosely to avoid importing groq at module load
    limiter: GroqRateLimiter
    rpm: MinuteLimiter                # per-minute requests AND tokens (name kept for compatibility)
    last_ts: Optional[float] = None   # set by acquire(); identifies the reserved event
    last_minute_ev: Optional[list] = None
    exhausted: bool = False           # last-known state, so we only log on changes

    def record_actual(self, total_tokens: int) -> None:
        """Replace this slot's reserved estimate with the real token count from the response."""
        self.limiter.record_actual(self.last_ts, total_tokens)
        self.rpm.correct(self.last_minute_ev, total_tokens)

    def mark_day_exhausted(self, retry_after_sec: Optional[float]) -> None:
        """Groq answered 429 'per day' for this key: skip it until the reset (default 1 h)."""
        self.limiter.block_until(time.time() + (retry_after_sec if retry_after_sec else 3600.0))


def _split_keys(text: str) -> List[str]:
    """Split on commas/whitespace/newlines; skip empty entries and '#' comment lines."""
    keys = []
    for line in text.splitlines():
        if line.strip().startswith("#"):
            continue
        keys += [k for k in re.split(r"[,\s]+", line) if k]
    return keys


def _dedupe(keys: List[str]) -> List[str]:
    """Drop repeated keys (order kept). A duplicate would share one limiter across two slots
    and double-count the same budget."""
    return list(dict.fromkeys(keys))


class GroqKeyPool:
    def __init__(self, api_keys: List[str],
                 daily_token_budget: int = 190_000,
                 daily_request_budget: int = 950,
                 rpm_per_key: int = 25,
                 tpm_per_key: int = 7000,
                 status_every: int = 20):
        api_keys = _dedupe([k.strip() for k in api_keys if k and k.strip()])
        if not api_keys:
            raise ValueError("[groq_key_pool] no API keys provided.")
        import groq as _groq          # deferred so the module imports without groq installed

        self.slots: List[GroqSlot] = []
        for key in api_keys:
            limiter = GroqRateLimiter.get_for_key(
                key, daily_token_budget=daily_token_budget,
                daily_request_budget=daily_request_budget,
            )
            self.slots.append(GroqSlot(
                # max_retries=0: rate limits are handled here (rotate on a per-day 429), not by
                # the SDK's blind retries on the same key
                key=key, client=_groq.Client(api_key=key, max_retries=0),
                limiter=limiter, rpm=MinuteLimiter(rpm=rpm_per_key, tpm=tpm_per_key),
            ))
        self._rr_idx = 0               # next slot to try first (round-robin)
        self.status_every = status_every
        self._acquire_count = 0
        self._all_exhausted_logged = False
        logger.info(f"[groq_key_pool] {len(self.slots)} key(s) loaded.\n{self.status_table()}")

    @classmethod
    def from_env(cls, env_var: str = "GROQ_API_KEYS", **kwargs) -> "GroqKeyPool":
        """Collect keys from `env_var` (comma-separated) and the file named by GROQ_API_KEYS_FILE
        (one or more per line, '#' comments allowed). Falls back to the single GROQ_API_KEY so
        one-account setups work unchanged."""
        keys = _split_keys(os.environ.get(env_var, ""))
        key_file = os.environ.get("GROQ_API_KEYS_FILE", "").strip()
        if key_file and Path(key_file).is_file():
            keys += _split_keys(Path(key_file).read_text())
        if not keys:
            single = os.environ.get("GROQ_API_KEY", "").strip()
            if single:
                keys = [single]
        if not keys:
            raise RuntimeError(f"[groq_key_pool] none of {env_var}, GROQ_API_KEYS_FILE, GROQ_API_KEY is set.")
        return cls(keys, **kwargs)

    def acquire(self, estimated_tokens: int) -> Optional[GroqSlot]:
        """Scan round-robin for a key with headroom and reserve capacity on it (never sleeps
        on the daily budget). Returns None only when every key is exhausted."""
        n = len(self.slots)
        for i in range(n):
            idx = (self._rr_idx + i) % n
            slot = self.slots[idx]
            # quiet=True: the pool reports exhaustion itself, once per state change
            ts = slot.limiter.reserve(estimated_tokens, block=False, quiet=True)
            if ts is not None:
                if slot.exhausted:      # key recovered (window rolled over or budget was raised)
                    logger.info(f"[groq_key_pool] key{idx} back under budget "
                                f"({slot.limiter.remaining_requests()} req / "
                                f"{slot.limiter.remaining_budget()} tok left) — resuming use.")
                    slot.exhausted = False
                slot.last_ts = ts
                self._all_exhausted_logged = False
                slot.last_minute_ev = slot.rpm.wait(estimated_tokens)   # per-minute req + token cap
                self._rr_idx = (idx + 1) % n     # next call starts after the key just used
                self._acquire_count += 1
                if self._acquire_count % self.status_every == 0:
                    logger.info(f"[groq_key_pool] status check-in "
                                f"({self._acquire_count} calls so far):\n{self.status_table()}")
                return slot
            if not slot.exhausted:      # log only on the OK -> exhausted transition
                logger.warning(f"[groq_key_pool] key{idx} just hit its daily budget "
                               f"({slot.limiter.remaining_requests()} req / "
                               f"{slot.limiter.remaining_budget()} tok left) — rotating past it.")
                slot.exhausted = True

        if not self._all_exhausted_logged:      # once per exhaustion, not on every later call
            logger.warning(f"[groq_key_pool] ALL {n} keys exhausted:\n{self.status_table()}")
            self._all_exhausted_logged = True
        return None

    def status_table(self) -> str:
        """One line per key: OK/EXHAUSTED tag, requests used/total, tokens used/total."""
        lines = []
        for i, s in enumerate(self.slots):
            lim = s.limiter
            used_req = lim.daily_request_budget - lim.remaining_requests()
            used_tok = lim.daily_token_budget - lim.remaining_budget()
            tag = "EXHAUSTED" if s.exhausted else "OK"
            lines.append(f"  key{i}: {tag:<9} req {used_req:>4}/{lim.daily_request_budget} "
                         f"| tok {used_tok:>7}/{lim.daily_token_budget}")
        return "\n".join(lines)

    def remaining_summary(self) -> str:
        """Compact single-line summary (kept for backward compatibility)."""
        return " | ".join(f"key{i}: {s.limiter.remaining_requests()}req/{s.limiter.remaining_budget()}tok"
                          for i, s in enumerate(self.slots))