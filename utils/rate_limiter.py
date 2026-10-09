# utils/rate_limiter.py
"""
Groq free-tier rate limiting.

- GroqRateLimiter: rolling 24h token + request budgets, persisted to disk (one usage file per API key).
- RPMLimiter:      in-process rolling 60s requests-per-minute cap.
- MinuteLimiter:   in-process rolling 60s requests AND tokens per minute (Oct 2026; the free tier's
                   8K TPM binds long before 30 RPM for batched calls, which caused avoidable 429s).
- classify_rate_limit: tells a per-minute 429 (wait retry-after, same key) from a per-DAY 429
                   (key is done until reset -> rotate to the next key instead of retrying it).
- call_with_backoff: retry helper for 429 / rate-limit errors (now honours retry-after).

Typical free-tier limits (gpt-oss-20b): 8K TPM / 200K TPD / 1000 RPD / 30 RPM.
Either the token or the request budget can bind first, so both are tracked.

Reserve/correct flow:
    ts = limiter.reserve(estimate, block=False)       # None -> budget exhausted
    resp = client.chat.completions.create(...)
    limiter.record_actual(ts, resp.usage.total_tokens)  # replace the estimate with the real count
"""
from __future__ import annotations
import hashlib
import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

from loguru import logger

T = TypeVar("T")
_USAGE_PATH = Path("data/groq_usage.json")
_USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Add a default log handler only if the application hasn't configured loguru itself
if not logger._core.handlers:
    logger.add(
        sys.stderr, level="INFO", colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>",
    )


class BudgetExhausted(Exception):
    """Optional: for callers that prefer raising over checking reserve()'s None return."""


class GroqRateLimiter:
    """Daily token + request budget for ONE API key, persisted to disk.

    Each key gets its own instance and usage file (data/groq_usage__{key_hash}.json)
    via get_for_key(). get() returns the legacy single-key instance ("default" hash).
    """
    _instances: Dict[str, "GroqRateLimiter"] = {}
    _registry_lock = threading.Lock()     # guards _instances only

    def __init__(self, key_hash: str = "default", daily_token_budget: int = 190_000,
                 window_hours: float = 24.0, daily_request_budget: int = 950):
        self.key_hash = key_hash
        self.daily_token_budget = daily_token_budget
        self.daily_request_budget = daily_request_budget
        self.window_seconds = window_hours * 3600
        self._usage_path = (_USAGE_PATH if key_hash == "default"
                            else _USAGE_PATH.with_name(f"groq_usage__{key_hash}.json"))
        # Per-instance lock: keys don't block each other. RLock so helpers can nest safely.
        self._lock = threading.RLock()
        self._events: List[Tuple[float, int]] = self._load()   # (timestamp, tokens) per call
        self._blocked_until = 0.0   # set when Groq itself answers "daily limit reached"

    # ── instance management ────────────────────────────────────────────────
    @classmethod
    def get(cls) -> "GroqRateLimiter":
        """Single-key singleton (used by llm_prior.py)."""
        return cls.get_for_key_hash("default")

    @classmethod
    def get_for_key(cls, api_key: str, **kwargs) -> "GroqRateLimiter":
        """Limiter for a specific API key; only a short hash of the key is ever used or stored."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:12]
        return cls.get_for_key_hash(key_hash, **kwargs)

    @classmethod
    def get_for_key_hash(cls, key_hash: str, **kwargs) -> "GroqRateLimiter":
        with cls._registry_lock:
            inst = cls._instances.get(key_hash)
            if inst is None:
                inst = cls._instances[key_hash] = cls(key_hash=key_hash, **kwargs)
            elif kwargs:
                inst.configure(**kwargs)    # a later caller may adjust budgets (e.g. from config)
            return inst

    def configure(self, daily_token_budget: Optional[int] = None,
                  daily_request_budget: Optional[int] = None,
                  window_hours: Optional[float] = None) -> None:
        with self._lock:
            if daily_token_budget is not None:
                self.daily_token_budget = daily_token_budget
            if daily_request_budget is not None:
                self.daily_request_budget = daily_request_budget
            if window_hours is not None:
                self.window_seconds = window_hours * 3600

    # ── persistence ────────────────────────────────────────────────────────
    def _load(self) -> List[Tuple[float, int]]:
        if not self._usage_path.exists():
            return []
        try:
            with open(self._usage_path) as f:
                return [(e["ts"], e["tokens"]) for e in json.load(f)]
        except Exception as e:
            logger.warning(f"[rate_limiter] could not read {self._usage_path} ({e}) — starting empty.")
            return []

    def _save(self) -> None:
        """Atomic write (temp file + rename) so a kill mid-write can't corrupt the usage file."""
        try:
            self._usage_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._usage_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump([{"ts": ts, "tokens": tok} for ts, tok in self._events], f)
            os.replace(tmp, self._usage_path)
        except Exception as e:
            logger.debug(f"[rate_limiter] usage save failed: {e}")   # tracking is best-effort

    # ── budget accounting (callers must hold self._lock) ───────────────────
    def _prune(self) -> None:
        """Drop events older than the rolling window."""
        cutoff = time.time() - self.window_seconds
        self._events = [(ts, tok) for ts, tok in self._events if ts >= cutoff]

    def _used(self) -> int:
        self._prune()
        return sum(tok for _, tok in self._events)

    def _request_count(self) -> int:
        self._prune()
        return len(self._events)

    def block_until(self, ts: float) -> None:
        """Groq said this key's DAILY limit is reached (its own count can differ from ours, e.g.
        usage from another machine): refuse reservations until `ts`."""
        with self._lock:
            self._blocked_until = max(self._blocked_until, ts)

    def is_blocked(self) -> bool:
        return time.time() < self._blocked_until

    def _try_reserve(self, estimated_tokens: int) -> Tuple[Optional[float], int, int]:
        """Record the call if it fits both budgets. Returns (timestamp or None, tokens used, requests used)."""
        used, n_reqs = self._used(), self._request_count()
        if time.time() < self._blocked_until:
            return None, used, n_reqs
        if (used + estimated_tokens <= self.daily_token_budget
                and n_reqs + 1 <= self.daily_request_budget):
            ts = time.time()
            self._events.append((ts, estimated_tokens))
            self._save()
            return ts, used, n_reqs
        return None, used, n_reqs

    # ── public API ─────────────────────────────────────────────────────────
    def reserve(self, estimated_tokens: int, block: bool = True,
                max_wait_sec: float = 300.0, quiet: bool = False) -> Optional[float]:
        """Reserve capacity for one call against BOTH daily budgets.

        block=True:  sleep until capacity frees up; returns the event timestamp.
        block=False: never sleep; returns the timestamp, or None if the budget is exhausted
                     (caller decides, e.g. fall back to a local model).
        quiet=True:  suppress the exhaustion warning (GroqKeyPool logs pool-level status itself).
        """
        with self._lock:
            ts, used, n_reqs = self._try_reserve(estimated_tokens)
            if ts is not None:
                return ts
            if not block:
                if not quiet:
                    logger.warning(f"[rate_limiter] budget exhausted "
                                   f"({used}/{self.daily_token_budget} tok, "
                                   f"{n_reqs}/{self.daily_request_budget} req) — "
                                   f"non-blocking caller notified, no sleep.")
                return None
            # A single call larger than the whole budget could never fit -> would sleep forever
            if estimated_tokens > self.daily_token_budget or self.daily_request_budget < 1:
                raise ValueError(f"estimated_tokens={estimated_tokens} can never fit the daily budget "
                                 f"({self.daily_token_budget} tok / {self.daily_request_budget} req).")

        # Blocking path: sleep until the oldest event leaves the window, then retry.
        # Each sleep is capped at max_wait_sec so KeyboardInterrupt lands promptly.
        while True:
            with self._lock:
                ts, used, n_reqs = self._try_reserve(estimated_tokens)
                if ts is not None:
                    return ts
                oldest_ts = min(t for t, _ in self._events) if self._events else time.time()
                wait = min(max(1.0, (oldest_ts + self.window_seconds) - time.time()), max_wait_sec)
            logger.warning(f"[rate_limiter] {used}/{self.daily_token_budget} tok, "
                           f"{n_reqs}/{self.daily_request_budget} req — sleep {wait:.0f}s")
            time.sleep(wait)

    def record_actual(self, ts: Optional[float], actual_tokens: int) -> None:
        """Replace a reserved estimate with the real usage (response.usage.total_tokens).
        No-op if ts is None (the reserve failed). Prevents budget drift over long runs."""
        if ts is None:
            return
        with self._lock:
            for i, (t, _) in enumerate(self._events):
                if t == ts:
                    self._events[i] = (t, actual_tokens)
                    self._save()
                    return

    def remaining_budget(self) -> int:
        with self._lock:
            return max(0, self.daily_token_budget - self._used())

    def remaining_requests(self) -> int:
        with self._lock:
            return max(0, self.daily_request_budget - self._request_count())

    def has_capacity(self, estimated_tokens: int) -> bool:
        """Non-mutating check — lets callers skip building a prompt when already out of budget."""
        with self._lock:
            return (self.daily_token_budget - self._used() >= estimated_tokens
                    and self.daily_request_budget - self._request_count() >= 1)

    @staticmethod
    def estimate_tokens(text: str, max_output_tokens: int = 200) -> int:
        """Rough estimate: ~4 chars per input token plus the max output allowance."""
        return max(1, len(text) // 4) + max_output_tokens


class RPMLimiter:
    """Rolling 60s requests-per-minute cap (Groq free tier allows 30/min).
    In-process only — not persisted; a fresh process starts below the cap."""
    def __init__(self, rpm: int = 25):      # default stays under Groq's 30
        self.rpm = rpm
        self._timestamps: List[float] = []
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until a request slot is free, then record this request."""
        with self._lock:
            now = time.time()
            self._timestamps = [t for t in self._timestamps if now - t < 60]
            if len(self._timestamps) >= self.rpm:
                sleep_for = 60 - (now - self._timestamps[0]) + 0.1
                logger.warning(f"[rate_limiter] RPM cap ({self.rpm}/min) — sleep {sleep_for:.1f}s")
                time.sleep(max(sleep_for, 0))
            self._timestamps.append(time.time())


class MinuteLimiter:
    """Rolling 60s cap on requests AND tokens for one key (Groq free tier gpt-oss-20b: 30 RPM,
    8K TPM). wait(est) blocks until both fit, then reserves `est`; correct() swaps in the real
    count. In-process only."""
    def __init__(self, rpm: int = 25, tpm: int = 7000):
        self.rpm, self.tpm = rpm, tpm
        self._events: List[List[float]] = []     # [timestamp, tokens]
        self._lock = threading.Lock()

    def wait(self, est_tokens: int = 0) -> List[float]:
        est = min(int(est_tokens), self.tpm)     # a single call may never exceed one minute's cap
        while True:
            with self._lock:
                now = time.time()
                self._events = [e for e in self._events if now - e[0] < 60]
                used = sum(e[1] for e in self._events)
                if len(self._events) < self.rpm and used + est <= self.tpm:
                    ev = [now, est]
                    self._events.append(ev)
                    return ev
                sleep_for = 60 - (now - self._events[0][0]) + 0.1 if self._events else 1.0
            logger.info(f"[rate_limiter] per-minute cap ({len(self._events)}/{self.rpm} req, "
                        f"{used}/{self.tpm} tok) - sleep {sleep_for:.1f}s")
            time.sleep(max(sleep_for, 0.1))

    def correct(self, ev: Optional[List[float]], actual_tokens: int) -> None:
        if ev is not None:
            with self._lock:
                ev[1] = int(actual_tokens)


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    """retry-after from the HTTP response, else 'try again in 1m2.5s' from the message."""
    import re
    resp = getattr(exc, "response", None)
    hdrs = getattr(resp, "headers", None)
    if hdrs is not None:
        try:
            v = hdrs.get("retry-after")
            if v is not None:
                return float(v)
        except (TypeError, ValueError):
            pass
    m = re.search(r"try again in ([0-9hms.]+)", str(exc))
    if not m:
        return None
    tot = 0.0
    for num, unit in re.findall(r"([\d.]+)(ms|h|m|s)", m.group(1).rstrip(".")):
        tot += float(num) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return tot


def classify_rate_limit(exc: Exception) -> Tuple[Optional[str], Optional[float]]:
    """-> (kind, retry_after_sec). kind: "day" (TPD/RPD reached: this key is done until reset),
    "minute" (TPM/RPM: wait and retry the same key), or None (not a rate-limit error)."""
    msg = str(exc)
    status = getattr(exc, "status_code", None)
    if status != 429 and "429" not in msg and "rate limit" not in msg.lower():
        return None, None
    low = msg.lower()
    kind = "day" if ("per day" in low or "(tpd)" in low or "(rpd)" in low) else "minute"
    return kind, _retry_after_seconds(exc)


def call_with_backoff(fn: Callable[[], T], max_retries: int = 5, base_wait: float = 3.0) -> T:
    """Call fn(), retrying with exponential backoff + jitter on 429 / rate-limit errors.
    Honours retry-after; a per-DAY 429 is re-raised at once (retrying the same key cannot help).
    Any other exception is re-raised immediately."""
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            kind, ra = classify_rate_limit(e)
            if kind is None or kind == "day":
                raise
            if attempt == max_retries - 1:
                break                        # out of retries: don't sleep before raising
            wait = (ra + 0.5) if ra is not None else base_wait * (2 ** attempt) + random.uniform(0, 1.5)
            logger.warning(f"[rate_limiter] 429 attempt {attempt + 1}/{max_retries}, sleep {wait:.1f}s: {e}")
            time.sleep(wait)
    raise last_exc