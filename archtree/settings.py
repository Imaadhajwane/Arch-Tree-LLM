# archtree/settings.py
"""
Single place where the search constants are read from config.yaml.

Before Oct 2026 these constants were hard-coded in search.py / ucb.py and several
config.yaml keys (ucb.C, trie.*, hw.*) were never read. Every module now imports
them from here, so config.yaml is the real source of truth.

If config.yaml is missing a key, the default below is used (these defaults are the
values the Sep-2026 runs actually used, except PRUNE_RULE, see trie.py).
"""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    with open(PROJECT_ROOT / "config.yaml") as _f:
        CFG = yaml.safe_load(_f) or {}
except FileNotFoundError:
    CFG = {}


def _get(*keys, default=None):
    node = CFG
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


# ── UCB ──────────────────────────────────────────────────────────────────────
C = float(_get("ucb", "C", default=1.414))                       # exploration constant
REWARD_MODE = _get("ucb", "reward", default="raw")              # "rank" (v2) | "raw" (v1, accuracy/100)
MUTATION_PROB = float(_get("search", "mutation_prob", default=0.0))   # v2: 0.5, v1: 0
MUTATION_TOURNAMENT = 5                                          # tournament size for the parent
MUTATION_WARMUP = 10                                             # real evals before mutation starts
HELDOUT_SEEDS = list(_get("search", "heldout_seeds", default=[]))
BACKUP_MODE_P12 = _get("ucb", "backup_mode", default="mean")     # phases 1/2
BACKUP_MODE_P34 = _get("ucb", "backup_mode_phase3", default="max")  # phase 3/4 + studies

# ── Trie pruning ─────────────────────────────────────────────────────────────
PRUNE_PERCENTILE = int(_get("trie", "prune_percentile", default=15))
PRUNE_EVERY_N = int(_get("trie", "prune_every_n", default=50))
PRUNE_RULE = _get("trie", "prune_rule", default="value")          # "value" | "visits" (legacy)
PRUNE_MIN_VISITS = int(_get("trie", "prune_min_visits", default=2))

# ── Proxy gate ───────────────────────────────────────────────────────────────
PROXY_THRESHOLD_PERCENTILE = int(_get("zc_proxy", "proxy_threshold_percentile", default=35))
WARMUP_REAL_EVALS = int(_get("zc_proxy", "warmup_real_evals", default=10))
CALIBRATION_MODE = _get("zc_proxy", "calibration_mode", default="supervised")
CALIBRATION_ARCHS = int(_get("zc_proxy", "calibration_archs", default=1500))

# ── LLM ──────────────────────────────────────────────────────────────────────
ALPHA_INIT = float(_get("llm", "alpha_init", default=0.0))
MAX_LLM_CALLS = int(_get("llm", "max_calls", default=60))
LLM_SHARED_CACHE = bool(_get("llm", "shared_cache", default=True))   # (model, prompt) answers reused across seeds/runs
GROQ_RPM_PER_KEY = int(_get("llm", "groq", "rpm_per_key", default=25))   # free tier: 30 RPM
GROQ_TPM_PER_KEY = int(_get("llm", "groq", "tpm_per_key", default=7000)) # free tier gpt-oss-20b: 8K TPM

# ── Hardware reward ──────────────────────────────────────────────────────────
LATENCY_PENALTY = float(_get("hw", "latency_penalty", default=0.05))
LATENCY_NORM_MS = float(_get("hw", "latency_norm_ms", default=50.0))

# ── Benchmark metric ─────────────────────────────────────────────────────────
# Search signal: "valid" (standard, no test leakage). Reported number: "test".
NB201_SEARCH_SPLIT = _get("benchmarks", "nb201_search_split", default="valid")
REPORT_METRIC = _get("benchmarks", "report_metric", default="best_test_acc")
