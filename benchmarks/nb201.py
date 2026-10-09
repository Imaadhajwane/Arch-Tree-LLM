# benchmarks/nb201.py
# isort: skip_file
"""
NAS-Bench-201 wrapper backed by the JSON cache built by tools/build_nb201_cache.py.

Cache v2 (Oct 2026, the one to use):
    {"_meta": {...}, "<canonical idx>": {"cifar10": {"valid": .., "test": .., "train_sec": ..}, ...}}
    query(idx, ds)      -> search signal  (config benchmarks.nb201_search_split, default "valid")
    query_test(idx, ds) -> report metric  (200-epoch test accuracy)
    train_sec(idx, ds)  -> cost of one real evaluation (seconds of training)

Cache v1 (Sep 2026, legacy): {"<idx>": {"cifar10": acc, ...}} = 12-epoch numbers, cifar10 = TRAIN
accuracy. Still loadable so old results can be reproduced, but a loud warning is printed and
query_test() returns the same number as query().

Index = canonical base-5 index of the 6 tokens (token 0 most significant), the same bijection as
archtree/encoder.py::index_from_tokens_nb201. index_to_tokens() decodes it mathematically.
"""

import json
import random
from pathlib import Path
from typing import List

OPS = ["none", "skip_connect", "conv_1x1", "conv_3x3", "avg_pool_3x3"]


def _decode(idx: int) -> List[str]:
    i, ops = int(idx) % 15625, []
    for _ in range(6):
        ops.append(OPS[i % 5])
        i //= 5
    return ops[::-1]


def _encode(tokens) -> int:
    i = 0
    for op in tokens:
        i = i * 5 + OPS.index(op)
    return i


def arch_key_to_index(arch) -> int:
    """eval_history 'arch' field -> canonical index. Accepts 'op|op|..' strings, token lists,
    or a digit string / int (baselines that log the index)."""
    if isinstance(arch, (list, tuple)):
        return _encode(arch)
    s = str(arch)
    if s.isdigit():
        return int(s) % 15625
    return _encode(s.split("|"))


def best_from_history(bench, history, dataset: str) -> dict:
    """The architecture with the best SEARCH metric (`acc`) in an eval_history, and its TEST
    accuracy. This is the honest number to report: selection uses validation, report uses test."""
    if not history:
        return {"best_acc": 0.0, "best_test_acc": 0.0, "best_arch": None}
    best = max(history, key=lambda e: e["acc"])
    idx = arch_key_to_index(best["arch"])
    return {"best_acc": float(best["acc"]), "best_test_acc": float(bench.query_test(idx, dataset)),
            "best_arch": "|".join(_decode(idx))}


class NB201:
    name = "nb201"
    datasets = ["cifar10", "cifar100", "ImageNet16-120"]
    total_archs = 15625
    OPS = OPS

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.api_path = cfg["paths"]["nb201"]
        self._cache = None
        self._version = None
        self._cache_path = Path(cfg["paths"].get("nb201_cache", "data/nb201_cache_v2.json"))
        self.search_split = cfg.get("benchmarks", {}).get("nb201_search_split", "valid")

    # ── load ─────────────────────────────────────────────────────────────────
    def _load(self):
        if self._cache is not None:
            return
        if not self._cache_path.exists():
            raise FileNotFoundError(
                f"[nb201] cache {self._cache_path} not found. Build it once with "
                f"`python tools/build_nb201_cache.py` (needs {self.api_path}).")
        import time
        t0 = time.time()
        with open(self._cache_path) as f:
            raw = json.load(f)
        meta = raw.pop("_meta", None)
        self._version = (meta or {}).get("version", 1)
        self._cache = {int(k): v for k, v in raw.items()}
        print(f"[nb201] cache v{self._version} loaded in {time.time() - t0:.2f}s ({len(self._cache)} archs)")
        if self._version < 2:
            print("[nb201] WARNING: v1 cache = 12-epoch results and cifar10 TRAIN accuracy. "
                  "Rebuild with tools/build_nb201_cache.py before producing paper numbers.")

    def _entry(self, idx, dataset):
        self._load()
        e = self._cache.get(int(idx) % self.total_archs)
        if e is None or dataset not in e:
            raise KeyError(f"[nb201] no cache entry for idx={idx} dataset={dataset}")
        return e[dataset]

    # ── public API ───────────────────────────────────────────────────────────
    def query(self, arch_index: int, dataset: str = "cifar10") -> float:
        """Search signal (validation accuracy with the v2 cache)."""
        e = self._entry(arch_index, dataset)
        return float(e[self.search_split]) if isinstance(e, dict) else float(e)

    def query_test(self, arch_index: int, dataset: str = "cifar10") -> float:
        """Report metric (test accuracy). Never use this to guide a search."""
        e = self._entry(arch_index, dataset)
        return float(e["test"]) if isinstance(e, dict) else float(e)

    def train_sec(self, arch_index: int, dataset: str = "cifar10"):
        e = self._entry(arch_index, dataset)
        return e.get("train_sec") if isinstance(e, dict) else None

    def query_by_tokens(self, tokens: List[str], dataset: str = "cifar10") -> float:
        return self.query(_encode(tokens), dataset)

    def index_to_tokens(self, idx: int) -> List[str]:
        return _decode(idx)

    def tokens_to_index(self, tokens: List[str]) -> int:
        return _encode(tokens)

    def mutate_tokens(self, tokens: List[str], rng: random.Random) -> List[str]:
        """Regularized-evolution mutation used in the NB201 paper: one random edge gets a
        different op (used by baselines/regularized_evolution.py)."""
        t = list(tokens)
        k = rng.randrange(len(t))
        t[k] = rng.choice([o for o in OPS if o != t[k]])
        return t

    def random_sample(self, n: int = 1, replace: bool = False) -> List[int]:
        """n architecture indices; WITHOUT replacement by default (Oct 2026 fix)."""
        if replace or n > self.total_archs:
            return [random.randint(0, self.total_archs - 1) for _ in range(n)]
        return random.sample(range(self.total_archs), n)
