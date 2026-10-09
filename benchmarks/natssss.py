# benchmarks/natssss.py
"""
NATS-Bench size search space (Dong et al., TPAMI 2021) - third benchmark, added Oct 3, 2026.

The cell topology is fixed; each of 5 layers (stem, cell stage 1, residual block 1, cell stage 2,
residual block 2) picks a width from {8, 16, ..., 64}: 8^5 = 32,768 networks on cifar10, cifar100
and ImageNet16-120 (90-epoch schedule). Read from data/natssss_cache.json, built once by
tools/build_natsbench_sss_cache.py (no nats_bench / GPU needed afterwards).

Search-space hooks (the generic path of archtree/search.py, as for NB101):
    tokens = 5 width tokens "c8".."c64" (trie depth 5, 8 children per node), every string is valid
    index  = base-8 code of the 5 width choices, first layer most significant
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

CHANNELS = [8, 16, 24, 32, 40, 48, 56, 64]
TOKENS = [f"c{c}" for c in CHANNELS]
DEPTH = 5


def tokens_to_channels(tokens: Sequence[str]) -> Optional[List[int]]:
    try:
        chs = [int(str(t).lstrip("c")) for t in tokens]
    except ValueError:
        return None
    if len(chs) != DEPTH or any(c not in CHANNELS for c in chs):
        return None
    return chs


class NATSSSS:
    name = "natssss"
    datasets = ["cifar10", "cifar100", "ImageNet16-120"]
    total_archs = 8 ** DEPTH
    depth = DEPTH
    OPS = TOKENS

    def __init__(self, cfg: dict):
        self.cfg = cfg
        p = Path(cfg["paths"].get("natssss_cache", "data/natssss_cache.json"))
        if not p.is_absolute():
            from archtree.settings import PROJECT_ROOT
            p = PROJECT_ROOT / p
        self._cache_path = p
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        if not self._cache_path.exists():
            raise FileNotFoundError(
                f"[natssss] cache {self._cache_path} not found. Download and extract NATS-sss-v1_0-50262-simple.tar "
                f"into data\\ and run `python tools/build_natsbench_sss_cache.py` once (README §6).")
        with open(self._cache_path) as f:
            raw = json.load(f)
        raw.pop("_meta", None)
        self._e: Dict[int, dict] = {int(k): v for k, v in raw.items()}
        if len(self._e) != self.total_archs:
            raise RuntimeError(f"[natssss] cache has {len(self._e)} archs, expected {self.total_archs}")
        self._loaded = True
        print(f"[natssss] cache loaded ({len(self._e)} archs)")

    # ── search-space hooks ───────────────────────────────────────────────────
    def token_vocab(self) -> List[List[str]]:
        return [list(TOKENS) for _ in range(DEPTH)]

    def tokens_to_index(self, tokens: Sequence[str]) -> Optional[int]:
        chs = tokens_to_channels(tokens)
        if chs is None:
            return None
        i = 0
        for c in chs:
            i = i * 8 + CHANNELS.index(c)
        return i

    def index_to_tokens(self, idx: int) -> List[str]:
        idx, out = int(idx), []
        for _ in range(DEPTH):
            out.append(TOKENS[idx % 8])
            idx //= 8
        return out[::-1]

    def mutate_tokens(self, tokens, rng: random.Random) -> List[str]:
        """Change the width of one random layer to a different value."""
        t = list(tokens)
        i = rng.randrange(DEPTH)
        t[i] = rng.choice([x for x in TOKENS if x != t[i]])
        return t

    def random_unevaluated_index(self, evaluated: set, rng=random) -> int:
        for _ in range(10000):
            i = rng.randrange(self.total_archs)
            if i not in evaluated:
                return i
        raise RuntimeError("[natssss] could not find an unevaluated architecture")

    # ── metrics ──────────────────────────────────────────────────────────────
    def _entry(self, idx, dataset):
        self._load()
        return self._e[int(idx)][dataset]

    def query(self, arch_index: int, dataset: str = "cifar10") -> float:
        """Search signal: validation accuracy (0-100)."""
        return float(self._entry(arch_index, dataset)["valid"])

    def query_test(self, arch_index: int, dataset: str = "cifar10") -> float:
        """Report metric: test accuracy. Never use it to guide a search."""
        return float(self._entry(arch_index, dataset)["test"])

    def train_sec(self, arch_index: int, dataset: str = "cifar10") -> float:
        return float(self._entry(arch_index, dataset)["train_sec"])

    def params(self, arch_index: int) -> float:
        """Trainable parameters in millions (cifar10 head) - known without training."""
        self._load()
        return float(self._e[int(arch_index)]["params"])

    def query_by_tokens(self, tokens, dataset: str = "cifar10") -> float:
        return self.query(self.tokens_to_index(tokens), dataset)

    def random_sample(self, n: int = 1, replace: bool = False) -> List[int]:
        if replace or n > self.total_archs:
            return [random.randrange(self.total_archs) for _ in range(n)]
        return random.sample(range(self.total_archs), n)

    def arch_key_to_index(self, arch) -> Optional[int]:
        if isinstance(arch, int):
            return arch
        if isinstance(arch, str) and arch.isdigit():
            return int(arch)
        toks = arch if isinstance(arch, (list, tuple)) else str(arch).split("|")
        return self.tokens_to_index(toks)

    def best_from_history(self, history, dataset: str = "cifar10") -> dict:
        if not history:
            return {"best_acc": 0.0, "best_test_acc": 0.0, "best_arch": None}
        best = max(history, key=lambda e: e["acc"])
        idx = self.arch_key_to_index(best["arch"])
        return {"best_acc": float(best["acc"]), "best_test_acc": self.query_test(idx, dataset),
                "best_arch": "|".join(self.index_to_tokens(idx))}

    def oracle(self, dataset: str = "cifar10") -> dict:
        self._load()
        bv = max(range(self.total_archs), key=lambda i: self.query(i, dataset))
        return {"best_test": max(self.query_test(i, dataset) for i in range(self.total_archs)),
                "test_of_best_valid": self.query_test(bv, dataset)}
