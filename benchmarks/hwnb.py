# benchmarks/hwnb.py
"""
HW-NAS-Bench wrapper: same interface as NB201 (query, query_test, index_to_tokens,
random_sample, total_archs) plus hw_query() for (accuracy, latency) pairs.

INDEX FIX (Oct 2026) — read this
--------------------------------
Everything in this project indexes architectures by the CANONICAL base-5 index of the six
tokens. HW-NAS-Bench indexes them in NAS-Bench-201's ORIGINAL order. The two orders differ for
15,624 of 15,625 architectures. The Sep-2026 code passed the canonical index straight to
HW-NAS-Bench, so every (accuracy, latency) pair combined the accuracy of one architecture with
the latency of another (Spearman between latency and the arch's own conv count: 0.001 instead of
0.624). Phase 4, the true Pareto fronts, the cross-device analysis and the TCR-H replay from
before this fix are invalid.

Now: on load, each HW-NAS-Bench entry's `config.arch_str` is converted to tokens and a
canonical->HW-NAS-Bench index map is built (and verified to be a bijection). hw_query() always
translates through it.

Negative-latency sentinel: HW-NAS-Bench returns a NEGATIVE latency for (arch, device) pairs it
did not measure (e.g. edgetpu). hw_query() raises HWMetricUnavailable for these; callers still
count the real evaluation but keep the arch out of Pareto / HV / IGD.

    git clone https://github.com/RICE-EIC/HW-NAS-Bench.git third_party/HW-NAS-Bench
"""
import os
import random
import sys
from typing import Dict, List, Tuple

from benchmarks.nb201 import NB201, _encode

HW_TARGETS = ["edgegpu", "raspi4", "edgetpu", "pixel3", "eyeriss", "fpga"]
_LATENCY_KEY = "{hw}_latency"
_OP_NAME_MAP = {"nor_conv_1x1": "conv_1x1", "nor_conv_3x3": "conv_3x3", "none": "none",
                "skip_connect": "skip_connect", "avg_pool_3x3": "avg_pool_3x3"}


class HWMetricUnavailable(Exception):
    """HW-NAS-Bench reports 'not supported / not measured' (negative latency) for this pair."""


def _arch_str_to_tokens(arch_str: str) -> List[str]:
    toks = []
    for part in arch_str.strip().split("+"):
        for op_node in part.strip("|").split("|"):
            op = op_node.split("~")[0].strip()
            if op:
                toks.append(_OP_NAME_MAP[op])
    return toks[:6]


class HWNASBench:
    name = "hwnb"
    datasets = ["cifar10"]
    total_archs = 15625

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.repo_path = cfg["paths"]["hwnb_repo"]
        self._api = None
        self._nb201 = NB201(cfg)              # accuracy + token <-> index
        self._canon_to_hw: Dict[str, List[int]] = {}

    # ── load ─────────────────────────────────────────────────────────────────
    def _load(self):
        if self._api is not None:
            return
        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)
        pickle_path = self._find_pickle()
        try:
            from hw_nas_bench_api import HWNASBenchAPI as HWAPI
            self._api = HWAPI(pickle_path, search_space="nasbench201")
            print(f"[hwnb] Loaded HW-NAS-Bench from {pickle_path}")
        except Exception as e:
            raise RuntimeError(f"[hwnb] Failed to load {pickle_path}: {e}")
        self._nb201._load()

    def _index_map(self, dataset: str) -> List[int]:
        """canonical index -> HW-NAS-Bench index, built from the benchmark's own arch strings."""
        if dataset not in self._canon_to_hw:
            self._load()
            cfgs = self._api.HW_metrics["nasbench201"][dataset]["config"]
            m = [-1] * self.total_archs
            for hw_idx, c in enumerate(cfgs):
                ci = _encode(_arch_str_to_tokens(c["arch_str"]))
                if m[ci] != -1:
                    raise RuntimeError(f"[hwnb] duplicate arch in HW-NAS-Bench config (canon {ci})")
                m[ci] = hw_idx
            if -1 in m:
                raise RuntimeError(f"[hwnb] index map incomplete for {dataset}")
            self._canon_to_hw[dataset] = m
            moved = sum(1 for ci, h in enumerate(m) if ci != h)
            print(f"[hwnb] index map for {dataset}: {moved}/{self.total_archs} indices differ from canonical")
        return self._canon_to_hw[dataset]

    def _find_pickle(self) -> str:
        for root, _, files in os.walk(self.repo_path):
            for f in files:
                if f.endswith(".pickle") and "HW-NAS-Bench" in f:
                    return os.path.join(root, f)
        for root, _, files in os.walk(self.repo_path):
            for f in files:
                if f.endswith(".pickle"):
                    return os.path.join(root, f)
        raise FileNotFoundError(f"[hwnb] no .pickle under {self.repo_path}")

    # ── NB201-compatible interface ───────────────────────────────────────────
    def index_to_tokens(self, idx: int) -> List[str]:
        return self._nb201.index_to_tokens(idx)

    def tokens_to_index(self, tokens: List[str]) -> int:
        return _encode(tokens)

    def random_sample(self, n: int = 1, replace: bool = False) -> List[int]:
        return self._nb201.random_sample(n, replace=replace)

    def query(self, arch_index: int, dataset: str = "cifar10") -> float:
        return self._nb201.query(arch_index, dataset)

    def query_test(self, arch_index: int, dataset: str = "cifar10") -> float:
        return self._nb201.query_test(arch_index, dataset)

    def latency(self, arch_index: int, hardware: str, dataset: str = "cifar10") -> float:
        """Latency (ms) of the CANONICAL arch `arch_index` on `hardware`."""
        self._load()
        hw_idx = self._index_map(dataset)[int(arch_index) % self.total_archs]
        try:
            metrics = self._api.query_by_index(hw_idx, dataset)
            latency = float(metrics[_LATENCY_KEY.format(hw=hardware)])
        except Exception as e:
            raise RuntimeError(f"[hwnb] latency lookup failed idx={arch_index} hw={hardware}: {e}")
        if latency < 0:
            raise HWMetricUnavailable(f"idx={arch_index} hardware={hardware}: sentinel latency {latency}")
        return latency

    def hw_query(self, arch_index: int, hardware: str = "edgegpu",
                 dataset: str = "cifar10") -> Tuple[float, float]:
        """(search accuracy, latency_ms) for the canonical arch. Raises HWMetricUnavailable."""
        acc = self.query(arch_index, dataset)
        return acc, self.latency(arch_index, hardware, dataset)

    def hw_query_by_tokens(self, tokens: List[str], hardware: str = "edgegpu",
                           dataset: str = "cifar10") -> Tuple[float, float]:
        return self.hw_query(_encode(tokens), hardware, dataset)

    def check_alignment(self, dataset: str = "cifar10", hardware: str = "edgegpu") -> float:
        """Sanity check: Spearman(latency, #conv ops of the SAME arch). ~0.6 when aligned, ~0 when not."""
        from scipy.stats import spearmanr
        lats, convs = [], []
        for i in range(self.total_archs):
            try:
                lats.append(self.latency(i, hardware, dataset))
            except HWMetricUnavailable:
                continue
            convs.append(sum(t.startswith("conv") for t in self.index_to_tokens(i)))
        return float(spearmanr(lats, convs)[0])
