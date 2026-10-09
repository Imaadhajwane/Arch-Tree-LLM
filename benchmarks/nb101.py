# benchmarks/nb101.py
# isort: skip_file
"""
NAS-Bench-101 (Ying et al., 2019) without TensorFlow (Oct 2026).

The official `nasbench` package needs TensorFlow 1.x, which does not install on a current
Windows/Python setup. Instead, tools/build_nb101_cache.py reads data/nasbench_only108.tfrecord
with a pure-Python reader once and writes data/nb101_cache.npz. This module loads that cache and
re-implements the three pieces of the official API the search needs, ported line by line:
    ModelSpec._prune          -> prune()
    graph_util.hash_module    -> hash_module()     (isomorphism-invariant MD5 hash)
    NASBench._check_spec      -> spec_hash() is None for out-of-space specs
The builder re-hashes every one of the 423,624 graphs with hash_module() and refuses to write the
cache unless every hash matches the one stored in the tfrecord, so the port is verified on the
real data, not just on unit tests.

Search space as tokens (26 positions, per-position vocabulary):
    positions 0-4   op of interior vertex 1..5   in OPS = conv3x3-bn-relu | conv1x1-bn-relu | maxpool3x3
    positions 5-25  upper-triangular adjacency bits of the 7x7 matrix ("0" | "1"), row-major
Ops come first so the trie shares statistics over op choices (which drive NB101 accuracy) before
it branches on wiring; validity is only decided by the adjacency bits at the bottom of the trie.
Many token sequences are invalid (disconnected or > 9 edges) and many map to the same graph
(isomorphic or with dangling vertices); tokens_to_index() returns None for invalid ones and the same
index for equivalent ones, so de-duplication is by graph, not by token string.

Metrics (108-epoch training, mean over the 3 repeats, deterministic like the NB201 cache):
    query(idx)       -> mean VALIDATION accuracy (search signal)
    query_test(idx)  -> mean TEST accuracy       (report metric)
    train_sec(idx)   -> mean training time (seconds, TPU v2) = cost of one real evaluation
    params(idx)      -> trainable parameters (zero-cost; used by the params gate)
Dataset: CIFAR-10 only.
"""
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

OPS = ["conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3"]   # official order = hash labels 0,1,2
N_VERTICES = 7
MAX_EDGES = 9
N_OPS = N_VERTICES - 2                                        # 5 interior vertices
EDGE_POS = [(i, j) for i in range(N_VERTICES) for j in range(i + 1, N_VERTICES)]   # 21
DEPTH = N_OPS + len(EDGE_POS)                                 # 26
BITS = ["0", "1"]
VOCAB: List[List[str]] = [OPS] * N_OPS + [BITS] * len(EDGE_POS)


# ── official-API ports (pure numpy) ───────────────────────────────────────────
def hash_module(matrix: np.ndarray, labeling: Sequence[int]) -> str:
    """Port of nasbench.lib.graph_util.hash_module (identical string formatting)."""
    vertices = np.shape(matrix)[0]
    in_edges = np.sum(matrix, axis=0).tolist()
    out_edges = np.sum(matrix, axis=1).tolist()
    assert len(in_edges) == len(out_edges) == len(labeling)
    hashes = list(zip(out_edges, in_edges, labeling))
    hashes = [hashlib.md5(str(h).encode("utf-8")).hexdigest() for h in hashes]
    for _ in range(vertices):
        new_hashes = []
        for v in range(vertices):
            in_neighbors = [hashes[w] for w in range(vertices) if matrix[w, v]]
            out_neighbors = [hashes[w] for w in range(vertices) if matrix[v, w]]
            new_hashes.append(hashlib.md5(
                ("".join(sorted(in_neighbors)) + "|" + "".join(sorted(out_neighbors)) + "|"
                 + hashes[v]).encode("utf-8")).hexdigest())
        hashes = new_hashes
    return hashlib.md5(str(sorted(hashes)).encode("utf-8")).hexdigest()


def prune(matrix: np.ndarray, ops: List[str]):
    """Port of ModelSpec._prune: drop vertices not on an input->output path.
    Returns (matrix, ops) or (None, None) if input and output are disconnected."""
    n = np.shape(matrix)[0]
    from_in, frontier = {0}, [0]
    while frontier:
        top = frontier.pop()
        for v in range(top + 1, n):
            if matrix[top, v] and v not in from_in:
                from_in.add(v)
                frontier.append(v)
    from_out, frontier = {n - 1}, [n - 1]
    while frontier:
        top = frontier.pop()
        for v in range(0, top):
            if matrix[v, top] and v not in from_out:
                from_out.add(v)
                frontier.append(v)
    extraneous = set(range(n)).difference(from_in.intersection(from_out))
    if len(extraneous) > n - 2:
        return None, None
    m = np.delete(np.delete(matrix, list(extraneous), axis=0), list(extraneous), axis=1)
    ops = list(ops)
    for index in sorted(extraneous, reverse=True):
        del ops[index]
    return m, ops


def spec_hash(matrix: np.ndarray, ops: List[str]) -> Optional[str]:
    """Hash of a raw (unpruned) spec, or None if it is outside the NB101 space (= not NASBench.is_valid)."""
    m, o = prune(np.asarray(matrix), ops)
    if m is None:
        return None
    if len(o) > N_VERTICES or int(np.sum(m)) > MAX_EDGES:
        return None
    if o[0] != "input" or o[-1] != "output" or any(op not in OPS for op in o[1:-1]):
        return None
    return hash_module(m, [-1] + [OPS.index(op) for op in o[1:-1]] + [-2])


def is_valid(matrix, ops) -> bool:
    return spec_hash(matrix, ops) is not None


# ── tokens <-> spec ───────────────────────────────────────────────────────────
def tokens_to_spec(tokens: Sequence[str]):
    """26 tokens -> (7x7 int matrix, 7 ops incl. input/output)."""
    if len(tokens) != DEPTH:
        raise ValueError(f"[nb101] expected {DEPTH} tokens, got {len(tokens)}")
    m = np.zeros((N_VERTICES, N_VERTICES), dtype=np.int64)
    for (i, j), b in zip(EDGE_POS, tokens[N_OPS:]):
        m[i, j] = 1 if str(b) == "1" else 0
    return m, ["input"] + list(tokens[:N_OPS]) + ["output"]


def spec_to_tokens(matrix: np.ndarray, ops: List[str]) -> List[str]:
    """(pruned or full) spec -> 26 tokens. A pruned graph with k < 7 vertices keeps its first k-1
    vertices in place and moves the output vertex to position 6; the unused interior vertices get
    no edges and the default op OPS[0] (they are pruned away again, so the graph is unchanged)."""
    matrix = np.asarray(matrix)
    k = matrix.shape[0]
    full = np.zeros((N_VERTICES, N_VERTICES), dtype=np.int64)
    pos = list(range(k - 1)) + [N_VERTICES - 1]          # old vertex -> new vertex
    for a in range(k):
        for b in range(k):
            if matrix[a, b]:
                full[pos[a], pos[b]] = 1
    interior = list(ops[1:-1]) + [OPS[0]] * (N_OPS - (k - 2))
    return list(interior[:N_OPS]) + [str(int(full[i, j])) for (i, j) in EDGE_POS]


def canonical_tokens(tokens: Sequence[str]) -> Optional[List[str]]:
    """Tokens of the pruned graph (same graph, dangling vertices removed). None if invalid.
    Used as the proxy-cache key so equivalent token strings share one proxy evaluation."""
    m, o = tokens_to_spec(tokens)
    if spec_hash(m, o) is None:
        return None
    pm, po = prune(m, o)
    return spec_to_tokens(pm, po)


def mutate_tokens(tokens: Sequence[str], rng: random.Random, mutation_rate: float = 1.0,
                  max_tries: int = 1000) -> List[str]:
    """Port of the NB101 regularized-evolution mutation (nasbench example notebook): flip each
    adjacency bit with prob rate/7, change each interior op with prob rate/5, retry until valid
    and different from the parent graph."""
    m0, o0 = tokens_to_spec(tokens)
    h0 = spec_hash(m0, o0)
    for _ in range(max_tries):
        m, o = m0.copy(), list(o0)
        for (i, j) in EDGE_POS:
            if rng.random() < mutation_rate / N_VERTICES:
                m[i, j] = 1 - m[i, j]
        for v in range(1, N_VERTICES - 1):
            if rng.random() < mutation_rate / N_OPS:
                o[v] = rng.choice([x for x in OPS if x != o[v]])
        h = spec_hash(m, o)
        if h is not None and h != h0:
            return list(o[1:-1]) + [str(int(m[i, j])) for (i, j) in EDGE_POS]
    return list(tokens)


# ── benchmark ─────────────────────────────────────────────────────────────────
class NB101:
    name = "nb101"
    datasets = ["cifar10"]
    total_archs = 423624          # overwritten by the cache size on load
    depth = DEPTH
    OPS = OPS

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.api_path = cfg["paths"].get("nb101", "data/nasbench_only108.tfrecord")
        self._cache_path = Path(cfg["paths"].get("nb101_cache", "data/nb101_cache.npz"))
        if not self._cache_path.is_absolute():
            from archtree.settings import PROJECT_ROOT
            self._cache_path = PROJECT_ROOT / self._cache_path
        self._loaded = False

    # ── load ─────────────────────────────────────────────────────────────────
    def _load(self):
        if self._loaded:
            return
        if not self._cache_path.exists():
            raise FileNotFoundError(
                f"[nb101] cache {self._cache_path} not found. Build it once with "
                f"`python tools/build_nb101_cache.py` (reads {self.api_path}, no TensorFlow needed).")
        import time
        t0 = time.time()
        z = np.load(self._cache_path, allow_pickle=False)
        self.hashes = [str(h) for h in z["hashes"]]
        self._adj = z["adj"]                 # [N, 7, 7] uint8, pruned graph (k x k in the top-left)
        self._ops = z["ops"]                 # [N, 5] int8 interior op ids, -1 = unused vertex
        self._nv = z["n_vertices"]           # [N] vertices of the pruned graph
        self._params = z["params"]
        self._valid = z["valid"].astype(np.float64).mean(1) * 100.0
        self._test = z["test"].astype(np.float64).mean(1) * 100.0
        self._time = z["train_time"].astype(np.float64).mean(1)
        self._h2i: Dict[str, int] = {h: i for i, h in enumerate(self.hashes)}
        self.total_archs = len(self.hashes)
        self._loaded = True
        print(f"[nb101] cache loaded in {time.time() - t0:.1f}s ({self.total_archs} archs)")

    # ── search-space hooks used by archtree/search.py and the baselines ──────
    def token_vocab(self) -> List[List[str]]:
        return VOCAB

    def tokens_to_index(self, tokens: Sequence[str]) -> Optional[int]:
        """Index of the graph, or None if the tokens are not a valid NB101 cell."""
        self._load()
        m, o = tokens_to_spec(tokens)
        h = spec_hash(m, o)
        if h is None:
            return None
        idx = self._h2i.get(h)
        if idx is None:
            raise KeyError(f"[nb101] valid spec with hash {h} missing from the cache "
                           "(cache built from a different tfrecord?)")
        return idx

    def index_to_tokens(self, idx: int) -> List[str]:
        self._load()
        idx = int(idx)
        k = int(self._nv[idx])
        m = self._adj[idx][:k, :k].astype(np.int64)
        ops = ["input"] + [OPS[int(o)] for o in self._ops[idx][:k - 2]] + ["output"]
        return spec_to_tokens(m, ops)

    def mutate_tokens(self, tokens, rng: random.Random) -> List[str]:
        return mutate_tokens(tokens, rng)

    def random_unevaluated_index(self, evaluated: set, rng=random) -> int:
        self._load()
        for _ in range(10000):
            i = rng.randrange(self.total_archs)
            if i not in evaluated:
                return i
        raise RuntimeError("[nb101] could not find an unevaluated architecture")

    # ── metrics ──────────────────────────────────────────────────────────────
    def query(self, arch_index: int, dataset: str = "cifar10") -> float:
        """Search signal: mean 108-epoch validation accuracy (0-100)."""
        self._load()
        return float(self._valid[int(arch_index)])

    def query_test(self, arch_index: int, dataset: str = "cifar10") -> float:
        """Report metric: mean 108-epoch test accuracy. Never use it to guide a search."""
        self._load()
        return float(self._test[int(arch_index)])

    def train_sec(self, arch_index: int, dataset: str = "cifar10") -> float:
        self._load()
        return float(self._time[int(arch_index)])

    def params(self, arch_index: int) -> int:
        self._load()
        return int(self._params[int(arch_index)])

    def query_by_tokens(self, tokens, dataset: str = "cifar10") -> float:
        idx = self.tokens_to_index(tokens)
        if idx is None:
            raise ValueError("[nb101] invalid cell")
        return self.query(idx, dataset)

    def random_sample(self, n: int = 1, replace: bool = False) -> List[int]:
        self._load()
        if replace or n > self.total_archs:
            return [random.randrange(self.total_archs) for _ in range(n)]
        return random.sample(range(self.total_archs), n)

    def arch_key_to_index(self, arch) -> Optional[int]:
        """eval_history 'arch' ('tok|tok|..' or token list or index) -> index."""
        if isinstance(arch, (int, np.integer)):
            return int(arch)
        if isinstance(arch, str) and arch.isdigit():
            return int(arch)
        toks = arch if isinstance(arch, (list, tuple)) else str(arch).split("|")
        return self.tokens_to_index(toks)

    def best_from_history(self, history, dataset: str = "cifar10") -> dict:
        """Best SEARCH metric in the history and the TEST accuracy of that architecture."""
        if not history:
            return {"best_acc": 0.0, "best_test_acc": 0.0, "best_arch": None}
        best = max(history, key=lambda e: e["acc"])
        idx = self.arch_key_to_index(best["arch"])
        return {"best_acc": float(best["acc"]), "best_test_acc": self.query_test(idx, dataset),
                "best_arch": self.hashes[idx]}

    def oracle(self) -> dict:
        """Best possible numbers (for regret): best test overall, and test of the best-valid arch."""
        self._load()
        bv = int(np.argmax(self._valid))
        return {"best_valid": float(self._valid[bv]), "test_of_best_valid": float(self._test[bv]),
                "best_test": float(self._test.max())}
