# archtree/zc_proxy.py
"""
Zero-cost proxy ensemble (C8) for NAS-Bench-201 style cells.

Components (all computed on an untrained ProxyNet):
    E - SVD entropy of the final feature map
    S - SynFlow (data-free saliency)
    T - NASWOT (log-det of the activation-pattern kernel), averaged over several batches
    C - complexity penalty (params + weighted op cost)

Caching:
    Only the RAW components {E, S, T, C} are cached (outputs/_zc_raw_cache.json),
    shared process-wide. The final score is aggregated on read from each
    instance's own weights / norm_stats, so differently calibrated ensembles
    (per dataset, or the equal-weight baselines) never see each other's numbers.
    The old outputs/_zc_score_cache.json is no longer read and can be deleted.

Helper API for the proxy-composition ablation:
    raw(tokens), fit_subset(...), from_fit(...), flush()

Calibration modes (Oct 2026, config zc_proxy.calibration_mode):
    supervised    z-score stats + least-squares weights fit on `n` architectures' TRUE
                  accuracies. Those n labels are real evaluations and must be counted as cost
                  (blob["labels_used"]; tools/analyze_proxy_cost.py does this).
    unsupervised  z-score stats from `n` UNLABELED architectures, fixed weights
                  E = S = T = 1/3, C = 0 (the three standard training-free proxies, equal vote).
                  Zero labels. Used by the zico-style / rz_nas-style baselines and as the fair
                  "no extra information" arm of experiments/run_gated_baselines.py.

NAS-Bench-101 (Oct 2026): 26-token NB101 cells get NB101ProxyNet (a proxy-size port of the official
NB101 network) and are cached under their pruned-graph tokens. The raw cache is shared; NB201 keys
(6 tokens) and NB101 keys (26 tokens) never collide.

NATS-Bench size space (Oct 3, 2026): 5 width tokens "c8".."c64" get SSSProxyNet - the NATS SSS
network with one cell per stage: stem (w1) -> fixed cell (w1->w2) -> residual block /2 (w2->w3)
-> fixed cell (w3->w4) -> residual block /2 (w4->w5) -> fixed cell (w5) -> GAP -> FC, the fixed
cell being the NATS SSS topology. Keys have 5 tokens, so they never collide with NB201 / NB101.

Known limitation (documented, not changed): the proxy network always sees CIFAR-10 images and
has a 10-class head, also when the search target is cifar100 or ImageNet16-120.
"""
import atexit
import json
import os
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import archtree.storage as storage

# NB201 cell edges as (destination node, source node)
EDGES = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]

N_NASWOT_BATCHES = 3      # NASWOT is averaged over this many random data batches
NASWOT_BATCH_SIZE = 256   # larger N gives a better-conditioned kernel determinant
PROXY_CHANNELS = 32       # wider untrained net gives a stronger proxy signal
ALLOW_SYNTHETIC_DATA = False  # if False, fail loudly when CIFAR-10 can't be loaded

# --------------------------------------------------------------------------
# Persistent raw-component cache
# --------------------------------------------------------------------------
_RAW_CACHE_PATH = storage.PROJECT_ROOT / "outputs" / "_zc_raw_cache.json"   # absolute: cwd-independent
_RAW_KEYS = ("E", "S", "T", "C")
SYNFLOW_VERSION = 2   # raw-cache entries without "sv" == 2 hold the broken train-mode-BN SynFlow:
                      # their S is recomputed on first use (E, T, C stay valid)
_RAW_FLUSH_EVERY = 25  # write to disk after this many new entries ... (also flushed at exit)
_RAW_FLUSH_MIN_SEC = 120.0  # ... but at most every 2 min: the whole JSON (6-15 MB) is rewritten each time
_RAW_LAST_FLUSH = 0.0
_SHARED_RAW: Optional[Dict[Tuple[str, ...], Dict[str, float]]] = None
_RAW_DIRTY = 0  # new entries not yet written to disk
_ATEXIT_REGISTERED = False


def _shared_raw_cache() -> Dict[Tuple[str, ...], Dict[str, float]]:
    """Return the process-wide raw cache, loading it from disk on first use."""
    global _SHARED_RAW, _ATEXIT_REGISTERED
    if _SHARED_RAW is None:
        _SHARED_RAW = {}
        if _RAW_CACHE_PATH.exists():
            try:
                with open(_RAW_CACHE_PATH) as f:
                    raw = json.load(f)
                # JSON keys are "tok1|tok2|..." strings; convert back to tuples
                _SHARED_RAW = {tuple(k.split("|")): v for k, v in raw.items()}
            except (json.JSONDecodeError, OSError) as e:
                print(f"[zc_proxy] WARNING: could not read raw cache ({e}) — starting empty.")
        if not _ATEXIT_REGISTERED:
            atexit.register(_flush_shared_raw_cache)
            _ATEXIT_REGISTERED = True
    return _SHARED_RAW


class _FileLock:
    """Tiny cross-process lock (O_CREAT | O_EXCL lock file) for the read-merge-write of the raw
    cache, so parallel workers never overwrite each other's new entries (Oct 4, 2026)."""
    def __init__(self, path, timeout=120.0, stale=300.0):
        self.path, self.timeout, self.stale = str(path), timeout, stale

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.path) > self.stale:
                        os.remove(self.path)          # left behind by a killed process
                        continue
                except OSError:
                    pass
                if time.time() - t0 > self.timeout:
                    raise TimeoutError(f"[zc_proxy] could not lock {self.path}")
                time.sleep(0.05)

    def __exit__(self, *exc):
        os.close(self.fd)
        try:
            os.remove(self.path)
        except OSError:
            pass


def _flush_shared_raw_cache() -> None:
    """Atomically write the raw cache (temp file + replace), retrying on transient errors."""
    global _RAW_DIRTY
    if _SHARED_RAW is None or _RAW_DIRTY == 0:
        return
    with _FileLock(_RAW_CACHE_PATH.with_suffix(".lock")):
        _flush_locked()


def _flush_locked() -> None:
    global _RAW_DIRTY
    try:
        _RAW_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Oct 4, 2026: merge what other processes wrote meanwhile (parallel workers share this
        # file) - union of both, our entries win; nothing another worker computed is lost.
        if _RAW_CACHE_PATH.exists():
            try:
                with open(_RAW_CACHE_PATH) as f:
                    for k, v in json.load(f).items():
                        _SHARED_RAW.setdefault(tuple(k.split("|")), v)
            except (json.JSONDecodeError, OSError):
                pass
        serializable = {"|".join(k): v for k, v in _SHARED_RAW.items()}
        tmp_path = _RAW_CACHE_PATH.with_suffix(f".{os.getpid()}.tmp")   # per-process temp file
        with open(tmp_path, "w") as f:
            json.dump(serializable, f)

        # On Windows, replace() can transiently fail (antivirus, OneDrive, an editor
        # holding the file), so retry with a short backoff.
        last_err = None
        for attempt in range(5):
            try:
                tmp_path.replace(_RAW_CACHE_PATH)
                _RAW_DIRTY = 0
                return
            except OSError as e:
                last_err = e
                time.sleep(0.05 * (attempt + 1))
        raise last_err
    except OSError as e:
        print(f"[zc_proxy] WARNING: could not write raw cache ({e}) — continuing without persist.")


if not torch.cuda.is_available():
    raise RuntimeError("[zc_proxy] GPU REQUIRED — no CUDA device found.")
DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------------
# Proxy network
# --------------------------------------------------------------------------

class _Zero(nn.Module):
    """The 'none' op: outputs zeros with the input's shape."""
    def forward(self, x):
        return x * 0.0


def _op(name: str, C: int) -> nn.Module:
    """Build one NB201 edge operation with C channels."""
    if name == "none":
        return _Zero()
    if name == "skip_connect":
        return nn.Identity()
    if name == "avg_pool_3x3":
        return nn.AvgPool2d(3, stride=1, padding=1)
    if name == "conv_1x1":
        return nn.Sequential(nn.Conv2d(C, C, 1, bias=False), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
    if name == "conv_3x3":
        return nn.Sequential(nn.Conv2d(C, C, 3, padding=1, bias=False), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
    raise ValueError(f"unknown op {name}")


class NB201Cell(nn.Module):
    """4-node DAG cell: node i is the sum of op(node j) over all j < i."""
    def __init__(self, tokens: List[str], C: int = PROXY_CHANNELS):
        super().__init__()
        self.ops = nn.ModuleDict({f"{d}_{s}": _op(tokens[i], C) for i, (d, s) in enumerate(EDGES)})

    def forward(self, x) -> List[torch.Tensor]:
        nodes = [x]
        for i in range(1, 4):
            nodes.append(sum(self.ops[f"{i}_{j}"](nodes[j]) for j in range(i)))
        return nodes


class ProxyNet(nn.Module):
    """Stem -> 3 stacked cells -> GAP -> FC. Returns (logits, list of intermediate activations)."""
    def __init__(self, tokens: List[str], C: int = PROXY_CHANNELS, n_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, C, 3, padding=1, bias=False), nn.BatchNorm2d(C))
        self.cells = nn.ModuleList([NB201Cell(tokens, C) for _ in range(3)])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(C, n_classes)

    def forward(self, x):
        x = self.stem(x)
        acts = [x]
        for cell in self.cells:
            nodes = cell(x)
            x = nodes[-1]  # last node is the cell output
            acts.extend(nodes[1:])
        out = self.fc(self.gap(x).flatten(1))
        return out, acts


# --------------------------------------------------------------------------
# NAS-Bench-101 proxy network (Oct 2026)
# --------------------------------------------------------------------------
# Proxy-size version of the NB101 network (Ying et al. 2019, nasbench/lib/model_builder.py):
# stem conv3x3 -> 3 stacks with max-pool downsampling and channel doubling, 1 cell per stack
# (the real net uses 128 channels and 3 cells per stack; the proxy only needs the cell's
# inductive bias). Cell = the official build_module(): projections from the input vertex, channel
# split at the output concat (compute_vertex_channels), truncation of wider predecessors.
NB101_STEM_CHANNELS = 16


def _nb101_vertex_channels(cin: int, cout: int, matrix) -> List[int]:
    """Port of nasbench model_builder.compute_vertex_channels."""
    n = len(matrix)
    vc = [0] * n
    vc[0], vc[n - 1] = cin, cout
    if n == 2:
        return vc
    in_degree = np.sum(matrix[1:], axis=0)
    interior, correction = cout // in_degree[n - 1], cout % in_degree[n - 1]
    for v in range(1, n - 1):
        if matrix[v, n - 1]:
            vc[v] = int(interior)
            if correction:
                vc[v] += 1
                correction -= 1
    for v in range(n - 3, 0, -1):
        if not matrix[v, n - 1]:
            for dst in range(v + 1, n - 1):
                if matrix[v, dst]:
                    vc[v] = max(vc[v], vc[dst])
    return vc


def _cbr(cin: int, cout: int, k: int) -> nn.Module:
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class NB101Cell(nn.Module):
    def __init__(self, matrix, ops: List[str], cin: int, cout: int):
        super().__init__()
        self.m, self.n = matrix, len(matrix)
        self.vc = _nb101_vertex_channels(cin, cout, matrix)
        self.proj = nn.ModuleDict()
        self.vops = nn.ModuleDict()
        for t in range(1, self.n - 1):
            if matrix[0, t]:
                self.proj[str(t)] = _cbr(cin, self.vc[t], 1)
            c = self.vc[t]
            self.vops[str(t)] = {"conv3x3-bn-relu": lambda: _cbr(c, c, 3),
                                 "conv1x1-bn-relu": lambda: _cbr(c, c, 1),
                                 "maxpool3x3": lambda: nn.MaxPool2d(3, stride=1, padding=1)}[ops[t]]()
        if matrix[0, self.n - 1]:
            self.proj[str(self.n - 1)] = _cbr(cin, cout, 1)

    def forward(self, x):
        n, m, vc = self.n, self.m, self.vc
        tensors, concat = [x], []
        for t in range(1, n - 1):
            add_in = [tensors[s][:, :vc[t]] for s in range(1, t) if m[s, t]]
            if m[0, t]:
                add_in.append(self.proj[str(t)](x))
            v = self.vops[str(t)](sum(add_in))
            tensors.append(v)
            if m[t, n - 1]:
                concat.append(v)
        if not concat:
            out = self.proj[str(n - 1)](x)
        else:
            out = torch.cat(concat, 1) if len(concat) > 1 else concat[0]
            if m[0, n - 1]:
                out = out + self.proj[str(n - 1)](x)
        return out, tensors[1:] + [out]


class NB101ProxyNet(nn.Module):
    """26 NB101 tokens -> (logits, list of intermediate activations), like ProxyNet."""
    def __init__(self, tokens: List[str], C: int = NB101_STEM_CHANNELS, n_classes: int = 10):
        super().__init__()
        from benchmarks.nb101 import prune, tokens_to_spec
        matrix, ops = prune(*tokens_to_spec(tokens))
        if matrix is None:
            raise ValueError("[zc_proxy] invalid NB101 cell")
        self.stem = _cbr(3, C, 3)
        self.down = nn.ModuleList()
        self.cells = nn.ModuleList()
        cin = C
        for s in range(3):
            cout = C * (2 ** s)
            self.down.append(nn.MaxPool2d(2, 2) if s > 0 else nn.Identity())
            self.cells.append(NB101Cell(matrix, ops, cin, cout))
            cin = cout
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, n_classes)
        self.interior_ops = ops[1:-1]

    def forward(self, x):
        x = self.stem(x)
        acts = [x]
        for down, cell in zip(self.down, self.cells):
            x, nodes = cell(down(x))
            acts.extend(nodes)
        return self.fc(self.gap(x).flatten(1)), acts


# --------------------------------------------------------------------------
# NATS-Bench size-space proxy network (Oct 3, 2026)
# --------------------------------------------------------------------------
# Fixed cell of NATS-Bench SSS: |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|
SSS_CELL = {(1, 0): "conv_3x3", (2, 0): "conv_3x3", (2, 1): "conv_3x3",
            (3, 0): "skip_connect", (3, 1): "conv_3x3", (3, 2): "conv_3x3"}


def _sss_op(name: str, cin: int, cout: int) -> nn.Module:
    if name == "conv_3x3":
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                             nn.ReLU(inplace=True))
    if name == "skip_connect":
        if cin == cout:
            return nn.Identity()
        return nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
    raise ValueError(name)


class SSSCell(nn.Module):
    """The fixed NATS SSS cell; edges leaving the input node map cin -> cout."""
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.ops = nn.ModuleDict({f"{d}_{s}": _sss_op(op, cin if s == 0 else cout, cout)
                                  for (d, s), op in SSS_CELL.items()})

    def forward(self, x) -> List[torch.Tensor]:
        nodes = [x]
        for i in range(1, 4):
            nodes.append(sum(self.ops[f"{i}_{j}"](nodes[j]) for j in range(i)))
        return nodes


class _ResBlock(nn.Module):
    """NATS / NB201 residual reduction block (stride 2)."""
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.a = nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                               nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.b = nn.Sequential(nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout))
        self.short = nn.Sequential(nn.AvgPool2d(2, 2), nn.Conv2d(cin, cout, 1, bias=False))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.b(self.a(x)) + self.short(x))


class SSSProxyNet(nn.Module):
    """5 width tokens -> (logits, list of intermediate activations), like ProxyNet."""
    def __init__(self, tokens: List[str], n_classes: int = 10):
        super().__init__()
        from benchmarks.natssss import tokens_to_channels
        w = tokens_to_channels(tokens)
        if w is None:
            raise ValueError(f"[zc_proxy] invalid NATS-SSS tokens {tokens}")
        self.stem = nn.Sequential(nn.Conv2d(3, w[0], 3, padding=1, bias=False), nn.BatchNorm2d(w[0]))
        self.cell1, self.res1 = SSSCell(w[0], w[1]), _ResBlock(w[1], w[2])
        self.cell2, self.res2 = SSSCell(w[2], w[3]), _ResBlock(w[3], w[4])
        self.cell3 = SSSCell(w[4], w[4])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(w[4], n_classes)

    def forward(self, x):
        x = self.stem(x)
        acts = [x]
        for cell, res in ((self.cell1, self.res1), (self.cell2, self.res2), (self.cell3, None)):
            nodes = cell(x)
            acts.extend(nodes[1:])
            x = nodes[-1]
            if res is not None:
                x = res(x)
                acts.append(x)
        return self.fc(self.gap(x).flatten(1)), acts


def _make_net(tokens: List[str]) -> nn.Module:
    """NB201 (6 tokens) -> ProxyNet, NATS-SSS (5 width tokens) -> SSSProxyNet,
    NB101 (26 tokens) -> NB101ProxyNet."""
    if len(tokens) == 6:
        return ProxyNet(tokens)
    if len(tokens) == 5:
        return SSSProxyNet(tokens)
    return NB101ProxyNet(tokens)


def _cache_key(tokens: List[str]) -> Tuple[str, ...]:
    """NB201 / NATS-SSS: the tokens. NB101: the tokens of the PRUNED graph, so equivalent
    spellings of one cell share a cache entry (and get the same network)."""
    if len(tokens) in (5, 6):
        return tuple(tokens)
    from benchmarks.nb101 import canonical_tokens
    c = canonical_tokens(tokens)
    if c is None:
        raise ValueError("[zc_proxy] invalid NB101 cell")
    return tuple(c)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

_MB_CACHE: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}


def get_mini_batch(batch_size: int = NASWOT_BATCH_SIZE, variant: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a cached CIFAR-10 mini-batch on the GPU. Different `variant`s are different batches."""
    if variant in _MB_CACHE:
        return _MB_CACHE[variant]
    try:
        import torchvision
        import torchvision.transforms as T
        ds = torchvision.datasets.CIFAR10(root=str(storage.PROJECT_ROOT / "data" / "cifar10"), train=True, download=True,
                                          transform=T.Compose([T.ToTensor()]))
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        x, y = next(iter(loader))
        print(f"[zc_proxy] using real CIFAR-10 mini-batch (variant={variant})")
    except Exception as e:
        if not ALLOW_SYNTHETIC_DATA:
            raise RuntimeError(f"[zc_proxy] CIFAR-10 unavailable ({e}). Synthetic fallback disabled.")
        print(f"[zc_proxy] CIFAR-10 unavailable ({e}) — using synthetic batch (ALLOWED)")
        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.randint(0, 10, (batch_size,))
    _MB_CACHE[variant] = (x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True))
    return _MB_CACHE[variant]


# --------------------------------------------------------------------------
# Proxy scores
# --------------------------------------------------------------------------

class _NASWOTHooks:
    """Accumulates the NASWOT Hamming kernel over the ReLU layers of one forward pass.

    Oct 2026 (memory): the kernel is summed layer by layer, K += c c^T + (1-c)(1-c)^T, instead of
    storing every layer's binary codes and concatenating them. The kernel entries are integer counts
    (< 2^24), so the sum is exact: raw E/S/T/C are bit-identical to before (checked on 8 archs) and
    outputs/_zc_raw_cache.json stays valid. Peak GPU memory drops from
    ~1.5 GB (batch 256, 18 conv layers: codes + a float copy + 1-K) to a few tens of MB, which
    matters on 4 GB laptop GPUs."""
    def __init__(self, net: nn.Module):
        self.K: Optional[torch.Tensor] = None
        self.n_layers = 0
        self.handles = [m.register_forward_hook(self._hook)
                        for m in net.modules() if isinstance(m, nn.ReLU)]

    def _hook(self, module, inp, out):
        c = (out > 0).flatten(1).float()
        k = c @ c.t()
        c = 1.0 - c
        k += c @ c.t()
        self.K = k if self.K is None else self.K + k
        self.n_layers += 1

    def remove(self):
        for h in self.handles:
            h.remove()

    def naswot_score(self) -> float:
        """log|det K|, where K[i, j] counts ReLU units on which inputs i and j agree (Hamming kernel)."""
        if self.K is None:
            return 0.0
        kernel = self.K + 1e-3 * torch.eye(self.K.shape[0], device=self.K.device)  # numerical jitter
        sign, logdet = torch.linalg.slogdet(kernel)
        if sign <= 0:
            return 0.0  # degenerate kernel: treat as zero score
        return float(logdet)


def _synflow_score(tokens: List[str], input_shape=(1, 3, 32, 32)) -> float:
    """
    Data-free SynFlow: abs() all weights, forward an all-ones input, backprop the
    output sum, and score = sum(|param * grad|). Returned on a log scale because
    raw values span many orders of magnitude.
    """
    torch.manual_seed(1234)  # same fixed init as the other components (NOTE: sets the global seed)
    net = _make_net(tokens).to(DEVICE)
    net.double()  # SynFlow is numerically sensitive, so use fp64

    # Linearise the net: keep the signs so they can be restored, then take abs of all weights
    signs = {}
    for name, p in net.named_parameters():
        signs[name] = p.data.sign()
        p.data = p.data.abs()

    net.zero_grad()
    x = torch.ones(input_shape, dtype=torch.float64, device=DEVICE)
    # BatchNorm in EVAL mode (running stats mean 0 / var 1 = near-identity), as in the reference
    # implementation which removes BN for SynFlow. Oct 2026 fix: train-mode BN on a single all-ones
    # image normalises the signal away, which made S useless (Kendall tau vs NB201 cifar10
    # accuracy -0.03 on 300 archs; eval mode: +0.52, in line with the literature).
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    out, _ = net(x)
    out.sum().backward()

    score = 0.0
    for name, p in net.named_parameters():
        if p.grad is not None:
            score += float((p * p.grad).abs().sum())
        p.data = p.data * signs[name]  # restore signs (net is discarded anyway)

    del net
    return math.log1p(max(score, 0.0))


# --------------------------------------------------------------------------
# Ensemble
# --------------------------------------------------------------------------

class ZCProxyEnsemble:
    def __init__(self, weights: Optional[Dict[str, float]] = None, use_disk_cache: bool = True):
        self.weights = weights or {"E": 0.25, "S": 0.25, "T": 0.25, "C": 0.25}
        # {component: (mean, std)} set by calibrate()/from_saved(); None means raw un-normalised sum
        self._norm_stats: Optional[Dict[str, Tuple[float, float]]] = None
        self._init_seed = 1234  # fixed weight-init seed for E, T and C
        self._use_disk_cache = use_disk_cache
        self._raw_cache: Dict[Tuple[str, ...], Dict[str, float]] = (
            _shared_raw_cache() if use_disk_cache else {}
        )
        self.labels_used = 0   # true accuracies consumed by calibration (cost accounting)

    def flush(self) -> None:
        """Force-write the shared raw cache to disk (also happens automatically at exit)."""
        if self._use_disk_cache:
            _flush_shared_raw_cache()

    def raw(self, tokens: List[str]) -> Dict[str, float]:
        try:
            return self._raw_cached(tokens)
        except torch.cuda.OutOfMemoryError:
            # free the cached allocator blocks and try once more before giving up
            torch.cuda.empty_cache()
            try:
                return self._raw_cached(tokens)
            except torch.cuda.OutOfMemoryError as e:
                raise RuntimeError(f"[zc_proxy] CUDA out of memory twice on {tokens}. Close other GPU "
                                   f"programs, or lower NASWOT_BATCH_SIZE in archtree/zc_proxy.py "
                                   f"(changes the proxy: delete outputs/_zc_raw_cache.json then).") from e

    def _raw_cached(self, tokens: List[str]) -> Dict[str, float]:
        """Raw components {E, S, T, C} for one architecture, computed once and cached."""
        global _RAW_DIRTY
        key = _cache_key(tokens)
        hit = self._raw_cache.get(key)
        if hit is not None and hit.get("sv") == SYNFLOW_VERSION:
            return {k: hit[k] for k in _RAW_KEYS}
        if hit is not None:                       # old entry: only S was wrong
            hit["S"] = _synflow_score(list(key))
            hit["sv"] = SYNFLOW_VERSION
            r = hit
        else:
            r = self._raw(list(key))
            r["sv"] = SYNFLOW_VERSION
            self._raw_cache[key] = r
        if self._use_disk_cache:
            global _RAW_LAST_FLUSH
            _RAW_DIRTY += 1
            now = time.time()
            if _RAW_DIRTY >= _RAW_FLUSH_EVERY and now - _RAW_LAST_FLUSH >= _RAW_FLUSH_MIN_SEC:
                _flush_shared_raw_cache()
                _RAW_LAST_FLUSH = now
        return {k: r[k] for k in _RAW_KEYS}

    def _raw(self, tokens: List[str]) -> Dict[str, float]:
        """Compute all four raw components from scratch (no caching)."""
        # T: NASWOT averaged over several data batches (weight init stays fixed)
        t_scores = []
        last_acts, last_net = None, None
        for variant in range(N_NASWOT_BATCHES):
            x, _ = get_mini_batch(variant=variant)
            torch.manual_seed(self._init_seed)
            net = _make_net(tokens).to(DEVICE)
            for m in net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
            hooks = _NASWOTHooks(net)
            with torch.no_grad():
                _, acts = net(x)
            t_scores.append(hooks.naswot_score())
            hooks.remove()

            if variant == 0:
                # Keep only what E and C need: the final feature map of batch 0, and the net
                last_acts, last_net = [acts[-1]], net
            else:
                del net
            del acts
        T_score = sum(t_scores) / len(t_scores)

        # E: entropy of the singular-value spectrum of the centred final feature map
        feat = last_acts[-1].flatten(1).detach()
        feat = feat - feat.mean(0, keepdim=True)
        try:
            s = torch.linalg.svdvals(feat)
            p = (s / (s.sum() + 1e-12)).clamp_min(1e-12)
            E = float(-(p * p.log()).sum())
        except Exception:
            E = 0.0  # SVD failed to converge

        # S: SynFlow (data-free)
        S = _synflow_score(tokens)

        # C: complexity penalty = -(log params) - weighted op cost
        n_params = sum(p.numel() for p in last_net.parameters())
        if len(tokens) == 6:
            flop_w = {"none": 0, "skip_connect": 0, "avg_pool_3x3": 1, "conv_1x1": 2, "conv_3x3": 6}
            flops = sum(flop_w[t] for t in tokens)
        elif len(tokens) == 5:   # NATS-SSS: fixed cell, only widths change -> params carry it all
            flops = 0
        else:   # NB101: interior ops of the pruned cell
            flop_w = {"maxpool3x3": 1, "conv1x1-bn-relu": 2, "conv3x3-bn-relu": 6}
            flops = sum(flop_w[t] for t in last_net.interior_ops)
        C_score = -math.log1p(n_params) - flops

        del last_net
        return {"E": E, "S": S, "T": T_score, "C": C_score}

    def score(self, tokens: List[str]) -> float:
        return self._aggregate(self.raw(tokens))

    def score_many(self, token_lists: List[List[str]]) -> List[float]:
        # Sequential on purpose: SynFlow's backward pass must not run concurrently
        return [self.score(t) for t in token_lists]

    def _aggregate(self, raw: Dict[str, float]) -> float:
        """Weighted sum of components, z-scored first if calibration stats exist."""
        if self._norm_stats:
            z = {k: (raw[k] - self._norm_stats[k][0]) / (self._norm_stats[k][1] + 1e-8)
                 for k in _RAW_KEYS}
        else:
            z = {k: raw[k] for k in _RAW_KEYS}
        return sum(self.weights[k] * z[k] for k in _RAW_KEYS)

    def calibrate_unsupervised(self, bench, n: int = 1500, phase: str = "phase2",
                               dataset: str = "cifar10", save: bool = True, seed: int = 0) -> None:
        """Label-free calibration: z-score stats from n unlabeled archs, weights E=S=T=1/3, C=0."""
        import random as _random
        idxs = _random.Random(seed).sample(range(bench.total_archs), n)
        raws = [self.raw(bench.index_to_tokens(i)) for i in idxs]
        keys = list(_RAW_KEYS)
        mat = np.array([[r[k] for k in keys] for r in raws])
        mu, sd = mat.mean(0), mat.std(0) + 1e-8
        self._norm_stats = {k: (float(mu[i]), float(sd[i])) for i, k in enumerate(keys)}
        self.weights = {"E": 1 / 3, "S": 1 / 3, "T": 1 / 3, "C": 0.0}
        self.labels_used = 0
        if save:
            storage.save_zc_calibration(phase, bench.name, dataset,
                                        {"weights": self.weights, "norm_stats": self._norm_stats,
                                         "mode": "unsupervised", "labels_used": 0, "n_archs": n},
                                        mode="unsupervised")
        self.flush()

    @classmethod
    def unsupervised(cls, bench, dataset: str, n: int = None, phase: str = "phase2") -> "ZCProxyEnsemble":
        """Load the saved label-free calibration, or build (and save) it."""
        from archtree import settings
        n = n or settings.CALIBRATION_ARCHS
        try:
            return cls.from_saved(phase, bench.name, dataset, mode="unsupervised")
        except FileNotFoundError:
            inst = cls()
            inst.calibrate_unsupervised(bench, n=n, phase=phase, dataset=dataset)
            return inst

    def calibrate(self, bench, n: int = 1500, phase: str = "phase2", dataset: str = "cifar10",
                  mode: str = None) -> None:
        """
        Fit z-score stats and component weights against ground-truth accuracy on
        `n` random architectures (sampled WITHOUT replacement), then save them per
        (phase, bench, dataset). mode="unsupervised" -> calibrate_unsupervised().
        """
        from archtree import settings
        mode = mode or settings.CALIBRATION_MODE
        if mode == "unsupervised":
            return self.calibrate_unsupervised(bench, n=n, phase=phase, dataset=dataset)
        from scipy.stats import kendalltau, spearmanr

        import random as _random
        idxs = _random.Random(0).sample(range(bench.total_archs), n)
        token_lists = [bench.index_to_tokens(idx) for idx in idxs]
        raws = [self.raw(tokens) for tokens in token_lists]  # cached, so re-calibration is cheap
        gts = [bench.query(idx, dataset) for idx in idxs]

        keys = list(_RAW_KEYS)
        mat = np.array([[r[k] for k in keys] for r in raws])
        mu, sd = mat.mean(0), mat.std(0) + 1e-8
        self._norm_stats = {k: (float(mu[i]), float(sd[i])) for i, k in enumerate(keys)}
        z = (mat - mu) / sd
        gt_arr = np.array(gts)

        # Per-component rank correlation with ground truth (logged to CSV)
        component_tau = {}
        for i, k in enumerate(keys):
            tau_k, _ = kendalltau(z[:, i], gt_arr)
            rho_k, _ = spearmanr(z[:, i], gt_arr)
            component_tau[k] = tau_k
            storage.append_result_row(phase, "phase2_proxy_calibration.csv", {
                "benchmark": bench.name, "dataset": dataset, "seed": 0,
                "proxy_name": k, "kendall_tau": tau_k, "spearman_rho": rho_k,
            })
            print(f"[zc_proxy] component {k}: tau={tau_k:.3f} rho={rho_k:.3f}")

        # Exclude near-zero-signal components from the least-squares fit
        DEAD_THRESHOLD = 0.05
        live_idx = [i for i, k in enumerate(keys) if abs(component_tau[k]) >= DEAD_THRESHOLD]

        if live_idx:
            # Least squares with intercept, then L1-normalise the weights
            A = np.hstack([z[:, live_idx], np.ones((len(z), 1))])
            coef, *_ = np.linalg.lstsq(A, gt_arr, rcond=None)
            w_live = coef[:len(live_idx)]
            w_live = w_live / (np.abs(w_live).sum() + 1e-8)
            w = np.zeros(len(keys))
            for pos, i in enumerate(live_idx):
                w[i] = w_live[pos]
        else:
            print("[zc_proxy] WARNING: no component cleared dead-signal threshold — equal weights fallback.")
            w = np.full(len(keys), 0.25)

        self.weights = {k: float(w[i]) for i, k in enumerate(keys)}
        self.labels_used = n
        storage.save_zc_calibration(phase, bench.name, dataset,
                                    {"weights": self.weights, "norm_stats": self._norm_stats,
                                     "mode": "supervised", "labels_used": n, "n_archs": n,
                                     "label_indices": [int(i) for i in idxs]})

        # Log the fitted ensemble's own correlation with ground truth
        scores = [self._aggregate(r) for r in raws]
        tau, _ = kendalltau(scores, gts)
        rho, _ = spearmanr(scores, gts)
        storage.append_result_row(phase, "phase2_proxy_calibration.csv", {
            "benchmark": bench.name, "dataset": dataset, "seed": 0,
            "proxy_name": "ensemble", "kendall_tau": tau, "spearman_rho": rho,
        })
        print(f"[zc_proxy] calibrated on {n} archs — ensemble tau={tau:.3f} weights={self.weights}")

        dead = [k for k, t in component_tau.items() if abs(t) < DEAD_THRESHOLD]
        if dead:
            print(f"[zc_proxy] components excluded from lstsq fit (near-zero signal): {dead}")
        self.flush()

    # ── Composition-ablation helpers (no file I/O, no side effects) ──────────
    @staticmethod
    def fit_subset(raws: List[Dict[str, float]], gts, components,
                   dead_threshold: Optional[float] = None):
        """
        Same recipe as calibrate() (z-score, least squares with intercept,
        L1-normalised weights), restricted to `components`; the others get weight 0.

        Returns (weights, norm_stats, component_tau):
            weights        full {E, S, T, C} dict
            norm_stats     {k: (mu, sd)} for all four keys
            component_tau  Kendall tau of each z-scored component vs gts

        dead_threshold=None keeps the requested subset exactly. Pass e.g. 0.05
        to also drop near-zero-signal components (falls back to the full subset
        if that would drop everything).
        """
        from scipy.stats import kendalltau
        keys = list(_RAW_KEYS)
        unknown = [c for c in components if c not in keys]
        assert not unknown and len(components) > 0, f"bad components {components}"

        mat = np.array([[r[k] for k in keys] for r in raws], dtype=float)
        mu, sd = mat.mean(0), mat.std(0) + 1e-8
        z = (mat - mu) / sd
        gt = np.asarray(gts, dtype=float)
        norm_stats = {k: (float(mu[i]), float(sd[i])) for i, k in enumerate(keys)}

        tau = {}
        for i, k in enumerate(keys):
            t, _ = kendalltau(z[:, i], gt)
            tau[k] = float(t)

        live = [keys.index(c) for c in components]
        if dead_threshold is not None:
            live = [i for i in live if abs(tau[keys[i]]) >= dead_threshold] or live

        A = np.hstack([z[:, live], np.ones((len(z), 1))])
        coef, *_ = np.linalg.lstsq(A, gt, rcond=None)
        w_live = coef[:len(live)]
        w_live = w_live / (np.abs(w_live).sum() + 1e-8)
        w = np.zeros(len(keys))
        for pos, i in enumerate(live):
            w[i] = w_live[pos]
        weights = {k: float(w[i]) for i, k in enumerate(keys)}
        return weights, norm_stats, tau

    @classmethod
    def from_fit(cls, weights: Dict[str, float],
                 norm_stats: Dict[str, Tuple[float, float]]) -> "ZCProxyEnsemble":
        """Build an instance from fit_subset() output (shares the raw cache)."""
        inst = cls(weights=weights)
        inst._norm_stats = norm_stats
        return inst

    @classmethod
    def from_saved(cls, phase: str, bench_name: str, dataset: str, mode: str = None) -> "ZCProxyEnsemble":
        """Load calibration saved by calibrate(). `dataset` is required: calibrations are per dataset.
        mode=None -> config zc_proxy.calibration_mode."""
        from archtree import settings
        mode = mode or settings.CALIBRATION_MODE
        blob = storage.load_zc_calibration(phase, bench_name, dataset, mode=mode)
        inst = cls(weights=blob["weights"])
        inst._norm_stats = blob["norm_stats"]
        inst.labels_used = blob.get("labels_used", 1500 if mode == "supervised" else 0)
        return inst