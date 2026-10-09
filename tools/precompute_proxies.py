# tools/precompute_proxies.py
"""
Pre-compute the zero-cost proxy scores (E, S, T, C) of many architectures in PARALLEL GPU worker
processes and store them in outputs/_zc_raw_cache.json (Oct 4, 2026 - speed-up).

Why: in the search experiments the proxies are computed one architecture at a time, in one
process, while the GPU is mostly idle (the proxy networks are tiny). Filling the cache up front
with 2-4 processes sharing the GPU makes every later step that needs these architectures read
them from the cache instantly. Scores are identical (same code, same fixed seeds); the cache file
is merged safely between processes (archtree/zc_proxy.py, merge-on-flush).

What (default `--set default`):
    natssss   ALL 32,768 NATS-Bench SSS architectures (the whole space; step 22 uses most of them)
    nb101     the step-22 analysis sample (2,000 archs, same seed as run_equal_gpu_hours.py) and the
              label-free calibration sample (1,500 archs)
    nb201     nothing to do (all 15,625 are cached since Oct 2) - checked and reported

Usage
    python tools/precompute_proxies.py                          # natssss + nb101, 3 workers
    python tools/precompute_proxies.py --bench natssss --workers 4
Needs a GPU (as every proxy computation). Resumable: cached architectures are skipped.
"""
import argparse
import os
import random
import sys
import time
from multiprocessing import get_context
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np  # noqa: E402
import yaml         # noqa: E402

cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))


def _bench(name):
    if name == "nb201":
        from benchmarks.nb201 import NB201
        b = NB201(cfg)
    elif name == "nb101":
        from benchmarks.nb101 import NB101
        b = NB101(cfg)
    else:
        from benchmarks.natssss import NATSSSS
        b = NATSSSS(cfg)
    b._load()
    return b


def indices_for(name, b):
    if name == "natssss":
        return list(range(b.total_archs))
    if name == "nb101":
        from archtree import settings
        ids = set(int(i) for i in np.random.default_rng(7).choice(b.total_archs, 2000, replace=False))
        ids |= set(random.Random(0).sample(range(b.total_archs), settings.CALIBRATION_ARCHS))
        return sorted(ids)
    return list(range(b.total_archs))


def _missing(name, b, ids):
    from archtree.zc_proxy import _shared_raw_cache, _cache_key, SYNFLOW_VERSION
    cache = _shared_raw_cache()
    out = []
    for i in ids:
        t = b.index_to_tokens(i)
        try:
            k = _cache_key(t)
        except ValueError:
            continue
        hit = cache.get(k)
        if hit is None or hit.get("sv") != SYNFLOW_VERSION:
            out.append(i)
    return out


def _worker(args):
    name, ids, wid = args
    import torch
    torch.set_num_threads(1)                       # several processes: no CPU oversubscription
    from archtree.zc_proxy import ZCProxyEnsemble, get_mini_batch, _flush_shared_raw_cache
    import archtree.zc_proxy as zp
    get_mini_batch()
    b = _bench(name)
    ens = ZCProxyEnsemble()
    t0 = time.time()
    for n, i in enumerate(ids, 1):
        ens.raw(b.index_to_tokens(i))
        if n % 200 == 0:
            zp._RAW_LAST_FLUSH = 0.0               # force a (merged) flush every 200 archs
            _flush_shared_raw_cache()
            print(f"  [w{wid}] {name}: {n}/{len(ids)}  ({(time.time() - t0) / n:.2f} s/arch)", flush=True)
    zp._RAW_LAST_FLUSH = 0.0
    _flush_shared_raw_cache()
    return len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="+", default=["natssss", "nb101", "nb201"],
                    choices=["natssss", "nb101", "nb201"])
    ap.add_argument("--workers", type=int, default=int(cfg.get("runtime", {}).get("proxy_workers", 3)))
    ap.add_argument("--limit", type=int, default=0, help="only the first N wanted archs (testing)")
    a = ap.parse_args()
    for name in a.bench:
        try:
            b = _bench(name)
        except FileNotFoundError as e:
            print(f"[precompute] {name} skipped: {e}")
            continue
        ids = indices_for(name, b)
        if a.limit:
            ids = ids[:a.limit]
        todo = _missing(name, b, ids)
        print(f"[precompute] {name}: {len(ids)} archs wanted, {len(ids) - len(todo)} already cached, "
              f"{len(todo)} to compute with {a.workers} worker(s)")
        if not todo:
            continue
        t0 = time.time()
        shards = [(name, todo[w::a.workers], w) for w in range(a.workers) if todo[w::a.workers]]
        if len(shards) == 1:
            _worker(shards[0])
        else:
            with get_context("spawn").Pool(len(shards)) as pool:
                pool.map(_worker, shards)
        print(f"[precompute] {name}: done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
