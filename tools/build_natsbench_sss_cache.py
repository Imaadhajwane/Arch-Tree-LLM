# tools/build_natsbench_sss_cache.py
"""
Run ONCE: python tools/build_natsbench_sss_cache.py
Builds data/natssss_cache.json (config paths.natssss_cache) from the NATS-Bench size search space
(Dong et al., TPAMI 2021) - the third benchmark (Oct 3, 2026).

Why this benchmark: in NATS-Bench SSS the cell is FIXED and only the channel widths of 5 layers
change (8..64 each, 8^5 = 32,768 networks). Training cost therefore varies a lot and mostly
independently of "cell design" - the right place to test the finding that zero-cost proxies are
largely cost detectors and that correcting the gate for cost hurts (README.md §9.5).

Input (download once, see data/README.md) - preferred, low RAM (one small file per arch, fast_mode):
    data/NATS-sss-v1_0-50262-simple/            (extract NATS-sss-v1_0-50262-simple.tar, 1.01 GB)
  or the single file (874 MB, unpacked into RAM at once, ~10 GB+):
    data/NATS-sss-v1_0-50262.pickle.pbz2        (config paths.natssss)
Needs:  pip install nats_bench      (no GPU)

Stored per architecture (hp = "90": the 90-epoch schedule of the NATS-Bench paper, averaged over
the benchmark's training seeds, is_random=False), the same split protocol as our NB201 cache:
    cifar10         valid = cifar10-valid / x-valid     test = cifar10 / ori-test
    cifar100        valid = x-valid                     test = x-test
    ImageNet16-120  valid = x-valid                     test = x-test
    train_sec       total training time on the search split (cost of one real evaluation)
    params / flops  from get_cost_info (cifar10 head; the zero-cost #params gate uses params)
Keys are the CANONICAL index: base-8 code of the 5 channel choices (8,16,...,64), first layer most
significant - the same bijection as benchmarks/natssss.py.
"""
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
HP = "90"
CHANNELS = [8, 16, 24, 32, 40, 48, 56, 64]
N_ARCHS = 8 ** 5


def canon_index(chs):
    i = 0
    for c in chs:
        i = i * 8 + CHANNELS.index(int(c))
    return i


def _path(key, default):
    p = Path(cfg["paths"].get(key, default))
    return p if p.is_absolute() else PROJECT_ROOT / p


def extract(api, n=None, log_every=2000):
    """api: a nats_bench size-space API (or a test double with the same methods)."""
    cache = {}
    n = n or len(api)
    for i in range(n):
        arch = api.arch(i)                                      # e.g. "64:32:16:48:8"
        chs = [int(c) for c in str(arch).split(":")]
        if len(chs) != 5 or any(c not in CHANNELS for c in chs):
            raise RuntimeError(f"unexpected arch string {arch!r} at index {i}")
        ci = canon_index(chs)
        if str(ci) in cache:
            raise RuntimeError(f"canonical index collision at {i} ({arch})")
        c10v = api.get_more_info(i, "cifar10-valid", hp=HP, is_random=False)
        c10 = api.get_more_info(i, "cifar10", hp=HP, is_random=False)
        entry = {"arch": str(arch), "nats_index": i,
                 "cifar10": {"valid": float(c10v["valid-accuracy"]), "test": float(c10["test-accuracy"]),
                             "train_sec": float(c10v["train-all-time"])}}
        for ds in ("cifar100", "ImageNet16-120"):
            info = api.get_more_info(i, ds, hp=HP, is_random=False)
            entry[ds] = {"valid": float(info["valid-accuracy"]), "test": float(info["test-accuracy"]),
                         "train_sec": float(info["train-all-time"])}
        cost = api.get_cost_info(i, "cifar10", hp=HP)
        entry["params"] = float(cost["params"])                # millions
        entry["flops"] = float(cost["flops"])                  # MFLOPs
        cache[str(ci)] = entry
        infos = getattr(api, "arch2infos_dict", None)
        if isinstance(infos, dict) and getattr(api, "_fast_mode", False):
            infos.pop(i, None)                                  # fast_mode: free this arch again (low RAM)
        if log_every and i % log_every == 0:
            print(f"  {i}/{n}  {arch}  c10 valid={entry['cifar10']['valid']:.2f} test={entry['cifar10']['test']:.2f} "
                  f"train={entry['cifar10']['train_sec']:.0f}s params={entry['params']:.3f}M")
    return cache


def sanity(cache):
    if len(cache) != N_ARCHS:
        raise RuntimeError(f"expected {N_ARCHS} archs, got {len(cache)}")
    big = cache[str(canon_index([64] * 5))]
    small = cache[str(canon_index([8] * 5))]
    print(f"widest 64:64:64:64:64  c10 test {big['cifar10']['test']:.2f}  train {big['cifar10']['train_sec']:.0f}s  "
          f"params {big['params']:.3f}M")
    print(f"narrowest 8:8:8:8:8    c10 test {small['cifar10']['test']:.2f}  train {small['cifar10']['train_sec']:.0f}s  "
          f"params {small['params']:.3f}M")
    if not (big["params"] > small["params"] and big["cifar10"]["test"] > small["cifar10"]["test"]):
        raise RuntimeError("sanity check failed: the widest network should be larger and more accurate")
    for ds in ("cifar10", "cifar100", "ImageNet16-120"):
        v = [e[ds]["valid"] for e in cache.values()]
        t = [e[ds]["train_sec"] for e in cache.values()]
        print(f"{ds:15s} valid {min(v):.2f}-{max(v):.2f}   train_sec {min(t):.0f}-{max(t):.0f}")


def main():
    src = _path("natssss", "data/NATS-sss-v1_0-50262.pickle.pbz2")
    out = _path("natssss_cache", "data/natssss_cache.json")
    alt = src.parent / "NATS-sss-v1_0-50262-simple"
    if alt.is_dir():                                            # preferred: extracted tar, low RAM
        metas = sorted(alt.rglob("meta.pickle.pbz2"))
        if not metas:
            sys.exit(f"[natssss] {alt} has no meta.pickle.pbz2 - is the tar fully extracted?")
        src = metas[0].parent                                   # handles an extra folder level
    elif not src.exists():
        sys.exit(f"[natssss] {src} not found. Download NATS-sss-v1_0-50262-simple.tar (1.01 GB, "
                 f"https://github.com/D-X-Y/NATS-Bench) and extract it into data\\ first (data/README.md).")
    from nats_bench import create
    t0 = time.time()
    print(f"Loading NATS-Bench SSS from {src} (a few minutes) ...")
    api = create(str(src), "sss", fast_mode=src.is_dir(), verbose=False)
    print(f"Loaded {len(api)} archs in {time.time() - t0:.0f}s; extracting hp={HP} ...")
    cache = extract(api)
    sanity(cache)
    cache["_meta"] = {"version": 1, "hp": HP, "source": src.name, "built": time.strftime("%Y-%m-%d %H:%M"),
                      "splits": {"cifar10": ["cifar10-valid/x-valid", "cifar10/ori-test"],
                                 "cifar100": ["x-valid", "x-test"], "ImageNet16-120": ["x-valid", "x-test"]}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(cache, f)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
