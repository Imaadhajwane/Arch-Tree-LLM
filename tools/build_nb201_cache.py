# tools/build_nb201_cache.py  (v2, Oct 2026)
"""
Run ONCE: python tools/build_nb201_cache.py
Writes data/nb201_cache_v2.json (path from config.yaml paths.nb201_cache).

Why v2 (verified against the nas_201_api source, Oct 2026)
-----------------------------------------------------------
The v1 cache had two problems:
  1. `api.query_by_index(idx)` defaults to hp='12' (the 12-epoch schedule), not the
     200-epoch results every NAS-Bench-201 paper reports.
  2. For dataset 'cifar10' the splits 'x-test' and 'test' do not exist (only 'train' and
     'ori-test'), so the old fallback chain silently returned TRAINING accuracy for cifar10.
     Evidence in the v1 file: the all-'none' architecture scores 9.852 on cifar10 (a constant
     classifier on the balanced test set scores exactly 10.00), while cifar100 / ImageNet16-120
     give exactly 1.00 / 0.833 (their x-test split was found).

v2 stores, per architecture and dataset, both a SEARCH metric and a REPORT metric, 200 epochs,
averaged over the benchmark's training seeds:
    cifar10         valid = cifar10-valid / x-valid     test = cifar10 / ori-test
    cifar100        valid = cifar100 / x-valid          test = cifar100 / x-test
    ImageNet16-120  valid = ImageNet16-120 / x-valid    test = ImageNet16-120 / x-test
plus train_sec (total training time on the search dataset, used to cost a real evaluation).
This is the split protocol of the NAS-Bench-201 paper: search on validation, report test.

Keys are the CANONICAL base-5 index of the token sequence (same bijection as
archtree/encoder.py::index_from_tokens_nb201), not the API's own index.

No GPU needed (CPU load of the .pth needs ~10-16 GB RAM; CUDA is used if available).
After rebuilding: re-run Phase 2 calibration (old calibrations were fit on the old metric).
"""
import functools
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
HP = "200"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_orig_load = torch.load


@functools.wraps(_orig_load)
def _load(f, *a, **kw):
    kw.setdefault("map_location", DEVICE)
    kw.setdefault("weights_only", False)
    return _orig_load(f, *a, **kw)


torch.load = _load

OPS_NB201 = ["none", "skip_connect", "conv_1x1", "conv_3x3", "avg_pool_3x3"]
_OP_NAME_MAP = {"nor_conv_1x1": "conv_1x1", "nor_conv_3x3": "conv_3x3", "none": "none",
                "skip_connect": "skip_connect", "avg_pool_3x3": "avg_pool_3x3"}

# dataset -> ((api dataset, split) for search, (api dataset, split) for report)
SPLITS = {
    "cifar10":        (("cifar10-valid", "x-valid"), ("cifar10", "ori-test")),
    "cifar100":       (("cifar100", "x-valid"), ("cifar100", "x-test")),
    "ImageNet16-120": (("ImageNet16-120", "x-valid"), ("ImageNet16-120", "x-test")),
}


def arch_str_to_tokens(arch_str):
    toks = []
    for part in arch_str.strip().split("+"):
        for op_node in part.strip("|").split("|"):
            op = op_node.split("~")[0].strip()
            if op:
                if op not in _OP_NAME_MAP:
                    raise ValueError(f"unknown op {op!r} in {arch_str!r}")
                toks.append(_OP_NAME_MAP[op])
    return toks[:6]


def canon_index(tokens):
    i = 0
    for op in tokens:
        i = i * 5 + OPS_NB201.index(op)
    return i


def main():
    from nas_201_api import NASBench201API
    print(f"Loading NB201 API from {cfg['paths']['nb201']} on {DEVICE} (several minutes)...")
    t0 = time.time()
    api = NASBench201API(cfg["paths"]["nb201"], verbose=False)
    print(f"Loaded in {time.time() - t0:.0f}s, {len(api)} archs. Extracting hp={HP} ...")

    cache = {}
    for api_idx in range(len(api)):
        info = api.query_by_index(api_idx, hp=HP)          # ArchResults, 200-epoch schedule
        tokens = arch_str_to_tokens(info.arch_str)
        ci = canon_index(tokens)
        if ci in cache:
            raise RuntimeError(f"canonical index collision at api_idx={api_idx} ({info.arch_str})")
        entry = {}
        for ds, ((vds, vsplit), (tds, tsplit)) in SPLITS.items():
            v = info.get_metrics(vds, vsplit)               # is_random=False -> mean over seeds
            t = info.get_metrics(tds, tsplit)
            tr = info.get_metrics(vds, "train")
            entry[ds] = {"valid": float(v["accuracy"]), "test": float(t["accuracy"]),
                         "train_sec": float(tr["all_time"]) if tr.get("all_time") is not None else None}
        cache[ci] = entry
        if api_idx % 2000 == 0:
            print(f"  {api_idx}/{len(api)}  canon={ci}  cifar10 valid={entry['cifar10']['valid']:.2f} "
                  f"test={entry['cifar10']['test']:.2f}")

    if len(cache) != 15625:
        raise RuntimeError(f"expected 15625 archs, got {len(cache)}")

    # Sanity checks that would have caught the v1 problems
    c10_none = cache[canon_index(["none"] * 6)]["cifar10"]["test"]
    best = max(v["cifar10"]["test"] for v in cache.values())
    print(f"\nall-'none' cifar10 test = {c10_none:.3f} (expect 10.00)   "
          f"best cifar10 test = {best:.2f} (expect ~94.4)")
    if abs(c10_none - 10.0) > 0.05 or best < 93.5:
        raise RuntimeError("sanity check failed - this is not the 200-epoch test metric")

    out = PROJECT_ROOT / cfg["paths"]["nb201_cache"]
    payload = {"_meta": {"version": 2, "hp": HP,
                         "splits": {k: [list(a), list(b)] for k, (a, b) in SPLITS.items()},
                         "built": time.strftime("%Y-%m-%d %H:%M")}}
    payload.update({str(k): v for k, v in cache.items()})
    with open(out, "w") as f:
        json.dump(payload, f)
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print("NEXT: re-run Phase 2 calibration (delete outputs/phase2/weights/zc_calibration_weights__*.pt first).")


if __name__ == "__main__":
    main()
