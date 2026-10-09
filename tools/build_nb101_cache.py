# tools/build_nb101_cache.py
"""
Build data/nb101_cache.npz from data/nasbench_only108.tfrecord WITHOUT TensorFlow (Oct 2026).

The tfrecord is read with a ~20-line TFRecord reader, each record's JSON row is decoded exactly as
nasbench/api.py does, and the ModelMetrics protobuf is decoded by a minimal wire-format parser
(field numbers from nasbench/lib/model_metrics.proto). Nothing from TF / protobuf is imported.

Checks (the script refuses to write the cache if any fails):
  1. every graph appears with exactly 3 repeats at 108 epochs
  2. every stored module_hash is reproduced by benchmarks/nb101.py::hash_module() from the stored
     adjacency + ops  -> proves the pure-Python hash port matches the official one on ALL graphs
  3. 423,624 unique graphs
It then prints the oracle numbers (best mean validation / test accuracy) for the paper.

Usage:
    python tools/build_nb101_cache.py                       # paths from config.yaml
    python tools/build_nb101_cache.py --tfrecord data/nasbench_only108.tfrecord --out data/nb101_cache.npz
Takes ~3-8 minutes and ~2 GB RAM. The 522 MB tfrecord can be moved out of the project afterwards.
Download (if missing): https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord
"""
import argparse
import base64
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.nb101 import OPS, hash_module  # noqa: E402

EXPECTED_ARCHS = 423624
EXPECTED_REPEATS = 3


# ── TFRecord + protobuf readers ───────────────────────────────────────────────
def iter_tfrecord(path):
    """Yield raw record payloads. Format: uint64 len | uint32 crc(len) | data | uint32 crc(data)."""
    with open(path, "rb") as f:
        while True:
            head = f.read(12)
            if not head:
                return
            if len(head) < 12:
                raise IOError("truncated tfrecord header")
            (length,) = struct.unpack("<Q", head[:8])
            data = f.read(length)
            if len(data) != length or len(f.read(4)) != 4:
                raise IOError("truncated tfrecord record")
            yield data


def _varint(b, i):
    shift = result = 0
    while True:
        c = b[i]
        i += 1
        result |= (c & 0x7F) << shift
        if not c & 0x80:
            return result, i
        shift += 7


def _fields(b):
    """Yield (field_number, wire_type, value) for one protobuf message."""
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1:
            v = struct.unpack("<d", b[i:i + 8])[0]
            i += 8
        elif wt == 2:
            ln, i = _varint(b, i)
            v = b[i:i + ln]
            i += ln
        elif wt == 5:
            v = struct.unpack("<f", b[i:i + 4])[0]
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wt}")
        yield fn, wt, v


def parse_model_metrics(b):
    """ModelMetrics {repeated EvaluationData evaluation_data = 1; int32 trainable_parameters = 2;}
    EvaluationData {double current_epoch=1, training_time=2, train_accuracy=3,
                    validation_accuracy=4, test_accuracy=5; string checkpoint_path=6}"""
    evals, params = [], 0
    for fn, wt, v in _fields(b):
        if fn == 1 and wt == 2:
            e = {}
            for f2, w2, v2 in _fields(v):
                if w2 == 1:
                    e[f2] = v2
            evals.append(e)
        elif fn == 2 and wt == 0:
            params = v
    return evals, params


# ── build ─────────────────────────────────────────────────────────────────────
def build(tfrecord, out, epochs_keep=108):
    t0 = time.time()
    index = {}                      # hash -> slot
    adj, ops, nv, params = [], [], [], []
    valid, test, ttime = [], [], []  # lists of per-slot repeat lists
    n_rec = 0
    for raw in iter_tfrecord(tfrecord):
        module_hash, epochs, raw_adj, raw_ops, raw_metrics = json.loads(raw.decode("utf-8"))
        n_rec += 1
        if epochs != epochs_keep:
            continue
        evals, p = parse_model_metrics(base64.b64decode(raw_metrics))
        final = evals[2]            # same index nasbench/api.py uses for the final evaluation
        slot = index.get(module_hash)
        if slot is None:
            dim = int(round(len(raw_adj) ** 0.5))
            a = np.array([int(c) for c in raw_adj], dtype=np.uint8).reshape(dim, dim)
            o = raw_ops.split(",")
            full = np.zeros((7, 7), dtype=np.uint8)
            full[:dim, :dim] = a
            op_ids = np.full(5, -1, dtype=np.int8)
            for k, name in enumerate(o[1:-1]):
                op_ids[k] = OPS.index(name)
            slot = len(adj)
            index[module_hash] = slot
            adj.append(full)
            ops.append(op_ids)
            nv.append(dim)
            params.append(p)
            valid.append([])
            test.append([])
            ttime.append([])
        valid[slot].append(final.get(4, 0.0))
        test[slot].append(final.get(5, 0.0))
        ttime[slot].append(final.get(2, 0.0))
        if n_rec % 200000 == 0:
            print(f"  read {n_rec:,} records, {len(adj):,} graphs ({time.time() - t0:.0f}s)")
    print(f"read {n_rec:,} records, {len(adj):,} unique graphs at {epochs_keep} epochs "
          f"({time.time() - t0:.0f}s)")

    # check 1: repeats
    bad_rep = [h for h, s in index.items() if len(valid[s]) != EXPECTED_REPEATS]
    print(f"check 1  graphs without exactly {EXPECTED_REPEATS} repeats: {len(bad_rep)}")
    # check 2: hash port reproduces every stored hash
    t1 = time.time()
    bad_hash = []
    for h, s in index.items():
        k = nv[s]
        labeling = [-1] + [int(x) for x in ops[s][:k - 2]] + [-2]
        if hash_module(adj[s][:k, :k].astype(np.int64), labeling) != h:
            bad_hash.append(h)
    print(f"check 2  hashes NOT reproduced by benchmarks/nb101.py::hash_module: {len(bad_hash)} "
          f"of {len(index):,} ({time.time() - t1:.0f}s)")
    # check 3: size
    print(f"check 3  unique graphs = {len(index):,} (expected {EXPECTED_ARCHS:,})")
    if bad_rep or bad_hash or len(index) != EXPECTED_ARCHS:
        print("FAILED - cache NOT written. Send this output to Claude.")
        if bad_hash:
            print("first bad hashes:", bad_hash[:3])
        sys.exit(1)

    order = sorted(index, key=lambda h: h)       # index = position in sorted hash list (stable)
    slots = np.array([index[h] for h in order])
    arr = lambda x, dt: np.asarray(x, dtype=dt)[slots]   # noqa: E731
    v, te, tt = arr(valid, np.float32), arr(test, np.float32), arr(ttime, np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, hashes=np.array(order), adj=arr(adj, np.uint8), ops=arr(ops, np.int8),
                        n_vertices=arr(nv, np.int8), params=arr(params, np.int64),
                        valid=v, test=te, train_time=tt)
    vm, tm = v.mean(1) * 100, te.mean(1) * 100
    bv = int(np.argmax(vm))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")
    print(f"sanity  best mean valid = {vm.max():.2f}  (its test = {tm[bv]:.2f})   "
          f"best mean test = {tm.max():.2f}   (paper: best mean test ~94.3)")
    print(f"sanity  mean test over all graphs = {tm.mean():.2f}   "
          f"mean train time = {tt.mean():.0f}s   params range = {arr(params, np.int64).min():,}-"
          f"{arr(params, np.int64).max():,}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    import yaml
    with open(PROJECT_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfrecord", default=cfg["paths"].get("nb101", "data/nasbench_only108.tfrecord"))
    ap.add_argument("--out", default=cfg["paths"].get("nb101_cache", "data/nb101_cache.npz"))
    a = ap.parse_args()
    src = Path(a.tfrecord) if Path(a.tfrecord).is_absolute() else PROJECT_ROOT / a.tfrecord
    dst = Path(a.out) if Path(a.out).is_absolute() else PROJECT_ROOT / a.out
    if not src.exists():
        sys.exit(f"{src} not found. Download https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord "
                 f"into data/ (522 MB).")
    build(src, dst)
