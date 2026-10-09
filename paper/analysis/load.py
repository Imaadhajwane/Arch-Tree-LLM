"""Shared loaders for the raw benchmark caches (read-only)."""
import json, numpy as np, random
import os
ROOT = os.environ.get("ARCHTREE_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # project root
NB201_OPS = ["none", "skip_connect", "conv_1x1", "conv_3x3", "avg_pool_3x3"]
DS = ["cifar10", "cifar100", "ImageNet16-120"]
NB101_OPS = ["conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3"]

def nb201_tokens(i):
    i, ops = int(i), []
    for _ in range(6):
        ops.append(NB201_OPS[i % 5]); i //= 5
    return ops[::-1]

def load_nb201():
    d = json.load(open(f"{ROOT}/data/nb201_cache_v2.json"))
    out = {}
    for ds in DS:
        v = np.array([d[str(i)][ds]["valid"] for i in range(15625)])
        t = np.array([d[str(i)][ds]["test"] for i in range(15625)])
        s = np.array([d[str(i)][ds]["train_sec"] for i in range(15625)])
        out[ds] = dict(valid=v, test=t, sec=s)
    return out

def load_nats():
    d = json.load(open(f"{ROOT}/data/natssss_cache.json"))
    n = len(d) - ("_meta" in d)
    out = {}
    for ds in DS:
        out[ds] = dict(valid=np.array([d[str(i)][ds]["valid"] for i in range(n)]),
                       test=np.array([d[str(i)][ds]["test"] for i in range(n)]),
                       sec=np.array([d[str(i)][ds]["train_sec"] for i in range(n)]))
    out["params"] = np.array([d[str(i)]["params"] for i in range(n)])
    out["flops"] = np.array([d[str(i)]["flops"] for i in range(n)])
    out["arch"] = [d[str(i)]["arch"] for i in range(n)]
    return out

def load_nb101():
    z = np.load(f"{ROOT}/data/nb101_cache.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}

def load_raw():
    return json.load(open(f"{ROOT}/outputs/_zc_raw_cache.json"))

def label_free_score(raw_rows, calib_rows):
    """Eq. (2): mean of z-scored E, S, T with z-stats from unlabelled calibration rows."""
    M = np.array([[r[k] for k in "EST"] for r in calib_rows])
    mu, sd = M.mean(0), M.std(0) + 1e-8
    X = np.array([[r[k] for k in "EST"] for r in raw_rows])
    return ((X - mu) / sd).mean(1), X
