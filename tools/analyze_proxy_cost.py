# tools/analyze_proxy_cost.py
# isort: skip_file
"""
Reviewer item A7 — count the "zero-cost" proxy's real cost and report a cost-adjusted efficiency.

Per search (headline system, Phase 3 logs) the cost has three parts:
    real evaluations      real_eval_count x t_real          (t_real = mean NB201 training time per
                                                            arch, from the v2 cache train_sec, or
                                                            --real-eval-sec)
    proxy evaluations     proxy_eval_count x t_proxy        (t_proxy measured here with --measure N
                                                            on the GPU, cache disabled, or
                                                            --proxy-sec)
    calibration labels    labels_used x t_real              (supervised calibration consumed
                                                            1,500 TRUE accuracies per dataset;
                                                            0 for the unsupervised mode)
Reported two ways: per search counting the full calibration (conservative, a single user running
one search), and with calibration amortised over all searches that reuse it.

Output: outputs/final/results/proxy_cost_summary.csv (+ proxy_timing.json if measured)

Usage:
    python tools/analyze_proxy_cost.py --measure 200            # GPU: time the proxy
    python tools/analyze_proxy_cost.py --proxy-sec 0.35         # reuse a measured value
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402
import yaml           # noqa: E402

import archtree.storage as storage  # noqa: E402

cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))


def measure_proxy_sec(n):
    from archtree.zc_proxy import ZCProxyEnsemble, get_mini_batch
    import random
    for v in range(3):
        get_mini_batch(variant=v)
    zc = ZCProxyEnsemble(use_disk_cache=False)
    rng = random.Random(0)
    OPS = ["none", "skip_connect", "conv_1x1", "conv_3x3", "avg_pool_3x3"]
    archs = [[rng.choice(OPS) for _ in range(6)] for _ in range(n)]
    zc._raw(archs[0])                                  # warm-up (cudnn autotune)
    t0 = time.perf_counter()
    for a in archs:
        zc._raw(a)
    sec = (time.perf_counter() - t0) / n
    out = storage.get_output_dir("final", "results") / "proxy_timing.json"
    out.write_text(json.dumps({"sec_per_proxy_eval": sec, "n": n}, indent=2))
    print(f"proxy: {sec:.4f} s per architecture (n={n}) -> {out}")
    return sec


def mean_train_sec(dataset):
    from benchmarks.nb201 import NB201
    b = NB201(cfg)
    vals = [b.train_sec(i, dataset) for i in range(0, 15625, 25)]
    vals = [v for v in vals if v]
    return float(np.mean(vals)) if vals else None


def main(a):
    t_proxy = a.proxy_sec
    if a.measure:
        t_proxy = measure_proxy_sec(a.measure)
    if t_proxy is None:
        tj = storage.get_output_dir("final", "results") / "proxy_timing.json"
        if tj.exists():
            t_proxy = json.loads(tj.read_text())["sec_per_proxy_eval"]
    if t_proxy is None:
        sys.exit("need --measure N (GPU) or --proxy-sec S")

    rows = []
    logs = glob.glob(str(PROJECT_ROOT / "outputs" / "phase3" / "logs" / "archtree_llm__nb201__*__budget*.json"))
    n_searches = {}
    recs = []
    for f in logs:
        j = json.load(open(f))
        # proxy SCORINGS = every proposal after the warm-up (proxy_eval_count in the log counts only
        # the skipped ones); repeated proposals hit the raw cache, so this is an upper bound
        warm = cfg.get("zc_proxy", {}).get("warmup_real_evals", 10)
        scorings = max(0, j["budget"] - warm) if j.get("use_proxy", True) else 0
        recs.append((j["dataset"], j["budget"], j["seed"], j.get("real_eval_count", 0), scorings))
        n_searches[j["dataset"]] = n_searches.get(j["dataset"], 0) + 1
    if not recs:
        sys.exit("no phase3 logs found")
    df = pd.DataFrame(recs, columns=["dataset", "budget", "seed", "real", "proxy"])

    for ds, g in df.groupby("dataset"):
        t_real = a.real_eval_sec or mean_train_sec(ds)
        if not t_real:
            sys.exit("no train_sec in the cache (v1 cache?) - pass --real-eval-sec")
        try:
            from archtree.zc_proxy import ZCProxyEnsemble
            labels = ZCProxyEnsemble.from_saved("phase2", "nb201", ds).labels_used
        except Exception:
            labels = cfg.get("zc_proxy", {}).get("calibration_archs", 1500)
        # every dataset's calibration is reused by all searches of every study on that dataset
        n_reuse = a.amortise_over or n_searches[ds]
        for b, gb in g.groupby("budget"):
            real, proxy = gb["real"].mean(), gb["proxy"].mean()
            eq_proxy = proxy * t_proxy / t_real
            rows.append({
                "dataset": ds, "budget": b, "n_runs": len(gb),
                "real_evals": real, "proxy_evals": proxy,
                "t_real_sec": t_real, "t_proxy_sec": t_proxy,
                "proxy_cost_in_real_evals": eq_proxy,
                "calibration_labels": labels,
                "cost_real_eval_equiv_amortised": real + eq_proxy + labels / n_reuse,
                "cost_real_eval_equiv_full_calibration": real + eq_proxy + labels,
                "nominal_budget_saving_pct_amortised": 100 * (1 - (real + eq_proxy + labels / n_reuse) / b),
                "nominal_budget_saving_pct_full_calibration": 100 * (1 - (real + eq_proxy + labels) / b),
            })
    out = pd.DataFrame(rows)
    path = storage.get_output_dir("final", "results") / "proxy_cost_summary.csv"
    out.to_csv(path, index=False)
    pd.set_option("display.width", 200)
    print(out.round(2).to_string(index=False))
    print(f"\n-> {path}\nRead: with full calibration counted, a single supervised-gated search costs MORE "
          f"real-eval equivalents than its budget whenever calibration_labels > budget. Report both columns.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", type=int, default=0, help="time N proxy evaluations on the GPU")
    ap.add_argument("--proxy-sec", type=float, default=None)
    ap.add_argument("--real-eval-sec", type=float, default=None, help="override NB201 training time per arch")
    ap.add_argument("--amortise-over", type=int, default=0, help="searches sharing one calibration (default: all phase3 runs)")
    main(ap.parse_args())
