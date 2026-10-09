# experiments/run_ucb_sensitivity.py
# isort: skip_file
"""
Reviewer item A5 — sensitivity of the headline system to the UCB exploration constant C and the
backup rule, now that every traversal uses the same accuracy/100 scale.

Grid: C in {0.1, 0.25, 0.5, 1.0, 1.414, 2.0, 4.0} x backup in {mean, max}; everything else is the
headline configuration (alpha=0, proxy gate top-35%, value pruning, v2 rank reward + mutation).
The default cell is (config ucb.C, max) = (0.3, max) since ArchTree v2.

Outputs (outputs/ucb_sensitivity/results/):
    ucb_runs.csv          one row per (dataset, budget, seed, C, backup)  [resumable]
    ucb_summary.csv       mean/std report metric, real evals, duplicate proposals
    ucb_paired_tests.csv  every cell vs the default (config ucb.C, max), Wilcoxon + Holm

Usage:
    python experiments/run_ucb_sensitivity.py
    python experiments/run_ucb_sensitivity.py --datasets cifar10 --budgets 100 200
    python experiments/run_ucb_sensitivity.py --summarize-only
"""
import argparse
import gc
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402
import yaml                 # noqa: E402
from loguru import logger   # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from dotenv import load_dotenv    # noqa: E402
load_dotenv()

import archtree.storage as storage   # noqa: E402
from archtree import settings        # noqa: E402

with open(PROJECT_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

PHASE = "ucb_sensitivity"
RUNS = "ucb_runs.csv"
C_GRID = [0.1, 0.2, 0.3, 0.5, 1.0, 1.414, 2.0]   # v2 default C=0.3 (rank reward) is in the grid
BACKUPS = ["mean", "max"]
DEFAULT = (settings.C, "max")     # headline cell (config ucb.C)

logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>")


def _backend():
    try:
        import ollama  # noqa: F401
        return dict(backend="ollama")
    except ImportError:
        keys = [os.environ.get(k, "").strip() for k in ("GROQ_API_KEYS_FILE", "GROQ_API_KEYS", "GROQ_API_KEY")]
        if not any(keys):
            sys.exit("LLMPrior needs `ollama` or a GROQ key just to construct (no call is made).")
        return dict(backend="groq", groq_model=cfg["llm"]["groq"]["model"], use_key_pool=bool(keys[0] or keys[1]))


def run(args):
    from benchmarks.nb201 import NB201
    from archtree.search import search_phase3
    bench = NB201(cfg)
    bench._load()
    kw = _backend()
    p = PROJECT_ROOT / "outputs" / PHASE / "results" / RUNS
    done = set()
    if p.exists():
        d = pd.read_csv(p)
        done = set(zip(d["dataset"], d["budget"], d["seed"], d["C"], d["backup"]))
    for ds in args.datasets:
        for b in args.budgets:
            for s in args.seeds:
                for C in args.c_grid:
                    for bk in BACKUPS:
                        if (ds, b, s, C, bk) in done:
                            continue
                        t0 = time.time()
                        r = search_phase3(bench, b, ds, s, phase=PHASE, alpha=0.0, use_llm=False, C=C,
                                          backup_mode=bk, ablation_id=f"U_C{C}_{bk}",
                                          proxy_threshold_percentile=settings.PROXY_THRESHOLD_PERCENTILE,
                                          results_filename="U_raw_runs.csv", tcr_filename="U_raw_tcr.csv",
                                          cache_stats_filename="U_raw_cache.csv", dedup_filename="U_dedup.csv", **kw)
                        storage.append_result_row(PHASE, RUNS, {
                            "dataset": ds, "budget": b, "seed": s, "C": C, "backup": bk,
                            "best_acc": r["best_acc"], "best_test_acc": r.get("best_test_acc"),
                            "real_eval_count": r["real_eval_count"], "dup_proposals": r.get("dup_proposals"),
                            "wall_time_sec": time.time() - t0})
                        gc.collect()
        logger.info(f"{ds} done")


def summarize():
    p = PROJECT_ROOT / "outputs" / PHASE / "results" / RUNS
    if not p.exists():
        return logger.error("no runs yet")
    df = storage.to_report_metric(pd.read_csv(p)).drop_duplicates(["dataset", "budget", "seed", "C", "backup"], keep="last")
    res = storage.get_output_dir(PHASE, "results")
    summ = (df.groupby(["dataset", "budget", "backup", "C"])
              .agg(n=("seed", "nunique"), acc_mean=("best_acc", "mean"), acc_std=("best_acc", "std"),
                   real_evals=("real_eval_count", "mean"), dup_proposals=("dup_proposals", "mean"))
              .reset_index())
    summ.to_csv(res / "ucb_summary.csv", index=False)
    rows = []
    for (ds, b), g in df.groupby(["dataset", "budget"]):
        g = g.assign(cell=list(zip(g["C"], g["backup"])))
        piv = g.pivot_table(index="seed", columns="cell", values="best_acc")
        if DEFAULT not in piv:
            continue
        for cell in [c for c in piv.columns if c != DEFAULT]:
            both = piv[[DEFAULT, cell]].dropna()
            if len(both) < 2:
                continue
            d = (both[cell] - both[DEFAULT]).values
            try:
                pv = wilcoxon(both[cell], both[DEFAULT]).pvalue if np.any(d != 0) else 1.0
            except ValueError:
                pv = 1.0
            rows.append({"dataset": ds, "budget": b, "C": cell[0], "backup": cell[1], "n": len(both),
                         "mean_diff_vs_default": float(d.mean()), "p_value": pv})
    t = pd.DataFrame(rows)
    if len(t):
        p_ = t["p_value"].values
        o, m, adj, r = np.argsort(p_), len(p_), np.empty(len(p_)), 0.0
        for k, i in enumerate(o):
            r = max(r, (m - k) * p_[i])
            adj[i] = min(1.0, r)
        t["p_holm"] = adj
        t["significant_holm"] = t["p_holm"] < 0.05
    t.to_csv(res / "ucb_paired_tests.csv", index=False)
    logger.info("\n" + summ.pivot_table(index=["dataset", "budget", "backup"], columns="C", values="acc_mean").round(2).to_string())
    if len(t):
        logger.info(f"{int(t['significant_holm'].sum())} of {len(t)} cells differ significantly from C=1.414/max (Holm)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=cfg["benchmarks"]["datasets"])
    ap.add_argument("--budgets", nargs="+", type=int, default=[100, 200])
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg["search"]["seeds"])
    ap.add_argument("--c-grid", nargs="+", type=float, default=C_GRID)
    a = ap.parse_args()
    if not a.summarize_only:
        run(a)
    summarize()
