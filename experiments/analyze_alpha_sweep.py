# experiments/analyze_alpha_sweep.py
"""
Reads outputs/alpha_sweep/results/alpha_sweep_results.csv, ranks every
non-zero alpha against alpha=0.0 (LLM-off baseline) via paired Wilcoxon
per (dataset, budget), same statistical shape as run_phase6_aggregate.py's
main test (paired-per-seed, not pooled-across-budgets).

Oct 3, 2026: Holm correction over all (alpha, dataset, budget) comparisons; alphas whose runs are
identical seed-by-seed (the prior saturates: alpha >= 0.2 gives the same search) are reported as one
group; and alpha is NOT to be picked from this table for the headline (that would select on the
test set) - the sweep is reported as the C7 study, the headline keeps alpha_init = 0.

Usage:
    python experiments/analyze_alpha_sweep.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from loguru import logger

import archtree.storage as storage

logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>")

df = storage.load_report_csv("alpha_sweep", "alpha_sweep_results.csv")   # test accuracy when available
df.columns = [c.strip() for c in df.columns]

logger.info("=== mean best_acc by (alpha, dataset, budget) ===")
grp = df.groupby(["alpha", "dataset", "budget"])["best_acc"].agg(["mean", "std", "count"])
print(grp.round(3).to_string())

rows = []
for dataset in df["dataset"].unique():
    for budget in sorted(df["budget"].unique()):
        base = df[(df.alpha == 0.0) & (df.dataset == dataset) & (df.budget == budget)]
        base = base.set_index("seed")["best_acc"]
        if base.empty:
            continue
        for alpha in sorted(a for a in df["alpha"].unique() if a != 0.0):
            cand = df[(df.alpha == alpha) & (df.dataset == dataset) & (df.budget == budget)]
            cand = cand.set_index("seed")["best_acc"]
            common = sorted(set(base.index) & set(cand.index))
            if len(common) < 2:
                continue
            a_vals = cand.loc[common].values
            b_vals = base.loc[common].values
            diff = a_vals - b_vals
            if np.all(diff == 0):
                continue
            stat, p = wilcoxon(a_vals, b_vals)
            sd = diff.std(ddof=1)
            d = float(diff.mean() / sd) if sd > 0 else float("nan")
            rows.append({
                "alpha": alpha, "dataset": dataset, "budget": budget,
                "n_paired_seeds": len(common),
                "mean_diff_vs_alpha0": diff.mean(),
                "p_value": p, "cohens_d": d,
            })

res = pd.DataFrame(rows)


def _holm(p):
    p = np.asarray(p, float)
    o, m, out, run_ = np.argsort(p), len(p), np.empty(len(p)), 0.0
    for k, i in enumerate(o):
        run_ = max(run_, (m - k) * p[i])
        out[i] = min(1.0, run_)
    return out


if len(res):
    # identical runs (same result for every seed) are ONE comparison, not several
    piv = df.pivot_table(index=["dataset", "budget", "seed"], columns="alpha", values="best_acc")
    groups, rep = {}, {}
    for a in sorted(c for c in piv.columns if c != 0.0):
        key = next((g for g in groups if piv[g].equals(piv[a])), a)
        groups.setdefault(key, []).append(a)
        rep[a] = key
    res["same_runs_as_alpha"] = res["alpha"].map(rep)
    first = res["alpha"] == res["same_runs_as_alpha"]
    res["p_holm"] = np.nan
    res.loc[first, "p_holm"] = _holm(res.loc[first, "p_value"].values)
    res["p_holm"] = res.groupby(["same_runs_as_alpha", "dataset", "budget"])["p_holm"].transform("max")
    res["significant"] = res["p_holm"] < 0.05
    res["beats_alpha0_significantly"] = res["significant"] & (res["mean_diff_vs_alpha0"] > 0)
    sat = [g for g, m in groups.items() if len(m) > 1]
    for g in sat:
        logger.info(f"alphas {groups[g]} give IDENTICAL runs for every seed (prior saturates) - one comparison.")
    distinct = res[first]
    logger.info(f"sign consistency: {int((distinct.mean_diff_vs_alpha0 > 0).sum())} of {len(distinct)} distinct "
                f"comparisons favour the LLM prior (mean {distinct.mean_diff_vs_alpha0.mean():+.3f} pts)")
out_path = "outputs/alpha_sweep/results/alpha_vs_baseline_stats.csv"
res.to_csv(out_path, index=False)
logger.success(f"written -> {out_path}")

logger.info("=== full comparison table (alpha vs alpha=0.0) ===")
print(res.to_string(index=False))

winners = res[res.beats_alpha0_significantly]
logger.info("=== alpha values that SIGNIFICANTLY beat alpha=0.0 (Holm) ===")
if winners.empty:
    logger.warning("NONE after Holm - no alpha beats the no-LLM baseline significantly at any "
                    "(dataset, budget). Keep alpha_init=0.0; report the direction / effect sizes as C7.")
else:
    print(winners.to_string(index=False))
    best = winners.loc[winners.cohens_d.idxmax()]
    logger.success(f"Best candidate: alpha={best.alpha} "
                    f"(dataset={best.dataset}, budget={best.budget}, "
                    f"cohens_d={best.cohens_d:.3f}, p={best.p_value:.4f})")
    logger.info("Report this as the C7 result. Do NOT set llm.alpha_init from this table for the "
                "headline runs: alpha would then be chosen on the test set.")