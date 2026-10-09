# tools/analyze_rank_equivalence.py
"""
Reviewer-grade statistics across ALL methods (Oct 4, 2026):

1. Friedman test + average ranks + Nemenyi critical difference (Demsar, JMLR 2006), with a
   critical-difference (CD) diagram. Blocks = (dataset, budget, seed); within a block the methods
   are ranked by test accuracy (1 = best, ties share the mean rank).
2. TOST equivalence (two one-sided paired t-tests, Schuirmann 1987): is ArchTree v2 EQUIVALENT to a
   method within +-delta test-accuracy points? "No significant difference" is not evidence of
   equality; TOST is. delta = 0.5 and 0.25 points; Holm over the cells of each (method, delta).
3. Probability of improvement P(ArchTree v2 > method) over paired (cell, seed) runs (ties count
   1/2), with a stratified bootstrap 95% CI (seeds resampled within cells; Agarwal et al.,
   NeurIPS 2021).

Sources (whatever exists):
    nb201_full_budget   Phase 0 baselines (full budget) vs the Phase 3 headline (gate, ~40% evals)
    nb201_equal_gpu_h   step 22, NB201 arms at equal GPU-hours
    nb101_equal_gpu_h   step 22, NB101
    natssss_equal_gpu_h step 22, NATS-SSS
    nb101_same_budget   step 21, label-free gate, archtree vs random / RE

Outputs (outputs/final/results/): stats_friedman.csv, stats_avg_ranks.csv, stats_tost.csv,
stats_prob_improvement.csv; figures outputs/final/plots/figures/cd_<source>.png
    python tools/analyze_rank_equivalence.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np                                       # noqa: E402
import pandas as pd                                      # noqa: E402
from scipy.stats import friedmanchisquare, rankdata, t as tdist, studentized_range  # noqa: E402

OUT = PROJECT_ROOT / "outputs"
RES = OUT / "final" / "results"
FIG = OUT / "final" / "plots" / "figures"
REF = "archtree_v2"
DELTAS = (0.5, 0.25)


def _holm(p):
    p = np.asarray(p, float)
    o, m, out, run_ = np.argsort(p), len(p), np.empty(len(p)), 0.0
    for k, i in enumerate(o):
        run_ = max(run_, (m - k) * p[i])
        out[i] = min(1.0, run_)
    return out


# ── data: long format (source, cell, seed, method, acc) ──────────────────────
def load_sources():
    out = []
    p0, p3 = OUT / "phase0/results/baseline_summary.csv", OUT / "phase3/results/phase3_main_results.csv"
    if p0.exists() and p3.exists():
        b = pd.read_csv(p0)[["method", "dataset", "budget", "seed", "best_test_acc"]]
        h = pd.read_csv(p3)[["dataset", "budget", "seed", "best_test_acc"]].assign(method=REF)
        d = pd.concat([b, h], ignore_index=True)
        d["cell"] = d["dataset"] + "/b" + d["budget"].astype(str)
        out.append(d.assign(source="nb201_full_budget")[["source", "cell", "seed", "method", "best_test_acc"]])
    pe = OUT / "equal_gpu_hours/results/egh_runs.csv"
    if pe.exists():
        e = pd.read_csv(pe).drop_duplicates(["benchmark", "dataset", "seed", "arm", "k"], keep="last")
        e["method"] = e["arm"].replace({"archtree_zc_unsup": REF})
        e["cell"] = e["dataset"] + "/k" + e["k"].astype(str)
        for bn, g in e.groupby("benchmark"):
            out.append(g.assign(source=f"{bn}_equal_gpu_h")[["source", "cell", "seed", "method", "best_test_acc"]])
    pn = OUT / "nb101/results/nb101_runs.csv"
    if pn.exists():
        n = pd.read_csv(pn)
        n = n[n["gate"] == "zc_unsup"].copy()
        n["method"] = n["proposer"].replace({"archtree": REF})
        n["cell"] = "cifar10/b" + n["budget"].astype(str)
        out.append(n.assign(source="nb101_same_budget")[["source", "cell", "seed", "method", "best_test_acc"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ── 1. Friedman + Nemenyi ────────────────────────────────────────────────────
def friedman(d):
    piv = d.pivot_table(index=["cell", "seed"], columns="method", values="best_test_acc").dropna()
    methods = list(piv.columns)
    k, n = len(methods), len(piv)
    ranks = np.vstack([rankdata(-row) for row in piv.values])     # 1 = best
    avg = ranks.mean(0)
    stat, p = friedmanchisquare(*[piv[m].values for m in methods]) if k >= 3 else (np.nan, np.nan)
    q = studentized_range.ppf(0.95, k, 1e6) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    return methods, avg, stat, p, cd, n


def cd_diagram(name, methods, avg, cd, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from plots.figure_export import install as _gallery; _gallery()   # copy + data CSV to plots/figures
    except ImportError:
        return
    order = np.argsort(avg)
    k = len(methods)
    fig, ax = plt.subplots(figsize=(8, 0.35 * k + 1.6))
    lo, hi = 1, k
    ax.set_xlim(lo - 2.2, hi + 0.3)
    ax.set_ylim(0, k + 1.5)
    ax.invert_xaxis()
    ax.axhline(k + 0.6, color="black", lw=1)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [k + 0.6, k + 0.8], color="black", lw=1)
        ax.text(r, k + 0.95, str(r), ha="center", fontsize=8)
    ax.plot([lo, lo + cd], [k + 1.3, k + 1.3], color="#c0392b", lw=2)
    ax.text(lo + cd / 2, k + 1.4, f"CD = {cd:.2f}", ha="center", fontsize=8, color="#c0392b")
    for row, i in enumerate(order):
        y = k - row
        col = "#c0392b" if methods[i] == REF else "#1f3a5f"
        ax.plot([avg[i], avg[i]], [y, k + 0.6], color=col, lw=0.8)
        ax.plot(avg[i], y, "o", color=col, ms=4)
        ax.text(avg[i] - 0.12, y, f"{methods[i]} ({avg[i]:.2f})", va="center", fontsize=8, color=col,
                ha="left")       # x axis is inverted: "left" = text to the right of the dot
    # groups not significantly different from the best: bar over methods within CD of the best
    best = avg[order[0]]
    same = [i for i in order if avg[i] - best <= cd]
    if len(same) > 1:
        ax.plot([avg[same[0]], avg[same[-1]]], [0.5, 0.5], color="black", lw=3)
        ax.text(avg[same[0]], 0.15, "not significantly worse than the best", fontsize=7)
    ax.axis("off")
    ax.set_title(f"Average rank (1 = best), {name}", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ── 2. TOST ──────────────────────────────────────────────────────────────────
def tost(diff, delta):
    """Paired TOST on diffs (ref - other): H0 |mu| >= delta. Returns the TOST p-value."""
    n = len(diff)
    if n < 3:
        return np.nan
    m, sd = diff.mean(), diff.std(ddof=1)
    if sd == 0:
        return 0.0 if abs(m) < delta else 1.0
    se = sd / np.sqrt(n)
    p_low = 1 - tdist.cdf((m + delta) / se, n - 1)     # H0: mu <= -delta
    p_up = tdist.cdf((m - delta) / se, n - 1)          # H0: mu >= +delta
    return float(max(p_low, p_up))


# ── 3. probability of improvement ────────────────────────────────────────────
def prob_improvement(piv, other, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    cells = piv.index.get_level_values(0).unique()

    def poi(frame):
        d = frame[REF].values - frame[other].values
        return float(np.mean((d > 0) + 0.5 * (d == 0)))

    est = poi(piv)
    boots = []
    groups = {c: piv.xs(c, level=0) for c in cells}
    for _ in range(n_boot):
        parts = [g.iloc[rng.integers(0, len(g), len(g))] for g in groups.values()]
        boots.append(poi(pd.concat(parts)))
    return est, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    d = load_sources()
    if d.empty:
        print("no results found")
        return
    RES.mkdir(parents=True, exist_ok=True)
    fr_rows, rank_rows, tost_rows, poi_rows = [], [], [], []
    for src, g in d.groupby("source"):
        if REF not in set(g["method"]):
            continue
        methods, avg, stat, p, cd, n = friedman(g)
        fr_rows.append({"source": src, "n_methods": len(methods), "n_blocks": n, "friedman_chi2": stat,
                        "friedman_p": p, "nemenyi_cd": cd,
                        "archtree_v2_avg_rank": float(avg[methods.index(REF)]),
                        "best_method": methods[int(np.argmin(avg))], "best_avg_rank": float(avg.min())})
        for m_, r in zip(methods, avg):
            rank_rows.append({"source": src, "method": m_, "avg_rank": float(r),
                              "within_cd_of_archtree": bool(abs(r - avg[methods.index(REF)]) <= cd)})
        cd_diagram(src, methods, avg, cd, FIG / f"cd_{src}.png")
        piv = g.pivot_table(index=["cell", "seed"], columns="method", values="best_test_acc")
        for other in [m_ for m_ in methods if m_ != REF]:
            both = piv[[REF, other]].dropna()
            for delta in DELTAS:
                ps, cells_ = [], []
                for c, gc in both.groupby(level=0):
                    diff = (gc[REF] - gc[other]).values
                    ps.append(tost(diff, delta))
                    cells_.append((c, float(diff.mean()), len(diff)))
                ph = _holm(np.nan_to_num(ps, nan=1.0))
                for (c, md, nn), pv, h in zip(cells_, ps, ph):
                    tost_rows.append({"source": src, "other": other, "delta": delta, "cell": c, "n_seeds": nn,
                                      "mean_diff": md, "tost_p": pv, "tost_p_holm": h, "equivalent": bool(h < 0.05)})
            est, lo, hi = prob_improvement(both, other)
            poi_rows.append({"source": src, "other": other, "n_pairs": len(both),
                             "P_archtree_better": est, "ci95_lo": lo, "ci95_hi": hi})
    fr, rk, ts, pi = map(pd.DataFrame, (fr_rows, rank_rows, tost_rows, poi_rows))
    fr.to_csv(RES / "stats_friedman.csv", index=False)
    rk.to_csv(RES / "stats_avg_ranks.csv", index=False)
    ts.to_csv(RES / "stats_tost.csv", index=False)
    pi.to_csv(RES / "stats_prob_improvement.csv", index=False)
    print("=" * 78)
    for r in fr.itertuples():
        print(f"[{r.source}] Friedman p={r.friedman_p:.2g}  CD={r.nemenyi_cd:.2f}  ArchTree v2 avg rank "
              f"{r.archtree_v2_avg_rank:.2f} of {r.n_methods} (best: {r.best_method} {r.best_avg_rank:.2f})")
    if len(ts):
        s = (ts.groupby(["source", "other", "delta"])["equivalent"]
               .agg(n_equiv="sum", n_cells="count").reset_index())
        print("TOST equivalence (cells equivalent / cells), ArchTree v2 vs:")
        for r in s.itertuples():
            print(f"  {r.source:20s} {r.other:20s} +-{r.delta:.2f} pts: {r.n_equiv}/{r.n_cells}")
    for r in pi.itertuples():
        print(f"  P(ArchTree v2 > {r.other:20s}) [{r.source}] = {r.P_archtree_better:.2f} "
              f"[{r.ci95_lo:.2f}, {r.ci95_hi:.2f}]")
    print(f"written: {RES}/stats_*.csv, {FIG}/cd_*.png")


if __name__ == "__main__":
    main()
