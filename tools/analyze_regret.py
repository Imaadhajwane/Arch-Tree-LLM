# tools/analyze_regret.py
# isort: skip_file
"""
§3.1 — Evidence for the informal regret argument (docs/REGRET_ARGUMENT.md). No GPU, no new search.

A. Landscape structure (exact; NB201 is a 15,625-leaf table)
     eta2        share of accuracy variance explained by the first k tokens (0..1)
     within_std  mean std of leaf accuracy inside a depth-k subtree (pp)
     N_k(eps)    # depth-k prefixes whose subtree has a leaf within eps pp of the global optimum
     gamma(eps)  growth exponent of N_k(eps): N_k ~ 5^(gamma*k)
                 (~0 = near-optimal set doesn't multiply with depth; ~1 = no structure to exploit)

B. Empirical simple regret of finished runs (from existing per-run logs)
     r_t = f* - best_so_far_t after t REAL evals; fit r_t ~ a * t^(-b) on the seed-mean curve
     for t >= T0, with a 95% bootstrap CI over seeds. "random_exact" is the analytic expected
     regret of uniform random search without replacement, used as the reference curve.
     NOTE: descriptive summary only; a power-law fit on 25-200 evals is NOT a regret-bound check.

Outputs: outputs/final/results/landscape_smoothness.csv, regret_fit.csv, regret_curves.csv
Figures: outputs/final/plots/figures/regret_landscape.png, regret_curves_{dataset}.png

Usage:
    python tools/analyze_regret.py
    python tools/analyze_regret.py --datasets cifar10 --boot 500
"""
import argparse
import glob
import json
import sys
from math import log
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402
import yaml           # noqa: E402
from loguru import logger  # noqa: E402
from scipy.special import gammaln  # noqa: E402

import archtree.storage as storage  # noqa: E402

with open(PROJECT_ROOT / "config.yaml") as _fh:
    cfg = yaml.safe_load(_fh)
OUT = PROJECT_ROOT / "outputs"
L, K = 6, 5          # trie depth (edges) and branching factor (ops)
N = K ** L           # 15,625 leaves
EPS = (0.25, 0.5, 1.0, 2.0, 4.0)  # "near-optimal" thresholds in accuracy pp
T0 = 5               # first eval index used in the power-law fit

logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>")

RUN_GLOBS = [
    str(OUT / "phase0" / "logs" / "*__nb201__*__seed*__budget*.json"),
    str(OUT / "phase1" / "logs" / "archtree_v1__nb201__*__seed*__budget*.json"),
    str(OUT / "phase2" / "logs" / "archtree_zc__nb201__*__seed*__budget*.json"),
    str(OUT / "phase3" / "logs" / "archtree_llm__nb201__*__seed*__budget*.json"),
    str(OUT / "ablations" / "logs" / "*A7*__nb201__*__seed*__budget*.json"),
]


def accuracy_table(dataset: str) -> np.ndarray:
    """Accuracy of all N architectures, indexed by NB201 index."""
    from benchmarks.nb201 import NB201
    b = NB201(cfg)
    b._load()
    return np.array([b.query(i, dataset) for i in range(N)], dtype=float)


# ── A. landscape ────────────────────────────────────────────────────────────
def landscape(acc: np.ndarray, dataset: str) -> list:
    fstar, var = acc.max(), acc.var()
    rows = []
    for k in range(1, L + 1):
        # Token 0 is the most significant base-5 digit of the index, so the depth-k prefix id
        # is just the index integer-divided by K^(L-k).
        pid = np.arange(N) // (K ** (L - k))
        n_pre = K ** k
        cnt = np.bincount(pid, minlength=n_pre)
        mean = np.bincount(pid, weights=acc, minlength=n_pre) / cnt
        sq = np.bincount(pid, weights=acc ** 2, minlength=n_pre) / cnt - mean ** 2  # within-prefix variance
        sub_max = np.full(n_pre, -np.inf)
        np.maximum.at(sub_max, pid, acc)  # best leaf in each subtree
        between = np.average((mean - acc.mean()) ** 2, weights=cnt)  # between-prefix variance

        row = {"dataset": dataset, "depth": k, "n_prefixes": n_pre,
               "eta2": between / var if var > 0 else np.nan,
               "within_std": float(np.sqrt(np.clip(sq, 0, None)).mean())}
        for e in EPS:
            row[f"N_k_eps{e:g}"] = int((sub_max >= fstar - e).sum())
        rows.append(row)

    df = pd.DataFrame(rows)
    for e in EPS:  # gamma = slope of ln N_k vs depth, in units of ln K
        y = np.log(df[f"N_k_eps{e:g}"].clip(lower=1).values)
        df[f"gamma_eps{e:g}"] = np.polyfit(df["depth"].values, y, 1)[0] / log(K)
    return df.to_dict("records")


# ── B. regret ───────────────────────────────────────────────────────────────
def random_expected_best(acc: np.ndarray, T: int) -> np.ndarray:
    """E[max of t uniform draws WITHOUT replacement] for t = 1..T (exact).
    P(best of t draws is the j-th best overall) = C(N-j, t-1) / C(N, t), computed in log space."""
    s = np.sort(acc)[::-1]          # s[j-1] = j-th best accuracy
    j = np.arange(1, N + 1)
    out = np.empty(T)
    for t in range(1, T + 1):
        valid = (N - j) >= (t - 1)  # C(n, k) = 0 when n < k
        logp = np.full(N, -np.inf)
        logp[valid] = gammaln(N - j[valid] + 1) - gammaln(t) - gammaln(N - j[valid] - t + 2)
        logp -= gammaln(N + 1) - gammaln(t + 1) - gammaln(N - t + 1)
        out[t - 1] = float((np.exp(logp) * s).sum())
    return out


def fit_power(mean_regret: np.ndarray, t0: int = T0):
    """Fit r_t = a * t^(-b) by least squares in log-log space over t >= t0. Returns (b, a)."""
    t = np.arange(1, len(mean_regret) + 1)
    m = t >= t0
    y = np.log(np.clip(mean_regret[m], 1e-3, None))  # clip: regret hits 0 once f* is found
    slope, icpt = np.polyfit(np.log(t[m]), y, 1)
    return -slope, float(np.exp(icpt))


def load_runs(datasets):
    rows = []
    for pat in RUN_GLOBS:
        for f in glob.glob(pat):
            try:
                with open(f) as fh:
                    j = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            h = j.get("eval_history") or []
            if not h or (datasets and j.get("dataset") not in datasets):
                continue
            m = j.get("method", Path(f).name.split("__")[0])
            if j.get("ablation_id"):
                m = f"{m}[{j['ablation_id']}]"
            rows.append({"method": m, "dataset": j["dataset"], "budget": j["budget"],
                         "seed": j["seed"], "acc": np.array([e["acc"] for e in h], float)})
    return rows


def regret_analysis(runs, fstars, args):
    df = pd.DataFrame(runs)
    fit_rows, curve_rows = [], []
    rng = np.random.default_rng(0)  # fixed seed -> reproducible bootstrap CIs
    for (m, ds, b), g in df.groupby(["method", "dataset", "budget"]):
        T = int(max(len(a) for a in g["acc"]))
        if T < T0 + 3:  # too short to fit
            continue
        # Best-so-far curve per seed; shorter runs are padded with their final best
        curves = np.array([np.maximum.accumulate(np.concatenate([a, np.full(T - len(a), a.max())]))
                           for a in g["acc"]])
        reg = np.clip(fstars[ds] - curves, 0, None)  # (n_seeds, T) simple regret
        mean_reg = reg.mean(0)

        b_hat, a_hat = fit_power(mean_reg)
        boots = [fit_power(reg[rng.integers(0, len(reg), len(reg))].mean(0))[0]  # resample seeds
                 for _ in range(args.boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        fit_rows.append({"method": m, "dataset": ds, "budget": b, "n_seeds": len(reg), "T_real_evals": T,
                         "b_exponent": b_hat, "b_ci_lo": lo, "b_ci_hi": hi, "a_scale": a_hat,
                         "simple_regret_at_T": mean_reg[-1], "avg_regret_per_eval": float(reg.mean())})
        for t in range(T):
            curve_rows.append({"method": m, "dataset": ds, "budget": b, "t": t + 1,
                               "mean_regret": mean_reg[t],
                               "sem": reg[:, t].std(ddof=1) / np.sqrt(len(reg)) if len(reg) > 1 else 0.0})
    return pd.DataFrame(fit_rows), pd.DataFrame(curve_rows)


# ── main ────────────────────────────────────────────────────────────────────
def main(args) -> None:
    res = storage.get_output_dir("final", "results")
    datasets = args.datasets or cfg["benchmarks"]["datasets"]

    # A. landscape
    tables, fstars, land = {}, {}, []
    for ds in datasets:
        tables[ds] = accuracy_table(ds)
        fstars[ds] = tables[ds].max()
        land += landscape(tables[ds], ds)
    ldf = pd.DataFrame(land)
    ldf.to_csv(res / "landscape_smoothness.csv", index=False)
    logger.info("=" * 78)
    logger.info("A. landscape (NB201, exact).  eta2 = variance explained by first k tokens; "
                "gamma ~ growth of near-optimal cells")
    for ds, g in ldf.groupby("dataset"):
        e = " ".join(f"k{r.depth}:{r.eta2:.2f}" for r in g.itertuples())
        logger.info(f"  {ds:15s} eta2  {e}")
        logger.info(f"  {'':15s} gamma(1pp)={g['gamma_eps1'].iloc[0]:.2f}  gamma(2pp)={g['gamma_eps2'].iloc[0]:.2f}  "
                    f"N_6(1pp)={int(g['N_k_eps1'].iloc[-1])} leaves within 1pp of optimum")

    # B. regret
    runs = load_runs(datasets)
    if not runs:
        logger.warning("No run logs found - landscape only.")
        _plot_landscape(ldf)
        return
    fit, curves = regret_analysis(runs, fstars, args)

    # Analytic random-search reference, evaluated up to the longest run seen per dataset
    ref_rows = []
    for ds in datasets:
        sub = curves[curves["dataset"] == ds] if len(curves) else curves
        T = int(sub["t"].max()) if len(sub) else 200
        reg = fstars[ds] - random_expected_best(tables[ds], T)
        b_hat, a_hat = fit_power(reg)
        ref_rows.append({"method": "random_exact", "dataset": ds, "budget": T, "n_seeds": 0, "T_real_evals": T,
                         "b_exponent": b_hat, "b_ci_lo": np.nan, "b_ci_hi": np.nan, "a_scale": a_hat,
                         "simple_regret_at_T": reg[-1], "avg_regret_per_eval": float(reg.mean())})
        curves = pd.concat([curves, pd.DataFrame({"method": "random_exact", "dataset": ds, "budget": T,
                                                   "t": np.arange(1, T + 1), "mean_regret": reg, "sem": 0.0})])
    fit = pd.concat([fit, pd.DataFrame(ref_rows)], ignore_index=True)
    fit.to_csv(res / "regret_fit.csv", index=False)
    curves.to_csv(res / "regret_curves.csv", index=False)

    logger.info("=" * 78)
    logger.info("B. simple-regret power-law exponent b (r_t ~ t^-b), largest budget per method; "
                "larger b = faster decay")
    for ds in datasets:
        g = fit[fit["dataset"] == ds]
        gb = g.sort_values("budget").groupby("method").tail(1).sort_values("b_exponent", ascending=False)
        for r in gb.itertuples():
            ci = "" if np.isnan(r.b_ci_lo) else f" [{r.b_ci_lo:.2f},{r.b_ci_hi:.2f}]"
            logger.info(f"  {ds:14s} {r.method:26s} b={r.b_exponent:5.2f}{ci}  "
                        f"r_T={r.simple_regret_at_T:5.2f}pp  (T={r.T_real_evals})")
    _plot_landscape(ldf)
    _plot_curves(curves, datasets)
    logger.success("landscape_smoothness.csv / regret_fit.csv / regret_curves.csv written.")


# ── plots ───────────────────────────────────────────────────────────────────
def _plt():
    """pyplot with a headless backend, or None if matplotlib isn't installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from plots.figure_export import install as _gallery; _gallery()   # copy + data CSV to plots/figures
        return plt
    except ImportError:
        return None


def _plot_landscape(ldf):
    plt = _plt()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for ds, g in ldf.groupby("dataset"):
        ax[0].plot(g["depth"], g["eta2"], marker="o", label=ds)
        ax[1].plot(g["depth"], g["N_k_eps1"], marker="o", label=ds)
    ax[0].set(xlabel="prefix length k", ylabel="variance explained (eta²)", title="prefix locality")
    ax[1].set(xlabel="prefix length k", ylabel="# depth-k cells within 1pp of optimum", yscale="log",
              title="near-optimal cells")
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(storage.get_figure_path("final", "regret_landscape"), dpi=180)
    plt.close(fig)


def _plot_curves(curves, datasets):
    plt = _plt()
    if plt is None:
        return
    for ds in datasets:
        g = curves[curves["dataset"] == ds]
        if g.empty:
            continue
        fig, ax = plt.subplots(figsize=(5.4, 4))
        for m, gm in g.groupby("method"):
            gm = gm[gm["budget"] == gm["budget"].max()]  # largest budget per method only
            ls = "--" if m == "random_exact" else "-"
            ax.plot(gm["t"], np.clip(gm["mean_regret"], 1e-3, None), ls, lw=1.3, label=m)
        ax.set(xscale="log", yscale="log", xlabel="real evals t", ylabel="simple regret (pp)", title=ds)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(storage.get_figure_path("final", f"regret_curves_{ds}"), dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--boot", type=int, default=200, help="bootstrap resamples for the exponent CI")
    main(ap.parse_args())