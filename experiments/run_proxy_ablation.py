# experiments/run_proxy_ablation.py
# isort: skip_file
"""
Ensemble composition ablation: 1 vs 2 vs 3+ proxy components.

Components: E (SVD expressivity), S (SynFlow), T (NASWOT), C (complexity).
For every non-empty subset (15) and every dataset:
  1. Ranking quality (no search): fit weights on a calibration sample, score a disjoint
     held-out sample -> Kendall tau, Spearman rho, gate recall (top-5% / top-1%).
  2. Search quality: run search_phase3 per seed at --budget -> best_acc, real_eval_count.
Reference arms: `none` (proxy gate off) and `saved` (calibrated ensemble from the main pipeline).

Requires the v5 archtree/zc_proxy.py (raw cache + fit_subset) and the patched search.py (zc_override).

Outputs (outputs/proxy_ablation/results/): ranking, search (append-only, resumable),
summary, paired_tests, size_summary, rank_vs_search CSVs, plus a figure in plots/figures/.

Usage:
    python experiments/run_proxy_ablation.py
    python experiments/run_proxy_ablation.py --budget 100 --datasets cifar10 --seeds 42 111 123
    python experiments/run_proxy_ablation.py --summarize-only
"""
import argparse
import gc
import itertools
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np           # noqa: E402
import pandas as pd          # noqa: E402
import yaml                  # noqa: E402
from loguru import logger    # noqa: E402
from scipy.stats import kendalltau, spearmanr, wilcoxon  # noqa: E402
from dotenv import load_dotenv    # noqa: E402
load_dotenv()

import archtree.storage as storage  # noqa: E402

with open(PROJECT_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

PHASE = "proxy_ablation"
KEYS = ("E", "S", "T", "C")          # proxy components
FULL = "".join(KEYS)                 # label of the full ensemble ("ESTC"), used as the test baseline
N_CAL = N_TEST = 1500                # calibration / held-out sample sizes
SEARCH_FILE = "proxy_ablation_search.csv"

# Logging: INFO to console, DEBUG to a rotating file
logger.remove()
LOG_DIR = PROJECT_ROOT / "outputs" / PHASE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>")
logger.add(LOG_DIR / "proxy_ablation_run.log", level="DEBUG", rotation="10 MB", retention=5)


def subsets():
    """All 15 non-empty component subsets."""
    return [c for r in range(1, len(KEYS) + 1) for c in itertools.combinations(KEYS, r)]


def label(c) -> str:
    return "".join(c)


def _llm_free_backend_kwargs() -> dict:
    """Backend kwargs needed only to construct the search object (use_llm=False, so no LLM call is made)."""
    try:
        import ollama  # noqa: F401
        return dict(backend="ollama")
    except ImportError:
        pass
    keys = [os.environ.get(k, "").strip() for k in ("GROQ_API_KEYS_FILE", "GROQ_API_KEYS", "GROQ_API_KEY")]
    if not any(keys):
        logger.error("Neither `ollama` nor a GROQ key is available (LLMPrior needs one to construct; "
                     "no call is made).")
        sys.exit(1)
    g = cfg["llm"]["groq"]
    return dict(backend="groq", groq_model=g["model"],
                groq_daily_token_budget=g.get("daily_token_budget", 190_000),
                groq_daily_request_budget=g.get("daily_request_budget", 950),
                groq_reasoning_effort=g.get("reasoning_effort", "low"),
                use_key_pool=bool(keys[0] or keys[1]))


def _done() -> set:
    """(dataset, arm, seed, budget) combos already in the search CSV, so reruns resume."""
    p = PROJECT_ROOT / "outputs" / PHASE / "results" / SEARCH_FILE
    if not p.exists():
        return set()
    d = pd.read_csv(p)
    return set(zip(d["dataset"], d["arm"], d["seed"], d["budget"]))


def _save_ranking(rows) -> None:
    """Merge one dataset's ranking rows into the CSV right away so an interrupted run loses nothing."""
    res = storage.get_output_dir(PHASE, "results")
    path = res / "proxy_ablation_ranking.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame(rows)]).drop_duplicates(["dataset", "subset"], keep="last").to_csv(path, index=False)


def gate_recall(scores, gts, pct, top):
    """Fraction of the true top-`top`% archs that pass the proxy gate (top `pct`% by proxy score)."""
    scores, gts = np.asarray(scores), np.asarray(gts)
    gate = scores >= np.percentile(scores, 100 - pct)
    top_mask = gts >= np.percentile(gts, 100 - top)
    return float((gate & top_mask).sum() / max(top_mask.sum(), 1))


def run(args) -> None:
    # Heavy imports are deferred so --summarize-only works without them
    from archtree.zc_proxy import ZCProxyEnsemble
    from archtree.search import search_phase3
    from benchmarks.nb201 import NB201

    bench = NB201(cfg)
    bench._load()
    pct = cfg.get("zc_proxy", {}).get("proxy_threshold_percentile", 35)
    base = ZCProxyEnsemble()                    # only used to compute/cache raw component scores
    backend_kw = _llm_free_backend_kwargs()
    done = _done()
    rank_rows = []

    for dataset in args.datasets:
        # Fixed seed -> same disjoint calibration / held-out split for every dataset
        rng = random.Random(0)
        pool = rng.sample(range(bench.total_archs), N_CAL + N_TEST)
        cal_idx, test_idx = pool[:N_CAL], pool[N_CAL:]

        logger.info(f"[{dataset}] scoring {len(pool)} architectures (cached after first pass) ...")
        t0 = time.time()
        raw_cal = [base.raw(bench.index_to_tokens(i)) for i in cal_idx]
        raw_test = [base.raw(bench.index_to_tokens(i)) for i in test_idx]
        base.flush()                            # persist the raw-score cache
        logger.info(f"[{dataset}] raw scoring done in {time.time() - t0:.0f}s")

        # Ground-truth accuracy for THIS dataset
        gt_cal = [bench.query(i, dataset) for i in cal_idx]
        gt_test = [bench.query(i, dataset) for i in test_idx]

        # ---- Part 1: ranking quality per subset ----
        fitted = {}
        for comps in subsets():
            w, ns, _ = ZCProxyEnsemble.fit_subset(raw_cal, gt_cal, comps)
            inst = ZCProxyEnsemble.from_fit(w, ns)
            sc = [inst._aggregate(r) for r in raw_test]
            tau = kendalltau(sc, gt_test)[0]
            rho = spearmanr(sc, gt_test)[0]
            rank_rows.append({"dataset": dataset, "subset": label(comps), "size": len(comps),
                              "kendall_tau": tau, "spearman_rho": rho,
                              "gate_recall_top5": gate_recall(sc, gt_test, pct, 5),
                              "gate_recall_top1": gate_recall(sc, gt_test, pct, 1),
                              "weights": json.dumps({k: round(v, 4) for k, v in w.items()})})
            fitted[label(comps)] = inst
            logger.info(f"  {label(comps):5s} tau={tau:+.3f} rho={rho:+.3f} "
                        f"gate_recall_top5={rank_rows[-1]['gate_recall_top5']:.2f}")

        _save_ranking([r for r in rank_rows if r["dataset"] == dataset])

        # ---- Part 2: search quality per arm and seed ----
        # `saved` and `none` have no fitted instance here (None): `saved` uses the pipeline's weights,
        # `none` disables the proxy gate entirely.
        arms = [(label(c), fitted[label(c)]) for c in subsets()]
        arms += [("saved", None), ("none", None)]
        for arm, inst in arms:
            is_subset = arm not in ("saved", "none")
            for seed in args.seeds:
                if (dataset, arm, seed, args.budget) in done:
                    continue                    # already finished in an earlier run
                logger.info(f"[PA] {dataset} arm={arm} seed={seed} budget={args.budget}")
                try:
                    t1 = time.time()
                    kw = dict(use_proxy=(arm != "none"), zc_override=inst if is_subset else None)
                    r = search_phase3(bench, args.budget, dataset, seed, phase=PHASE,
                                      alpha=0.0, use_llm=False, ablation_id=f"PA_{arm}",
                                      proxy_threshold_percentile=pct,
                                      results_filename="PA_raw_runs.csv", tcr_filename="PA_raw_tcr.csv",
                                      cache_stats_filename="PA_raw_cache_stats.csv",
                                      dedup_filename="PA_dedup_stats.csv", **kw, **backend_kw)
                    storage.append_result_row(PHASE, SEARCH_FILE, {
                        "dataset": dataset, "arm": arm, "size": len(arm) if is_subset else -1,
                        "seed": seed, "budget": args.budget, "best_acc": r["best_acc"],
                        "best_test_acc": r.get("best_test_acc"),
                        "real_eval_count": r["real_eval_count"], "proxy_eval_count": r["proxy_eval_count"],
                        "wall_time_sec": time.time() - t1})
                except KeyboardInterrupt:
                    logger.warning("Interrupted — append-only CSV, rerun to resume.")
                    sys.exit(0)
                except Exception:
                    logger.exception("    ERROR:")   # log and continue with the next run
                finally:
                    gc.collect()


def _wt(a, b):
    """Paired Wilcoxon p-value and Cohen's d (paired) for a vs b."""
    d = a - b
    if len(d) < 2 or np.all(d == 0):
        return np.nan, np.nan
    try:
        p = wilcoxon(a, b).pvalue
    except ValueError:
        p = np.nan
    sd = d.std(ddof=1)
    return p, (d.mean() / sd if sd > 0 else np.nan)


def summarize() -> None:
    res = storage.get_output_dir(PHASE, "results")
    sp, rp = res / SEARCH_FILE, res / "proxy_ablation_ranking.csv"
    if not sp.exists():
        logger.error("No search results yet.")
        return
    # Keep the latest row if a run was repeated
    s = storage.to_report_metric(pd.read_csv(sp)).drop_duplicates(["dataset", "arm", "seed", "budget"], keep="last")
    # Empty fallback must carry the merged columns, otherwise the merge below raises KeyError
    rk = pd.read_csv(rp) if rp.exists() else pd.DataFrame(
        columns=["dataset", "subset", "kendall_tau", "gate_recall_top5"])

    # Per (dataset, arm) mean/std over seeds, joined with held-out ranking metrics
    summ = (s.groupby(["dataset", "arm"])
              .agg(size=("size", "first"), n_seeds=("seed", "nunique"),
                   best_acc_mean=("best_acc", "mean"), best_acc_std=("best_acc", "std"),
                   real_evals_mean=("real_eval_count", "mean")).reset_index())
    summ = summ.merge(rk[["dataset", "subset", "kendall_tau", "gate_recall_top5"]]
                      .rename(columns={"subset": "arm"}), on=["dataset", "arm"], how="left")
    summ.to_csv(res / "proxy_ablation_summary.csv", index=False)

    # Paired tests: every arm vs the full ensemble, paired over seeds
    tests = []
    for ds, g in s.groupby("dataset"):
        piv = g.pivot(index="seed", columns="arm", values="best_acc")
        if FULL not in piv:
            continue
        for arm in piv.columns:
            if arm == FULL:
                continue
            both = piv[[FULL, arm]].dropna()
            if len(both) < 2:
                continue
            p, d = _wt(both[arm].values, both[FULL].values)
            tests.append({"dataset": ds, "arm": arm, "vs": FULL, "n_paired_seeds": len(both),
                          "mean_diff_pp": float((both[arm] - both[FULL]).mean()),
                          "wilcoxon_p": p, "cohens_d": d,
                          "significant": bool(p < 0.05) if not np.isnan(p) else False})
    tdf = pd.DataFrame(tests)
    if len(tdf):
        # B12: Holm over all arm-vs-full-ensemble tests of a dataset; flag the best single proxy
        tdf["p_holm"] = np.nan
        for ds, idx in tdf.groupby("dataset").groups.items():
            p = tdf.loc[idx, "wilcoxon_p"].fillna(1.0).values
            o, m, adj, r = np.argsort(p), len(p), np.empty(len(p)), 0.0
            for k, i in enumerate(o):
                r = max(r, (m - k) * p[i])
                adj[i] = min(1.0, r)
            tdf.loc[idx, "p_holm"] = adj
        tdf["significant_holm"] = tdf["p_holm"] < 0.05
        singles = tdf[tdf["arm"].isin(list(KEYS))]
        for ds, g in singles.groupby("dataset"):
            best = g.sort_values("mean_diff_pp", ascending=False).iloc[0]
            logger.info(f"   [{ds}] best single proxy {best['arm']} vs full ensemble: "
                        f"{best['mean_diff_pp']:+.2f}pp, p={best['wilcoxon_p']:.3f}, Holm p={best['p_holm']:.3f}")
    tdf.to_csv(res / "proxy_ablation_paired_tests.csv", index=False)

    # Accuracy by number of components (size > 0 excludes the `saved` / `none` reference arms)
    sub = summ[summ["size"] > 0]
    size_tbl = (sub.groupby(["dataset", "size"])
                   .agg(best_acc_mean=("best_acc_mean", "mean"), best_acc_best=("best_acc_mean", "max"),
                        best_acc_worst=("best_acc_mean", "min"), tau_mean=("kendall_tau", "mean"))
                   .reset_index())
    size_tbl.to_csv(res / "proxy_ablation_size_summary.csv", index=False)

    # Does ranking quality (tau / gate recall) predict search quality across subsets?
    rv = []
    for ds, g in sub.dropna(subset=["kendall_tau", "gate_recall_top5"]).groupby("dataset"):
        if len(g) >= 5:                         # too few points -> correlation is meaningless
            r, p = spearmanr(g["kendall_tau"], g["best_acc_mean"])
            r2, p2 = spearmanr(g["gate_recall_top5"], g["best_acc_mean"])
            rv.append({"dataset": ds, "n_subsets": len(g), "rho_tau_vs_best_acc": r, "p_tau": p,
                       "rho_gate_recall_vs_best_acc": r2, "p_gate_recall": p2})
    pd.DataFrame(rv).to_csv(res / "proxy_ablation_rank_vs_search.csv", index=False)

    # Console report
    logger.info("=" * 78)
    for ds, g in summ.groupby("dataset"):
        ref = g[g.arm == "none"]["best_acc_mean"]
        logger.info(f"[{ds}] no-proxy best_acc = {ref.iloc[0]:.2f}" if len(ref) else f"[{ds}]")
        for r in g.sort_values("best_acc_mean", ascending=False).head(6).itertuples():
            tau = r.kendall_tau if pd.notna(r.kendall_tau) else float("nan")
            logger.info(f"   {r.arm:6s} size={int(r.size):2d} best_acc={r.best_acc_mean:.2f}±{r.best_acc_std:.2f} "
                        f"tau={tau:+.3f}")
    for r in size_tbl.itertuples():
        logger.info(f"   size={int(r.size)} mean over subsets = {r.best_acc_mean:.2f} "
                    f"(best subset {r.best_acc_best:.2f}, worst {r.best_acc_worst:.2f})  [{r.dataset}]")
    for r in rv:
        verdict = "ranking quality predicts search quality" if r["rho_tau_vs_best_acc"] > 0.5 else "weak link"
        logger.info(f"   [{r['dataset']}] Spearman(tau, best_acc) across subsets = {r['rho_tau_vs_best_acc']:+.2f} "
                    f"(p={r['p_tau']:.2f}) — {verdict}")
    _plot(summ)
    logger.success("proxy_ablation_* CSVs written.")


def _plot(summ):
    """Scatter of held-out tau vs mean best_acc per subset, one panel per dataset."""
    try:
        import matplotlib
        matplotlib.use("Agg")                   # headless backend
        import matplotlib.pyplot as plt
        from plots.figure_export import install as _gallery; _gallery()   # copy + data CSV to plots/figures
    except ImportError:
        return                                  # plotting is optional
    ds_list = sorted(summ["dataset"].unique())
    if not ds_list:
        return
    fig, axes = plt.subplots(1, len(ds_list), figsize=(4.8 * len(ds_list), 3.8), squeeze=False)
    sc = None
    for ax, ds in zip(axes[0], ds_list):
        g = summ[(summ["dataset"] == ds) & (summ["size"] > 0)]
        sc = ax.scatter(g["kendall_tau"], g["best_acc_mean"], c=g["size"], cmap="viridis", s=42)
        for r in g.itertuples():
            ax.annotate(r.arm, (r.kendall_tau, r.best_acc_mean), fontsize=6, xytext=(2, 2), textcoords="offset points")
        # Reference lines: no proxy (dotted) and saved ensemble (dashed)
        for arm, ls in (("none", ":"), ("saved", "--")):
            v = summ[(summ["dataset"] == ds) & (summ["arm"] == arm)]["best_acc_mean"]
            if len(v):
                ax.axhline(v.iloc[0], ls=ls, color="gray", lw=1, label=arm)
        ax.set(title=ds, xlabel="held-out Kendall tau", ylabel="mean best_acc (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.colorbar(sc, ax=axes[0], label="# components", fraction=0.02)
    fig.savefig(storage.get_figure_path(PHASE, "proxy_ablation"), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize-only", action="store_true", help="skip runs, only rebuild CSVs/figure")
    ap.add_argument("--datasets", nargs="+", default=cfg["benchmarks"]["datasets"])
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg["search"]["seeds"])
    ap.add_argument("--budget", type=int, default=100)
    a = ap.parse_args()
    if not a.summarize_only:
        run(a)
    summarize()