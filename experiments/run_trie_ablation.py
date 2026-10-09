# experiments/run_trie_ablation.py
# isort: skip_file
"""
A7 — Trie vs flat search ablation.

Three arms, everything else identical (proxy calibration + gate, budget accounting,
UCB constants, backup_mode="max", LLM off, same seeds/datasets/budgets):

  trie       search_phase3(alpha=0, use_llm=False)   prefix-conditioned stats, pruning, trie dedup
  flat_set   search_flat_ucb(dedup="set")            position-wise UCB1, visited-set dedup only
  flat_none  search_flat_ucb(dedup="none")           same, no dedup (duplicates cost real evals)

The trie arm is re-run here (ablation_id "A7_trie") rather than reusing A1, so all arms
share the same code path and proxy cache. It should still match A1 as a cross-check.

Outputs (outputs/ablations/results/):
  ablation_A7_no_trie.csv   one row per (arm, dataset, budget, seed)  [append-only, resumable]
  A7_summary.csv            mean/std best_acc, real evals, duplicate counts
  A7_paired_tests.csv       trie - flat, paired Wilcoxon + Cohen's d
  A7_raw_*.csv              raw per-run files written by the search functions
outputs/ablations/plots/figures/A7_no_trie_ablation.png

Usage (from archtree-llm/ root):
    python experiments/run_trie_ablation.py                   # run + summarise (resumable)
    python experiments/run_trie_ablation.py --summarize-only  # no GPU needed
    python experiments/run_trie_ablation.py --budgets 50 100 --datasets cifar10

Requires Phase 2 calibration files for every dataset and a GPU (zc_proxy needs CUDA).
"""
import argparse
import gc
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _cal_path(ds):
    """Phase-2 calibration file the gate will load (respects zc_proxy.calibration_mode)."""
    from archtree import storage as _st, settings as _se
    return _st._zc_cal_path("phase2", "nb201", ds, _se.CALIBRATION_MODE)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402
import yaml                 # noqa: E402
from loguru import logger   # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from dotenv import load_dotenv    # noqa: E402
load_dotenv()

import archtree.storage as storage  # noqa: E402

with open(PROJECT_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

RESULTS_FILE = "ablation_A7_no_trie.csv"
ARMS = ["trie", "flat_set", "flat_none"]

# Logging: INFO to console, DEBUG to a rotating file
logger.remove()
LOG_DIR = PROJECT_ROOT / "outputs" / "ablations" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>")
logger.add(LOG_DIR / "A7_run.log", level="DEBUG", rotation="10 MB", retention=5)


def _llm_free_backend_kwargs() -> dict:
    """Backend kwargs needed only because LLMPrior.__init__ builds a client even with the LLM off.
    Prefers ollama (no server or key needed to construct); falls back to Groq keys."""
    try:
        import ollama  # noqa: F401
        return dict(backend="ollama")
    except ImportError:
        pass
    keys = [os.environ.get(k, "").strip() for k in ("GROQ_API_KEYS_FILE", "GROQ_API_KEYS", "GROQ_API_KEY")]
    if not any(keys):
        logger.error("Neither the `ollama` package nor any GROQ key is available — LLMPrior needs "
                     "one of them just to construct (no call is ever made in this script).")
        sys.exit(1)
    g = cfg["llm"]["groq"]
    return dict(backend="groq", groq_model=g["model"],
                groq_daily_token_budget=g.get("daily_token_budget", 190_000),
                groq_daily_request_budget=g.get("daily_request_budget", 950),
                groq_reasoning_effort=g.get("reasoning_effort", "low"),
                use_key_pool=bool(keys[0] or keys[1]))


def _done_keys() -> set:
    """(dataset, budget, seed, arm) combos already recorded, so reruns resume."""
    path = PROJECT_ROOT / "outputs" / "ablations" / "results" / RESULTS_FILE
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(zip(df["dataset"], df["budget"], df["seed"], df["param_value"]))


def run(args) -> None:
    # Fail early if any dataset lacks its Phase 2 proxy calibration
    missing = [ds for ds in args.datasets
               if not _cal_path(ds).exists()]
    if missing:
        logger.error(f"Missing Phase 2 calibration for: {missing}. Run run_phase2.py first.")
        sys.exit(1)

    # Heavy imports are deferred so --summarize-only works without them
    from benchmarks.nb201 import NB201
    from archtree.search import search_phase3, search_flat_ucb

    bench = NB201(cfg)
    bench._load()
    pct = cfg.get("zc_proxy", {}).get("proxy_threshold_percentile", 35)
    backend_kw = _llm_free_backend_kwargs()
    done = _done_keys()
    n_run = n_skip = n_err = 0
    logger.info(f"A7: arms={args.arms} datasets={args.datasets} budgets={args.budgets} "
                f"seeds={len(args.seeds)}  proxy_threshold_percentile={pct}")

    for dataset in args.datasets:
        for budget in args.budgets:
            for seed in args.seeds:
                for arm in args.arms:
                    if (dataset, budget, seed, arm) in done:
                        n_skip += 1
                        continue
                    logger.info(f"[A7] {dataset} | budget={budget} | seed={seed} | arm={arm}")
                    try:
                        t0 = time.time()
                        if arm == "trie":
                            r = search_phase3(
                                bench, budget, dataset, seed, phase="ablations",
                                alpha=0.0, use_llm=False, ablation_id="A7_trie",
                                proxy_threshold_percentile=pct,
                                results_filename="A7_raw_trie_runs.csv",
                                tcr_filename="A7_raw_trie_tcr.csv",
                                cache_stats_filename="A7_raw_trie_cache_stats.csv",
                                dedup_filename="A7_dedup_stats.csv",
                                **backend_kw)
                            # Trie dedup means every real eval is unique, so duplicate real evals are 0
                            uniq, dup_real = r["real_eval_count"], 0
                        else:
                            r = search_flat_ucb(
                                bench, budget, dataset, seed,
                                dedup="set" if arm == "flat_set" else "none",
                                phase="ablations", ablation_id="A7",
                                proxy_threshold_percentile=pct,
                                results_filename="A7_raw_flat_runs.csv",
                                dedup_filename="A7_dedup_stats.csv")
                            uniq, dup_real = r["unique_real_evals"], r["dup_real_evals"]
                        elapsed = time.time() - t0
                        storage.append_result_row("ablations", RESULTS_FILE, {
                            "ablation_id": "A7", "varied_param": "arm", "param_value": arm,
                            "benchmark": "nb201", "dataset": dataset, "seed": seed, "budget": budget,
                            "best_acc": r["best_acc"], "best_test_acc": r.get("best_test_acc"),
                            "real_eval_count": r["real_eval_count"],
                            "proxy_eval_count": r.get("proxy_eval_count"),
                            "unique_real_evals": uniq, "dup_real_evals": dup_real,
                            "dup_proposals": r.get("dup_proposals"),
                            "wall_time_sec": elapsed,
                        })
                        logger.success(f"    best_acc={r['best_acc']:.2f}%  real={r['real_eval_count']}  "
                                       f"unique={uniq}  dup_proposals={r.get('dup_proposals')}  {elapsed:.1f}s")
                        n_run += 1
                    except KeyboardInterrupt:
                        logger.warning("Interrupted — progress saved (append-only CSV), rerun to resume.")
                        sys.exit(0)
                    except Exception:
                        logger.exception("    ERROR:")   # log and continue with the next run
                        n_err += 1
                    finally:
                        gc.collect()
    logger.info(f"A7 done — ran={n_run} skipped={n_skip} errors={n_err}")


def _paired(a: np.ndarray, b: np.ndarray):
    """Paired Wilcoxon p-value and paired Cohen's d for a vs b."""
    diff = a - b
    if len(diff) < 2 or np.all(diff == 0):
        return float("nan"), float("nan")
    try:
        p = wilcoxon(a, b).pvalue
    except ValueError:
        p = float("nan")
    sd = diff.std(ddof=1)
    d = float(diff.mean() / sd) if sd > 0 else float("nan")
    return p, d


def summarize() -> None:
    path = PROJECT_ROOT / "outputs" / "ablations" / "results" / RESULTS_FILE
    if not path.exists():
        logger.error(f"{path} not found — run the ablation first.")
        return
    df = storage.to_report_metric(pd.read_csv(path))   # best_acc := test acc when available
    # A resumed/duplicated row would double-count; keep the last per key
    df = df.drop_duplicates(["dataset", "budget", "seed", "param_value"], keep="last")

    # Per (dataset, budget, arm) statistics over seeds
    summ = (df.groupby(["dataset", "budget", "param_value"])
              .agg(n_seeds=("seed", "nunique"),
                   best_acc_mean=("best_acc", "mean"), best_acc_std=("best_acc", "std"),
                   real_evals_mean=("real_eval_count", "mean"),
                   unique_real_mean=("unique_real_evals", "mean"),
                   dup_real_mean=("dup_real_evals", "mean"),
                   dup_proposals_mean=("dup_proposals", "mean"))
              .reset_index().rename(columns={"param_value": "arm"}))
    # Rewrite from scratch so stale rows never linger
    storage.clear_result_file("ablations", "A7_summary.csv")
    for _, row in summ.iterrows():
        storage.append_result_row("ablations", "A7_summary.csv", row.to_dict())

    # Paired tests: trie vs each flat arm, paired over seeds
    storage.clear_result_file("ablations", "A7_paired_tests.csv")
    logger.info("=" * 78)
    logger.info("A7 paired tests: trie - flat (positive = trie better); Wilcoxon + paired Cohen's d")
    for (dataset, budget), g in df.groupby(["dataset", "budget"]):
        piv = g.pivot(index="seed", columns="param_value", values="best_acc")
        if "trie" not in piv:
            continue
        for arm in ("flat_set", "flat_none"):
            if arm not in piv:
                continue
            both = piv[["trie", arm]].dropna()   # only seeds present in both arms
            if len(both) < 2:
                continue
            p, d = _paired(both["trie"].values, both[arm].values)
            diff = float((both["trie"] - both[arm]).mean())
            storage.append_result_row("ablations", "A7_paired_tests.csv", {
                "dataset": dataset, "budget": budget, "arm_a": "trie", "arm_b": arm,
                "n_paired_seeds": len(both), "mean_diff_pp": diff,
                "wilcoxon_p": p, "cohens_d": d,
                "significant": bool(p < 0.05) if not np.isnan(p) else False,
            })
            logger.info(f"  {dataset:15s} b={budget:<4d} trie-{arm:9s} diff={diff:+.2f}pp  "
                        f"p={p:.3f}  d={d:+.2f}  n={len(both)}")

    # How much of flat_none's real-eval budget was wasted on duplicates
    for (dataset, budget), g in summ.groupby(["dataset", "budget"]):
        fn = g[g["arm"] == "flat_none"]
        if len(fn):
            r = fn.iloc[0]
            logger.info(f"  {dataset:15s} b={budget:<4d} flat_none: {r['dup_real_mean']:.1f} of "
                        f"{r['real_evals_mean']:.1f} real evals were duplicates")

    _plot(summ)
    logger.success("A7_summary.csv / A7_paired_tests.csv written.")


def _plot(summ: pd.DataFrame) -> None:
    """Best accuracy vs budget per dataset, one line per arm (mean ± s.e.m. over seeds)."""
    try:
        import matplotlib
        matplotlib.use("Agg")                    # headless backend
        import matplotlib.pyplot as plt
        from plots.figure_export import install as _gallery; _gallery()   # copy + data CSV to plots/figures
    except ImportError:
        logger.warning("matplotlib missing — skipping A7 figure.")
        return
    datasets = sorted(summ["dataset"].unique())
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 3.8), squeeze=False)
    # arm -> (color, marker, legend label)
    style = {"trie": ("#1f3a5f", "o", "ArchTree (trie)"),
             "flat_set": ("#c0392b", "s", "flat UCB1 + set dedup"),
             "flat_none": ("#7f8c8d", "^", "flat UCB1, no dedup")}
    for ax, ds in zip(axes[0], datasets):
        for arm, (color, marker, label) in style.items():
            g = summ[(summ["dataset"] == ds) & (summ["arm"] == arm)].sort_values("budget")
            if g.empty:
                continue
            # std -> standard error of the mean
            ax.errorbar(g["budget"], g["best_acc_mean"], yerr=g["best_acc_std"].fillna(0) / np.sqrt(g["n_seeds"]),
                        color=color, marker=marker, label=label, capsize=2, lw=1.4)
        ax.set_title(ds)
        ax.set_xlabel("budget (proposals)")
        ax.set_ylabel("mean best_acc (%)")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("A7 — trie vs flat UCB1 (LLM off, proxy on, mean ± s.e.m.)")
    fig.tight_layout()
    out = storage.get_figure_path("ablations", "A7_no_trie_ablation")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    logger.info(f"figure -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize-only", action="store_true", help="skip runs, only rebuild CSVs/figure")
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--datasets", nargs="+", default=cfg["benchmarks"]["datasets"])
    ap.add_argument("--budgets", nargs="+", type=int, default=cfg["search"]["budgets"])
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg["search"]["seeds"])
    a = ap.parse_args()
    if not a.summarize_only:
        run(a)
    summarize()