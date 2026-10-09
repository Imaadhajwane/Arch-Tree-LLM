# experiments/run_phase6_aggregate.py
# isort: skip_file
"""
PHASE 6 AGGREGATE — run after the ablations finish.
Reads all phase CSVs, builds the final tables and generates the paper figures.
Only reads upstream outputs, so it is safe to rerun at any time.

Outputs (outputs/final/results/):
  main_results_table.csv                     mean/std per (method, benchmark, dataset, budget)
  statistical_tests.csv                      archtree_llm vs each baseline, paired per seed at a fixed budget
  eval_efficiency_summary.csv                real evals vs accuracy at budget 100 (cifar10)
  equal_compute_comparison[_strict].csv      accuracy at matched REAL-eval cost
  equal_compute_statistical_tests[_strict].csv
  extended_findings_summary.csv              headline tests from A7, proxy ablation, convergence, LLM-size

Statistical test: for each (benchmark, dataset, budget), pair archtree_llm and the
baseline on matching seeds, run a Wilcoxon signed-rank test and report paired
Cohen's d (mean diff / std of diffs).

Equal-compute comparison: archtree_llm's proxy gating means it spends fewer real
evals than its nominal budget, while baselines spend all of theirs. Each
archtree_llm cell is therefore compared with baselines at a budget from the
config grid chosen to match archtree_llm's real-eval count:
  * nearest : closest grid budget (can be BELOW archtree's real evals)
  * strict  : smallest grid budget >= archtree's real evals (baseline never gets
              less compute) — use this one for paper claims

Schema note: missing columns (e.g. tcr in phase0/1) are NaN-filled, and the
phase4 table is derived from pareto + hv_igd (there is no phase4_main_results.csv).

Oct 2026 changes
  * Every accuracy used here is the REPORT metric (config benchmarks.report_metric, default
    best_test_acc = test accuracy of the arch chosen by validation accuracy) via
    storage.to_report_metric(); search-metric values are kept as best_acc_search.
  * Headline comparisons use the 9 EXTERNAL baselines only (incl. regularized evolution, local search and RE + zero-cost warm-up since Oct 3) (comparator_type == "external");
    our own variants (archtree_v1 / archtree_zc / ...) are tested separately ("internal").
  * Every test row has: outcome (win / tie / loss), Holm-adjusted p within its family, and a
    95% bootstrap CI of the mean paired difference.
  * Strict equal-compute matching uses the MAX real-eval count over seeds (every seed's
    baseline run got at least as many real evaluations), not the mean.
"""
import sys
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402
import yaml                           # noqa: E402
from scipy.stats import wilcoxon      # noqa: E402

import archtree.storage as storage    # noqa: E402

N_BOOT = 5000
_RNG = np.random.default_rng(0)


def comparator_type(method: str) -> str:
    return "internal" if str(method).startswith(("archtree", "flat_ucb")) else "external"


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def boot_ci(diff, n=N_BOOT):
    diff = np.asarray(diff, float)
    if len(diff) < 2:
        return np.nan, np.nan
    means = diff[_RNG.integers(0, len(diff), (n, len(diff)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def outcome(mean_diff, p, alpha=0.05):
    if p < alpha:
        return "win" if mean_diff > 0 else "loss"
    return "tie"


def add_family_columns(df: pd.DataFrame, family_cols) -> pd.DataFrame:
    """Holm-adjust p within each family and derive win/tie/loss (raw and Holm)."""
    if df.empty:
        return df
    df = df.copy()
    df["p_holm"] = np.nan
    for _, idx in df.groupby(family_cols).groups.items():
        df.loc[idx, "p_holm"] = holm(df.loc[idx, "p_value"].values)
    diff_col = "mean_diff" if "mean_diff" in df.columns else "mean_diff_archtree_minus_baseline"
    df["outcome_raw"] = [outcome(d, p) for d, p in zip(df[diff_col], df["p_value"])]
    df["outcome_holm"] = [outcome(d, p) for d, p in zip(df[diff_col], df["p_holm"])]
    return df

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "outputs" / "final" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(
    sys.stderr, level="INFO", colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>",
)
logger.add(LOG_DIR / "phase6_aggregate_run.log", level="DEBUG", rotation="10 MB", retention=5)

with open(PROJECT_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)
storage.verify_phase_inputs("final")

RESULTS_DIR = PROJECT_ROOT / "outputs" / "final" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# every table below is written with DataFrame.to_csv (full rebuild), so reruns never duplicate rows

REQUIRED_GROUP_COLS = ["method", "benchmark", "dataset", "budget"]
AGG_COLS = ["best_acc", "tcr", "wall_time_sec"]
BASELINE_BUDGET_GRID = sorted(cfg["search"]["budgets"])   # budgets the baselines actually ran at


# ── Helpers ───────────────────────────────────────────────────────────────────
def _grouped(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std per (method, benchmark, dataset, budget); NaN-fills missing metric columns."""
    for col in AGG_COLS:
        if col not in df.columns:
            df[col] = np.nan
    grp = df.groupby(REQUIRED_GROUP_COLS).agg(
        {"best_acc": ["mean", "std"], "tcr": ["mean", "std"], "wall_time_sec": ["mean"]})
    grp.columns = ["best_acc_mean", "best_acc_std", "tcr_mean", "tcr_std", "wall_time_mean"]
    return grp.reset_index()


def _paired_vals(a_df: pd.DataFrame, b_df: pd.DataFrame):
    """Align two frames on shared seeds. Returns (seeds, a_vals, b_vals) or None if < 2 shared
    seeds (Wilcoxon is degenerate below that). Seeds missing on either side are dropped rather
    than relying on row order."""
    seeds = sorted(set(a_df["seed"]) & set(b_df["seed"]))
    if len(seeds) < 2:
        return None
    a = a_df.set_index("seed").loc[seeds]["best_acc"].values
    b = b_df.set_index("seed").loc[seeds]["best_acc"].values
    return seeds, a, b


def _wilcoxon(a: np.ndarray, b: np.ndarray):
    """Wilcoxon signed-rank; None when all diffs are zero or scipy rejects the input."""
    if np.all(a - b == 0):
        return None
    try:
        return wilcoxon(a, b)
    except ValueError:
        return None


def _paired_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """mean(diff) / std(diff); NaN when the diffs have zero variance."""
    diff = a - b
    sd = diff.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(diff.mean() / sd)


# ── Load phases & build the main results table ────────────────────────────────
all_rows = []     # grouped tables, for the paper table
raw_frames = []   # raw per-seed frames, for the paired tests

try:
    df0 = storage.load_report_csv("phase0", "baseline_summary.csv")
    all_rows.append(_grouped(df0))
    raw_frames.append(df0)
    logger.info(f"phase0 baselines: {len(df0)} rows -> {df0['method'].nunique()} methods")
except FileNotFoundError:
    logger.warning("outputs/phase0/results/baseline_summary.csv missing — baseline comparison will be incomplete.")

for phase in ["phase1", "phase2", "phase3"]:
    try:
        df = storage.load_report_csv(phase, f"{phase}_main_results.csv")
        all_rows.append(_grouped(df))
        raw_frames.append(df)
        logger.info(f"{phase}: {len(df)} rows")
    except FileNotFoundError:
        logger.warning(f"{phase}_main_results.csv missing — skipping {phase}.")

# Phase 4 has no main-results file: derive best_acc per run from the pareto rows and join HV/IGD.
# Dataset is suffixed with the hardware target so each (dataset, hardware) is its own group.
try:
    pareto = storage.load_results_csv("phase4", "phase4_pareto_results.csv")
    hv = storage.load_results_csv("phase4", "phase4_hv_igd_summary.csv")
    best_per_run = (pareto.groupby(["hardware", "dataset", "seed", "budget"])["accuracy"]
                    .max().reset_index().rename(columns={"accuracy": "best_acc"}))
    p4 = best_per_run.merge(hv, on=["hardware", "dataset", "seed", "budget"], how="left")
    p4["method"] = "archtree_llm_hw"
    p4["benchmark"] = "hwnb"
    p4["dataset"] = p4["dataset"] + "__" + p4["hardware"]
    p4["tcr"] = np.nan
    all_rows.append(_grouped(p4))
    logger.info(f"phase4 (derived from pareto+hv_igd): {len(p4)} rows across {p4['hardware'].nunique()} hardware targets")
except FileNotFoundError as e:
    logger.warning(f"phase4 pareto/hv_igd results missing ({e}) — skipping phase4.")

if not all_rows:
    logger.error("no phase results loaded at all — nothing to aggregate.")
    sys.exit(1)

final_df = pd.concat(all_rows, ignore_index=True)
final_df.to_csv(RESULTS_DIR / "main_results_table.csv", index=False)
storage.save_plot_data("final", "main_results_table", final_df)
logger.success(f"main_results_table.csv written ({len(final_df)} rows)")

# Raw per-seed rows (frames that have the full schema) for the paired tests
raw_needed_cols = ["method", "benchmark", "dataset", "seed", "budget", "best_acc"]
raw_all = pd.concat(
    [df[raw_needed_cols] for df in raw_frames if all(c in df.columns for c in raw_needed_cols)],
    ignore_index=True,
)

# ── Statistical tests: archtree_llm vs every other method ────────────────────
# Equal NOMINAL budget (proposals). Kept for completeness; the headline is the
# equal-real-compute table further down.
_all_baseline_methods = sorted(m for m in raw_all["method"].unique() if m != "archtree_llm")
logger.info(f"Testing archtree_llm vs {len(_all_baseline_methods)} methods (paired per seed, per budget); "
            f"external = {[m for m in _all_baseline_methods if comparator_type(m) == 'external']}")

test_rows = []
for method_b in _all_baseline_methods:
    for bench_name in raw_all["benchmark"].unique():
        for dataset in raw_all["dataset"].unique():
            sub_a = raw_all[(raw_all["method"] == "archtree_llm") & (raw_all["benchmark"] == bench_name) &
                            (raw_all["dataset"] == dataset)]
            sub_b = raw_all[(raw_all["method"] == method_b) & (raw_all["benchmark"] == bench_name) &
                            (raw_all["dataset"] == dataset)]
            if sub_a.empty or sub_b.empty:
                continue
            for budget in sorted(set(sub_a["budget"]) & set(sub_b["budget"])):
                paired = _paired_vals(sub_a[sub_a["budget"] == budget], sub_b[sub_b["budget"] == budget])
                if paired is None:
                    continue
                seeds, a_vals, b_vals = paired
                res = _wilcoxon(a_vals, b_vals)
                diff = a_vals - b_vals
                lo, hi = boot_ci(diff)
                test_rows.append({
                    "method_a": "archtree_llm", "method_b": method_b,
                    "comparator_type": comparator_type(method_b),
                    "benchmark": bench_name, "dataset": dataset, "budget": budget,
                    "metric": "best_acc(report)", "n_paired_seeds": len(seeds),
                    "mean_diff": float(diff.mean()), "ci95_lo": lo, "ci95_hi": hi,
                    "wilcoxon_stat": res.statistic if res is not None else np.nan,
                    "p_value": res.pvalue if res is not None else 1.0,
                    "cohens_d": _paired_cohens_d(a_vals, b_vals),
                })
st_df = add_family_columns(pd.DataFrame(test_rows), ["comparator_type"])
if len(st_df):
    st_df["significant"] = st_df["p_holm"] < 0.05
st_df.to_csv(RESULTS_DIR / "statistical_tests.csv", index=False)
logger.success(f"statistical_tests.csv written ({len(st_df)} tests, Holm within external / internal)")

# ── Eval-efficiency comparison (cifar10, budget 100) ─────────────────────────
try:
    eff_rows = []

    p3 = storage.load_report_csv("phase3", "phase3_main_results.csv")
    p3_c10_100 = p3[(p3["dataset"] == "cifar10") & (p3["budget"] == 100)]
    if not p3_c10_100.empty:
        eff_rows.append({
            "method": "archtree_llm (proxy ON)",
            "mean_best_acc": p3_c10_100["best_acc"].mean(),
            "mean_real_evals": p3_c10_100["real_eval_count"].mean(),
            "budget": 100,
        })

    try:
        a2 = storage.load_report_csv("ablations", "ablation_A2_no_zc.csv")
        # Ablations run on all datasets, so filter to cifar10 to match the rows above
        a2_b100 = a2[(a2["budget"] == 100) & (a2["dataset"] == "cifar10")]
        if not a2_b100.empty:
            eff_rows.append({
                "method": "A2 no_zc (proxy OFF)",
                "mean_best_acc": a2_b100["best_acc"].mean(),
                "mean_real_evals": a2_b100["real_eval_count"].mean(),
                "budget": 100,
            })
    except FileNotFoundError:
        pass

    # Baselines spend their full budget as real evals (zico/rz_nas also spend proxy evaluations:
    # see baseline_summary.csv proxy_evals)
    baseline_100 = final_df[(final_df["budget"] == 100) & (final_df["dataset"] == "cifar10") &
                            (final_df["method"] != "archtree_llm")]
    for _, row in baseline_100.iterrows():
        eff_rows.append({
            "method": row["method"],
            "mean_best_acc": row["best_acc_mean"],
            "mean_real_evals": 100,
            "budget": 100,
        })

    if eff_rows:
        eff_df = pd.DataFrame(eff_rows)
        eff_df["eval_reduction_pct"] = 100.0 * (1 - eff_df["mean_real_evals"] / 100.0)
        eff_df.to_csv(RESULTS_DIR / "eval_efficiency_summary.csv", index=False)
        storage.save_plot_data("final", "eval_efficiency_summary", eff_df)
        logger.success("eval_efficiency_summary.csv written.")
        logger.info("\n" + eff_df.to_string(index=False))
    else:
        logger.warning("eval_efficiency_summary SKIP — no data found.")
except FileNotFoundError as e:
    logger.warning(f"eval_efficiency_summary SKIP — {e}")

# ── Paper figures ─────────────────────────────────────────────────────────────
try:
    from plots.plot_results import plot_final_all
    plot_final_all()
except ImportError:
    logger.warning("plots/plot_results.py::plot_final_all not found — skipping figures.")

# ── Exit summary of the main statistical tests (informational) ───────────────
if len(st_df):
    ext = st_df[st_df["comparator_type"] == "external"]
    logger.info(f"Nominal-budget tests vs external baselines (Holm): "
                f"{(ext['outcome_holm'] == 'win').sum()} wins / {(ext['outcome_holm'] == 'tie').sum()} ties / "
                f"{(ext['outcome_holm'] == 'loss').sum()} losses of {len(ext)}")


# ══════════════════════════════════════════════════════════════════════════════
# Equal-real-compute comparison (nearest-budget and strict/ceiling-budget)
# ══════════════════════════════════════════════════════════════════════════════
def _nearest_budget(x: float) -> int:
    return min(BASELINE_BUDGET_GRID, key=lambda g: abs(g - x))


def _ceiling_budget(x: float) -> int:
    """Smallest grid budget >= x. If x exceeds the whole grid, returns the max
    (that cell is then NOT strictly fair; the caller reports it)."""
    for g in BASELINE_BUDGET_GRID:
        if g >= x:
            return g
    return max(BASELINE_BUDGET_GRID)


def equal_compute(real_stats: pd.DataFrame, match_fn, tag: str, budget_col: str, match_on: str) -> None:
    """Compare archtree_llm (at its nominal budget) with each baseline at the grid budget chosen by
    match_fn(real evals). match_on = "mean" (nearest table, descriptive) or "max" (strict: the
    largest real-eval count of any seed, so EVERY paired baseline run had at least as much compute)."""
    rows, test_rows, uncapped = [], [], []
    baselines = sorted(m for m in raw_all["method"].unique() if m != "archtree_llm")

    for _, mr in real_stats.iterrows():
        dataset, nominal = mr["dataset"], mr["nominal_budget"]
        real_ref = mr[match_on]
        matched = match_fn(real_ref)
        if matched < real_ref:
            uncapped.append((dataset, nominal, real_ref))

        at_nominal = raw_all[(raw_all["method"] == "archtree_llm") & (raw_all["dataset"] == dataset) &
                             (raw_all["budget"] == nominal)]
        if at_nominal.empty:
            continue

        common = {"dataset": dataset, "archtree_nominal_budget": nominal,
                  "archtree_mean_real_evals": round(float(mr["mean"]), 1),
                  "archtree_max_real_evals": int(mr["max"]), budget_col: matched}
        rows.append({**common, "method": "archtree_llm",
                     "mean_best_acc": at_nominal["best_acc"].mean(),
                     "n_seeds": at_nominal["seed"].nunique()})

        for baseline in baselines:
            b_sub = raw_all[(raw_all["method"] == baseline) & (raw_all["dataset"] == dataset) &
                            (raw_all["budget"] == matched)]
            if b_sub.empty:
                continue
            rows.append({**common, "method": baseline,
                         "mean_best_acc": b_sub["best_acc"].mean(),
                         "n_seeds": b_sub["seed"].nunique()})
            paired = _paired_vals(at_nominal, b_sub)
            if paired is None:
                continue
            seeds, a_vals, b_vals = paired
            res = _wilcoxon(a_vals, b_vals)
            diff = a_vals - b_vals
            lo, hi = boot_ci(diff)
            test_rows.append({
                **common, "baseline": baseline, "comparator_type": comparator_type(baseline),
                "n_paired_seeds": len(seeds),
                "mean_diff_archtree_minus_baseline": float(diff.mean()), "ci95_lo": lo, "ci95_hi": hi,
                "wilcoxon_stat": res.statistic if res is not None else np.nan,
                "p_value": res.pvalue if res is not None else 1.0,
                "cohens_d": _paired_cohens_d(a_vals, b_vals),
                "baseline_got_ge_archtree_evals": matched >= real_ref,
            })

    if not rows:
        logger.warning(f"equal_compute{tag} SKIP — no overlapping (dataset, budget) rows found.")
        return

    cmp_df = pd.DataFrame(rows).drop_duplicates(["dataset", "archtree_nominal_budget", "method"], keep="last")
    cmp_df.to_csv(RESULTS_DIR / f"equal_compute_comparison{tag}.csv", index=False)
    storage.save_plot_data("final", f"equal_compute_comparison{tag}", cmp_df)

    tests_df = add_family_columns(pd.DataFrame(test_rows), ["comparator_type"])
    if len(tests_df):
        tests_df["significant"] = tests_df["p_holm"] < 0.05
        # backward-compatible column for plots/backend: not a significant loss (Holm)
        tests_df["archtree_wins_or_ties"] = tests_df["outcome_holm"] != "loss"
    tests_df.to_csv(RESULTS_DIR / f"equal_compute_statistical_tests{tag}.csv", index=False)
    logger.success(f"equal_compute{tag}: {len(cmp_df)} comparison rows, {len(tests_df)} tests written.")
    if tests_df.empty:
        return

    logger.info("=" * 78)
    for ctype, g in tests_df.groupby("comparator_type"):
        logger.info(f"EQUAL-COMPUTE{tag or ' (nearest, descriptive)'} vs {ctype} comparators: "
                    f"{(g['outcome_holm'] == 'win').sum()} wins / {(g['outcome_holm'] == 'tie').sum()} ties / "
                    f"{(g['outcome_holm'] == 'loss').sum()} losses of {len(g)} (Holm); unadjusted "
                    f"{(g['outcome_raw'] == 'win').sum()} / {(g['outcome_raw'] == 'tie').sum()} / "
                    f"{(g['outcome_raw'] == 'loss').sum()}")
        if ctype == "external":
            per = g.groupby("baseline")["outcome_holm"].value_counts().unstack(fill_value=0)
            logger.info("  per baseline (Holm):\n" + per.to_string())
    if tag:
        logger.info(f"  every paired baseline run got >= archtree's real-eval count (max over seeds): "
                    f"{bool(tests_df['baseline_got_ge_archtree_evals'].all())}")
        if uncapped:
            logger.warning(f"{len(uncapped)} cell(s) exceed the largest grid budget ({max(BASELINE_BUDGET_GRID)}) - "
                           f"NOT strictly fair: " + ", ".join(f"{d}/b={b} (real={r:.0f})" for d, b, r in uncapped))


try:
    p3_full = storage.load_report_csv("phase3", "phase3_main_results.csv")
    if "real_eval_count" not in p3_full.columns:
        logger.warning("equal_compute SKIP — phase3_main_results.csv has no real_eval_count column.")
    else:
        real_stats = (p3_full.groupby(["dataset", "budget"])["real_eval_count"]
                      .agg(["mean", "max"]).reset_index().rename(columns={"budget": "nominal_budget"}))
        equal_compute(real_stats, _nearest_budget, "", "matched_baseline_budget", match_on="mean")
        equal_compute(real_stats, _ceiling_budget, "_strict", "matched_baseline_budget_strict", match_on="max")
except FileNotFoundError as e:
    logger.warning(f"equal_compute SKIP — {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Extended findings summary (A7, proxy ablation, convergence, LLM-size)
# ══════════════════════════════════════════════════════════════════════════════
# Re-export only: the source CSVs stay authoritative (see `source_file`). Each
# experiment is optional and skipped with a warning if it hasn't been run yet.
EXTENDED_SOURCES = [
    dict(exp="gated_baselines", phase="gated_baselines", file="gated_paired_tests.csv",
         comparison=lambda r: f"archtree_vs_{r['other']}__gate_{r['gate']}", metric=lambda r: "best_acc(report)",
         budget=lambda r: r["budget"], diff="mean_diff_archtree_minus_other",
         hint="experiments/run_gated_baselines.py"),
    dict(exp="design_ablation_heldout", phase="design_ablation", file="design_paired_tests.csv",
         comparison=lambda r: f"{r['arm']}_vs_v2", metric=lambda r: "best_test_acc",
         budget=lambda r: r["budget"], diff="mean_diff_arm_minus_v2",
         hint="experiments/run_design_ablation.py (pipeline step 11)"),
    dict(exp="nb101_same_gate", phase="nb101", file="nb101_paired_tests.csv",
         comparison=lambda r: f"nb101_archtree_vs_{r['other']}__gate_{r['gate']}",
         metric=lambda r: "best_test_acc", budget=lambda r: r["budget"],
         diff="mean_diff_archtree_minus_other", hint="experiments/run_nb101.py (pipeline step 21)"),
    dict(exp="nb101_equal_compute", phase="nb101", file="nb101_equal_compute_tests.csv",
         comparison=lambda r: f"nb101_archtree_{r['gate']}_vs_{r['other']}_b{r['other_budget']}",
         metric=lambda r: "best_test_acc", budget=lambda r: r["archtree_budget"],
         diff="mean_diff_archtree_minus_other", hint="experiments/run_nb101.py (pipeline step 21)"),
    dict(exp="equal_gpu_hours", phase="equal_gpu_hours", file="egh_paired_tests.csv",
         comparison=lambda r: f"{r['benchmark']}_archtree_zc_unsup_vs_{r['other']}__k{r['k_mean_arch_costs']}",
         metric=lambda r: "best_test_acc@equal_gpu_hours", budget=lambda r: r["budget"],
         diff="mean_diff_archtree_minus_other", hint="experiments/run_equal_gpu_hours.py (pipeline step 22)"),
    dict(exp="A7_trie_vs_flat_ucb1", phase="ablations", file="A7_paired_tests.csv",
         comparison=lambda r: f"trie_vs_{r['arm_b']}", metric=lambda r: "best_acc",
         budget=lambda r: r["budget"], diff="mean_diff_pp",
         hint="experiments/run_trie_ablation.py"),
    dict(exp="proxy_composition_ablation", phase="proxy_ablation", file="proxy_ablation_paired_tests.csv",
         comparison=lambda r: f"{r['arm']}_vs_full_ensemble", metric=lambda r: "best_acc",
         budget=lambda r: np.nan, diff="mean_diff_pp",
         hint="experiments/run_proxy_ablation.py"),
    dict(exp="convergence_study", phase="convergence", file="convergence_paired_tests.csv",
         comparison=lambda r: f"{r.get('arm_a', 'llm_on')}_vs_{r.get('arm_b', 'llm_off')}",
         metric=lambda r: r["metric"],
         budget=lambda r: r["budget"], diff="mean_diff",
         hint="experiments/run_convergence_study.py (arms llm_on / size_prior / llm_described)"),
    dict(exp="llm_contamination_probe", phase="llm_contamination", file="contamination_tests.csv",
         comparison=lambda r: r["comparison"], metric=lambda r: "op_ranking_spearman",
         budget=lambda r: np.nan, diff="mean_diff",
         hint="experiments/run_llm_contamination.py (Groq)"),
    dict(exp="llm_size_sensitivity", phase="llm_size", file="llm_size_paired_tests.csv",
         comparison=lambda r: f"{r['arm']}_vs_off", metric=lambda r: r["metric"],
         budget=lambda r: r["budget"], diff="mean_diff",
         hint="the LLM-size study (may be waiting on a Groq daily token-budget reset)"),
]

ext_rows = []
for src in EXTENDED_SOURCES:
    try:
        tdf = storage.load_results_csv(src["phase"], src["file"])
    except (FileNotFoundError, pd.errors.EmptyDataError):
        logger.warning(f"[extended] {src['file']} not found — run {src['hint']} first.")
        continue
    for _, r in tdf.iterrows():
        ext_rows.append({
            "experiment": src["exp"], "dataset": r["dataset"], "budget": src["budget"](r),
            "comparison": src["comparison"](r), "metric": src["metric"](r),
            "n_paired_seeds": r["n_paired_seeds"], "mean_diff": r[src["diff"]],
            "p_value": r["wilcoxon_p"], "cohens_d": r["cohens_d"], "significant": r["significant"],
            "source_file": f"outputs/{src['phase']}/results/{src['file']}",
        })
    logger.info(f"[extended] {src['exp']}: {len(tdf)} comparisons folded in.")

if ext_rows:
    ext_df = pd.DataFrame(ext_rows)
    ext_df.to_csv(RESULTS_DIR / "extended_findings_summary.csv", index=False)
    storage.save_plot_data("final", "extended_findings_summary", ext_df)
    logger.success(f"extended_findings_summary.csv written ({len(ext_df)} rows across "
                   f"{ext_df['experiment'].nunique()} experiments).")
    logger.info("=" * 78)
    logger.info("EXTENDED FINDINGS — significant results:")
    sig = ext_df[ext_df["significant"].fillna(False).astype(bool)]
    if len(sig):
        for exp, g in sig.groupby("experiment"):
            logger.info(f"  {exp}: {len(g)} of {(ext_df['experiment'] == exp).sum()} comparisons significant")
    else:
        logger.info("  none — every extended experiment came back non-significant.")
else:
    logger.warning("extended_findings_summary SKIP — none of the extended experiments have been run yet.")

logger.success("Phase 6 DONE — outputs/final/ written. Ready for Phase 7 (paper writing).")