# plots/plot_results.py
"""
Central plotting module. Called by:
  - experiments/run_phase3.py         -> plot_phase3(phase="phase3")
  - experiments/run_phase4.py         -> plot_phase4(phase="phase4")
  - experiments/run_ablations.py      -> plot_ablations()
  - experiments/run_phase6_aggregate.py -> plot_final_all()
  - pipeline step 18: `python plots/plot_results.py all` -> every figure (Oct 4, 2026), incl. the
    equal-GPU-hours study, the gate-cost model check, the LLM contamination probe and the 4-arm
    convergence study. NOTE: since Oct 2, 2026 every baseline runs on all 3 datasets; the
    "cifar10 only" remarks below describe the original fig1 design (one dataset per panel).

Every phase script already tolerates ImportError from this module (each
wraps its plot_*() call in try/except ImportError and just logs a
warning) — so this file is optional infra, never a hard blocker if it's
missing or a single figure fails partway.

── DESIGN NOTE — dataset-aware throughout ──────────────────────────────────
This project runs 3 datasets everywhere (cifar10/cifar100/ImageNet16-120 —
see phase0-4 + run_ablations.py's "full dataset sweep" patch notes). Every
figure function here follows the SAME rule the driver scripts already
settled on (RESULTS_REVIEW.md #1): headline/comparison numbers filter to
dataset=="cifar10" (the only one with a baseline_summary.csv to compare
against), while every OTHER dataset gets its own informational panel/line —
never blended into one mean, never silently dropped. Concretely: multi-
dataset figures use one subplot per dataset (or one color per dataset in a
line/bar group) rather than filtering down to cifar10 only, EXCEPT the
headline "vs baselines" comparison (fig1, ablation deltas), which stays
cifar10-only for apples-to-apples since baselines only ever ran cifar10.

Each figure function is independently wrapped in try/except so one missing
upstream CSV (e.g. a phase run still in progress) never kills the rest of
the figure set. Every function is idempotent — safe to rerun anytime after
partial/cold-resumed runs, always regenerates from whatever CSVs currently
exist.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plots.figure_export import install as _install_gallery
_install_gallery()   # every figure is also copied to plots/figures/ with its data-point CSV

import archtree.storage as storage

DATASET_COLORS = {
    "cifar10": "#2b7a4b",
    "cifar100": "#c9772e",
    "ImageNet16-120": "#4361b8",
}
DATASET_ORDER = ["cifar10", "cifar100", "ImageNet16-120"]


def _color_for(dataset: str) -> str:
    return DATASET_COLORS.get(dataset, "#777777")


def _safe(fn_name, fn):
    """Run one figure function, log+continue on any failure (missing CSV,
    empty frame, etc.) instead of aborting the whole plot_*() call."""
    try:
        fn()
        print(f"[plot_results] OK — {fn_name}")
    except FileNotFoundError as e:
        print(f"[plot_results] SKIP {fn_name} — missing input: {e}")
    except Exception as e:
        print(f"[plot_results] SKIP {fn_name} — {type(e).__name__}: {e}")


def _present_datasets(df: pd.DataFrame) -> list:
    """Order datasets consistently (DATASET_ORDER first, then anything else
    found), rather than relying on df's own row order."""
    found = list(df["dataset"].unique())
    ordered = [d for d in DATASET_ORDER if d in found]
    ordered += [d for d in found if d not in ordered]
    return ordered


# ══════════════════════════════════════════════════════════════════════════
# PHASE 3 — LLM warm-start UCB
# ══════════════════════════════════════════════════════════════════════════
def plot_phase3(phase: str = "phase3") -> None:
    out_dir = storage.get_output_dir(phase, "plots/figures")

    def _acc_vs_budget():
        df = storage.load_report_csv(phase, "phase3_main_results.csv")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for ds in _present_datasets(df):
            sub = df[df["dataset"] == ds].groupby("budget")["best_acc"].agg(["mean", "std"]).reset_index()
            ax.errorbar(sub["budget"], sub["mean"], yerr=sub["std"].fillna(0),
                        marker="o", capsize=3, label=ds, color=_color_for(ds))
        ax.set_xlabel("Budget (real evals)")
        ax.set_ylabel("Best accuracy (%)")
        ax.set_title("Phase 3 — archtree_llm best_acc vs budget, by dataset")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "phase3_acc_vs_budget.png", dpi=300)
        plt.close(fig)

    def _eval_efficiency():
        df = storage.load_report_csv(phase, "phase3_main_results.csv")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for ds in _present_datasets(df):
            sub = df[df["dataset"] == ds].groupby("budget").agg(
                real=("real_eval_count", "mean"), proxy=("proxy_eval_count", "mean")
            ).reset_index()
            frac = sub["real"] / (sub["real"] + sub["proxy"]).replace(0, np.nan)
            ax.plot(sub["budget"], frac, marker="o", label=ds, color=_color_for(ds))
        ax.set_xlabel("Budget")
        ax.set_ylabel("Real-eval fraction (proxy-gated)")
        ax.set_title("Phase 3 — fraction of iterations that were real evals")
        ax.set_ylim(0, 1.05)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "phase3_eval_efficiency.png", dpi=300)
        plt.close(fig)

    def _cache_hit_rate():
        df = storage.load_report_csv(phase, "phase3_llm_cache_stats.csv")
        fig, ax = plt.subplots(figsize=(6, 4))
        for ds in _present_datasets(df):
            vals = df[df["dataset"] == ds]["cache_hit_rate"]
            ax.scatter([ds] * len(vals), vals, color=_color_for(ds), alpha=0.6)
        ax.axhline(0.80, color="crimson", linestyle="--", linewidth=1, label="80% target")
        ax.set_ylabel("Cache hit rate")
        ax.set_title("Phase 3 — LLM prior cache hit rate by dataset")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "phase3_cache_hit_rate.png", dpi=300)
        plt.close(fig)

    _safe("phase3_acc_vs_budget", _acc_vs_budget)
    _safe("phase3_eval_efficiency", _eval_efficiency)
    _safe("phase3_cache_hit_rate", _cache_hit_rate)


# ══════════════════════════════════════════════════════════════════════════
# PHASE 4 — hardware-aware Pareto search
# ══════════════════════════════════════════════════════════════════════════
def plot_phase4(phase: str = "phase4") -> None:
    out_dir = storage.get_output_dir(phase, "plots/figures")

    def _hv_vs_budget():
        df = storage.load_report_csv(phase, "phase4_hv_igd_summary.csv")
        hw_list = sorted(df["hardware"].unique())
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        cmap = plt.get_cmap("tab10")
        for i, hw in enumerate(hw_list):
            sub = df[df["hardware"] == hw].groupby("budget")["hypervolume"].mean().reset_index()
            ax.plot(sub["budget"], sub["hypervolume"], marker="o", label=hw, color=cmap(i % 10))
        ax.set_xlabel("Budget")
        ax.set_ylabel("Hypervolume")
        ax.set_title("Phase 4 — hypervolume vs budget, by hardware target")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "phase4_hv_vs_budget.png", dpi=300)
        plt.close(fig)

    def _pareto_front_size():
        df = storage.load_report_csv(phase, "phase4_hv_igd_summary.csv")
        hw_list = sorted(df["hardware"].unique())
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        cmap = plt.get_cmap("tab10")
        for i, hw in enumerate(hw_list):
            sub = df[df["hardware"] == hw].groupby("budget")["pareto_front_size"].mean().reset_index()
            ax.plot(sub["budget"], sub["pareto_front_size"], marker="s", label=hw, color=cmap(i % 10))
        ax.set_xlabel("Budget")
        ax.set_ylabel("Mean Pareto front size")
        ax.set_title("Phase 4 — Pareto front size vs budget")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "phase4_pareto_front_size.png", dpi=300)
        plt.close(fig)

    def _igd_vs_budget():
        df = storage.load_report_csv(phase, "phase4_hv_igd_summary.csv")
        df = df[df["igd"].notna() & np.isfinite(df["igd"])]
        if df.empty:
            raise FileNotFoundError("igd column empty/NaN for every row — "
                                     "run tools/build_igd_reference_front.py first")
        hw_list = sorted(df["hardware"].unique())
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        cmap = plt.get_cmap("tab10")
        for i, hw in enumerate(hw_list):
            sub = df[df["hardware"] == hw].groupby("budget")["igd"].mean().reset_index()
            ax.plot(sub["budget"], sub["igd"], marker="^", label=hw, color=cmap(i % 10))
        ax.set_xlabel("Budget")
        ax.set_ylabel("IGD (lower is better)")
        ax.set_title("Phase 4 — IGD vs true Pareto front, by hardware target")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "phase4_igd_vs_budget.png", dpi=300)
        plt.close(fig)

    def _pareto_scatter():
        pareto = storage.load_report_csv(phase, "phase4_pareto_results.csv")
        hw_list = sorted(pareto["hardware"].unique())
        n = len(hw_list)
        ncols = min(3, n) or 1
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)
        for idx, hw in enumerate(hw_list):
            ax = axes[idx // ncols][idx % ncols]
            sub = pareto[pareto["hardware"] == hw]
            ax.scatter(sub["latency_ms"], sub["accuracy"], s=6, alpha=0.25, color="#999999", label="visited")
            front = sub[sub["on_pareto_front"]]
            ax.scatter(front["latency_ms"], front["accuracy"], s=14, color="crimson", label="Pareto front")
            ax.set_title(hw, fontsize=9)
            ax.set_xlabel("Latency (ms)", fontsize=8)
            ax.set_ylabel("Accuracy (%)", fontsize=8)
            ax.tick_params(labelsize=7)
        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
        fig.suptitle("Phase 4 — accuracy vs latency, visited archs + Pareto front", fontsize=11)
        fig.tight_layout(rect=[0, 0.04, 1, 0.96])
        fig.savefig(out_dir / "phase4_pareto_scatter.png", dpi=300)
        plt.close(fig)

    _safe("phase4_hv_vs_budget", _hv_vs_budget)
    _safe("phase4_pareto_front_size", _pareto_front_size)
    _safe("phase4_igd_vs_budget", _igd_vs_budget)
    _safe("phase4_pareto_scatter", _pareto_scatter)


# ══════════════════════════════════════════════════════════════════════════
# ABLATIONS — A1/A2/A3 (toggle deltas), A5 (prune %), A6 (budget sensitivity)
# ══════════════════════════════════════════════════════════════════════════
def plot_ablations() -> None:
    out_dir = storage.get_output_dir("ablations", "plots/figures")

    def _load(fname):
        return storage.load_report_csv("ablations", fname)

    def _toggle_deltas():
        """A1/A2/A3 vs the full archtree_llm (phase3) — cifar10-only headline
        (baselines/full-method both only ever ran cifar10), each dataset the
        ablation ITSELF ran (cifar10/cifar100/ImageNet16-120) still gets its
        own informational bar group rather than being dropped."""
        full = storage.load_report_csv("phase3", "phase3_main_results.csv")
        files = {
            "A1_no_llm": "ablation_A1_no_llm.csv",
            "A2_no_zc": "ablation_A2_no_zc.csv",
            "A3_no_weight_inherit": "ablation_A3_no_weight_inherit.csv",
        }
        rows = []
        for aid, fname in files.items():
            try:
                df = _load(fname)
            except FileNotFoundError:
                continue
            for ds in _present_datasets(df) if "dataset" in df.columns else ["cifar10"]:
                sub = df[df["dataset"] == ds] if "dataset" in df.columns else df
                full_sub = full[full["dataset"] == ds]
                if sub.empty or full_sub.empty:
                    continue
                rows.append({
                    "ablation": aid, "dataset": ds,
                    "best_acc_mean": sub["best_acc"].mean(),
                    "best_acc_std": sub["best_acc"].std(),
                    "delta_vs_full": sub["best_acc"].mean() - full_sub["best_acc"].mean(),
                })
        if not rows:
            raise FileNotFoundError("no ablation A1/A2/A3 CSVs found yet")
        tdf = pd.DataFrame(rows)
        tdf.to_csv(storage.get_output_dir("ablations", "plots/data") / "toggle_ablations.csv", index=False)

        ablation_ids = list(files.keys())
        datasets = _present_datasets(tdf)
        width = 0.8 / max(len(datasets), 1)
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        x = np.arange(len(ablation_ids))
        for i, ds in enumerate(datasets):
            sub = tdf[tdf["dataset"] == ds].set_index("ablation").reindex(ablation_ids)
            offset = (i - (len(datasets) - 1) / 2) * width
            ax.bar(x + offset, sub["delta_vs_full"], width=width,
                   yerr=sub["best_acc_std"].fillna(0), capsize=3,
                   label=ds, color=_color_for(ds))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(ablation_ids, rotation=15, ha="right")
        ax.set_ylabel("Δ best_acc vs full archtree_llm (pp)")
        ax.set_title("Ablations A1-A3 — component removal impact, by dataset")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "toggle_ablations.png", dpi=300)
        plt.close(fig)

    def _prune_threshold():
        df = _load("ablation_A5_prune_threshold.csv")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for ds in _present_datasets(df):
            sub = df[df["dataset"] == ds].groupby("param_value")["best_acc"].agg(["mean", "std"]).reset_index()
            sub = sub.sort_values("param_value")
            ax.errorbar(sub["param_value"], sub["mean"], yerr=sub["std"].fillna(0),
                        marker="o", capsize=3, label=ds, color=_color_for(ds))
        ax.set_xlabel("Prune percentile")
        ax.set_ylabel("Best accuracy (%)")
        ax.set_title("A5 — prune threshold sensitivity (budget=100), by dataset")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "ablation_A5_prune_threshold.png", dpi=300)
        plt.close(fig)

    def _budget_sensitivity():
        df = _load("ablation_A6_budget_sensitivity.csv")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for ds in _present_datasets(df):
            sub = df[df["dataset"] == ds].groupby("param_value")["best_acc"].agg(["mean", "std"]).reset_index()
            sub = sub.sort_values("param_value")
            ax.errorbar(sub["param_value"], sub["mean"], yerr=sub["std"].fillna(0),
                        marker="o", capsize=3, label=ds, color=_color_for(ds))
        ax.set_xlabel("Budget")
        ax.set_ylabel("Best accuracy (%)")
        ax.set_title("A6 — budget sensitivity, by dataset")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "ablation_A6_budget_sensitivity.png", dpi=300)
        plt.close(fig)

    def _alpha_sensitivity():
        df = storage.load_report_csv("ablations", "ablation_A4_alpha_sensitivity.csv")
        # A4 only ever ran cifar10/budget=100 (see run_phase3.py) — single panel.
        sub = df.groupby("alpha")["best_acc"].agg(["mean", "std"]).reset_index().sort_values("alpha")
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.errorbar(sub["alpha"], sub["mean"], yerr=sub["std"].fillna(0),
                    marker="o", capsize=3, color=_color_for("cifar10"))
        ax.set_xlabel("alpha_init")
        ax.set_ylabel("Best accuracy (%)")
        ax.set_title("A4 — LLM prior alpha sensitivity (cifar10, budget=100)")
        fig.tight_layout()
        fig.savefig(out_dir / "ablation_A4_alpha_sensitivity.png", dpi=300)
        plt.close(fig)

    _safe("toggle_ablations (A1-A3)", _toggle_deltas)
    _safe("ablation_A4_alpha_sensitivity", _alpha_sensitivity)
    _safe("ablation_A5_prune_threshold", _prune_threshold)
    _safe("ablation_A6_budget_sensitivity", _budget_sensitivity)


# ══════════════════════════════════════════════════════════════════════════
# FINAL — main paper figures (Phase 6 aggregate)
# ══════════════════════════════════════════════════════════════════════════
def plot_final_all() -> None:
    out_dir = storage.get_output_dir("final", "plots/figures")
    results_dir = PROJECT_ROOT / "outputs" / "final" / "results"

    def _fig1_main_results():
        """Headline bar chart — cifar10-only, budget=100 (baselines only ever
        ran cifar10 — see RESULTS_REVIEW.md #1). This is the ONE figure that
        stays cifar10-only by design, not an oversight."""
        main = pd.read_csv(results_dir / "main_results_table.csv")
        sub = main[(main["benchmark"] == "nb201") & (main["budget"] == 100)
                   & (main["dataset"] == "cifar10")]
        if sub.empty:
            raise FileNotFoundError("no nb201/cifar10/budget=100 rows in main_results_table.csv")
        order = sub.sort_values("best_acc_mean")
        colors = ["crimson" if m != "archtree_llm" else "seagreen" for m in order["method"]]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh(order["method"], order["best_acc_mean"],
                xerr=order["best_acc_std"].fillna(0), color=colors, capsize=3)
        ax.set_xlabel("Best accuracy (%)")
        ax.set_title("Fig 1 — NB201/CIFAR-10, budget=100, vs all baselines")
        fig.tight_layout()
        fig.savefig(out_dir / "fig1_main_results.png", dpi=300)
        plt.close(fig)

    def _fig2_other_datasets():
        """Informational companion to fig1 — archtree_llm on cifar100 /
        ImageNet16-120, no baseline comparison exists for these (baselines
        never ran them), so this is archtree_llm-only across budgets."""
        p3 = storage.load_report_csv("phase3", "phase3_main_results.csv")
        other = p3[p3["dataset"] != "cifar10"]
        if other.empty:
            raise FileNotFoundError("no non-cifar10 rows in phase3_main_results.csv")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for ds in _present_datasets(other):
            sub = other[other["dataset"] == ds].groupby("budget")["best_acc"].agg(["mean", "std"]).reset_index()
            ax.errorbar(sub["budget"], sub["mean"], yerr=sub["std"].fillna(0),
                        marker="o", capsize=3, label=ds, color=_color_for(ds))
        ax.set_xlabel("Budget")
        ax.set_ylabel("Best accuracy (%)")
        ax.set_title("Fig 2 — archtree_llm on cifar100/ImageNet16-120 (no baseline exists)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "fig2_other_datasets.png", dpi=300)
        plt.close(fig)

    def _fig3_eval_efficiency():
        eff = pd.read_csv(results_dir / "eval_efficiency_summary.csv")
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(eff)))
        ax.bar(eff["method"], eff["eval_reduction_pct"], color=colors)
        ax.set_ylabel("Eval reduction vs full budget (%)")
        ax.set_title("Fig 3 — real-eval efficiency (budget=100, cifar10)")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        fig.tight_layout()
        fig.savefig(out_dir / "fig3_eval_efficiency.png", dpi=300)
        plt.close(fig)

    def _fig4_statistical_tests():
        st = pd.read_csv(results_dir / "statistical_tests.csv")
        st_100 = st[st["budget"] == 100]
        if st_100.empty:
            st_100 = st
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        colors = ["seagreen" if s else "crimson" for s in st_100["significant"]]
        ax.barh(st_100["method_b"] + " / " + st_100["dataset"], st_100["cohens_d"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Cohen's d (archtree_llm vs baseline)")
        ax.set_title("Fig 4 — effect size vs baselines (green = p<0.05)")
        fig.tight_layout()
        fig.savefig(out_dir / "fig4_statistical_tests.png", dpi=300)
        plt.close(fig)

    def _fig5_ablation_summary():
        toggle_path = PROJECT_ROOT / "outputs" / "ablations" / "plots" / "data" / "toggle_ablations.csv"
        tdf = pd.read_csv(toggle_path)
        datasets = _present_datasets(tdf)
        ablation_ids = list(tdf["ablation"].unique())
        width = 0.8 / max(len(datasets), 1)
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        x = np.arange(len(ablation_ids))
        for i, ds in enumerate(datasets):
            sub = tdf[tdf["dataset"] == ds].set_index("ablation").reindex(ablation_ids)
            offset = (i - (len(datasets) - 1) / 2) * width
            ax.bar(x + offset, sub["delta_vs_full"], width=width, label=ds, color=_color_for(ds))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(ablation_ids, rotation=15, ha="right")
        ax.set_ylabel("Δ best_acc vs full (pp)")
        ax.set_title("Fig 5 — ablation summary (component removal impact)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "fig5_ablation_summary.png", dpi=300)
        plt.close(fig)

    def _fig6_hw_pareto():
        hv = storage.load_report_csv("phase4", "phase4_hv_igd_summary.csv")
        b100 = hv[hv["budget"] == hv["budget"].max()]
        agg = b100.groupby("hardware").agg(
            hypervolume=("hypervolume", "mean"),
            pareto_front_size=("pareto_front_size", "mean"),
        ).reset_index().sort_values("hypervolume")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        axes[0].barh(agg["hardware"], agg["hypervolume"], color="#4361b8")
        axes[0].set_xlabel("Hypervolume")
        axes[0].set_title(f"Mean HV @ budget={int(b100['budget'].iloc[0])}")
        axes[1].barh(agg["hardware"], agg["pareto_front_size"], color="#c9772e")
        axes[1].set_xlabel("Pareto front size")
        axes[1].set_title("Mean Pareto front size")
        fig.suptitle("Fig 6 — hardware-aware search summary (Phase 4)")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(out_dir / "fig6_hw_pareto_summary.png", dpi=300)
        plt.close(fig)

    _safe("fig1_main_results", _fig1_main_results)
    _safe("fig2_other_datasets", _fig2_other_datasets)
    _safe("fig3_eval_efficiency", _fig3_eval_efficiency)
    _safe("fig4_statistical_tests", _fig4_statistical_tests)
    _safe("fig5_ablation_summary", _fig5_ablation_summary)
    _safe("fig6_hw_pareto_summary", _fig6_hw_pareto)


# ══════════════════════════════════════════════════════════════════════════
# A7 — trie vs flat-UCB1 (dedup-vs-structure)
# ══════════════════════════════════════════════════════════════════════════
def plot_trie_ablation() -> None:
    out_dir = storage.get_output_dir("ablations", "plots/figures")

    def _acc_by_arm():
        df = storage.load_report_csv("ablations", "ablation_A7_no_trie.csv")
        summ = (df.groupby(["dataset", "budget", "param_value"])["best_acc"]
                  .agg(["mean", "std"]).reset_index().rename(columns={"param_value": "arm"}))
        datasets = _present_datasets(summ)
        fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=True)
        axes = np.atleast_1d(axes)
        arm_colors = {"trie": "#2b7a4b", "flat_set": "#4361b8", "flat_none": "#c9772e"}
        for ax, ds in zip(axes, datasets):
            for arm in sorted(summ["arm"].unique()):
                g = summ[(summ["dataset"] == ds) & (summ["arm"] == arm)].sort_values("budget")
                ax.errorbar(g["budget"], g["mean"], yerr=g["std"].fillna(0), marker="o",
                            capsize=3, label=arm, color=arm_colors.get(arm, "#777"))
            ax.set_title(ds)
            ax.set_xlabel("Budget")
        axes[0].set_ylabel("Best accuracy (%)")
        axes[0].legend(fontsize=8)
        fig.suptitle("A7 — trie vs flat_set (dedup) vs flat_none (no dedup)")
        fig.tight_layout()
        fig.savefig(out_dir / "A7_trie_vs_flat.png", dpi=300)
        plt.close(fig)

    def _dedup_waste():
        df = storage.load_report_csv("ablations", "A7_dedup_stats.csv")
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for i, arm in enumerate(sorted(df["arm"].unique()) if "arm" in df.columns else []):
            vals = df[df["arm"] == arm]["dup_frac"] if "dup_frac" in df.columns else df[df["arm"] == arm].iloc[:, -1]
            ax.scatter([arm] * len(vals), vals, alpha=0.6, color=plt.get_cmap("tab10")(i % 10))
        ax.set_ylabel("Fraction of evals that were duplicates")
        ax.set_title("A7 — duplicate-eval waste by arm")
        fig.tight_layout()
        fig.savefig(out_dir / "A7_dedup_waste.png", dpi=300)
        plt.close(fig)

    _safe("A7_trie_vs_flat", _acc_by_arm)
    _safe("A7_dedup_waste", _dedup_waste)


# ══════════════════════════════════════════════════════════════════════════
# Proxy composition ablation
# ══════════════════════════════════════════════════════════════════════════
def plot_proxy_ablation() -> None:
    out_dir = storage.get_output_dir("proxy_ablation", "plots/figures")

    def _acc_by_proxy_arm():
        df = storage.load_report_csv("proxy_ablation", "proxy_ablation_summary.csv")
        datasets = _present_datasets(df)
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        arms = sorted(df["arm"].unique())
        x = np.arange(len(arms))
        width = 0.8 / max(len(datasets), 1)
        for i, ds in enumerate(datasets):
            sub = df[df["dataset"] == ds].set_index("arm").reindex(arms)
            offset = (i - (len(datasets) - 1) / 2) * width
            ax.bar(x + offset, sub["best_acc_mean"], width=width,
                   yerr=sub.get("best_acc_std", pd.Series(0, index=sub.index)).fillna(0),
                   label=ds, color=_color_for(ds), capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(arms, rotation=20, ha="right")
        ax.set_ylabel("Best accuracy (%)")
        ax.set_title("Proxy composition ablation — accuracy by arm")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "proxy_ablation_accuracy.png", dpi=300)
        plt.close(fig)

    def _tau_vs_search_quality():
        df = storage.load_report_csv("proxy_ablation", "proxy_ablation_summary.csv")
        if "kendall_tau" not in df.columns:
            raise FileNotFoundError("kendall_tau column missing (rank_vs_search merge not run)")
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for ds in _present_datasets(df):
            sub = df[df["dataset"] == ds]
            ax.scatter(sub["kendall_tau"], sub["best_acc_mean"], label=ds, color=_color_for(ds))
        ax.set_xlabel("Kendall tau (proxy ranking quality)")
        ax.set_ylabel("Best accuracy (%)")
        ax.set_title("Proxy ablation — ranking quality vs search quality")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "proxy_ablation_tau_vs_quality.png", dpi=300)
        plt.close(fig)

    _safe("proxy_ablation_accuracy", _acc_by_proxy_arm)
    _safe("proxy_ablation_tau_vs_quality", _tau_vs_search_quality)


# ══════════════════════════════════════════════════════════════════════════
# Convergence study — C7 (LLM prior) early-convergence effect
# ══════════════════════════════════════════════════════════════════════════
def plot_convergence_study() -> None:
    """Oct 4, 2026: rewritten for the 4-arm study (llm_off / llm_on / size_prior / llm_described;
    columns arm, evals_to_1.0pp, ... and paired tests with arm_a / arm_b / p_holm)."""
    out_dir = storage.get_output_dir("convergence", "plots/figures")
    arm_colors = {"llm_off": "#1f3a5f", "llm_on": "#c0392b", "size_prior": "#7f8c8d", "llm_described": "#e67e22"}

    def _early_convergence():
        df = storage.load_report_csv("convergence", "convergence_per_run_metrics.csv")
        b = int(df["budget"].min())
        df = df[df["budget"] == b]
        datasets = _present_datasets(df)
        arms = [a for a in arm_colors if a in set(df["arm"])]
        fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 4), sharey=False)
        axes = np.atleast_1d(axes)
        for ax, ds in zip(axes, datasets):
            sub = df[df["dataset"] == ds]
            g = sub.groupby("arm")["evals_to_1.0pp"].agg(["mean", "sem"]).reindex(arms)
            ax.bar(range(len(arms)), g["mean"], yerr=g["sem"], capsize=3,
                   color=[arm_colors[a] for a in arms])
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels(arms, rotation=20, ha="right", fontsize=8)
            ax.set_title(ds)
        axes[0].set_ylabel("real evaluations to reach (best - 1 pt)\n(lower = faster), mean +- s.e.")
        fig.suptitle(f"Convergence: LLM prior vs knowledge-free size prior vs hidden-names prompt (budget {b})")
        fig.tight_layout()
        fig.savefig(out_dir / "convergence_early_speed.png", dpi=300)
        plt.close(fig)

    def _paired_effect_sizes():
        df = storage.load_report_csv("convergence", "convergence_paired_tests.csv")
        if "arm_a" not in df.columns:
            df["arm_a"], df["arm_b"] = "llm_on", "llm_off"
        df = df[df["metric"] == "evals_to_1.0pp"]
        df = df[df["budget"] == df["budget"].min()]
        sig = df["p_holm"] < 0.05 if "p_holm" in df.columns else df["significant"]
        label = df["arm_a"] + " vs " + df["arm_b"] + " | " + df["dataset"].astype(str)
        fig, ax = plt.subplots(figsize=(8, 0.32 * len(df) + 1.5))
        ax.barh(label, df["mean_diff"], color=["seagreen" if x else "#bbbbbb" for x in sig])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("difference in evaluations to (best - 1 pt): arm_a - arm_b  (negative = arm_a faster)")
        ax.set_title("Convergence study - paired differences (green = significant after Holm)")
        fig.tight_layout()
        fig.savefig(out_dir / "convergence_effect_sizes.png", dpi=300)
        plt.close(fig)

    _safe("convergence_early_speed", _early_convergence)
    _safe("convergence_effect_sizes", _paired_effect_sizes)


# ══════════════════════════════════════════════════════════════════════════
# Equal GPU-hours (step 22) + gate-cost model  (Oct 4, 2026)
# ══════════════════════════════════════════════════════════════════════════
EGH_ARMS = {   # arm -> (label, colour, linestyle)
    "archtree_zc_unsup": ("ArchTree v2 + label-free gate", "#c0392b", "-"),
    "archtree_none":     ("ArchTree v2, no gate", "#c0392b", ":"),
    "re_zc_unsup":       ("RE + same gate", "#1f3a5f", "-"),
    "re_none":           ("RE", "#1f3a5f", ":"),
    "re_zcwarm":         ("RE + zero-cost warm-up", "#16a085", "-"),
    "ls_none":           ("local search", "#8e44ad", ":"),
    "ls_zc_unsup":       ("local search + same gate", "#8e44ad", "-"),
    "random_none":       ("random", "#7f8c8d", ":"),
    "archtree_costpen":  ("ArchTree v2 + cost-penalised gate", "#e67e22", "--"),
}


def plot_equal_gpu_hours() -> None:
    out_dir = storage.get_output_dir("equal_gpu_hours", "plots/figures")

    def _acc_vs_gpu_hours():
        df = storage.load_report_csv("equal_gpu_hours", "egh_summary.csv")
        panels = list(df.groupby(["benchmark", "dataset"]).groups)
        n = len(panels)
        cols = min(3, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.8 * rows), squeeze=False)
        for ax, (bn, ds) in zip(axes.flat, panels):
            sub = df[(df["benchmark"] == bn) & (df["dataset"] == ds)]
            for arm, (lab, col, ls) in EGH_ARMS.items():
                g = sub[sub["arm"] == arm].sort_values("k")
                if len(g):
                    ax.plot(g["k"], g["test_mean"], ls, color=col, marker="o", ms=3, lw=1.6 if "archtree_zc" in arm else 1.0,
                            label=lab)
            ax.set_title(f"{bn} / {ds}", fontsize=10)
            ax.set_xlabel("GPU-hour budget (x mean training time of one architecture)")
            ax.set_ylabel("test accuracy of best-valid arch (%)")
            ax.grid(alpha=0.3)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), fontsize=8, frameon=False)
        fig.suptitle("Accuracy at EQUAL GPU-hours (12 seeds)", fontsize=12)
        fig.tight_layout(rect=[0, 0.07, 1, 0.96])
        fig.savefig(out_dir / "egh_acc_vs_gpu_hours.png", dpi=300)
        plt.close(fig)

    def _model_check():
        v = storage.load_report_csv("equal_gpu_hours", "gate_cost_validation.csv").dropna(subset=["predicted_delta"])
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        for (prop, lam), g in v.groupby(["proposer", "lambda"]):
            ax.scatter(g["predicted_delta"], g["measured_delta"], s=28,
                       marker="o" if prop == "random" else "^", label=f"{prop}, lambda={lam:g}")
        lim = np.nanmax(np.abs(v[["predicted_delta", "measured_delta"]].values)) * 1.1 + 1e-6
        ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.8, ls="--")
        ax.axhline(0, color="#999", lw=0.6)
        ax.axvline(0, color="#999", lw=0.6)
        ax.set_xlabel("predicted change of a cost-penalised gate (model, pts)")
        ax.set_ylabel("measured change (pts)")
        ax.set_title("Gate-cost model: predicted vs measured")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "egh_gate_cost_model_check.png", dpi=300)
        plt.close(fig)

    def _proxy_cost():
        c = storage.load_report_csv("equal_gpu_hours", "cost_correlation.csv")
        c = c[c["score"].isin(["E", "S", "T", "ensemble_unsup", "params", "valid_acc (oracle)"])]
        piv = c.groupby(["benchmark", "score"])["spearman_with_train_time"].mean().unstack(0)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        piv.plot.bar(ax=ax)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("Spearman rho with training time")
        ax.set_title("Zero-cost proxies are cost detectors")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(out_dir / "egh_proxy_cost_correlation.png", dpi=300)
        plt.close(fig)

    _safe("egh_acc_vs_gpu_hours", _acc_vs_gpu_hours)
    _safe("egh_gate_cost_model_check", _model_check)
    _safe("egh_proxy_cost_correlation", _proxy_cost)


def plot_llm_contamination() -> None:
    out_dir = storage.get_output_dir("llm_contamination", "plots/figures")

    def _probe():
        df = storage.load_report_csv("llm_contamination", "contamination_summary.csv")
        modes = [m for m in ("standard", "described", "anonymous", "size_prior") if m in set(df["mode"])]
        datasets = _present_datasets(df)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        x = np.arange(len(modes))
        w = 0.8 / max(len(datasets), 1)
        for i, ds in enumerate(datasets):
            sub = df[df["dataset"] == ds].set_index("mode").reindex(modes)
            off = (i - (len(datasets) - 1) / 2) * w
            axes[0].bar(x + off, sub["spearman_mean"], w, color=_color_for(ds), label=ds)
            axes[1].bar(x + off, sub["top1_hit_rate"], w, color=_color_for(ds), label=ds)
        for ax, t, chance in ((axes[0], "op-ranking quality (Spearman rho)", 0.0), (axes[1], "top-1 hit rate", 0.2)):
            ax.set_xticks(x)
            ax.set_xticklabels(modes)
            ax.axhline(chance, color="crimson", ls="--", lw=1, label=f"chance ({chance:g})")
            ax.set_title(t)
        axes[0].legend(fontsize=7)
        fig.suptitle("LLM contamination test: named vs described vs anonymous prompt vs size prior")
        fig.tight_layout()
        fig.savefig(out_dir / "llm_contamination_probe.png", dpi=300)
        plt.close(fig)

    _safe("llm_contamination_probe", _probe)


# ══════════════════════════════════════════════════════════════════════════
# LLM-size sensitivity study
# ══════════════════════════════════════════════════════════════════════════
def plot_llm_size_sensitivity() -> None:
    out_dir = storage.get_output_dir("llm_size", "plots/figures")

    def _probe_quality():
        # llm_size_summary.csv is the PROBE-stage ranking-quality file (not accuracy):
        # columns = model, dataset, n, spearman_mean, top1_hit_rate,
        #           uniform_fallback_rate, name_mismatch_per_call, latency_mean_sec
        df = storage.load_report_csv("llm_size", "llm_size_summary.csv")
        datasets = _present_datasets(df)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        models = sorted(df["model"].unique())
        x = np.arange(len(models))
        width = 0.8 / max(len(datasets), 1)
        for i, ds in enumerate(datasets):
            sub = df[df["dataset"] == ds].set_index("model").reindex(models)
            offset = (i - (len(datasets) - 1) / 2) * width
            axes[0].bar(x + offset, sub["spearman_mean"], width=width, label=ds, color=_color_for(ds))
            axes[1].bar(x + offset, sub["top1_hit_rate"], width=width, label=ds, color=_color_for(ds))
        for ax, title, ylabel in [(axes[0], "Ranking quality (rho)", "Spearman rho"),
                                   (axes[1], "Top-1 hit rate", "Top-1 hit rate")]:
            ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
            ax.set_title(title); ax.set_ylabel(ylabel)
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[1].axhline(0.20, color="crimson", linestyle="--", linewidth=1, label="chance (0.20)")
        axes[0].legend(fontsize=7)
        axes[1].legend(fontsize=7)
        fig.suptitle("LLM-size probe — proxy ranking quality by model")
        fig.tight_layout()
        fig.savefig(out_dir / "llm_size_probe_quality.png", dpi=300)
        plt.close(fig)

    def _search_accuracy_delta():
        # llm_size_paired_tests.csv: dataset, budget, metric, arm(=model), n_paired_seeds,
        #                             mean_on, mean_off, mean_diff, wilcoxon_p, cohens_d, significant
        df = storage.load_report_csv("llm_size", "llm_size_paired_tests.csv")
        df = df[df["metric"] == "best_acc"] if "metric" in df.columns else df
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        label = df["arm"] + " / " + df["dataset"].astype(str) + " b=" + df["budget"].astype(str)
        colors = ["seagreen" if s else "crimson" for s in df["significant"]]
        ax.barh(label, df["mean_diff"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("mean_diff = LLM-on(model) − LLM-off, best_acc (pp)")
        ax.set_title("LLM-size search — accuracy delta vs no-LLM (green = p<0.05)")
        fig.tight_layout()
        fig.savefig(out_dir / "llm_size_search_delta.png", dpi=300)
        plt.close(fig)

    _safe("llm_size_probe_quality", _probe_quality)
    _safe("llm_size_search_delta", _search_accuracy_delta)


# ══════════════════════════════════════════════════════════════════════════
# Cross-device Pareto transfer analysis
# ══════════════════════════════════════════════════════════════════════════
def plot_cross_device_pareto() -> None:
    out_dir = storage.get_output_dir("final", "plots/figures")

    def _truth_jaccard():
        df = storage.load_report_csv("final", "cross_device_truth_overlap.csv")
        fig, ax = plt.subplots(figsize=(7, 5))
        piv = df.groupby(["hw_a", "hw_b"])["jaccard"].mean().unstack() if {"hw_a", "hw_b"}.issubset(df.columns) \
            else df.pivot(index=df.columns[0], columns=df.columns[1], values="jaccard")
        im = ax.imshow(piv.values, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        fig.colorbar(im, ax=ax, label="Front Jaccard similarity")
        ax.set_title("Cross-device TRUE Pareto-front Jaccard overlap")
        fig.tight_layout()
        fig.savefig(out_dir / "cross_device_truth_jaccard.png", dpi=300)
        plt.close(fig)

    def _transfer_precision():
        df = storage.load_report_csv("final", "cross_device_found_overlap.csv")
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.hist(df["transfer_precision_a_to_b"].dropna(), bins=20, color="#c9772e")
        ax.set_xlabel("Transfer precision A→B")
        ax.set_ylabel("Count")
        ax.set_title("Cross-device transfer precision (own-front on other device's true front)")
        fig.tight_layout()
        fig.savefig(out_dir / "cross_device_transfer_precision.png", dpi=300)
        plt.close(fig)

    _safe("cross_device_truth_jaccard", _truth_jaccard)
    _safe("cross_device_transfer_precision", _transfer_precision)


# ══════════════════════════════════════════════════════════════════════════
# TCR-H vs accuracy correlation
# ══════════════════════════════════════════════════════════════════════════
def plot_tcr_h_correlation() -> None:
    out_dir = storage.get_output_dir("final", "plots/figures")

    def _corr_bars():
        df = storage.load_report_csv("final", "tcr_h_accuracy_correlation.csv")
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        x = np.arange(len(df))
        width = 0.35
        ax.bar(x - width / 2, df["r1"] if "r1" in df.columns else df.iloc[:, 2], width, label="TCR-H vs acc", color="#2b7a4b")
        ax.bar(x + width / 2, df["r2"] if "r2" in df.columns else df.iloc[:, 3], width, label="plain TCR vs acc", color="#4361b8")
        labels = (df["dataset"].astype(str) + "/" + df["budget"].astype(str)) if "budget" in df.columns else df["dataset"]
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Spearman r")
        ax.set_title("TCR-H vs plain-TCR correlation with best_acc")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "tcr_h_accuracy_correlation.png", dpi=300)
        plt.close(fig)

    _safe("tcr_h_accuracy_correlation", _corr_bars)


# ══════════════════════════════════════════════════════════════════════════
# Regret decay
# ══════════════════════════════════════════════════════════════════════════
def plot_regret_decay() -> None:
    out_dir = storage.get_output_dir("final", "plots/figures")

    def _curves():
        df = storage.load_report_csv("final", "regret_curves.csv")
        datasets = _present_datasets(df)
        fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, ds in zip(axes, datasets):
            sub = df[df["dataset"] == ds]
            for m in sorted(sub["method"].unique()):
                g = sub[sub["method"] == m].groupby("t")["regret"].mean().reset_index() \
                    if "regret" in sub.columns else sub[sub["method"] == m]
                ycol = "regret" if "regret" in g.columns else g.columns[-1]
                ax.plot(g["t"], g[ycol], label=m)
            ax.set_title(ds)
            ax.set_xlabel("Real evals (t)")
        axes[0].set_ylabel("Regret")
        axes[0].legend(fontsize=7)
        fig.suptitle("Regret decay by method (flat_ucb_none ≈ no decay confirms A7)")
        fig.tight_layout()
        fig.savefig(out_dir / "regret_decay_curves.png", dpi=300)
        plt.close(fig)

    _safe("regret_decay_curves", _curves)


# ══════════════════════════════════════════════════════════════════════════
# Strict equal-real-compute headline (§0.8) — the final win/tie/loss figure
# ══════════════════════════════════════════════════════════════════════════
def plot_strict_equal_compute() -> None:
    out_dir = storage.get_output_dir("final", "plots/figures")
    results_dir = PROJECT_ROOT / "outputs" / "final" / "results"

    def _win_tie_loss():
        df = pd.read_csv(results_dir / "equal_compute_statistical_tests_strict.csv")

        def outcome(r):
            # matches the script's own archtree_wins_or_ties = (diff>=0 or p>=0.05) definition
            if not r["significant"]:
                return "tie"
            return "win" if r["mean_diff_archtree_minus_baseline"] > 0 else "loss"
        df["outcome"] = df.apply(outcome, axis=1)
        counts = df["outcome"].value_counts().reindex(["win", "tie", "loss"]).fillna(0)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.bar(counts.index, counts.values, color=["seagreen", "#c9c9c9", "crimson"])
        for i, v in enumerate(counts.values):
            ax.text(i, v + 0.5, str(int(v)), ha="center")
        ax.set_ylabel("# baseline comparisons")
        ax.set_title(f"Strict equal-real-compute outcome (n={int(counts.sum())}, "
                     f"wins-or-ties={df['archtree_wins_or_ties'].mean():.0%})")
        fig.tight_layout()
        fig.savefig(out_dir / "strict_equal_compute_win_tie_loss.png", dpi=300)
        plt.close(fig)

    def _by_baseline():
        df = pd.read_csv(results_dir / "equal_compute_statistical_tests_strict.csv")
        agg = df.groupby("baseline")["mean_diff_archtree_minus_baseline"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        colors = ["crimson" if v < 0 else "seagreen" for v in agg.values]
        ax.barh(agg.index, agg.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean best_acc diff, archtree_llm − baseline (pp)")
        ax.set_title("Strict equal-compute — effect size per baseline")
        fig.tight_layout()
        fig.savefig(out_dir / "strict_equal_compute_by_baseline.png", dpi=300)
        plt.close(fig)

    _safe("strict_equal_compute_win_tie_loss", _win_tie_loss)
    _safe("strict_equal_compute_by_baseline", _by_baseline)


# ══════════════════════════════════════════════════════════════════════════
# F7 — budget-paired significance (ArchTree vs every method x dataset,
# Wilcoxon signed-rank ACROSS the shared budget grid, not across seeds)
# ══════════════════════════════════════════════════════════════════════════
def plot_significance_f7(archtree_method: str = "archtree_llm") -> None:
    """Pairs archtree_method's best_acc_mean(budget) against every other
    (method, dataset) row's best_acc_mean(budget) at the budgets they share,
    then runs one Wilcoxon signed-rank test per (method, dataset) over that
    small budget grid (n = # shared budgets, typically 3-4) rather than over
    seeds. This is intentionally a coarser, more conservative test than
    statistical_tests.csv's per-seed pairing — it answers 'does the ranking
    hold consistently across the whole budget sweep', not 'is any single
    budget's per-seed gap real'. With only 3-4 budgets the test is
    structurally underpowered (min two-sided p for n=3 is 0.25, for n=4 is
    0.125) — that's flagged on the plot itself, it's not a bug in the code.
    """
    from scipy.stats import wilcoxon
    out_dir = storage.get_output_dir("final", "plots/figures")
    results_dir = PROJECT_ROOT / "outputs" / "final" / "results"
    main = pd.read_csv(results_dir / "main_results_table.csv")

    a = main[main["method"] == archtree_method]
    if a.empty:
        raise FileNotFoundError(f"no '{archtree_method}' rows in main_results_table.csv")

    rows = []
    for (method, dataset), b in main[main["method"] != archtree_method].groupby(["method", "dataset"]):
        a_ds = a[a["dataset"] == dataset]
        common_budgets = sorted(set(a_ds["budget"]) & set(b["budget"]))
        if len(common_budgets) < 3:
            continue  # wilcoxon degenerate/meaningless below n=3 paired points
        a_vals = a_ds.set_index("budget").loc[common_budgets, "best_acc_mean"].values
        b_vals = b.set_index("budget").loc[common_budgets, "best_acc_mean"].values
        diff = a_vals - b_vals
        if np.all(diff == 0):
            continue
        try:
            stat, p = wilcoxon(a_vals, b_vals)
        except ValueError:
            continue
        rows.append({"method": method, "dataset": dataset, "n_paired_budgets": len(common_budgets),
                     "mean_diff": diff.mean(), "p_value": p, "significant": p < 0.05})

    if not rows:
        raise FileNotFoundError("no (method,dataset) pair had >=3 shared budgets with "
                                 f"{archtree_method} — nothing to test")
    res = pd.DataFrame(rows).sort_values("p_value", ascending=False)
    res["neglog10_p"] = -np.log10(res["p_value"].clip(lower=1e-300))
    res.to_csv(results_dir / "significance_f7_budget_paired.csv", index=False)

    n_min = int(res["n_paired_budgets"].min())
    min_achievable_p = 2 / (2 ** n_min)  # exact wilcoxon 2-sided floor for n paired points
    labels = [f"{m} ({d})" for m, d in zip(res["method"], res["dataset"])]

    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(res) + 1.6))
    y = np.arange(len(res))
    colors = ["#4a90d9" if not s else "#2b7a4b" for s in res["significant"]]
    ax.barh(y, res["neglog10_p"], color=colors, height=0.6)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    for yi, (p, sig) in enumerate(zip(res["neglog10_p"], res["significant"])):
        ax.text(p + 0.02, yi, "sig" if sig else "ns", va="center", fontweight="bold", fontsize=9)
    ax.axvline(-np.log10(0.05), color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$-\log_{10}(p)$  (dashed = p=0.05)")
    ax.set_title(f"F7 — Significance: {archtree_method} vs. Baselines (budget-paired)")

    ax.text(0.98, 0.97,
            f"n_paired_budgets={n_min} -> min achievable p = {min_achievable_p:.2f}.\n"
            "Underpowered design, not evidence of no effect.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, style="italic",
            bbox=dict(boxstyle="round", facecolor="#fdecea", edgecolor="crimson"))
    if (res["mean_diff"] > 0).all():
        ax.text(0.98, 0.03, "Gains are real, not seed noise",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="#888"))
    fig.tight_layout()
    fig.savefig(out_dir / "f7_significance_budget_paired.png", dpi=300)
    plt.close(fig)
    print("[plot_results] OK — f7_significance_budget_paired")


def plot_everything() -> None:
    """Run every figure function across every phase/experiment in one call."""
    for name, fn in [
        ("plot_phase3", lambda: plot_phase3("phase3")),
        ("plot_phase4", lambda: plot_phase4("phase4")),
        ("plot_ablations", plot_ablations),
        ("plot_trie_ablation", plot_trie_ablation),
        ("plot_proxy_ablation", plot_proxy_ablation),
        ("plot_convergence_study", plot_convergence_study),
        ("plot_llm_size_sensitivity", plot_llm_size_sensitivity),
        ("plot_cross_device_pareto", plot_cross_device_pareto),
        ("plot_tcr_h_correlation", plot_tcr_h_correlation),
        ("plot_regret_decay", plot_regret_decay),
        ("plot_strict_equal_compute", plot_strict_equal_compute),
        ("plot_significance_f7", plot_significance_f7),
        ("plot_equal_gpu_hours", plot_equal_gpu_hours),
        ("plot_llm_contamination", plot_llm_contamination),
        ("plot_final_all", plot_final_all),
    ]:
        print(f"\n=== {name} ===")
        try:
            fn()
        except Exception as e:
            print(f"[plot_results] top-level SKIP {name} — {type(e).__name__}: {e}")


if __name__ == "__main__":
    # manual invocation: python plots/plot_results.py [phase3|phase4|ablations|trie|proxy|
    #                        convergence|llmsize|crossdevice|tcrh|regret|strict|f7|egh|contamination|final|all]
    # (pipeline step 18 runs "all" since Oct 4, 2026)
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    {
        "phase3": lambda: plot_phase3("phase3"),
        "phase4": lambda: plot_phase4("phase4"),
        "ablations": plot_ablations,
        "trie": plot_trie_ablation,
        "proxy": plot_proxy_ablation,
        "convergence": plot_convergence_study,
        "llmsize": plot_llm_size_sensitivity,
        "crossdevice": plot_cross_device_pareto,
        "tcrh": plot_tcr_h_correlation,
        "regret": plot_regret_decay,
        "strict": plot_strict_equal_compute,
        "f7": plot_significance_f7,
        "egh": plot_equal_gpu_hours,
        "contamination": plot_llm_contamination,
        "final": plot_final_all,
        "all": plot_everything,
    }.get(target, plot_everything)()