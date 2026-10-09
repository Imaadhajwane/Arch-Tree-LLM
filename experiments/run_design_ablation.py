# experiments/run_design_ablation.py
# isort: skip_file
"""
ArchTree v1 -> v2 design ablation on HELD-OUT seeds (Oct 3, 2026).

Why: v2 (rank-normalised UCB reward, C=0.3, 50% evolutionary expansion) was chosen after looking
at results on the 12 main seeds. To report that choice honestly, every design step is re-tested on
12 seeds that were never used to choose it (config search.heldout_seeds), with the headline gate
(label-free ZC proxy, top-35% after 10 warm-up proposals).

Arms (all search_phase3, alpha=0, max backup, value pruning, same gate)
    v1        reward=raw  (accuracy/100), C=1.414, no mutation       - the Oct-2 system
    rank      reward=rank, C=config, no mutation                     - reward scale fix only
    v2        reward=rank, C=config, mutation=config (0.5)          - the headline (ArchTree v2)
    mut_only  reward=rank, C=config, mutation=1.0                    - no UCB proposals after warm-up:
                                                                       does the trie/UCB part add anything?
    re        regularized evolution with the same gate               - strongest baseline

Outputs (outputs/design_ablation/results/)
    design_runs.csv          one row per (dataset, budget, seed, arm)          [resumable]
    design_summary.csv       mean/std test accuracy, real evals, train hours
    design_paired_tests.csv  every arm vs v2, Wilcoxon, Holm within each arm (= family)

Usage
    python experiments/run_design_ablation.py                 # pipeline step 11
    python experiments/run_design_ablation.py --summarize-only
"""
import argparse
import gc
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

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

PHASE = "design_ablation"
RUNS = "design_runs.csv"
ARMS = {
    "v1": dict(reward_mode="raw", C=1.414, mutation_prob=0.0),
    "rank": dict(reward_mode="rank", C=settings.C, mutation_prob=0.0),
    "v2": dict(reward_mode="rank", C=settings.C, mutation_prob=settings.MUTATION_PROB),
    "mut_only": dict(reward_mode="rank", C=settings.C, mutation_prob=1.0),
    "re": None,
}

logger.remove()
LOG_DIR = PROJECT_ROOT / "outputs" / PHASE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>")
logger.add(LOG_DIR / "design_run.log", level="DEBUG", rotation="10 MB", retention=5)


def _llm_free_backend_kwargs():
    """LLMPrior is constructed by search_phase3 even with use_llm=False (no call is made)."""
    try:
        import ollama  # noqa: F401
        return dict(backend="ollama")
    except ImportError:
        pass
    keys = [os.environ.get(k, "").strip() for k in ("GROQ_API_KEYS_FILE", "GROQ_API_KEYS", "GROQ_API_KEY")]
    if not any(keys):
        logger.error("LLMPrior needs `ollama` or a GROQ key just to construct (no call is made).")
        sys.exit(1)
    return dict(backend="groq", groq_model=cfg["llm"]["groq"]["model"], use_key_pool=bool(keys[0] or keys[1]))


def run_re(bench, dataset, budget, seed, gate, pct, warmup):
    """Regularized evolution with the same gate rule (budget = proposals)."""
    from baselines.regularized_evolution import RegularizedEvolution
    from benchmarks.nb201 import best_from_history
    from archtree.search import _train_hours
    random.seed(seed)
    prop = RegularizedEvolution(bench, seed, budget)
    evaluated, scores, history = set(), [], []
    for i in range(budget):
        toks = prop.propose(evaluated)
        if i >= warmup:
            s = gate.score(toks)
            scores.append(s)
            if s < np.percentile(scores, 100 - pct):
                continue
        idx = bench.tokens_to_index(toks)
        acc = bench.query(idx, dataset)
        evaluated.add(idx)
        prop.update(toks, acc)
        history.append({"iter": i + 1, "arch": "|".join(toks), "acc": acc})
    rep = best_from_history(bench, history, dataset)
    return {"best_acc": rep["best_acc"], "best_test_acc": rep["best_test_acc"],
            "real_eval_count": len(history), "train_hours": _train_hours(bench, history, dataset),
            "mutation_proposals": np.nan}


def _done():
    p = PROJECT_ROOT / "outputs" / PHASE / "results" / RUNS
    if not p.exists():
        return set()
    d = pd.read_csv(p)
    return set(zip(d["dataset"], d["budget"], d["seed"], d["arm"]))


def run(args):
    from benchmarks.nb201 import NB201
    from archtree.search import search_phase3
    from archtree.zc_proxy import ZCProxyEnsemble, get_mini_batch
    bench = NB201(cfg)
    bench._load()
    get_mini_batch()
    pct, warmup = settings.PROXY_THRESHOLD_PERCENTILE, settings.WARMUP_REAL_EVALS
    backend_kw = _llm_free_backend_kwargs()
    done = _done()
    for dataset in args.datasets:
        gate = ZCProxyEnsemble.unsupervised(bench, dataset)       # label-free headline gate
        for budget in args.budgets:
            for seed in args.seeds:
                for arm in args.arms:
                    if (dataset, budget, seed, arm) in done:
                        continue
                    t0 = time.time()
                    try:
                        if arm == "re":
                            r = run_re(bench, dataset, budget, seed, gate, pct, warmup)
                        else:
                            a = ARMS[arm]
                            r = search_phase3(
                                bench, budget, dataset, seed, phase=PHASE, alpha=0.0, use_llm=False,
                                use_proxy=True, zc_override=gate, ablation_id=f"D_{arm}",
                                proxy_threshold_percentile=pct, backup_mode=settings.BACKUP_MODE_P34,
                                C=a["C"], reward_mode=a["reward_mode"], mutation_prob=a["mutation_prob"],
                                results_filename="D_raw_runs.csv", tcr_filename="D_raw_tcr.csv",
                                cache_stats_filename="D_raw_cache_stats.csv",
                                dedup_filename="D_dedup_stats.csv", **backend_kw)
                    except KeyboardInterrupt:
                        logger.warning("Interrupted - append-only CSV, rerun to resume.")
                        sys.exit(0)
                    except Exception:
                        logger.exception(f"ERROR {dataset} b={budget} s={seed} {arm}")
                        continue
                    storage.append_result_row(PHASE, RUNS, {
                        "dataset": dataset, "budget": budget, "seed": seed, "arm": arm,
                        "best_acc": r["best_acc"], "best_test_acc": r["best_test_acc"],
                        "real_eval_count": r["real_eval_count"],
                        "train_hours": r.get("train_hours", np.nan),
                        "wall_time_sec": time.time() - t0})
                    logger.info(f"[D] {dataset} b={budget} s={seed} {arm:8s} test={r['best_test_acc']:.2f} "
                                f"real={r['real_eval_count']}")
                    gc.collect()


def _holm(p):
    p = np.asarray(p, float)
    o, m, out, run_ = np.argsort(p), len(p), np.empty(len(p)), 0.0
    for k, i in enumerate(o):
        run_ = max(run_, (m - k) * p[i])
        out[i] = min(1.0, run_)
    return out


def summarize():
    p = PROJECT_ROOT / "outputs" / PHASE / "results" / RUNS
    if not p.exists():
        logger.error("no runs yet")
        return
    df = pd.read_csv(p).drop_duplicates(["dataset", "budget", "seed", "arm"], keep="last")
    res = storage.get_output_dir(PHASE, "results")
    (df.groupby(["dataset", "budget", "arm"])
       .agg(n=("seed", "nunique"), test_mean=("best_test_acc", "mean"), test_std=("best_test_acc", "std"),
            real_evals=("real_eval_count", "mean"), train_hours=("train_hours", "mean"))
       .reset_index().to_csv(res / "design_summary.csv", index=False))
    rows = []
    for (ds, b), grp in df.groupby(["dataset", "budget"]):
        piv = grp.pivot(index="seed", columns="arm", values="best_test_acc")
        if "v2" not in piv:
            continue
        for arm in [a for a in piv.columns if a != "v2"]:
            both = piv[[arm, "v2"]].dropna()
            if len(both) < 2:
                continue
            d = (both[arm] - both["v2"]).values
            try:
                pv = wilcoxon(both[arm], both["v2"]).pvalue if np.any(d != 0) else 1.0
            except ValueError:
                pv = 1.0
            rows.append({"dataset": ds, "budget": b, "arm": arm, "vs": "v2", "n_paired_seeds": len(both),
                         "mean_diff_arm_minus_v2": float(d.mean()), "wilcoxon_p": pv})
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm"] = t.groupby("arm")["wilcoxon_p"].transform(lambda s: _holm(s.values))
        t["significant"] = t["p_holm"] < 0.05
        t["outcome_vs_v2"] = np.where(t["significant"],
                                      np.where(t["mean_diff_arm_minus_v2"] > 0, "arm better", "v2 better"), "tie")
        t["cohens_d"] = np.nan
    t.to_csv(res / "design_paired_tests.csv", index=False)
    logger.info("=" * 78)
    logger.info("Design ablation on held-out seeds (each arm vs v2, Holm within arm):")
    for arm, g in (t.groupby("arm") if len(t) else []):
        c = g["outcome_vs_v2"].value_counts()
        logger.info(f"  {arm:9s} mean {g['mean_diff_arm_minus_v2'].mean():+.3f} pts   v2 better={c.get('v2 better', 0)} "
                    f"ties={c.get('tie', 0)} {arm} better={c.get('arm better', 0)} (of {len(g)})")
    logger.success("design_summary.csv / design_paired_tests.csv written.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--datasets", nargs="+", default=cfg["benchmarks"]["datasets"])
    ap.add_argument("--budgets", nargs="+", type=int, default=cfg["search"]["budgets"])
    ap.add_argument("--seeds", nargs="+", type=int, default=settings.HELDOUT_SEEDS or cfg["search"]["seeds"])
    a = ap.parse_args()
    if not a.summarize_only:
        run(a)
    summarize()
