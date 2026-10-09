# experiments/run_phase1.py
# isort: skip_file
"""
PHASE 1 — Core ArchTree v1 (Trie + UCB1 + Weight Inheritance).
Single process, loads NB201 once. Resume-safe: skips (dataset, seed, budget)
combos that already have a log file.

Usage (from archtree-llm/ root):
    python experiments/run_phase1.py
"""

import sys
import gc
import time
import yaml
import torch
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import archtree.storage as storage  # noqa: E402

with open(PROJECT_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "outputs" / "phase1" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
# Custom levels so RUN / SKIP lines stand out in the console
logger.level("RUN", no=22, color="<cyan>")
logger.level("SKIP", no=21, color="<yellow>")
logger.add(
    sys.stderr, level="INFO", colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>",
)
# File sink stays uncolored (escape codes would corrupt the log file)
logger.add(LOG_DIR / "phase1_run.log", level="DEBUG", rotation="10 MB", retention=5)

# ── Preflight ─────────────────────────────────────────────────────────────────
storage.verify_phase_inputs("phase1")
baseline_df = storage.load_results_csv("phase0", "baseline_summary.csv")
torch.set_num_threads(2)

from benchmarks.nb201 import NB201         # noqa: E402
from archtree.search import search_phase1  # noqa: E402

bench = NB201(cfg)
bench._load()
logger.info("Benchmark loaded OK")

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS       = cfg["search"]["seeds"]
BUDGETS     = cfg["search"]["budgets"]
DATASETS    = cfg["benchmarks"]["datasets"]
BACKUP_MODE = cfg.get("ucb", {}).get("backup_mode", "mean")


def already_done(seed, budget, dataset):
    p = LOG_DIR / f"archtree_v1__nb201__{dataset}__seed{seed}__budget{budget}.json"
    return p.exists()


# ── Main loop ─────────────────────────────────────────────────────────────────
run_num = skipped = errored = 0

for dataset in DATASETS:
    for seed in SEEDS:
        for budget in BUDGETS:
            run_num += 1

            if already_done(seed, budget, dataset):
                logger.log("SKIP", f"[{run_num:02d}] SKIP archtree_v1 | {dataset} | seed={seed} | budget={budget}")
                skipped += 1
                continue

            logger.log("RUN", f"[{run_num:02d}] archtree_v1 | {dataset} | seed={seed} | budget={budget}")
            try:
                t0 = time.time()
                result = search_phase1(bench, budget, dataset, seed,
                                       phase="phase1", backup_mode=BACKUP_MODE)
                elapsed = time.time() - t0
                logger.success(f"    best_acc={result['best_acc']:.2f}%  "
                               f"real_evals={result['real_eval_count']}/{budget}  time={elapsed:.1f}s")
            except KeyboardInterrupt:
                logger.warning("Interrupted — progress saved (CSV is append-only), rerun to resume.")
                sys.exit(0)
            except Exception:
                logger.exception(f"    ERROR in run {run_num}:")
                errored += 1
            finally:
                gc.collect()

# ── Summary ───────────────────────────────────────────────────────────────────
logger.info("=" * 50)
logger.info(f"Total: {run_num} | Skipped: {skipped} | Errors: {errored}")

try:
    results = storage.load_results_csv("phase1", "phase1_main_results.csv")

    # Phase 0 baselines only exist for cifar10, so every comparison below is
    # restricted to cifar10. Mixing datasets would skew the means.
    results_c10  = results[results["dataset"] == "cifar10"]
    baseline_c10 = baseline_df[baseline_df["dataset"] == "cifar10"]
    baseline_methods = baseline_c10["method"].unique()

    logger.info("Best_acc by budget — archtree_v1 vs phase0 baselines (cifar10 ONLY):")
    for budget in BUDGETS:
        at_acc = results_c10[results_c10["budget"] == budget]["best_acc"].mean()
        row = f"  budget={budget:<4} archtree_v1={at_acc:.2f}%"
        for m in baseline_methods:
            b_acc = baseline_c10[(baseline_c10["method"] == m) &
                                 (baseline_c10["budget"] == budget)]["best_acc"].mean()
            row += f"  {m}={b_acc:.2f}%"
        logger.info(row)

    overall_at     = results_c10["best_acc"].mean()
    overall_random = baseline_c10[baseline_c10["method"] == "random_search"]["best_acc"].mean()
    gain = overall_at - overall_random
    logger.info(f"Mean best_acc (cifar10 only): archtree_v1={overall_at:.2f}%  "
                f"random_search={overall_random:.2f}%  gain={gain:+.2f}pp")

    if gain <= 0:
        logger.warning("archtree_v1 not beating random_search on accuracy — "
                       "check UCB C / prune_percentile in config.yaml.")
    else:
        logger.success("Quality check passed — UCB traversal beats random search at equal budget.")

    # Other datasets: reported on their own, no baseline to compare against
    for ds in DATASETS:
        if ds != "cifar10":
            ds_acc = results[results["dataset"] == ds]["best_acc"].mean()
            logger.info(f"  (info, no baseline) archtree_v1 mean best_acc on {ds}: {ds_acc:.2f}%")

    # real_eval_count == budget is expected here: at this budget, repeat draws
    # from the 15625-arch space are too rare for trie dedup to save any evals.
    logger.info("(Note: real_eval_count == budget at this scale is expected, not a bug.)")

    logger.success("phase1 DONE — ready for Phase 2.")
except FileNotFoundError:
    logger.error("no results written — check errors above")
    sys.exit(1)