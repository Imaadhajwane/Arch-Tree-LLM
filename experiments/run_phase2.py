# experiments/run_phase2.py
# isort: skip_file
"""
PHASE 2 — Zero-Cost (ZC) Proxy Ensemble.
Single process, loads NB201 once. For each dataset, calibrates a ZC proxy
ensemble (or loads saved weights), then uses it to gate real evaluations.
Resume-safe: finished (dataset, seed, budget) combos and saved calibrations
are skipped.

Config keys used (config.yaml):
    zc_proxy.calibration_archs             archs used to calibrate the proxy
    zc_proxy.proxy_threshold_percentile    ~fraction of candidates that get a
                                           real eval (lower = more skipped =
                                           higher eval reduction)

Usage (from archtree-llm/ root):
    python experiments/run_phase2.py
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
LOG_DIR = PROJECT_ROOT / "outputs" / "phase2" / "logs"
WEIGHTS_DIR = PROJECT_ROOT / "outputs" / "phase2" / "weights"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.level("RUN", no=22, color="<cyan>")
logger.level("SKIP", no=21, color="<yellow>")
logger.add(
    sys.stderr, level="INFO", colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>",
)
logger.add(LOG_DIR / "phase2_run.log", level="DEBUG", rotation="10 MB", retention=5)

# ── Preflight ─────────────────────────────────────────────────────────────────
storage.verify_phase_inputs("phase2")
torch.set_num_threads(2)

from benchmarks.nb201 import NB201             # noqa: E402
from archtree.search import search_phase2      # noqa: E402
from archtree.zc_proxy import ZCProxyEnsemble  # noqa: E402

bench = NB201(cfg)
bench._load()
logger.info("Benchmark loaded OK")

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS       = cfg["search"]["seeds"]
BUDGETS     = cfg["search"]["budgets"]
DATASETS    = cfg["benchmarks"]["datasets"]
BACKUP_MODE = cfg.get("ucb", {}).get("backup_mode", "mean")
CAL_ARCHS   = cfg.get("zc_proxy", {}).get("calibration_archs", 1500)
CAL_MODE    = cfg.get("zc_proxy", {}).get("calibration_mode", "supervised")
PROXY_THRESHOLD_PCT = cfg.get("zc_proxy", {}).get("proxy_threshold_percentile", 35)
logger.info(f"proxy_threshold_percentile={PROXY_THRESHOLD_PCT} (from config.yaml)")


def already_done(seed, budget, dataset):
    p = LOG_DIR / f"archtree_zc__nb201__{dataset}__seed{seed}__budget{budget}.json"
    return p.exists()


# ── Main loop ─────────────────────────────────────────────────────────────────
run_num = skipped = errored = 0

for dataset in DATASETS:
    # Calibrate per dataset: a proxy fit on one dataset's accuracies must not
    # gate architectures for another. Saved weights are reused on rerun.
    # BOTH calibrations are always built: the configured one drives the gate. The supervised one
    # also writes the per-proxy Kendall tau table (phase2_proxy_calibration.csv) and is the
    # "labelled" arm of the gated-baseline comparison; the label-free one is used by the
    # zico/rz_nas-style baselines. (Oct 2: with calibration_mode=unsupervised the supervised fit was
    # skipped, so the tau table was never written and the summary failed.)
    for mode in ("supervised", "unsupervised"):
        cal_path = storage._zc_cal_path("phase2", bench.name, dataset, mode)
        if cal_path.exists():
            logger.info(f"[{dataset}] {mode} calibration found -> {cal_path}")
            continue
        logger.info(f"[{dataset}] calibrating ZC proxies ({mode}) on {CAL_ARCHS} archs ...")
        ZCProxyEnsemble().calibrate(bench, n=CAL_ARCHS, phase="phase2", dataset=dataset, mode=mode)
    zc = ZCProxyEnsemble.from_saved("phase2", bench.name, dataset, mode=CAL_MODE)
    logger.info(f"[{dataset}] gate uses the {CAL_MODE} calibration "
                f"({zc.labels_used} true accuracies consumed by calibration)")
    if CAL_MODE == "unsupervised":
        # Report how well the label-free gate ranks architectures (labels used for EVALUATION only,
        # on the same 1,500 archs as the supervised fit; they never touch the gate's weights).
        try:
            _done_rows = storage.load_results_csv("phase2", "phase2_proxy_calibration.csv")
            _have = ((_done_rows["dataset"] == dataset)
                     & (_done_rows["proxy_name"] == "ensemble_unsupervised")).any()
        except FileNotFoundError:
            _have = False
        if not _have:
            import random as _r
            from scipy.stats import kendalltau, spearmanr
            _idx = _r.Random(0).sample(range(bench.total_archs), CAL_ARCHS)
            _sc = [zc.score(bench.index_to_tokens(i)) for i in _idx]
            _gt = [bench.query(i, dataset) for i in _idx]
            _t, _ = kendalltau(_sc, _gt)
            _rho, _ = spearmanr(_sc, _gt)
            storage.append_result_row("phase2", "phase2_proxy_calibration.csv", {
                "benchmark": bench.name, "dataset": dataset, "seed": 0,
                "proxy_name": "ensemble_unsupervised", "kendall_tau": _t, "spearman_rho": _rho})
            logger.info(f"[{dataset}] label-free gate: Kendall tau={_t:.3f} (evaluation only)")

    for seed in SEEDS:
        for budget in BUDGETS:
            run_num += 1

            if already_done(seed, budget, dataset):
                logger.log("SKIP", f"[{run_num:02d}] SKIP archtree_zc | {dataset} | seed={seed} | budget={budget}")
                skipped += 1
                continue

            logger.log("RUN", f"[{run_num:02d}] archtree_zc | {dataset} | seed={seed} | budget={budget}")
            try:
                t0 = time.time()
                result = search_phase2(bench, budget, dataset, seed, zc,
                                       phase="phase2", backup_mode=BACKUP_MODE,
                                       proxy_threshold_percentile=PROXY_THRESHOLD_PCT)
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
    cal = storage.load_results_csv("phase2", "phase2_proxy_calibration.csv")
    ensemble_tau = cal[cal["proxy_name"] == "ensemble"].groupby("dataset")["kendall_tau"].mean()
    logger.info("Kendall tau (calibration, ensemble rows only) by dataset:")
    logger.info("\n" + ensemble_tau.round(3).to_string())

    red = storage.load_results_csv("phase2", "phase2_eval_reduction.csv")
    mean_reduction = red["reduction_pct"].mean()
    logger.info(f"Mean real-eval reduction vs Phase 1: {mean_reduction:.1f}%")

    # Phase 1 baselines are compared on cifar10 only (avoids blending datasets)
    p1 = storage.load_results_csv("phase1", "phase1_main_results.csv")
    p2 = storage.load_results_csv("phase2", "phase2_main_results.csv")
    p1_acc = p1[p1["dataset"] == "cifar10"]["best_acc"].mean()
    p2_acc = p2[p2["dataset"] == "cifar10"]["best_acc"].mean()
    logger.info(f"Mean best_acc (cifar10 only) — phase1={p1_acc:.2f}%  "
                f"phase2={p2_acc:.2f}%  delta={p2_acc - p1_acc:+.2f}pp")

    # Sanity checks (warnings only, never blocking)
    for ds, tau in ensemble_tau.items():
        if tau <= 0.3:
            logger.warning(f"[{ds}] ensemble kendall_tau={tau:.3f} <= 0.3 — still weak. "
                           f"Check per-component tau in phase2_proxy_calibration.csv.")
    if mean_reduction < 50:
        logger.warning(f"eval reduction < 50% — lower zc_proxy.proxy_threshold_percentile "
                       f"in config.yaml (currently {PROXY_THRESHOLD_PCT}).")

    logger.success("phase2 DONE — ready for Phase 3.")
except FileNotFoundError:
    logger.error("no results written — check errors above")
    sys.exit(1)