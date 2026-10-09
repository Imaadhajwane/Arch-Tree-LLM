# experiments/run_ablations.py
# isort: skip_file
"""
Ablations master (Phase 6.1) — run after Phase 4 completes.

Runs A1, A2, A3, A5, A6 over every dataset in config.yaml. A4 (alpha
sensitivity) is written by Phase 3 and skipped here.

  A1  no LLM              (alpha=0.0, use_llm=False)
  A2  no ZC proxy         (use_proxy=False)
  A3  no weight inherit   (use_weight_inherit=False)
  A5  prune threshold     (prune_percentile sweep, fixed budget=100)
  A6  budget sensitivity  (full method, budget sweep)

Cold-resume safe: each ablation appends to its own CSV and skips
(dataset, seed, param_value) triples already present.

Oct 2026 fix: every arm now runs the HEADLINE configuration (config.yaml alpha_init,
proxy_threshold_percentile) except for the one component it removes. Before, COMMON_KW passed
neither, so A1-A6 silently used a 50% gate (headline: 35%) and A2/A3/A5/A6 ran with the LLM prior
ON (search_phase3's default alpha=0.3). Rows now also carry best_test_acc and llm_fallbacks.
With alpha_init=0.0, A1 is the headline system itself (a reproducibility cross-check vs Phase 3).

Inputs:
  outputs/phase2/weights/zc_calibration_weights__nb201__{dataset}.pt

Outputs:
  outputs/ablations/results/ablation_*.csv
  outputs/ablations/plots/{figures,data}/

Usage (from archtree-llm/ root):
    python experiments/run_ablations.py
"""
import sys
import os
import gc
import time
import yaml
import torch
from pathlib import Path
from loguru import logger

from dotenv import load_dotenv
load_dotenv()  # load .env from cwd / parent dirs

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _cal_path(ds):
    """Phase-2 calibration file the gate will load (respects zc_proxy.calibration_mode)."""
    from archtree import storage as _st, settings as _se
    return _st._zc_cal_path("phase2", "nb201", ds, _se.CALIBRATION_MODE)
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import archtree.storage as storage  # noqa: E402

with open(PROJECT_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "outputs" / "ablations" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(
    sys.stderr, level="INFO", colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>",
)
logger.add(LOG_DIR / "ablations_run.log", level="DEBUG", rotation="10 MB", retention=5)

storage.verify_phase_inputs("ablations")

DATASETS = cfg["benchmarks"]["datasets"]  # e.g. cifar10, cifar100, ImageNet16-120

# ── Fail fast: every dataset needs its Phase 2 calibration file ──────────────
# A1/A3/A5/A6 run with use_proxy=True, so a missing file would otherwise
# crash mid-sweep.
_missing_cal = [
    str(_cal_path(ds))
    for ds in DATASETS
]
_missing_cal = [p for p in _missing_cal if not Path(p).exists()]
if _missing_cal:
    logger.error("Missing Phase 2 calibration file(s):")
    for m in _missing_cal:
        logger.error(f"  x  {m}")
    logger.error("Run Phase 2 for every dataset first.")
    sys.exit(1)

# ── LLM backend selection ────────────────────────────────────────────────────
llm_cfg = cfg["llm"]

# ARCHTREE_LLM_BACKEND overrides the prompt (useful for CI / non-interactive runs).
_env_backend = os.environ.get("ARCHTREE_LLM_BACKEND", "").strip().lower()
if _env_backend in ("groq", "ollama"):
    _backend = _env_backend
    logger.info(f"ARCHTREE_LLM_BACKEND={_backend} set — skipping prompt.")
else:
    print("\nPick LLM backend for ablations (A2/A3/A5/A6 — A1 skips the LLM entirely):")
    print("  1) Ollama (local, slow, free)")
    print("  2) Groq   (cloud, fast, rate-limited)")
    _backend = "groq" if input("Enter 1 or 2: ").strip() == "2" else "ollama"

if _backend == "groq":
    # Key priority: pool file > pool CSV env var > single key.
    _pool_file = os.environ.get("GROQ_API_KEYS_FILE", "").strip()
    _has_pool_file = bool(_pool_file)
    _has_pool_csv = bool(os.environ.get("GROQ_API_KEYS", "").strip())
    _has_single = bool(os.environ.get("GROQ_API_KEY", "").strip())
    _use_key_pool = _has_pool_file or _has_pool_csv

    if not (_use_key_pool or _has_single):
        logger.error("groq selected but none of GROQ_API_KEYS_FILE / GROQ_API_KEYS / "
                     "GROQ_API_KEY are set.")
        sys.exit(1)
    if _has_pool_file and not Path(_pool_file).exists():
        logger.warning(f"GROQ_API_KEYS_FILE points to a missing file: {_pool_file}")

    _llm_backend_kwargs = dict(
        backend="groq",
        groq_model=llm_cfg["groq"]["model"],
        groq_daily_token_budget=llm_cfg["groq"].get("daily_token_budget", 190_000),
        groq_daily_request_budget=llm_cfg["groq"].get("daily_request_budget", 950),
        groq_reasoning_effort=llm_cfg["groq"].get("reasoning_effort", "low"),
        use_key_pool=_use_key_pool,  # rotate accounts to avoid the single-key 25 RPM cap
    )
    if _use_key_pool:
        _source = "GROQ_API_KEYS_FILE" if _has_pool_file else "GROQ_API_KEYS"
        logger.info(f"Groq KEY POOL ({_source}), model='{llm_cfg['groq']['model']}'")
    else:
        logger.info(f"Groq single key, model='{llm_cfg['groq']['model']}' "
                    f"(set GROQ_API_KEYS or GROQ_API_KEYS_FILE to pool accounts)")
else:
    _llm_backend_kwargs = dict(backend="ollama")
    logger.info(f"Ollama model '{llm_cfg['model']}' at "
                f"{llm_cfg.get('host', 'http://localhost:11434')} "
                f"(ensure `ollama serve` is running and the model is pulled)")

torch.set_num_threads(2)

from benchmarks.nb201 import NB201         # noqa: E402
from archtree.search import search_phase3  # noqa: E402

bench = NB201(cfg)
bench._load()
logger.info("Benchmark loaded OK")

SEEDS = cfg["search"]["seeds"]
BUDGETS = cfg["search"]["budgets"]  # used by A1/A2/A3
logger.info(f"Budget grid for A1/A2/A3: {BUDGETS}")
logger.info(f"Dataset sweep for all ablations: {DATASETS}")

# kwargs shared by every search_phase3 call = the headline configuration
ALPHA = float(llm_cfg.get("alpha_init", 0.0))
PCT = int(cfg.get("zc_proxy", {}).get("proxy_threshold_percentile", 35))
COMMON_KW = dict(
    model=llm_cfg["model"], host=llm_cfg.get("host", "http://localhost:11434"),
    max_llm_calls=llm_cfg["max_calls"], phase="ablations",
    proxy_threshold_percentile=PCT,
    **_llm_backend_kwargs,
)
HEADLINE_LLM = dict(alpha=ALPHA, use_llm=ALPHA > 0.0)
logger.info(f"Ablations use the headline config: alpha={ALPHA}, gate top-{PCT}%")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _existing_rows(filename: str) -> set:
    """Return {(dataset, seed, param_value)} already recorded in an ablation CSV.

    The per-ablation CSV is the resume source. search_phase3's run-log JSON
    can't be used because its filename has no ablation_id, so different
    ablations overwrite each other's files.
    """
    path = PROJECT_ROOT / "outputs" / "ablations" / "results" / filename
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(zip(df["dataset"], df["seed"], df["param_value"]))


def _run_ablation(name: str, filename: str, ablation_id: str, varied_param: str,
                  param_grid: list, run_fn) -> None:
    """Run one ablation over dataset x param_grid x seed, appending one CSV row per run.

    run_fn(param_value, seed, dataset) -> result dict from search_phase3.
    Loop order (dataset > param > seed) matches phases 1-4.
    """
    done = _existing_rows(filename)
    n_run = n_skip = n_err = 0

    for dataset in DATASETS:
        for param_value in param_grid:
            for seed in SEEDS:
                if (dataset, seed, param_value) in done:
                    n_skip += 1
                    continue

                logger.info(f"[{name}] {dataset} | {varied_param}={param_value} | seed={seed}")
                try:
                    t0 = time.time()
                    result = run_fn(param_value, seed, dataset)
                    elapsed = time.time() - t0

                    best_acc = result.get("best_acc")
                    storage.append_result_row("ablations", filename, {
                        "best_test_acc": result.get("best_test_acc"),
                        "llm_fallbacks": result.get("llm_fallbacks", 0),
                        "ablation_id": ablation_id, "varied_param": varied_param,
                        "param_value": param_value, "benchmark": "nb201",
                        "dataset": dataset, "seed": seed,
                        "budget": result.get("budget"),
                        "best_acc": best_acc,
                        "real_eval_count": result.get("real_eval_count"),
                        "proxy_eval_count": result.get("proxy_eval_count"),
                        "llm_calls": result.get("llm_calls"),
                        "cache_hits": result.get("cache_hits"),
                        "wall_time_sec": elapsed,
                    })
                    acc_str = f"{best_acc:.2f}%" if best_acc is not None else "n/a"
                    logger.success(f"[{name}] best_acc={acc_str}  time={elapsed:.1f}s")
                    n_run += 1
                except KeyboardInterrupt:
                    # CSV is append-only, so rerunning resumes from here.
                    logger.warning(f"[{name}] Interrupted — progress saved, rerun to resume.")
                    sys.exit(0)
                except Exception:
                    # Failed runs write no row, so they are retried on the next launch.
                    logger.exception(f"[{name}] ERROR:")
                    n_err += 1
                finally:
                    gc.collect()

    logger.info(f"[{name}] done — ran={n_run} skipped={n_skip} errors={n_err}")


# ── A1 — no LLM ──────────────────────────────────────────────────────────────
# use_llm=False (not just alpha=0.0): alpha only weights the LLM scores, but
# score_branches() would still be called every iteration. use_llm=False makes
# the LLM prior return uniform scores with zero calls.
_run_ablation(
    "A1_no_llm", "ablation_A1_no_llm.csv", "A1", "budget", BUDGETS,
    lambda budget, seed, dataset: search_phase3(
        bench, budget, dataset, seed,
        alpha=0.0, use_llm=False, ablation_id="A1", **COMMON_KW),
)

# ── A2 — no ZC proxy ─────────────────────────────────────────────────────────
_run_ablation(
    "A2_no_zc", "ablation_A2_no_zc.csv", "A2", "budget", BUDGETS,
    lambda budget, seed, dataset: search_phase3(
        bench, budget, dataset, seed,
        use_proxy=False, ablation_id="A2", **HEADLINE_LLM, **COMMON_KW),
)

# ── A3 — no weight inheritance ───────────────────────────────────────────────
_run_ablation(
    "A3_no_weight_inherit", "ablation_A3_no_weight_inherit.csv", "A3",
    "budget", BUDGETS,
    lambda budget, seed, dataset: search_phase3(
        bench, budget, dataset, seed,
        use_weight_inherit=False, ablation_id="A3", **HEADLINE_LLM, **COMMON_KW),
)

# ── A5 — prune threshold sensitivity ─────────────────────────────────────────
# Fixed budget=100: the varied axis is the pruning threshold, not budget.
# ablation_id includes pct and dataset so cache namespaces don't collide.
# pct=0 now really means "no pruning" (value rule in trie.prune, Oct 2026).
_run_ablation(
    "A5_prune_threshold", "ablation_A5_prune_threshold.csv", "A5",
    "prune_percentile", [0, 15, 25, 35],
    lambda pv, seed, dataset: search_phase3(
        bench, 100, dataset, seed,
        prune_percentile=pv,
        ablation_id=f"A5_p{pv}_{dataset}", **HEADLINE_LLM, **COMMON_KW),
)

# ── A6 — budget sensitivity (full method) ────────────────────────────────────
# Hardcoded grid; keep in sync with config.yaml search.budgets if that changes.
_run_ablation(
    "A6_budget_sensitivity", "ablation_A6_budget_sensitivity.csv", "A6",
    "budget", [25, 50, 100, 200],
    lambda pv, seed, dataset: search_phase3(
        bench, pv, dataset, seed,
        ablation_id=f"A6_b{pv}_{dataset}", **HEADLINE_LLM, **COMMON_KW),
)

# ── Plots ────────────────────────────────────────────────────────────────────
# Import is guarded separately so an ImportError inside plot_ablations()
# itself is not mistaken for a missing module.
try:
    from plots.plot_results import plot_ablations
except ImportError:
    plot_ablations = None
    logger.warning("plots/plot_results.py::plot_ablations not found — skipping plots.")

if plot_ablations is not None:
    plot_ablations()

logger.success("Ablations DONE — outputs/ablations/ written. Ready for Phase 6 aggregate.")