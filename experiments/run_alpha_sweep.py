# experiments/run_alpha_sweep.py
"""
Alpha-sensitivity sweep (Groq backend, key-pool aware).

Finer alpha grid than ablation A4. Standalone: own phase name
("alpha_sweep"), own output dir, own cache files. Each alpha gets its own
LLM-cache namespace, so cached scores never leak between alpha values.

Cost model:
  - alpha=0.0 uses use_llm=False -> zero LLM calls.
  - every non-zero alpha starts from a fresh cache, so worst case is
    (n_alphas-1) * n_datasets * n_budgets * n_seeds runs, each up to
    SWEEP_MAX_LLM_CALLS real calls.
  - SWEEP_MAX_LLM_CALLS is lower than production (60): this sweep is for
    ranking alpha values, not for reportable results.

Resume: rerunning continues from the existing results CSV.
Pass --fresh to wipe the CSV and start over.

Usage:
    export GROQ_API_KEYS="key1,key2,..."      # or GROQ_API_KEY for one key
    python experiments/run_alpha_sweep.py [--fresh]
"""
import sys
import os
import time
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from loguru import logger
from dotenv import load_dotenv
load_dotenv()

import archtree.storage as storage
from archtree.search import search_phase3
from benchmarks.nb201 import NB201

with open(PROJECT_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "outputs" / "alpha_sweep" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.level("RUN", no=22, color="<cyan>")
logger.level("SKIP", no=21, color="<yellow>")
logger.add(
    sys.stderr, level="INFO", colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>",
)
logger.add(LOG_DIR / "alpha_sweep_run.log", level="DEBUG", rotation="10 MB", retention=5)

# ── Sweep grid (trim/widen based on key-pool size; see cost model above) ─────
ALPHAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
BUDGETS = [50, 100]
DATASETS = cfg["benchmarks"]["datasets"]   # all 3 (Oct 3): the prompt does not mention the dataset, so
                                          # most answers come from the shared cache (cifar10 rows are skipped)
SEEDS = cfg["search"]["seeds"]
SWEEP_MAX_LLM_CALLS = 25
RESULTS_FILE = "alpha_sweep_results.csv"
FRESH = "--fresh" in sys.argv[1:]

# ── Groq backend + key pool (Groq only; no Ollama fallback here) ─────────────
# Key priority: pool file > pool CSV env var > single key.
_pool_file = os.environ.get("GROQ_API_KEYS_FILE", "").strip()
_has_pool_file = bool(_pool_file)
_has_pool_csv = bool(os.environ.get("GROQ_API_KEYS", "").strip())
_has_single = bool(os.environ.get("GROQ_API_KEY", "").strip())
use_key_pool = _has_pool_file or _has_pool_csv

if not (use_key_pool or _has_single):
    logger.error("No Groq key source found — set GROQ_API_KEYS (pool), "
                 "GROQ_API_KEYS_FILE, or GROQ_API_KEY (single) before running.")
    sys.exit(1)
if _has_pool_file and not Path(_pool_file).exists():
    logger.warning(f"GROQ_API_KEYS_FILE points to a missing file: {_pool_file}")

llm_cfg = cfg["llm"]
backend_kwargs = dict(
    backend="groq",
    groq_model=llm_cfg["groq"]["model"],
    groq_daily_token_budget=llm_cfg["groq"].get("daily_token_budget", 190_000),
    groq_daily_request_budget=llm_cfg["groq"].get("daily_request_budget", 950),
    groq_reasoning_effort=llm_cfg["groq"].get("reasoning_effort", "low"),
    use_key_pool=use_key_pool,
)
if use_key_pool:
    source = "GROQ_API_KEYS_FILE" if _has_pool_file else "GROQ_API_KEYS"
    logger.info(f"Groq KEY POOL ({source}), model='{llm_cfg['groq']['model']}'")
else:
    logger.info(f"Groq single key, model='{llm_cfg['groq']['model']}'. "
                f"The full grid needs a multi-key pool (GROQ_API_KEYS) to finish in a day.")

# ── Grid size / worst-case cost ──────────────────────────────────────────────
n_alphas_nonzero = len(ALPHAS) - (1 if 0.0 in ALPHAS else 0)  # alpha=0.0 makes no LLM calls
worst_case_runs = n_alphas_nonzero * len(DATASETS) * len(BUDGETS) * len(SEEDS)
logger.info(f"Grid: {len(ALPHAS)} alphas x {len(DATASETS)} datasets x "
            f"{len(BUDGETS)} budgets x {len(SEEDS)} seeds")
logger.info(f"Worst-case real-LLM runs: {worst_case_runs}, calls: ~"
            f"{worst_case_runs * SWEEP_MAX_LLM_CALLS} (actual is lower due to cache reuse)")

storage.verify_phase_inputs("phase3")  # same upstream deps as a normal phase3 run

bench = NB201(cfg)
bench._load()
logger.info("Benchmark loaded OK")

# ── Resume handling ──────────────────────────────────────────────────────────
if FRESH:
    storage.clear_result_file("alpha_sweep", RESULTS_FILE)
    logger.warning("--fresh given: previous alpha_sweep results cleared.")


def _load_done() -> set:
    """Return {(alpha, dataset, budget, seed)} already in the results CSV."""
    path = PROJECT_ROOT / "outputs" / "alpha_sweep" / "results" / RESULTS_FILE
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(zip(df["alpha"], df["dataset"], df["budget"], df["seed"]))


done = _load_done()  # read once; updated in memory after each successful run
if done:
    logger.info(f"Resuming: {len(done)} run(s) already recorded.")

# ── Sweep ────────────────────────────────────────────────────────────────────
n_run = n_skip = n_err = 0
t_start = time.time()

for alpha in ALPHAS:
    for dataset in DATASETS:
        for budget in BUDGETS:
            for seed in SEEDS:
                key = (alpha, dataset, budget, seed)
                if key in done:
                    n_skip += 1
                    continue

                logger.log("RUN", f"alpha={alpha} | {dataset} | budget={budget} | seed={seed}")
                try:
                    t0 = time.time()
                    result = search_phase3(
                        bench, budget, dataset, seed,
                        phase="alpha_sweep",
                        alpha=alpha,
                        use_llm=(alpha > 0.0),                 # alpha=0 -> no LLM calls at all
                        ablation_id=f"asweep_a{alpha}",        # per-alpha cache namespace
                        proxy_threshold_percentile=cfg.get("zc_proxy", {}).get(
                            "proxy_threshold_percentile", 35),
                        max_llm_calls=SWEEP_MAX_LLM_CALLS,
                        results_filename="alpha_sweep_raw_runs.csv",
                        tcr_filename="alpha_sweep_raw_tcr.csv",
                        cache_stats_filename="alpha_sweep_raw_cache_stats.csv",
                        **backend_kwargs,
                    )
                    elapsed = time.time() - t0

                    best_acc = result.get("best_acc")
                    if alpha > 0 and result.get("llm_fallbacks", 0) > 0:
                        logger.warning(f"  {result['llm_fallbacks']} LLM fallbacks (keys exhausted?) - "
                                       f"row NOT written; rerun after the Groq budget resets.")
                        n_err += 1
                        continue
                    storage.append_result_row("alpha_sweep", RESULTS_FILE, {
                        "alpha": alpha, "dataset": dataset, "budget": budget, "seed": seed,
                        "best_acc": best_acc, "best_test_acc": result.get("best_test_acc"),
                        "llm_fallbacks": result.get("llm_fallbacks", 0),
                        "llm_calls": result.get("llm_calls"),
                        "cache_hits": result.get("cache_hits"),
                        "wall_time_sec": elapsed,
                    })
                    done.add(key)

                    acc_str = f"{best_acc:.2f}%" if best_acc is not None else "n/a"
                    logger.success(f"  best_acc={acc_str}  llm_calls={result.get('llm_calls')}  "
                                   f"cache_hits={result.get('cache_hits')}  time={elapsed:.1f}s")
                    n_run += 1
                except KeyboardInterrupt:
                    # CSV is append-only, so rerunning resumes from here.
                    logger.warning("Interrupted — progress saved, rerun to resume.")
                    sys.exit(0)
                except Exception:
                    # Failed runs write no row, so they are retried on the next launch.
                    logger.exception("  ERROR:")
                    n_err += 1

logger.info("=" * 50)
logger.info(f"Total: ran={n_run} skipped={n_skip} errors={n_err}  "
            f"wall_time={(time.time() - t_start) / 60:.1f} min")
logger.success("alpha_sweep DONE — outputs/alpha_sweep/results/alpha_sweep_results.csv written. "
               "Run experiments/analyze_alpha_sweep.py next.")