# experiments/run_phase0.py
# isort: skip_file
"""
PHASE 0 — baseline sweep (single process).
Loads NB201 once, runs every (dataset, seed, budget, method) combo in a loop.
Resume-safe: combos with an existing log file are skipped.

Usage (from archtree-llm/ root):
    python experiments/run_phase0.py
"""

import sys
import os
import gc
import json
import time
import yaml
import torch
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Make project packages importable before any project import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import archtree.storage as storage  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "outputs" / "phase0" / "logs"
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
logger.add(LOG_DIR / "phase0_run.log", level="DEBUG", rotation="10 MB", retention=5)


def log_free_ram(label):
    """Log available RAM if psutil is installed; silently skip otherwise."""
    try:
        import psutil
        logger.info(f"RAM free {label}: {psutil.virtual_memory().available/1e9:.1f} GB")
    except ImportError:
        pass


# ── Preflight ─────────────────────────────────────────────────────────────────
with open(PROJECT_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

cache_path = Path(cfg["paths"]["nb201_cache"])
if not cache_path.exists():
    logger.error(f"NB201 accuracy cache missing at: {cache_path.resolve()}")
    logger.error("Build it once: python tools/build_nb201_cache.py (needs data/NAS-Bench-201-v1_1-096897.pth)")
    sys.exit(1)

logger.info(f"NB201 cache: {cache_path}")
logger.info("Loading benchmark — takes 1-3 min on first load, please wait...")
log_free_ram("before load")

storage.verify_phase_inputs("phase0")
torch.set_num_threads(2)

# ── Imports that need the path fix / heavy deps ───────────────────────────────
from benchmarks.nb201 import NB201                 # noqa: E402
from baselines.random_search import RandomSearch   # noqa: E402
from baselines.enas_proxy import ENASProxy         # noqa: E402
from baselines.mcts_nas import MCTSNASProxy        # noqa: E402
from baselines.zico import ZiCoBaseline            # noqa: E402
from baselines.rz_nas import RZNASBaseline         # noqa: E402
from benchmarks.nb201 import best_from_history     # noqa: E402
from baselines.regularized_evolution import RegularizedEvolutionSearch, ZCWarmupRESearch  # noqa: E402
from baselines.local_search import LocalSearchSearch  # noqa: E402

# ── Methods ───────────────────────────────────────────────────────────────────
METHODS = {
    "random_search": RandomSearch,
    "enas":          ENASProxy,
    "mcts_nas":      MCTSNASProxy,
    "zico":          ZiCoBaseline,
    "rz_nas":        RZNASBaseline,
    "re":            RegularizedEvolutionSearch,   # regularized evolution (Real et al. 2019), added Oct 3
    "local_search":  LocalSearchSearch,            # local search (White et al. 2021), added Oct 3
    "re_zc_warmup":  ZCWarmupRESearch,             # RE + zero-cost warm-up (Abdelfattah et al. 2021), Oct 3
}

# llm_nas is slow (real LLM calls each iteration), so it is off by default.
# Preferred path: run_phase0_llm_nas_only.py. Set True to run it inline.
INCLUDE_LLM_NAS = False

_llm_nas_kwargs = {}
if INCLUDE_LLM_NAS:
    from baselines.llm_nas import LLMNASBaseline   # noqa: E402
    METHODS["llm_nas"] = LLMNASBaseline

    llm_cfg = cfg["llm"]

    # Backend comes from env var if set, otherwise ask interactively
    _env_backend = os.environ.get("ARCHTREE_LLM_BACKEND", "").strip().lower()
    if _env_backend in ("groq", "ollama"):
        _backend = _env_backend
        logger.info(f"ARCHTREE_LLM_BACKEND={_backend} set in environment — skipping prompt.")
    else:
        print("\nPick LLM backend for llm_nas baseline:")
        print("  1) Ollama (local, slow, free)")
        print("  2) Groq   (cloud, fast, rate-limited)")
        _backend = "groq" if input("Enter 1 or 2: ").strip() == "2" else "ollama"

    if _backend == "groq":
        if not os.environ.get("GROQ_API_KEY"):
            logger.error("groq selected but GROQ_API_KEY not set in environment.")
            sys.exit(1)
        _llm_nas_kwargs = dict(
            backend="groq",
            groq_model=llm_cfg["groq"]["model"],
            groq_daily_token_budget=llm_cfg["groq"].get("daily_token_budget", 190_000),
            groq_daily_request_budget=llm_cfg["groq"].get("daily_request_budget", 950),
            groq_reasoning_effort=llm_cfg["groq"].get("reasoning_effort", "low"),
        )
        logger.info(f"llm_nas using Groq backend, model='{llm_cfg['groq']['model']}'.")
    else:
        _host = llm_cfg.get("host", "http://localhost:11434")
        _llm_nas_kwargs = dict(backend="ollama", model=llm_cfg["model"], host=_host)
        logger.info(f"llm_nas using Ollama model '{llm_cfg['model']}' at {_host} — "
                    f"ensure `ollama serve` is running and the model is pulled.")

# ── Load benchmark once ───────────────────────────────────────────────────────
bench = NB201(cfg)
bench._load()
logger.info("Benchmark loaded OK")
log_free_ram("after load")

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS    = cfg["search"]["seeds"]
BUDGETS  = cfg["search"]["budgets"]
DATASETS = cfg["benchmarks"]["datasets"]   # e.g. cifar10, cifar100, ImageNet16-120
logger.info(f"Dataset sweep: {DATASETS}")


def log_path(method_name, dataset, seed, budget):
    return LOG_DIR / f"{method_name}__nb201__{dataset}__seed{seed}__budget{budget}.json"


# ── Main loop ─────────────────────────────────────────────────────────────────
run_num = skipped = errored = 0

for dataset in DATASETS:
    for seed in SEEDS:
        for budget in BUDGETS:
            for method_name, MethodCls in METHODS.items():
                run_num += 1
                out_file = log_path(method_name, dataset, seed, budget)

                # Resume support: a log file means this combo already finished
                if out_file.exists():
                    logger.log("SKIP", f"[{run_num:02d}] SKIP {method_name} | {dataset} | seed={seed} | budget={budget}")
                    skipped += 1
                    continue

                logger.log("RUN", f"[{run_num:02d}] {method_name} | {dataset} | seed={seed} | budget={budget}")

                method = result = None   # defined up-front so cleanup below is always safe
                try:
                    t0 = time.time()
                    extra = _llm_nas_kwargs if method_name == "llm_nas" else {}
                    method = MethodCls(bench, dataset, seed, **extra)
                    result = method.search(budget)
                    elapsed = time.time() - t0

                    logger.success(f"    best_acc={result['best_acc']:.2f}%  time={elapsed:.1f}s")

                    rep = best_from_history(bench, result["eval_history"], dataset)
                    row = {
                        "method": method_name, "benchmark": "nb201",
                        "dataset": dataset, "seed": seed, "budget": budget,
                        "best_acc": result["best_acc"],
                        "best_test_acc": rep["best_test_acc"],
                        "eval_count": result["eval_count"],
                        "proxy_evals": result.get("proxy_evals", 0),
                        "wall_time_sec": elapsed,
                    }

                    # Per-run JSON (full eval history) + one summary CSV row
                    with open(out_file, "w") as f:
                        json.dump({**row, "eval_history": result["eval_history"]}, f, indent=2)
                    storage.append_result_row("phase0", "baseline_summary.csv", row)

                except KeyboardInterrupt:
                    logger.warning("Interrupted — progress saved, rerun to resume.")
                    sys.exit(0)

                except Exception:
                    logger.exception(f"    ERROR in run {run_num}:")
                    errored += 1

                finally:
                    del method, result
                    gc.collect()

# ── Summary ───────────────────────────────────────────────────────────────────
logger.info("=" * 50)
logger.info(f"Total: {run_num} | Skipped: {skipped} | Errors: {errored}")

try:
    df = storage.load_results_csv("phase0", "baseline_summary.csv")
    logger.info(f"{len(df)} rows in baseline_summary.csv")
    # Group by (method, dataset): averaging across datasets would blend
    # incomparable accuracies into a meaningless number.
    logger.info("\n" + df.groupby(["method", "dataset"])["best_acc"].mean().round(2).to_string())
    logger.success("phase0 DONE — ready for Phase 1.")
except FileNotFoundError:
    logger.error("no results written — check errors above")
    sys.exit(1)