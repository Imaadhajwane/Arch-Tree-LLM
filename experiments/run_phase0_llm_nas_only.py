# experiments/run_phase0_llm_nas_only.py
# isort: skip_file
"""
Phase 0 — llm_nas only.

Split out from run_phase0.py because llm_nas is slow (real LLM calls per
iteration) while the other baselines run in under a second.

Sweeps every dataset in config.yaml x seeds x budgets. Cold-resume safe:
a (dataset, seed, budget) combo is skipped if its per-run JSON log exists.
The log is written last, so a run interrupted mid-write is retried.

LLM settings come from cfg["llm"] and cfg["llm"]["nas"] (batch_size,
top_k_context, use_key_pool, ollama_fallback_model, ollama_fallback_host).
With use_key_pool=true, baselines/llm_nas.py reads the Groq keys itself, so
this script only checks that a key source exists.

Rough cost: 12 seeds x 4 budgets x 3 datasets = 144 runs, ~1,700 Groq calls
at batch_size=8. If the daily key budget runs out, the Ollama fallback tier
(if configured) takes over.

Usage (from archtree-llm/ root):
    export GROQ_API_KEYS="key1,key2,..."     # or GROQ_API_KEYS_FILE / GROQ_API_KEY
    python experiments/run_phase0_llm_nas_only.py
"""
import sys
import os
import gc
import json
import time
import yaml
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
load_dotenv()  # load .env from cwd / parent dirs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import archtree.storage as storage  # noqa: E402

with open(PROJECT_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "outputs" / "phase0" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.level("RUN", no=22, color="<cyan>")
logger.level("SKIP", no=21, color="<yellow>")
logger.add(
    sys.stderr, level="INFO", colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>",
)
logger.add(LOG_DIR / "phase0_llm_nas_run.log", level="DEBUG", rotation="10 MB", retention=5)

# ── LLM backend selection ────────────────────────────────────────────────────
llm_cfg = cfg["llm"]
nas_cfg = llm_cfg.get("nas", {})   # batch size, key pool, Ollama fallback settings

# ARCHTREE_LLM_BACKEND overrides the prompt (useful for CI / non-interactive runs).
_env_backend = os.environ.get("ARCHTREE_LLM_BACKEND", "").strip().lower()
if _env_backend in ("groq", "ollama"):
    _backend = _env_backend
    logger.info(f"ARCHTREE_LLM_BACKEND={_backend} set — skipping prompt.")
else:
    print("\nPick LLM backend for llm_nas baseline:")
    print("  1) Ollama (local, slow, free)")
    print("  2) Groq   (cloud, fast, rate-limited)")
    _backend = "groq" if input("Enter 1 or 2: ").strip() == "2" else "ollama"

if _backend == "groq":
    _use_key_pool = nas_cfg.get("use_key_pool", True)
    _pool_file = os.environ.get("GROQ_API_KEYS_FILE", "").strip()
    _pool_csv = os.environ.get("GROQ_API_KEYS", "").strip()
    _has_pool = bool(_pool_file or _pool_csv)
    _has_single = bool(os.environ.get("GROQ_API_KEY", "").strip())

    # With the pool enabled, any key source works; otherwise a single key is required.
    if _use_key_pool and not (_has_pool or _has_single):
        logger.error("groq selected but none of GROQ_API_KEYS_FILE / GROQ_API_KEYS / "
                     "GROQ_API_KEY are set.")
        sys.exit(1)
    if not _use_key_pool and not _has_single:
        logger.error("groq selected with use_key_pool=false but GROQ_API_KEY not set.")
        sys.exit(1)
    if _pool_file and not Path(_pool_file).exists():
        logger.warning(f"GROQ_API_KEYS_FILE points to a missing file: {_pool_file}")

    _llm_nas_kwargs = dict(
        backend="groq",
        groq_model=llm_cfg["groq"]["model"],
        groq_daily_token_budget=llm_cfg["groq"].get("daily_token_budget", 190_000),
        groq_daily_request_budget=llm_cfg["groq"].get("daily_request_budget", 950),
        groq_reasoning_effort=llm_cfg["groq"].get("reasoning_effort", "low"),
        batch_size=nas_cfg.get("batch_size", 5),
        top_k_context=nas_cfg.get("top_k_context", 8),
        use_key_pool=_use_key_pool,
        ollama_fallback_model=nas_cfg.get("ollama_fallback_model"),
        ollama_fallback_host=nas_cfg.get("ollama_fallback_host", "http://localhost:11434"),
    )

    if _use_key_pool and _has_pool:
        if _pool_csv:
            n_keys = len([k for k in _pool_csv.split(",") if k.strip()])
            logger.info(f"llm_nas Groq KEY POOL ({n_keys} accounts, GROQ_API_KEYS), "
                        f"model='{llm_cfg['groq']['model']}', batch_size={_llm_nas_kwargs['batch_size']}.")
        else:
            logger.info(f"llm_nas Groq KEY POOL (GROQ_API_KEYS_FILE), "
                        f"model='{llm_cfg['groq']['model']}', batch_size={_llm_nas_kwargs['batch_size']}.")
    else:
        logger.info(f"llm_nas Groq single key, model='{llm_cfg['groq']['model']}', "
                    f"batch_size={_llm_nas_kwargs['batch_size']}.")
    if _llm_nas_kwargs["ollama_fallback_model"]:
        logger.info(f"Ollama fallback enabled: model='{_llm_nas_kwargs['ollama_fallback_model']}' "
                    f"(used only if every Groq key is exhausted for the day).")
else:
    _llm_nas_kwargs = dict(
        backend="ollama",
        model=llm_cfg["model"],
        host=llm_cfg.get("host", "http://localhost:11434"),
        batch_size=nas_cfg.get("batch_size", 5),
        top_k_context=nas_cfg.get("top_k_context", 8),
    )
    logger.info(f"llm_nas Ollama model '{llm_cfg['model']}' at "
                f"{llm_cfg.get('host', 'http://localhost:11434')} "
                f"(ensure `ollama serve` is running and the model is pulled).")

from benchmarks.nb201 import NB201            # noqa: E402
from baselines.llm_nas import LLMNASBaseline  # noqa: E402
from benchmarks.nb201 import best_from_history  # noqa: E402

bench = NB201(cfg)
bench._load()
logger.info("Benchmark loaded OK")

SEEDS = cfg["search"]["seeds"]
BUDGETS = cfg["search"]["budgets"]
DATASETS = cfg["benchmarks"]["datasets"]
logger.info(f"Dataset sweep: {DATASETS}")


def _log_path(dataset, seed, budget) -> Path:
    return LOG_DIR / f"llm_nas__nb201__{dataset}__seed{seed}__budget{budget}.json"


def already_done(dataset, seed, budget) -> bool:
    """A run counts as done once its JSON log exists (the log is written last)."""
    return _log_path(dataset, seed, budget).exists()


# ── Main sweep ───────────────────────────────────────────────────────────────
run_num = completed = skipped = errored = 0

for dataset in DATASETS:
    for seed in SEEDS:
        for budget in BUDGETS:
            run_num += 1
            if already_done(dataset, seed, budget):
                logger.log("SKIP", f"[{run_num:02d}] SKIP llm_nas | {dataset} | seed={seed} | budget={budget}")
                skipped += 1
                continue

            logger.log("RUN", f"[{run_num:02d}] llm_nas | {dataset} | seed={seed} | budget={budget}")
            method = result = None   # reset so `finally` never touches a previous iteration's objects
            try:
                t0 = time.time()
                method = LLMNASBaseline(bench, dataset, seed, **_llm_nas_kwargs)
                result = method.search(budget)
                elapsed = time.time() - t0

                best_acc = result.get("best_acc")
                acc_str = f"{best_acc:.2f}%" if best_acc is not None else "n/a"
                logger.success(f"    best_acc={acc_str}  time={elapsed:.1f}s")
                logger.info(f"    llm_calls={result.get('llm_calls', 0)}  "
                            f"parse_failures={result.get('parse_failures', 0)}  "
                            f"fallback_archs={result.get('fallback_archs', 0)}  "
                            f"ollama_fallback_archs={result.get('ollama_fallback_archs', 0)}")

                # CSV rows first, JSON log last: the log doubles as the "done" marker,
                # so a crash before it is written just means the run is retried.
                rep = best_from_history(bench, result.get("eval_history") or [], dataset)
                # SAME keys and order as run_phase0.py (shared CSV header)
                storage.append_result_row("phase0", "baseline_summary.csv", {
                    "method": "llm_nas", "benchmark": "nb201",
                    "dataset": dataset, "seed": seed, "budget": budget,
                    "best_acc": best_acc,
                    "best_test_acc": rep["best_test_acc"],
                    "eval_count": result.get("eval_count"),
                    "proxy_evals": 0,
                    "wall_time_sec": elapsed,
                })
                # Separate diagnostics CSV: keeps baseline_summary.csv's schema identical
                # to the other methods (they have no llm_calls field).
                storage.append_result_row("phase0", "llm_nas_call_diagnostics.csv", {
                    "dataset": dataset, "seed": seed, "budget": budget,
                    "llm_calls": result.get("llm_calls", 0),
                    "parse_failures": result.get("parse_failures", 0),
                    "fallback_archs": result.get("fallback_archs", 0),
                    "ollama_fallback_archs": result.get("ollama_fallback_archs", 0),
                    "llm_tokens": result.get("llm_tokens", 0),
                    "truncated_replies": result.get("truncated_replies", 0),
                })
                with open(_log_path(dataset, seed, budget), "w") as f:
                    json.dump({
                        "method": "llm_nas", "benchmark": "nb201",
                        "dataset": dataset, "seed": seed, "budget": budget,
                        "eval_history": result.get("eval_history"),
                        "best_acc": best_acc,
                        "eval_count": result.get("eval_count"),
                        "wall_time_sec": elapsed,
                        "llm_calls": result.get("llm_calls", 0),
                        "parse_failures": result.get("parse_failures", 0),
                        "fallback_archs": result.get("fallback_archs", 0),
                        "ollama_fallback_archs": result.get("ollama_fallback_archs", 0),
                    }, f, indent=2)
                completed += 1

            except KeyboardInterrupt:
                logger.warning("Interrupted — rerun to resume (a run is only marked done "
                               "once its full budget finishes).")
                logger.info(f"Completed this session: {completed} | Skipped (already done): {skipped}")
                sys.exit(0)

            except Exception:
                logger.exception(f"    ERROR in run {run_num}:")
                errored += 1

            finally:
                del method, result
                gc.collect()

# ── Summary ──────────────────────────────────────────────────────────────────
logger.info("=" * 50)
logger.info(f"Total: {run_num} | Ran: {completed} | Skipped: {skipped} | Errors: {errored}")
try:
    diag = storage.load_results_csv("phase0", "llm_nas_call_diagnostics.csv")
    logger.info(f"Total llm_calls across all runs: {diag['llm_calls'].sum()}  "
                f"parse_failures: {diag['parse_failures'].sum()}  "
                f"fallback_archs: {diag['fallback_archs'].sum()}  "
                f"ollama_fallback_archs: {diag['ollama_fallback_archs'].sum()}")
    if "dataset" in diag.columns:
        logger.info("\n" + diag.groupby("dataset")["llm_calls"].sum().to_string())
except FileNotFoundError:
    pass
logger.success("phase0-llm_nas DONE — results already appended to "
               "outputs/phase0/results/baseline_summary.csv (no extra step needed).")