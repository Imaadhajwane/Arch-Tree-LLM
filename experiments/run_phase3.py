"""
experiments/run_phase3.py — PHASE 3 (LLM warm-start UCB)

Runs the main sweep with LLM priors, then the alpha-sensitivity ablation (A4).
Resume-safe: finished runs are skipped, and the LLM cache is persisted per
call (archtree/llm_prior.py), so after a crash just restart.

Inputs:
    outputs/phase2/results/phase2_main_results.csv
    outputs/phase2/weights/zc_calibration_weights__{bench}__{dataset}.pt

Outputs:
    outputs/phase3/results/phase3_main_results.csv
    outputs/phase3/results/phase3_tcr_per_iter.csv
    outputs/phase3/results/phase3_llm_cache_stats.csv
    outputs/ablations/results/ablation_A4_alpha_sensitivity.csv
    outputs/ablations/results/phase3_A4_raw_*.csv   (per-run raw rows for A4)

Before running, set GROQ_API_KEY (single account), or GROQ_API_KEYS /
GROQ_API_KEYS_FILE (multi-account pool), or run Ollama locally.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import csv
import yaml
from loguru import logger
from dotenv import load_dotenv
load_dotenv()  # load .env before project imports / env-var checks

from archtree import storage
from archtree.search import search_phase3
from benchmarks.nb201 import NB201

PROJECT_ROOT = Path(__file__).resolve().parent.parent
with open(PROJECT_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "outputs" / "phase3" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.level("RUN", no=22, color="<cyan>")
logger.level("SKIP", no=21, color="<yellow>")
logger.add(
    sys.stderr, level="INFO", colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>",
)
logger.add(LOG_DIR / "phase3_run.log", level="DEBUG", rotation="10 MB", retention=5)


# ── LLM backend selection ─────────────────────────────────────────────────────
def pick_llm_backend(cfg) -> dict:
    """Ask for Ollama vs Groq and return the backend settings.
    Groq can use a single key (GROQ_API_KEY) or a multi-account key pool
    (GROQ_API_KEYS / GROQ_API_KEYS_FILE)."""
    print("\nPick LLM backend:")
    print("  1) Ollama (local, slow, free)")
    print("  2) Groq   (cloud, fast, rate-limited)")
    _env = os.environ.get("ARCHTREE_LLM_BACKEND", "").strip().lower()
    if _env in ("groq", "ollama"):
        choice = "2" if _env == "groq" else "1"
        print(f"  ARCHTREE_LLM_BACKEND={_env} set - skipping prompt.")
    else:
        choice = input("Enter 1 or 2: ").strip()
    llm_cfg = cfg["llm"]

    if choice != "2":
        return {"backend": "ollama", "model": llm_cfg["model"],
                "host": llm_cfg.get("host", "http://localhost:11434")}

    has_pool_file = bool(os.environ.get("GROQ_API_KEYS_FILE", "").strip())
    has_pool_csv  = bool(os.environ.get("GROQ_API_KEYS", "").strip())
    has_single    = bool(os.environ.get("GROQ_API_KEY", "").strip())
    use_key_pool  = has_pool_file or has_pool_csv

    if use_key_pool:
        source = "GROQ_API_KEYS_FILE" if has_pool_file else "GROQ_API_KEYS"
        print(f"  -> {source} detected — using multi-account key pool.")
    elif has_single:
        print("  -> single GROQ_API_KEY detected — using single-key mode.")
    else:
        print("  -> WARNING: no Groq key env var set yet (checked again before running).")

    groq_cfg = llm_cfg["groq"]
    return {
        "backend": "groq",
        "model": groq_cfg["model"],
        "groq_daily_token_budget": groq_cfg.get("daily_token_budget", 190_000),
        "groq_daily_request_budget": groq_cfg.get("daily_request_budget", 950),
        "groq_reasoning_effort": groq_cfg.get("reasoning_effort", "low"),
        "use_key_pool": use_key_pool,
    }


backend_cfg = pick_llm_backend(cfg)
logger.info(f"Backend selected: {backend_cfg['backend']} (model={backend_cfg['model']})"
            + (f" [key pool: {backend_cfg['use_key_pool']}]" if backend_cfg["backend"] == "groq" else ""))

# Groq needs at least one key source before we start
if backend_cfg["backend"] == "groq":
    if not any(os.environ.get(k, "").strip()
               for k in ("GROQ_API_KEY", "GROQ_API_KEYS", "GROQ_API_KEYS_FILE")):
        logger.error("backend=groq selected but none of GROQ_API_KEY / GROQ_API_KEYS / "
                     "GROQ_API_KEYS_FILE are set in environment.")
        sys.exit(1)

# ── Preflight & upstream data ─────────────────────────────────────────────────
storage.verify_phase_inputs("phase3")
phase2_results = storage.load_results_csv("phase2", "phase2_main_results.csv")  # unfiltered; filtered at summary time

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS   = cfg["search"]["seeds"]
BUDGETS = cfg["search"]["budgets"]
llm_cfg = cfg["llm"]
LLM_HOST = llm_cfg.get("host", "http://localhost:11434")

# Only NB201 is wired for Phase 3 (encoder/index logic is NB201-specific, and
# Phase 2 proxy calibration was never run for NB101).
bench = NB201(cfg)
BENCHES = [bench]

PROXY_THRESHOLD_PCT = cfg.get("zc_proxy", {}).get("proxy_threshold_percentile", 35)
logger.info(f"proxy_threshold_percentile={PROXY_THRESHOLD_PCT} (from config.yaml, same key as Phase 2)")

# Backend kwargs forwarded to every search_phase3() call
if backend_cfg["backend"] == "groq":
    _llm_backend_kwargs = dict(
        backend="groq",
        groq_model=backend_cfg["model"],
        groq_daily_token_budget=backend_cfg["groq_daily_token_budget"],
        groq_daily_request_budget=backend_cfg["groq_daily_request_budget"],
        groq_reasoning_effort=backend_cfg["groq_reasoning_effort"],
        use_key_pool=backend_cfg["use_key_pool"],
    )
else:
    _llm_backend_kwargs = dict(backend="ollama")


# ── Main Phase 3 sweep ────────────────────────────────────────────────────────
def _already_run(bench_name, dataset, seed, budget):
    p = LOG_DIR / f"archtree_llm__{bench_name}__{dataset}__seed{seed}__budget{budget}.json"
    return p.exists()


for bench in BENCHES:
    for dataset in bench.datasets:
        for seed in SEEDS:
            for budget in BUDGETS:
                if _already_run(bench.name, dataset, seed, budget):
                    logger.log("SKIP", f"SKIP {bench.name}/{dataset}/seed={seed}/budget={budget}")
                    continue

                logger.log("RUN", f"{bench.name} / {dataset} / seed={seed} / budget={budget}")
                try:
                    result = search_phase3(
                        bench, budget, dataset, seed, phase="phase3",
                        alpha=llm_cfg["alpha_init"], use_llm=True,
                        model=llm_cfg["model"], host=LLM_HOST,
                        max_llm_calls=llm_cfg["max_calls"],
                        proxy_threshold_percentile=PROXY_THRESHOLD_PCT,
                        **_llm_backend_kwargs)
                except KeyboardInterrupt:
                    logger.warning("Interrupted — rerun to resume (LLM cache is persisted).")
                    sys.exit(0)
                except Exception:
                    # One failed run shouldn't kill the whole sweep
                    logger.exception(f"    ERROR in {bench.name}/{dataset}/seed={seed}/budget={budget}:")
                    continue

                logger.success(f"    best_acc={result['best_acc']:.2f}%  "
                               f"real_evals={result['real_eval_count']}/{budget}  "
                               f"llm_calls={result['llm_calls']}  cache_hits={result['cache_hits']}  "
                               f"time={result['wall_time_sec']:.1f}s")


# ── Alpha-sensitivity ablation (A4) ───────────────────────────────────────────
A4_CSV = PROJECT_ROOT / "outputs" / "ablations" / "results" / "ablation_A4_alpha_sensitivity.csv"


def _load_done_ablation_combos() -> set:
    """(alpha, seed) pairs already in the A4 CSV. Rows are appended only after
    a combo fully finishes, so anything here is safe to skip on resume."""
    done = set()
    if A4_CSV.exists():
        with open(A4_CSV, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["alpha"], row["seed"]))
    return done


_done_ablation_combos = _load_done_ablation_combos()
if _done_ablation_combos:
    logger.info(f"A4 resume: {len(_done_ablation_combos)} (alpha, seed) combo(s) already done — will be skipped.")

# ARCHTREE_SKIP_A4=1 (set by run_full_pipeline.ps1 -Plan core / -NoGroq): run only the LLM-free main
# sweep now; the A4 grid needs Groq and is filled later by rerunning this script (main sweep is skipped
# as already done).
_SKIP_A4 = os.environ.get("ARCHTREE_SKIP_A4", "").strip() == "1"
if _SKIP_A4:
    logger.warning("ARCHTREE_SKIP_A4=1 - A4 alpha grid skipped (needs Groq). Rerun step 8 with Groq later.")

for alpha in ([] if _SKIP_A4 else [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]):
    for seed in SEEDS:
        # CSV values are strings, so compare as strings
        if (str(alpha), str(seed)) in _done_ablation_combos:
            logger.log("SKIP", f"SKIP A4 alpha={alpha} seed={seed} (already done)")
            continue

        try:
            result = search_phase3(
                bench, 100, "cifar10", seed, phase="ablations",
                alpha=alpha, use_llm=(alpha > 0.0),   # alpha=0 -> no-LLM control
                model=llm_cfg["model"], host=LLM_HOST,
                max_llm_calls=llm_cfg["max_calls"],
                ablation_id=f"A4_a{alpha}",
                proxy_threshold_percentile=PROXY_THRESHOLD_PCT,
                # Separate raw-output filenames so A4 rows don't mix into
                # phase3_main_results.csv
                results_filename="phase3_A4_raw_runs.csv",
                tcr_filename="phase3_A4_raw_tcr_per_iter.csv",
                cache_stats_filename="phase3_A4_raw_llm_cache_stats.csv",
                **_llm_backend_kwargs)
        except KeyboardInterrupt:
            logger.warning("Interrupted — rerun to resume.")
            sys.exit(0)
        except Exception:
            logger.exception(f"    ERROR in A4 alpha={alpha} seed={seed}:")
            continue

        if alpha > 0 and result.get("llm_fallbacks", 0) > 0:
            # Some LLM calls returned no model scores (keys exhausted / errors): this is not a
            # valid LLM-on run. Do not record it; rerun A4 when the Groq budget has reset.
            logger.warning(f"A4 alpha={alpha} seed={seed}: {result['llm_fallbacks']} LLM fallbacks "
                           f"- row NOT written (rerun later).")
            continue
        storage.append_result_row("ablations", "ablation_A4_alpha_sensitivity.csv", {
            "alpha": alpha, "benchmark": "nb201", "dataset": "cifar10",
            "seed": seed, "budget": 100, "best_acc": result["best_acc"],
            "best_test_acc": result["best_test_acc"],
            "eta": result["eta"], "llm_calls": result["llm_calls"],
            "cache_hits": result["cache_hits"], "llm_fallbacks": result.get("llm_fallbacks", 0),
        })

# ── Plots ─────────────────────────────────────────────────────────────────────
try:
    from plots.plot_results import plot_phase3
    plot_phase3("phase3")
except ImportError:
    logger.warning("plots/plot_results.py::plot_phase3 not found — skipping plots.")

# ── Exit criteria (warnings only, never blocking) ─────────────────────────────
cache_stats = storage.load_results_csv("phase3", "phase3_llm_cache_stats.csv")
hit_rate = cache_stats["cache_hit_rate"].mean()
if hit_rate <= 0.80:
    logger.warning(f"cache hit rate {hit_rate:.2%} <= 80% target.")

p3_results = storage.load_results_csv("phase3", "phase3_main_results.csv")

# Compare on cifar10 only — datasets have very different accuracy ranges,
# so a blended mean is meaningless.
phase2_c10 = phase2_results[phase2_results["dataset"] == "cifar10"]
p3_c10     = p3_results[p3_results["dataset"] == "cifar10"]

p2_acc = phase2_c10["best_acc"].mean()
p3_acc = p3_c10["best_acc"].mean()
gain = (p3_acc - p2_acc) / p2_acc   # relative gain
logger.info(f"Phase 2 mean best_acc (cifar10 only): {p2_acc:.2f}%")
logger.info(f"Phase 3 mean best_acc (cifar10 only): {p3_acc:.2f}%  (delta: {gain*100:+.2f}%)")
if gain < 0.15 and p3_acc < p2_acc:
    logger.warning("ETA not improved >=15% vs Phase 2 (cifar10 only, informational, not blocking).")

# Other datasets: reported separately, no pass/fail threshold defined
for ds in sorted(p3_results["dataset"].unique()):
    if ds == "cifar10":
        continue
    ds_p2 = phase2_results[phase2_results["dataset"] == ds]["best_acc"].mean()
    ds_p3 = p3_results[p3_results["dataset"] == ds]["best_acc"].mean()
    logger.info(f"  (info) {ds}: phase2={ds_p2:.2f}%  phase3={ds_p3:.2f}%")

logger.success("Phase 3 DONE — outputs/phase3/ written. Ready for Phase 4.")