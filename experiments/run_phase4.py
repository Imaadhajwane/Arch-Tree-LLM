# experiments/run_phase4.py
# isort: skip_file
"""
PHASE 4 — hardware-aware ArchTree search: drives archtree.search.search_phase4()
over (hardware, dataset, seed, budget) and writes the files that the ablation,
aggregation and cross-device analyses expect.

How search_phase4 behaves:
  * No ZC-proxy gate: every fresh leaf is a real HW-NAS-Bench query
    (accuracy + latency) via bench.hw_query().
  * The LLM prior is constructed but never called (alpha=0.0, max_calls=0).
    A backend is only needed so the object can be built, and it is
    auto-detected from the environment (no interactive prompt).
  * Reward = accuracy - LATENCY_PENALTY * min(latency/LATENCY_NORM_MS, 1) * 100.
    The Pareto archive tracks (accuracy max, latency min). Archs HW-NAS-Bench
    marks unsupported (negative latency) count as real evals but never enter
    the archive.

Files written by search_phase4 (append-only):
  outputs/phase4/results/phase4_pareto_results.csv    every evaluated (arch, acc, latency)
  outputs/phase4/results/phase4_tcr_h_per_iter.csv    TCR-H every 10 real evals
  outputs/phase4/results/phase4_hv_igd_summary.csv    one row per run (also the resume marker)
  outputs/phase4/logs/archtree_llm__nb201__{dataset}__{hardware}__seed{s}__budget{b}.json

Prerequisites:
  1. Phase 3 finished (verify_phase_inputs needs phase3_main_results.csv).
  2. data/true_pareto_front__{hardware}.json per hardware target, built by
     `python tools/build_igd_reference_front.py`. If missing, IGD is NaN for
     that hardware; the script warns and asks before continuing.

Resume: a combo is skipped if it already has a row in phase4_hv_igd_summary.csv.
If a run is killed between the per-eval rows and the summary row, that combo
may get duplicate pareto rows on rerun. Downstream analysis de-duplicates by
architecture, so results stay correct.

Usage (from archtree-llm/ root):
    python experiments/run_phase4.py
    python experiments/run_phase4.py --hardware edgegpu raspi4 --budgets 100 --seeds 42 111
    python experiments/run_phase4.py --yes    # don't prompt about missing true-front files
"""
import argparse
import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from archtree import storage  # noqa: E402
from archtree.search import search_phase4  # noqa: E402
from benchmarks.hwnb import HWNASBench  # noqa: E402

with open(PROJECT_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "outputs" / "phase4" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.level("RUN", no=22, color="<cyan>")
logger.level("SKIP", no=21, color="<yellow>")
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>")
logger.add(LOG_DIR / "phase4_run.log", level="DEBUG", rotation="10 MB", retention=5)

SUMMARY = "phase4_hv_igd_summary.csv"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "phase4" / "results" / SUMMARY


def construct_only_backend_kwargs() -> dict:
    """Pick any LLM backend the environment supports (Groq preferred, then Ollama).
    It is only constructed, never called."""
    llm = cfg["llm"]
    pool = bool(os.environ.get("GROQ_API_KEYS_FILE", "").strip() or os.environ.get("GROQ_API_KEYS", "").strip())
    single = bool(os.environ.get("GROQ_API_KEY", "").strip())
    if pool or single:
        g = llm["groq"]
        return dict(backend="groq", groq_model=g["model"],
                    groq_daily_token_budget=g.get("daily_token_budget", 190_000),
                    groq_daily_request_budget=g.get("daily_request_budget", 950),
                    groq_reasoning_effort=g.get("reasoning_effort", "low"),
                    use_key_pool=pool)
    try:
        import ollama  # noqa: F401
        return dict(backend="ollama", model=llm["model"], host=llm.get("host", "http://localhost:11434"))
    except ImportError:
        logger.error("No GROQ_API_KEY / GROQ_API_KEYS / GROQ_API_KEYS_FILE set and the `ollama` package "
                     "is not installed. LLMPrior needs one of them just to construct.")
        sys.exit(1)


def done_combos() -> set:
    """(hardware, dataset, seed, budget) tuples already present in the summary CSV."""
    if not SUMMARY_PATH.exists():
        return set()
    with open(SUMMARY_PATH, newline="") as f:
        return {(r["hardware"], r["dataset"], int(r["seed"]), int(r["budget"])) for r in csv.DictReader(f)}


def check_true_fronts(hardware, assume_yes: bool) -> None:
    """Warn (and optionally prompt) if IGD reference fronts are missing."""
    missing = [h for h in hardware if not (PROJECT_ROOT / "data" / f"true_pareto_front__{h}.json").exists()]
    if not missing:
        return
    logger.warning(f"data/true_pareto_front__{{hw}}.json missing for: {missing}. "
                   f"IGD will be NaN for those targets. Fix: python tools/build_igd_reference_front.py")
    if assume_yes:
        return
    if input("Continue anyway? [y/N]: ").strip().lower() != "y":
        sys.exit(1)


def print_summary() -> None:
    if not SUMMARY_PATH.exists():
        return
    # Keep the latest row per run in case a resumed run left duplicates
    df = pd.read_csv(SUMMARY_PATH).drop_duplicates(["hardware", "dataset", "seed", "budget"], keep="last")
    g = (df.groupby(["hardware", "budget"])
           .agg(n=("seed", "count"), hv=("hypervolume", "mean"), igd=("igd", "mean"),
                front=("pareto_front_size", "mean"), real_evals=("real_eval_count", "mean"))
           .reset_index())
    logger.info("mean over datasets x seeds:\n" + g.round(2).to_string(index=False))
    if df["igd"].isna().any():
        logger.warning("Some IGD values are NaN (missing data/true_pareto_front__{hw}.json).")
    if (g["real_evals"] < 0.5 * g["budget"]).any():
        logger.warning("real_evals << budget for some rows — possible exploration collapse in "
                       "search_phase4. Inspect before using those rows.")


def main(args) -> None:
    storage.verify_phase_inputs("phase4")
    hw = HWNASBench(cfg)
    hw._load()

    hardware = args.hardware or cfg["benchmarks"]["hw_targets"]
    datasets = args.datasets or hw.datasets
    check_true_fronts(hardware, args.yes)
    backend_kw = construct_only_backend_kwargs()
    logger.info(f"Phase 4: hardware={hardware} datasets={datasets} seeds={len(args.seeds)} "
                f"budgets={args.budgets} (LLM backend '{backend_kw['backend']}' is construct-only)")

    done = done_combos()
    n_run = n_skip = n_err = 0
    for hwn in hardware:
        for dataset in datasets:
            for seed in args.seeds:
                for budget in args.budgets:
                    if (hwn, dataset, seed, budget) in done:
                        logger.log("SKIP", f"SKIP {hwn}/{dataset}/seed={seed}/budget={budget}")
                        n_skip += 1
                        continue

                    logger.log("RUN", f"{hwn} / {dataset} / seed={seed} / budget={budget}")
                    try:
                        r = search_phase4(hw, budget, dataset, seed, hwn,
                                          llm_cache_phase="phase3", **backend_kw)
                        logger.success(f"    HV={r['hypervolume']:.1f}  front={r['pareto_front_size']}  "
                                       f"best_acc={r['best_acc']:.2f}%  real_evals={r['real_eval_count']}/{budget}  "
                                       f"unsupported={r['unsupported_hw_skipped']}  {r['wall_time_sec']:.1f}s")
                        n_run += 1
                    except KeyboardInterrupt:
                        logger.warning("Interrupted — completed combos are safe (summary row is the resume marker).")
                        sys.exit(0)
                    except Exception:
                        logger.exception("    ERROR:")
                        n_err += 1

    logger.info(f"Phase 4 done — ran={n_run} skipped={n_skip} errors={n_err}")
    print_summary()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", nargs="+", default=None)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg["search"]["seeds"])
    ap.add_argument("--budgets", nargs="+", type=int, default=cfg["search"]["budgets"])
    ap.add_argument("--yes", action="store_true", help="don't prompt about missing true-front files")
    main(ap.parse_args())