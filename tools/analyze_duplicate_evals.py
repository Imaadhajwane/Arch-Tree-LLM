# tools/analyze_duplicate_evals.py
# isort: skip_file
"""
§2.1 item 3 — Quantify duplicate evaluations (ArchTree vs. baselines, incl. MCTS-NAS).

No search is re-run: this reads existing per-run JSON logs and, for every method that logged an
eval_history, counts real evaluations that hit an already-evaluated architecture:

    n_dup_real = len(eval_history) - len(unique architectures)

Two quantities, kept separate on purpose:
  n_dup_real     duplicate REAL evals actually paid for (any method)
  dup_proposals  times the ArchTree UCB traversal proposed an already-evaluated leaf and the trie
                 redirected it ("duplicates avoided"). Only present in logs written by the patched
                 archtree/search.py; older logs -> NaN.

Log sources: outputs/phase0 (baselines), phase1/2/3 (ArchTree v1 / zc / llm), ablations (A7 arms).

Outputs: outputs/final/results/duplicate_evals_per_run.csv, duplicate_evals_summary.csv

Usage:
    python tools/analyze_duplicate_evals.py
"""
import glob
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402
from loguru import logger  # noqa: E402

import archtree.storage as storage  # noqa: E402
from archtree.encoder import OPS_NB201, index_from_tokens_nb201  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>")

OUT = PROJECT_ROOT / "outputs"
N = 5 ** 6  # NB201 search space size

# (source label, glob pattern)
GLOBS = [
    ("phase0", str(OUT / "phase0" / "logs" / "*__nb201__*__seed*__budget*.json")),
    ("phase1", str(OUT / "phase1" / "logs" / "archtree_v1__nb201__*__seed*__budget*.json")),
    ("phase2", str(OUT / "phase2" / "logs" / "archtree_zc__nb201__*__seed*__budget*.json")),
    ("phase3", str(OUT / "phase3" / "logs" / "archtree_llm__nb201__*__seed*__budget*.json")),
    ("A7",     str(OUT / "ablations" / "logs" / "*A7*__nb201__*__seed*__budget*.json")),
]


def _arch_key(s: str):
    """Canonical architecture id from an eval_history 'arch' string.
    Baselines log either the NB201 index ('4711') or the tokens joined by '|'."""
    s = str(s)
    if s.isdigit():
        return int(s) % N
    toks = s.split("|")
    if len(toks) == 6 and all(t in OPS_NB201 for t in toks):
        return index_from_tokens_nb201(toks)
    return s  # unknown format: still deduplicated on the raw string


def load_runs() -> pd.DataFrame:
    """One row per finished run log, with duplicate counts."""
    rows = []
    for src, pattern in GLOBS:
        files = sorted(glob.glob(pattern))
        if not files:
            logger.info(f"  {src:7s}: no logs matching {Path(pattern).name}")
            continue
        n_ok = 0
        for f in files:
            try:
                with open(f) as fh:
                    j = json.load(fh)
            except (json.JSONDecodeError, OSError):
                logger.warning(f"unreadable log skipped: {f}")
                continue
            hist = j.get("eval_history") or []
            if not hist:
                continue
            keys = [_arch_key(e["arch"]) for e in hist]
            n, uniq = len(keys), len(set(keys))
            method = j.get("method", Path(f).name.split("__")[0])
            abl = j.get("ablation_id", "")
            if abl:
                method = f"{method}[{abl}]"  # keep ablation arms separate
            rows.append({
                "source": src, "method": method,
                "dataset": j.get("dataset"), "seed": j.get("seed"), "budget": j.get("budget"),
                "n_real_evals": n, "n_unique": uniq, "n_dup_real": n - uniq,
                "dup_rate": (n - uniq) / n,
                "dup_proposals": j.get("dup_proposals", np.nan),
                "repeat_proxy_skips": j.get("repeat_proxy_skips", np.nan),
                "best_acc": j.get("best_acc", np.nan),
            })
            n_ok += 1
        logger.info(f"  {src:7s}: {n_ok} runs")
    return pd.DataFrame(rows)


def main() -> None:
    logger.info("Scanning run logs ...")
    df = load_runs()
    if df.empty:
        logger.error("No run logs found under outputs/*/logs/. Nothing to analyse.")
        sys.exit(1)

    res_dir = storage.get_output_dir("final", "results")
    df.to_csv(res_dir / "duplicate_evals_per_run.csv", index=False)

    summ = (df.groupby(["method", "dataset", "budget"])
              .agg(n_runs=("seed", "nunique"),
                   real_evals_mean=("n_real_evals", "mean"),
                   dup_real_mean=("n_dup_real", "mean"),
                   dup_real_max=("n_dup_real", "max"),
                   dup_rate_mean=("dup_rate", "mean"),
                   dup_proposals_mean=("dup_proposals", "mean"),
                   repeat_proxy_skips_mean=("repeat_proxy_skips", "mean"))
              .reset_index())
    summ.to_csv(res_dir / "duplicate_evals_summary.csv", index=False)
    logger.success(f"wrote {res_dir / 'duplicate_evals_per_run.csv'} and duplicate_evals_summary.csv")

    # Sanity check: trie-based runs should never pay for a duplicate real eval
    trie_methods = [m for m in summ["method"].unique() if m.startswith("archtree")]
    bad = summ[summ["method"].isin(trie_methods) & (summ["dup_real_max"] > 0)]
    if len(bad):
        logger.warning("Trie-based runs with duplicate REAL evals (should be 0 unless "
                       "random_unvisited() exhausted its retries):\n" + bad.to_string(index=False))

    # ── headline: duplicate real evals per method and budget, datasets pooled ──
    logger.info("=" * 78)
    logger.info("Duplicate real evals per run (mean), by method, at each budget - all datasets pooled")
    pooled = (df.groupby(["method", "budget"])
                .agg(dup=("n_dup_real", "mean"), of=("n_real_evals", "mean"),
                     rate=("dup_rate", "mean")).reset_index())
    for budget in sorted(pooled["budget"].dropna().unique()):
        logger.info(f"  budget={int(budget)}")
        for _, r in pooled[pooled["budget"] == budget].sort_values("dup", ascending=False).iterrows():
            logger.info(f"    {r['method']:28s} {r['dup']:6.2f} dup / {r['of']:6.1f} evals  ({r['rate']:.1%})")

    # ── "duplicates avoided" (ArchTree only; needs logs from the patched search.py) ──
    avoided = df.dropna(subset=["dup_proposals"])
    avoided = avoided[avoided["method"].str.startswith("archtree")]
    logger.info("-" * 78)
    if len(avoided):
        logger.info("ArchTree 'duplicates avoided' = UCB proposals redirected by the trie (mean per run):")
        for (m, b), g in avoided.groupby(["method", "budget"]):
            logger.info(f"    {m:28s} budget={int(b):<4d} {g['dup_proposals'].mean():6.2f} redirected "
                        f"(+{g['repeat_proxy_skips'].mean():.1f} repeat proxy-skips)")
    else:
        logger.info("No dup_proposals in any ArchTree log yet (logs predate the patched search.py). "
                    "Re-run those phases to populate it; n_dup_real above is still valid.")

    # ── interpretation warnings ──
    mcts = pooled[pooled["method"] == "mcts_nas"]
    if len(mcts) and mcts["dup"].max() < 0.5:
        logger.warning("MCTS-NAS had < 0.5 duplicate evals/run at every budget (its expansion pads paths with "
                       "random ops, so repeats are rare at budgets << 15,625). Don't lead the paper with "
                       "'trie avoids MCTS duplicates'; the dedup effect shows against rz_nas / flat UCB (A7).")
    rz = pooled[pooled["method"] == "rz_nas"]
    if len(rz) and rz["rate"].max() > 0.1:
        # rz_nas re-evaluates its current arch whenever it rejects a mutation
        logger.info(f"rz_nas re-evaluates its current arch on every rejected mutation: "
                    f"up to {rz['rate'].max():.0%} of its evals are duplicates.")


if __name__ == "__main__":
    main()