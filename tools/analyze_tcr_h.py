# tools/analyze_tcr_h.py
# isort: skip_file
"""
§2.1, item 2 — "Use TCR-H in results, correlate with accuracy."

Why a replay instead of editing search_phase1/2/3
-------------------------------------------------
TCR-H needs a latency for every visited architecture, and phases 1-3 never
query one (only search_phase4 does). Rather than bolt HW-NAS-Bench lookups
into three search loops (slower runs, and it changes nothing about the search),
this script REPLAYS each finished run: it walks that run's eval_history in
order, looks each architecture's latency up in HW-NAS-Bench, and rebuilds the
trie + Pareto archive with exactly the functions search_phase4 uses
(archtree.tcr.update_pareto_archive / tcr_h / hypervolume). Result: TCR-H for
phase 1, 2, 3 AND the phase-0 baselines, on the same footing, with no re-run.
The always-empty tcr_h column that phase3_tcr_per_iter.csv writes is left
alone (changing a CSV header mid-file would break append_result_row).

Notes on the metric as implemented in archtree/tcr.py
  - Objectives: max accuracy (from the run's own log), min latency (HW-NAS-Bench).
  - Architectures HW-NAS-Bench marks unsupported (negative-latency sentinel ->
    HWMetricUnavailable) never enter the Pareto archive, same as search_phase4;
    they still count as visited terminals with the worst rank weight.
  - Duplicate evals (e.g. rz_nas re-evaluating its current arch) are counted once.
  - "iter" in the outputs = real-eval index, not proposal index (proxy-skipped
    proposals do not appear in eval_history).

Outputs (outputs/final/results/):
  tcr_h_replay_per_iter.csv          checkpoint every --every real evals
  tcr_h_replay_summary.csv           one row per (method, dataset, hardware, seed, budget)
  tcr_h_accuracy_correlation.csv     Spearman/Kendall of final TCR-H vs best_acc, and plain TCR vs best_acc

Reading the correlation file: plain TCR = (#unique evals)/15625, i.e. it only
knows HOW MANY archs were evaluated. TCR-H also knows whether they were any good
on the accuracy/latency front. If TCR-H correlates with best_acc noticeably more
than TCR does at fixed budget, that is the evidence C9 adds information.

Usage (from archtree-llm/ root; no GPU needed):
    python tools/analyze_tcr_h.py
    python tools/analyze_tcr_h.py --hardware edgegpu raspi4 --phases phase3 phase0 --every 5
"""
import argparse
import glob
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402
import yaml           # noqa: E402
from loguru import logger  # noqa: E402
from scipy.stats import kendalltau, spearmanr  # noqa: E402

import archtree.storage as storage  # noqa: E402
from archtree.encoder import OPS_NB201, index_from_tokens_nb201  # noqa: E402
from archtree.tcr import hypervolume, tcr_h, update_pareto_archive  # noqa: E402
from archtree.trie import ArchTrie  # noqa: E402
from benchmarks.hwnb import HWMetricUnavailable, HWNASBench  # noqa: E402

cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
OUT = PROJECT_ROOT / "outputs"
TOTAL = 5 ** 6

logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>")

PHASE_GLOBS = {
    "phase0": str(OUT / "phase0" / "logs" / "*__nb201__*__seed*__budget*.json"),
    "phase1": str(OUT / "phase1" / "logs" / "archtree_v1__nb201__*__seed*__budget*.json"),
    "phase2": str(OUT / "phase2" / "logs" / "archtree_zc__nb201__*__seed*__budget*.json"),
    "phase3": str(OUT / "phase3" / "logs" / "archtree_llm__nb201__*__seed*__budget*.json"),
}


def _tokens(arch: str):
    arch = str(arch)
    if arch.isdigit():
        idx = int(arch) % TOTAL
        toks = []
        for _ in range(6):
            toks.append(OPS_NB201[idx % 5])
            idx //= 5
        return toks[::-1]
    toks = arch.split("|")
    return toks if len(toks) == 6 and all(t in OPS_NB201 for t in toks) else None


class LatencyCache:
    def __init__(self, hw: HWNASBench):
        self.hw, self._c = hw, {}

    def get(self, idx: int, hardware: str, dataset: str):
        k = (idx, hardware, dataset)
        if k not in self._c:
            try:
                self._c[k] = self.hw.hw_query(idx, hardware=hardware, dataset=dataset)[1]
            except HWMetricUnavailable:
                self._c[k] = None
        return self._c[k]


def replay(hist, hardware, dataset, lat: LatencyCache, every: int):
    trie = ArchTrie(max_depth=6)
    archive, per_iter = [], []
    best, n_unsupported = 0.0, 0
    n = len(hist)
    for k, e in enumerate(hist, start=1):
        toks = _tokens(e["arch"])
        if toks is None:
            continue
        already = trie.is_visited(toks)
        if not already:
            idx = index_from_tokens_nb201(toks)
            latency = lat.get(idx, hardware, dataset)
            node = trie.insert(toks)
            node.accuracy = float(e["acc"])
            node.latency = latency
            if latency is None:
                n_unsupported += 1
            else:
                archive = update_pareto_archive(archive, node)
        best = max(best, float(e["acc"]))
        if k % every == 0 or k == n:
            per_iter.append({
                "iter": k, "tcr_h": tcr_h(trie, archive),
                "tcr": trie.count_visited_terminals() / TOTAL,
                "pareto_size": len(archive), "best_acc_so_far": best,
            })
    final = per_iter[-1] if per_iter else {"tcr_h": np.nan, "tcr": np.nan, "pareto_size": 0, "best_acc_so_far": best}
    return per_iter, {**final, "hv": hypervolume(archive), "n_unsupported": n_unsupported,
                      "n_unique": trie.count_visited_terminals()}


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 4 or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return np.nan, np.nan, np.nan, np.nan
    r, p = spearmanr(x[m], y[m])
    t, tp = kendalltau(x[m], y[m])
    return r, p, t, tp


def main(args) -> None:
    hw = HWNASBench(cfg)
    hw._load()
    lat = LatencyCache(hw)
    logger.info(f"HW-NAS-Bench loaded; hardware={args.hardware}; phases={args.phases}")

    per_iter_rows, summary_rows = [], []
    for phase in args.phases:
        files = sorted(glob.glob(PHASE_GLOBS[phase]))
        if args.max_runs:
            files = files[:args.max_runs]
        logger.info(f"{phase}: {len(files)} run logs")
        for fi, f in enumerate(files, start=1):
            with open(f) as fh:
                j = json.load(fh)
            hist = j.get("eval_history") or []
            if not hist or (args.datasets and j.get("dataset") not in args.datasets):
                continue
            method = j.get("method", Path(f).name.split("__")[0])
            for hwn in args.hardware:
                pi, fin = replay(hist, hwn, j["dataset"], lat, args.every)
                key = {"phase": phase, "method": method, "dataset": j["dataset"], "hardware": hwn,
                       "seed": j["seed"], "budget": j["budget"]}
                for r in pi:
                    per_iter_rows.append({**key, **r})
                summary_rows.append({**key, **fin, "n_real_evals": len(hist),
                                     "best_acc": j.get("best_acc", fin["best_acc_so_far"])})
            if fi % 100 == 0:
                logger.info(f"  {phase}: {fi}/{len(files)}")

    if not summary_rows:
        logger.error("No runs replayed — check outputs/*/logs/ and --phases/--datasets.")
        sys.exit(1)

    res = storage.get_output_dir("final", "results")
    pi_df, sm_df = pd.DataFrame(per_iter_rows), pd.DataFrame(summary_rows)
    pi_df.to_csv(res / "tcr_h_replay_per_iter.csv", index=False)
    sm_df.to_csv(res / "tcr_h_replay_summary.csv", index=False)
    logger.success(f"replayed {len(sm_df)} (run x hardware) combos -> tcr_h_replay_per_iter.csv / _summary.csv")

    # ── correlation with accuracy ───────────────────────────────────────────
    rows = []
    for (ds, hwn, b), g in sm_df.groupby(["dataset", "hardware", "budget"]):
        r1, p1, t1, _ = _corr(g["tcr_h"], g["best_acc"])
        r2, p2, t2, _ = _corr(g["tcr"], g["best_acc"])
        rows.append({"level": "pooled_all_methods", "method": "*", "dataset": ds, "hardware": hwn,
                     "budget": b, "n": len(g), "rho_tcr_h": r1, "p_tcr_h": p1, "tau_tcr_h": t1,
                     "rho_tcr": r2, "p_tcr": p2, "tau_tcr": t2})
    for (m, ds, hwn, b), g in sm_df.groupby(["method", "dataset", "hardware", "budget"]):
        r1, p1, t1, _ = _corr(g["tcr_h"], g["best_acc"])
        r2, p2, t2, _ = _corr(g["tcr"], g["best_acc"])
        rows.append({"level": "per_method_across_seeds", "method": m, "dataset": ds, "hardware": hwn,
                     "budget": b, "n": len(g), "rho_tcr_h": r1, "p_tcr_h": p1, "tau_tcr_h": t1,
                     "rho_tcr": r2, "p_tcr": p2, "tau_tcr": t2})
    # within-run: does TCR-H(t) track best-so-far(t)? (autocorrelated; descriptive only)
    wr = []
    for _, g in pi_df.groupby(["phase", "method", "dataset", "hardware", "seed", "budget"]):
        if len(g) >= 4:
            r, _, _, _ = _corr(g["tcr_h"], g["best_acc_so_far"])
            wr.append(r)
    cor = pd.DataFrame(rows)
    cor.to_csv(res / "tcr_h_accuracy_correlation.csv", index=False)

    pooled = cor[cor["level"] == "pooled_all_methods"]
    logger.info("=" * 78)
    logger.info("Spearman rho vs best_acc at fixed budget, all methods pooled (mean over dataset x hardware):")
    for b, g in pooled.groupby("budget"):
        logger.info(f"  budget={int(b):<4d} TCR-H rho={g['rho_tcr_h'].mean():+.3f}   "
                    f"plain TCR rho={g['rho_tcr'].mean():+.3f}   (n_groups={len(g)})")
    if wr:
        logger.info(f"Within-run rho(TCR-H(t), best_so_far(t)): mean={np.nanmean(wr):+.3f} over {len(wr)} runs "
                    f"(both series drift upward with t — descriptive only, not a test)")
    logger.success("tcr_h_accuracy_correlation.csv written.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", nargs="+", default=cfg["benchmarks"]["hw_targets"])
    ap.add_argument("--phases", nargs="+", default=["phase0", "phase1", "phase2", "phase3"],
                    choices=list(PHASE_GLOBS))
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--every", type=int, default=10, help="checkpoint interval in real evals")
    ap.add_argument("--max-runs", type=int, default=0, help="debug: cap logs per phase")
    main(ap.parse_args())