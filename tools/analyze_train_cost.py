# tools/analyze_train_cost.py
"""
Real cost of each search = training time of the architectures it really evaluated (Oct 3, 2026).

Counting real evaluations understates the cost of proxy-gated search: the zero-cost gate prefers
larger architectures, which take longer to train (on NB201 about +18% per evaluation). This tool
reports GPU-hours (the benchmark's recorded training time, summed over the real evaluations) next
to the test accuracy, for every method that has them.

Sources (whatever exists):
    outputs/phase0/logs/*.json                   external baselines (eval_history -> NB201 train time)
    outputs/phase1|2|3/results/*_main_results.csv  column train_hours
    outputs/gated_baselines/results/gated_runs.csv  column train_hours
    outputs/design_ablation/results/design_runs.csv column train_hours
    outputs/nb101/results/nb101_runs.csv            column train_sec

Output: outputs/final/results/train_cost_summary.csv and a log of the headline ratios.
    python tools/analyze_train_cost.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"


def phase0_rows(bench):
    rows = []
    for f in sorted((OUT / "phase0" / "logs").glob("*__nb201__*.json")):
        try:
            d = json.load(open(f))
        except (OSError, json.JSONDecodeError):
            continue
        hist, ds = d.get("eval_history") or [], d.get("dataset")
        if not hist or ds is None:
            continue
        from benchmarks.nb201 import arch_key_to_index
        hours = sum(bench.train_sec(arch_key_to_index(e["arch"]), ds) or 0 for e in hist) / 3600
        rows.append({"source": "phase0", "method": d.get("method"), "gate": "none", "dataset": ds,
                     "budget": d.get("budget"), "seed": d.get("seed"), "real_evals": len(hist),
                     "test": d.get("best_test_acc"), "train_hours": hours})
    return rows


def csv_rows(path, source, method_col=None, method=None, gate=None, hours_col="train_hours", scale=1.0):
    if not path.exists():
        return []
    d = pd.read_csv(path)
    if hours_col not in d:
        return []
    out = pd.DataFrame({
        "source": source,
        "method": d[method_col] if method_col else method,
        "gate": d["gate"] if "gate" in d and gate is None else (gate or "none"),
        "dataset": d["dataset"], "budget": d["budget"], "seed": d["seed"],
        "real_evals": d["real_eval_count"], "test": d["best_test_acc"],
        "train_hours": d[hours_col] * scale})
    return out.to_dict("records")


def main():
    from benchmarks.nb201 import NB201
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    bench = NB201(cfg)
    rows = phase0_rows(bench)
    rows += csv_rows(OUT / "phase1/results/phase1_main_results.csv", "phase1", method="archtree_trie_only", gate="none")
    rows += csv_rows(OUT / "phase2/results/phase2_main_results.csv", "phase2", method="archtree_phase2", gate="zc")
    rows += csv_rows(OUT / "phase3/results/phase3_main_results.csv", "phase3", method="archtree_headline",
                     gate=cfg["zc_proxy"].get("calibration_mode", "supervised"))
    rows += csv_rows(OUT / "gated_baselines/results/gated_runs.csv", "gated", method_col="proposer")
    rows += csv_rows(OUT / "design_ablation/results/design_runs.csv", "design", method_col="arm", gate="zc_unsup")
    nb = OUT / "nb101/results/nb101_runs.csv"
    if nb.exists():
        d = pd.read_csv(nb)
        rows += [{"source": "nb101", "method": r.proposer, "gate": r.gate, "dataset": "nb101-cifar10",
                  "budget": r.budget, "seed": r.seed, "real_evals": r.real_eval_count,
                  "test": r.best_test_acc, "train_hours": r.train_sec / 3600} for r in d.itertuples()]
    if not rows:
        print("no results with training-time information yet")
        return
    df = pd.DataFrame(rows)
    summ = (df.groupby(["source", "method", "gate", "dataset", "budget"])
              .agg(n=("seed", "nunique"), real_evals=("real_evals", "mean"), test=("test", "mean"),
                   train_hours=("train_hours", "mean"))
              .reset_index())
    res = OUT / "final" / "results"
    res.mkdir(parents=True, exist_ok=True)
    summ.to_csv(res / "train_cost_summary.csv", index=False)
    print(f"wrote {res / 'train_cost_summary.csv'} ({len(summ)} rows)")

    # headline ratios at the largest budget
    b = max(cfg["search"]["budgets"])
    h = summ[(summ.source == "phase3") & (summ.budget == b)]
    for r in h.itertuples():
        ref = summ[(summ.source == "phase1") & (summ.budget == b) & (summ.dataset == r.dataset)]
        re_ = summ[(summ.source == "phase0") & (summ.method == "re") & (summ.budget == b) & (summ.dataset == r.dataset)]
        line = f"[{r.dataset}] b={b}: headline {r.train_hours:.1f} GPU-h (test {r.test:.2f}, {r.real_evals:.0f} evals)"
        if len(ref):
            line += f" | no-gate trie {ref.train_hours.iloc[0]:.1f} h -> saving {100 * (1 - r.train_hours / ref.train_hours.iloc[0]):.0f}%"
        if len(re_):
            line += f" | RE {re_.train_hours.iloc[0]:.1f} h (test {re_.test.iloc[0]:.2f})"
        print(line)


if __name__ == "__main__":
    main()
