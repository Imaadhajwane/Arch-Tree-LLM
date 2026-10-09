# experiments/run_phase4_baselines.py
# isort: skip_file
"""
Reviewer item B9 — baselines for the hardware-aware phase (Phase 4 was ArchTree-only).

Same budget (= real HW-NAS-Bench evaluations), same seeds, same devices, same HV / IGD code
(archtree.tcr) and the same reference fronts as search_phase4:
    random_hw   uniform random search without replacement
    nsga2_hw    NSGA-II-style evolutionary search (population 20, binary tournament on
                (non-domination rank, crowding distance), uniform crossover + one-point mutation,
                elitist (mu + lambda) survival). Objectives: max accuracy, min latency.

Requires the Oct-2026 benchmarks/hwnb.py (canonical -> HW-NAS-Bench index map) and fronts rebuilt
with it (python tools/build_igd_reference_front.py).

Outputs (outputs/phase4/results/):
    phase4_baselines_hv_igd_summary.csv   one row per (method, hardware, dataset, seed, budget)
    phase4_hw_paired_tests.csv            archtree_hw vs each baseline: HV and IGD, Wilcoxon + Holm

Usage:
    python experiments/run_phase4_baselines.py
    python experiments/run_phase4_baselines.py --hardware edgegpu raspi4 --budgets 100 200
    python experiments/run_phase4_baselines.py --summarize-only
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402
import yaml                 # noqa: E402
from loguru import logger   # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

import archtree.storage as storage                         # noqa: E402
from archtree.trie import TrieNode                         # noqa: E402
from archtree.tcr import update_pareto_archive, hypervolume, igd   # noqa: E402

with open(PROJECT_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

SUMMARY = "phase4_baselines_hv_igd_summary.csv"
OPS = ["none", "skip_connect", "conv_1x1", "conv_3x3", "avg_pool_3x3"]

logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>")


class Evaluator:
    """Real HW evaluations with dedup, Pareto archive and unsupported-latency handling."""
    def __init__(self, hw, hardware, dataset):
        from benchmarks.hwnb import HWMetricUnavailable
        self.hw, self.hardware, self.dataset, self.Unavail = hw, hardware, dataset, HWMetricUnavailable
        self.archive, self.seen, self.unsupported, self.best = [], {}, 0, 0.0

    def __call__(self, idx):
        """-> (acc, latency or None). Each distinct idx costs one real evaluation."""
        if idx in self.seen:
            return self.seen[idx]
        acc = self.hw.query(idx, self.dataset)
        try:
            lat = self.hw.latency(idx, self.hardware, self.dataset)
        except self.Unavail:
            lat = None
            self.unsupported += 1
        self.seen[idx] = (acc, lat)
        self.best = max(self.best, acc)
        if lat is not None:
            n = TrieNode()
            n.accuracy, n.latency = acc, lat
            self.archive = update_pareto_archive(self.archive, n)
        return acc, lat


def tok2idx(t):
    i = 0
    for o in t:
        i = i * 5 + OPS.index(o)
    return i


def idx2tok(i):
    out = []
    for _ in range(6):
        out.append(OPS[i % 5])
        i //= 5
    return out[::-1]


def random_hw(ev, budget, rng):
    for i in rng.sample(range(15625), budget):
        ev(i)


def _nd_sort(F):
    """F: list of (acc, lat) -> list of fronts (lists of indices), maximise acc, minimise lat."""
    n = len(F)
    dom = [[] for _ in range(n)]
    cnt = [0] * n
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            a, b = F[p], F[q]
            if a[0] >= b[0] and a[1] <= b[1] and (a[0] > b[0] or a[1] < b[1]):
                dom[p].append(q)
            elif b[0] >= a[0] and b[1] <= a[1] and (b[0] > a[0] or b[1] < a[1]):
                cnt[p] += 1
        if cnt[p] == 0:
            fronts[0].append(p)
    k = 0
    while fronts[k]:
        nxt = []
        for p in fronts[k]:
            for q in dom[p]:
                cnt[q] -= 1
                if cnt[q] == 0:
                    nxt.append(q)
        k += 1
        fronts.append(nxt)
    return fronts[:-1]


def _crowding(F, front):
    d = {i: 0.0 for i in front}
    for m in (0, 1):
        s = sorted(front, key=lambda i: F[i][m])
        d[s[0]] = d[s[-1]] = float("inf")
        span = (F[s[-1]][m] - F[s[0]][m]) or 1.0
        for j in range(1, len(s) - 1):
            d[s[j]] += (F[s[j + 1]][m] - F[s[j - 1]][m]) / span
    return d


def nsga2_hw(ev, budget, rng, pop_size=20):
    def fit(idx):
        acc, lat = ev(idx)
        return (acc, lat if lat is not None else float("inf"))

    pop = rng.sample(range(15625), min(pop_size, budget))
    for i in pop:
        fit(i)
    while len(ev.seen) < budget:
        F = [fit(i) for i in pop]
        fronts = _nd_sort(F)
        rank, crowd = {}, {}
        for r, fr in enumerate(fronts):
            for i in fr:
                rank[i] = r
            crowd.update(_crowding(F, fr))

        def tourney():
            a, b = rng.randrange(len(pop)), rng.randrange(len(pop))
            ka, kb = (rank[a], -crowd[a]), (rank[b], -crowd[b])
            return pop[a] if ka <= kb else pop[b]

        children, tries = [], 0
        while len(children) < pop_size and len(ev.seen) + len(children) < budget and tries < 2000:
            tries += 1
            pa, pb = idx2tok(tourney()), idx2tok(tourney())
            child = [x if rng.random() < 0.5 else y for x, y in zip(pa, pb)]
            pos = rng.randrange(6)
            child[pos] = rng.choice([o for o in OPS if o != child[pos]])
            ci = tok2idx(child)
            if ci not in ev.seen and ci not in children:
                children.append(ci)
        if not children:      # population converged: inject random unseen archs
            children = [i for i in rng.sample(range(15625), 50) if i not in ev.seen][:pop_size]
        for c in children:
            if len(ev.seen) >= budget:
                break
            fit(c)
        union = list(dict.fromkeys(pop + [c for c in children if c in ev.seen]))
        F = [fit(i) for i in union]
        new = []
        for fr in _nd_sort(F):
            if len(new) + len(fr) <= pop_size:
                new += [union[i] for i in fr]
            else:
                cd = _crowding(F, fr)
                new += [union[i] for i in sorted(fr, key=lambda i: -cd[i])[:pop_size - len(new)]]
                break
        pop = new


def run(args):
    from benchmarks.hwnb import HWNASBench
    hw = HWNASBench(cfg)
    hw._load()
    path = PROJECT_ROOT / "outputs" / "phase4" / "results" / SUMMARY
    done = set()
    if path.exists():
        d = pd.read_csv(path)
        done = set(zip(d["method"], d["hardware"], d["dataset"], d["seed"], d["budget"]))
    for hardware in args.hardware:
        tf_path = PROJECT_ROOT / "data" / f"true_pareto_front__{hardware}.json"
        true_front = [tuple(p) for p in json.load(open(tf_path))] if tf_path.exists() else None
        for dataset in args.datasets:
            for budget in args.budgets:
                for seed in args.seeds:
                    for method, fn in (("random_hw", random_hw), ("nsga2_hw", nsga2_hw)):
                        if (method, hardware, dataset, seed, budget) in done:
                            continue
                        t0 = time.time()
                        ev = Evaluator(hw, hardware, dataset)
                        fn(ev, budget, random.Random(seed))
                        hv = hypervolume(ev.archive)
                        ig = igd(ev.archive, true_front) if true_front else float("nan")
                        storage.append_result_row("phase4", SUMMARY, {
                            "method": method, "benchmark": "hwnb", "hardware": hardware, "dataset": dataset,
                            "seed": seed, "budget": budget, "hypervolume": hv, "igd": ig,
                            "pareto_front_size": len(ev.archive), "real_eval_count": len(ev.seen),
                            "best_acc": ev.best, "unsupported_hw_skipped": ev.unsupported,
                            "wall_time_sec": time.time() - t0})
                        logger.info(f"{method:9s} {hardware:8s} {dataset} b={budget} s={seed} "
                                    f"HV={hv:.0f} IGD={ig:.3f} front={len(ev.archive)}")


def _holm(p):
    p = np.asarray(p, float)
    o, m, out, r = np.argsort(p), len(p), np.empty(len(p)), 0.0
    for k, i in enumerate(o):
        r = max(r, (m - k) * p[i])
        out[i] = min(1.0, r)
    return out


def summarize():
    res = PROJECT_ROOT / "outputs" / "phase4" / "results"
    bp, ap_ = res / SUMMARY, res / "phase4_hv_igd_summary.csv"
    if not (bp.exists() and ap_.exists()):
        logger.error("need both phase4_hv_igd_summary.csv (run_phase4.py) and the baselines file")
        return
    b = pd.read_csv(bp)
    a = pd.read_csv(ap_).drop_duplicates(["hardware", "dataset", "seed", "budget"], keep="last")
    a["method"] = "archtree_hw"
    df = pd.concat([a, b], ignore_index=True)
    rows = []
    for (hwn, ds, bud), g in df.groupby(["hardware", "dataset", "budget"]):
        for metric, better in (("hypervolume", 1), ("igd", -1)):
            piv = g.pivot_table(index="seed", columns="method", values=metric)
            if "archtree_hw" not in piv:
                continue
            for other in [c for c in piv.columns if c != "archtree_hw"]:
                both = piv[["archtree_hw", other]].dropna()
                if len(both) < 2:
                    continue
                d = (both["archtree_hw"] - both[other]).values
                try:
                    pv = wilcoxon(both["archtree_hw"], both[other]).pvalue if np.any(d != 0) else 1.0
                except ValueError:
                    pv = 1.0
                rows.append({"hardware": hwn, "dataset": ds, "budget": bud, "metric": metric, "other": other,
                             "n": len(both), "mean_archtree": both["archtree_hw"].mean(),
                             "mean_other": both[other].mean(), "mean_diff": float(d.mean()),
                             "better_direction": "higher" if better > 0 else "lower", "p_value": pv})
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm"] = _holm(t["p_value"].values)
        good = np.where(t["better_direction"] == "higher", t["mean_diff"] > 0, t["mean_diff"] < 0)
        t["outcome_holm"] = np.where(t["p_holm"] < 0.05, np.where(good, "archtree_better", "archtree_worse"), "tie")
    t.to_csv(res / "phase4_hw_paired_tests.csv", index=False)
    if len(t):
        logger.info("\n" + t.groupby(["metric", "other"])["outcome_holm"].value_counts().unstack(fill_value=0).to_string())
    logger.success("phase4_hw_paired_tests.csv written.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--hardware", nargs="+", default=cfg["benchmarks"]["hw_targets"])
    ap.add_argument("--datasets", nargs="+", default=["cifar10"])
    ap.add_argument("--budgets", nargs="+", type=int, default=cfg["search"]["budgets"])
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg["search"]["seeds"])
    a = ap.parse_args()
    if not a.summarize_only:
        run(a)
    summarize()
