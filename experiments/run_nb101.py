# experiments/run_nb101.py
# isort: skip_file
"""
NAS-Bench-101 experiment (Oct 2026): does ArchTree carry over to a second, much larger and
structurally different search space (423,624 graphs, variable topology, 26 decisions)?

Same protocol as the NB201 runs: budget = number of proposals, gate = top-PCT% rule after
WARMUP proposals, search on mean VALIDATION accuracy, report the mean TEST accuracy of the best-
validation architecture, 12 paired seeds, Wilcoxon + Holm + bootstrap CI.

Proposers
    archtree   search_phase3(alpha=0, use_llm=False) - the headline system (trie + UCB + dedup +
               pruning; per-position vocabulary, invalid cells / isomorphic duplicates redirected)
    random     uniform over the 423,624 graphs, never re-evaluates a graph
    re         regularized evolution (Real et al. 2019), the strongest standard NB101 baseline
Gates
    none       every proposal is a real evaluation  (ArchTree = phase-1 style trie search)
    params     trainable parameters of the cell (zero-cost, zero labels)
    zc_unsup   label-free ZC ensemble (E,S,T z-scored, equal weights)       - GPU
    zc         supervised ZC ensemble fit on calibration_archs NB101 labels  - GPU, labels = cost
The LLM prior is not run on NB101 (its prompt is NB201-specific; alpha=0 is the headline anyway).

Outputs (outputs/nb101/results/)
    nb101_runs.csv                 one row per (budget, seed, proposer, gate)   [resumable]
    nb101_summary.csv              mean/std test acc, regret to the oracle, real evals
    nb101_paired_tests.csv         archtree vs random / re under the SAME gate and budget
    nb101_equal_compute_tests.csv  gated archtree vs UNGATED random / re that spent at least as
                                   many real evaluations (the efficiency claim on NB101)

Usage
    python tools/build_nb101_cache.py            # once
    python experiments/run_nb101.py              # everything (GPU needed for the zc gates)
    python experiments/run_nb101.py --gates none params        # CPU-only part
    python experiments/run_nb101.py --summarize-only
"""
import argparse
import gc
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402
import yaml                 # noqa: E402
from loguru import logger   # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from dotenv import load_dotenv    # noqa: E402
load_dotenv()

import archtree.storage as storage   # noqa: E402
from archtree import settings        # noqa: E402

with open(PROJECT_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)
NB101_CFG = cfg.get("benchmarks", {}).get("nb101", {}) or {}

PHASE = "nb101"
RUNS = "nb101_runs.csv"
DATASET = "cifar10"
PROPOSERS = ["archtree", "random", "re"]
GATES = ["none", "params", "zc_unsup", "zc"]

logger.remove()
LOG_DIR = PROJECT_ROOT / "outputs" / PHASE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>")
logger.add(LOG_DIR / "nb101_run.log", level="DEBUG", rotation="10 MB", retention=5)


class ParamsGate:
    """Zero-label, zero-GPU gate: trainable parameters of the cell (known without training)."""
    labels_used = 0

    def __init__(self, bench):
        self.bench = bench

    def score(self, tokens):
        return float(self.bench.params(self.bench.tokens_to_index(tokens)))


def make_gate(name, bench):
    if name == "none":
        return None
    if name == "params":
        return ParamsGate(bench)
    from archtree.zc_proxy import ZCProxyEnsemble, get_mini_batch
    get_mini_batch()
    if name == "zc_unsup":
        return ZCProxyEnsemble.unsupervised(bench, DATASET, phase=PHASE)
    if name == "zc":
        try:
            return ZCProxyEnsemble.from_saved(PHASE, bench.name, DATASET, mode="supervised")
        except FileNotFoundError:
            logger.info(f"calibrating the supervised ZC ensemble on {settings.CALIBRATION_ARCHS} NB101 "
                        f"labels (counted as cost: gate_labels_used)")
            z = ZCProxyEnsemble()
            z.calibrate(bench, n=settings.CALIBRATION_ARCHS, phase=PHASE, dataset=DATASET,
                        mode="supervised")
            return z
    raise ValueError(name)


class RandomProposer:
    def __init__(self, bench, seed):
        self.bench, self.rng = bench, random.Random(seed)

    def propose(self, evaluated):
        return self.bench.index_to_tokens(self.bench.random_unevaluated_index(evaluated, self.rng))

    def update(self, tokens, acc):
        pass


def run_flat(bench, budget, seed, proposer_name, gate, pct, warmup):
    """Gated search with a non-trie proposer; budget = number of proposals (as for ArchTree)."""
    from baselines.regularized_evolution import RegularizedEvolution
    prop = (RandomProposer(bench, seed) if proposer_name == "random"
            else RegularizedEvolution(bench, seed, budget))
    evaluated, scores, history = set(), [], []
    proxy_evals = 0
    for i in range(budget):
        toks = prop.propose(evaluated)
        take = True
        if gate is not None and i >= warmup:
            s = gate.score(toks)
            proxy_evals += 1
            scores.append(s)
            take = s >= np.percentile(scores, 100 - pct)
        if not take:
            continue
        idx = bench.tokens_to_index(toks)
        acc = bench.query(idx, DATASET)
        evaluated.add(idx)
        prop.update(toks, acc)
        history.append({"iter": i + 1, "arch": str(idx), "acc": acc})
    rep = bench.best_from_history(history, DATASET)
    return {"best_acc": rep["best_acc"], "best_test_acc": rep["best_test_acc"],
            "real_eval_count": len(history), "proxy_eval_count": proxy_evals,
            "train_sec": float(sum(bench.train_sec(int(h["arch"])) for h in history))}


def _llm_free_backend_kwargs():
    """LLMPrior is constructed by search_phase3 even with use_llm=False (no call is made)."""
    try:
        import ollama  # noqa: F401
        return dict(backend="ollama")
    except ImportError:
        pass
    keys = [os.environ.get(k, "").strip() for k in ("GROQ_API_KEYS_FILE", "GROQ_API_KEYS", "GROQ_API_KEY")]
    if not any(keys):
        logger.error("LLMPrior needs `ollama` or a GROQ key just to construct (no call is made).")
        sys.exit(1)
    g = cfg["llm"]["groq"]
    return dict(backend="groq", groq_model=g["model"], use_key_pool=bool(keys[0] or keys[1]))


def _done():
    p = PROJECT_ROOT / "outputs" / PHASE / "results" / RUNS
    if not p.exists():
        return set()
    d = pd.read_csv(p)
    return set(zip(d["budget"], d["seed"], d["proposer"], d["gate"]))


def run(args):
    from benchmarks.nb101 import NB101
    from archtree.search import search_phase3
    bench = NB101(cfg)
    bench._load()
    pct, warmup = settings.PROXY_THRESHOLD_PERCENTILE, settings.WARMUP_REAL_EVALS
    backend_kw = _llm_free_backend_kwargs() if "archtree" in args.proposers else {}
    done = _done()
    for gname in args.gates:                       # cheap gates first
        gate = make_gate(gname, bench)
        for budget in args.budgets:
            for seed in args.seeds:
                for pname in args.proposers:
                    if (budget, seed, pname, gname) in done:
                        continue
                    t0 = time.time()
                    try:
                        if pname == "archtree":
                            r = search_phase3(
                                bench, budget, DATASET, seed, phase=PHASE, alpha=0.0, use_llm=False,
                                use_proxy=gate is not None, zc_override=gate,
                                ablation_id=f"NB101_{gname}", proxy_threshold_percentile=pct,
                                backup_mode=settings.BACKUP_MODE_P34,
                                results_filename="nb101_raw_archtree_runs.csv",
                                tcr_filename="nb101_raw_tcr.csv",
                                cache_stats_filename="nb101_raw_cache_stats.csv",
                                dedup_filename="nb101_dedup_stats.csv", **backend_kw)
                            log = storage.get_output_dir(PHASE, "logs") / storage.build_filename(
                                f"archtree_llm__NB101_{gname}", bench.name, DATASET, seed, "json").replace(
                                ".json", f"__budget{budget}.json")
                            import json
                            hist = json.load(open(log))["eval_history"]
                            r["train_sec"] = float(sum(bench.train_sec(bench.arch_key_to_index(h["arch"]))
                                                       for h in hist))
                        else:
                            random.seed(seed)
                            r = run_flat(bench, budget, seed, pname, gate, pct, warmup)
                    except KeyboardInterrupt:
                        logger.warning("Interrupted - append-only CSV, rerun to resume.")
                        sys.exit(0)
                    except Exception:
                        logger.exception(f"ERROR b={budget} s={seed} {pname}/{gname}")
                        continue
                    storage.append_result_row(PHASE, RUNS, {
                        "benchmark": "nb101", "dataset": DATASET, "budget": budget, "seed": seed,
                        "proposer": pname, "gate": gname,
                        "best_acc": r["best_acc"], "best_test_acc": r["best_test_acc"],
                        "real_eval_count": r["real_eval_count"],
                        "proxy_eval_count": (max(0, budget - warmup) if gate is not None else 0)
                        if pname == "archtree" else r.get("proxy_eval_count", 0),
                        "gate_labels_used": getattr(gate, "labels_used", 0) if gate is not None else 0,
                        "train_sec": r.get("train_sec", np.nan),
                        "wall_time_sec": time.time() - t0})
                    logger.info(f"[NB101] b={budget} s={seed} {pname:8s} gate={gname:8s} "
                                f"valid={r['best_acc']:.2f} test={r['best_test_acc']:.2f} "
                                f"real={r['real_eval_count']}")
                    gc.collect()


def _holm(p):
    p = np.asarray(p, float)
    o, m, out, run_ = np.argsort(p), len(p), np.empty(len(p)), 0.0
    for k, i in enumerate(o):
        run_ = max(run_, (m - k) * p[i])
        out[i] = min(1.0, run_)
    return out


def _paired(a, b, rng):
    d = (a - b).values
    try:
        pv = wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        pv = 1.0
    boots = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), pv


def _finish(t, family):
    """Holm within each comparison family (Oct 3: per family, not over every test at once)."""
    if not len(t):
        return t
    t["p_holm"] = t.groupby(family)["p_value"].transform(lambda s: _holm(s.values))
    t["outcome_holm"] = np.where(t["p_holm"] < 0.05,
                                 np.where(t["mean_diff_archtree_minus_other"] > 0, "win", "loss"), "tie")
    # columns read by run_phase6_aggregate.py (extended_findings_summary.csv)
    t["dataset"] = DATASET
    t["n_paired_seeds"] = t["n"]
    t["wilcoxon_p"] = t["p_value"]
    t["cohens_d"] = np.nan
    t["significant"] = t["p_holm"] < 0.05
    return t


def summarize():
    from benchmarks.nb101 import NB101
    p = PROJECT_ROOT / "outputs" / PHASE / "results" / RUNS
    if not p.exists():
        logger.error("no NB101 runs yet")
        return
    bench = NB101(cfg)
    orc = bench.oracle()
    df = pd.read_csv(p).drop_duplicates(["budget", "seed", "proposer", "gate"], keep="last")
    df["regret_test"] = orc["test_of_best_valid"] - df["best_test_acc"]
    res = storage.get_output_dir(PHASE, "results")
    summ = (df.groupby(["gate", "budget", "proposer"])
              .agg(n=("seed", "nunique"), test_mean=("best_test_acc", "mean"),
                   test_std=("best_test_acc", "std"), valid_mean=("best_acc", "mean"),
                   regret_mean=("regret_test", "mean"), real_evals=("real_eval_count", "mean"),
                   proxy_evals=("proxy_eval_count", "mean"), labels=("gate_labels_used", "first"),
                   train_hours=("train_sec", lambda x: x.mean() / 3600)).reset_index())
    summ.to_csv(res / "nb101_summary.csv", index=False)

    rng = np.random.default_rng(0)
    rows = []
    for (g, b), grp in df.groupby(["gate", "budget"]):
        piv = grp.pivot(index="seed", columns="proposer", values="best_test_acc")
        if "archtree" not in piv:
            continue
        for other in [c for c in piv.columns if c != "archtree"]:
            both = piv[["archtree", other]].dropna()
            if len(both) < 2:
                continue
            md, lo, hi, pv = _paired(both["archtree"], both[other], rng)
            rows.append({"gate": g, "budget": b, "other": other, "n": len(both),
                         "mean_diff_archtree_minus_other": md, "ci95_lo": lo, "ci95_hi": hi,
                         "p_value": pv})
    t = _finish(pd.DataFrame(rows), ["gate", "other"])
    t.to_csv(res / "nb101_paired_tests.csv", index=False)

    # equal compute: gated archtree vs ungated baselines that spent >= as many real evals
    rows = []
    ung = df[df["gate"] == "none"]
    for (g, b), grp in df[(df["proposer"] == "archtree") & (df["gate"] != "none")].groupby(["gate", "budget"]):
        need = grp["real_eval_count"].max()
        cands = sorted(ung["budget"].unique())
        b2 = next((x for x in cands if x >= need), None)
        if b2 is None:
            continue
        for other in ["random", "re", "archtree"]:
            o = ung[(ung["proposer"] == other) & (ung["budget"] == b2)].set_index("seed")["best_test_acc"]
            a = grp.set_index("seed")["best_test_acc"]
            both = pd.concat([a, o], axis=1, keys=["a", "o"]).dropna()
            if len(both) < 2:
                continue
            md, lo, hi, pv = _paired(both["a"], both["o"], rng)
            rows.append({"gate": g, "archtree_budget": b, "archtree_real_evals_max": int(need),
                         "other": f"{other}_ungated", "other_budget": b2, "n": len(both),
                         "mean_diff_archtree_minus_other": md, "ci95_lo": lo, "ci95_hi": hi,
                         "p_value": pv})
    e = _finish(pd.DataFrame(rows), ["gate", "other"])
    e.to_csv(res / "nb101_equal_compute_tests.csv", index=False)

    logger.info("=" * 78)
    logger.info(f"NB101 oracle: best mean valid {orc['best_valid']:.2f} -> test {orc['test_of_best_valid']:.2f}; "
                f"best mean test {orc['best_test']:.2f}")
    for gate_name, grp in (t.groupby("gate") if len(t) else []):
        c = grp["outcome_holm"].value_counts()
        logger.info(f"  same gate={gate_name:9s}: archtree wins={c.get('win', 0)} ties={c.get('tie', 0)} "
                    f"losses={c.get('loss', 0)} (of {len(grp)})")
    if len(e):
        c = e["outcome_holm"].value_counts()
        logger.info(f"  equal real-eval compute (gated archtree vs ungated): wins={c.get('win', 0)} "
                    f"ties={c.get('tie', 0)} losses={c.get('loss', 0)} (of {len(e)})")
    logger.success("nb101_summary.csv / nb101_paired_tests.csv / nb101_equal_compute_tests.csv written.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--proposers", nargs="+", default=PROPOSERS, choices=PROPOSERS)
    ap.add_argument("--gates", nargs="+", default=GATES, choices=GATES)
    ap.add_argument("--budgets", nargs="+", type=int,
                    default=NB101_CFG.get("budgets", [50, 100, 200, 400]))
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg["search"]["seeds"])
    a = ap.parse_args()
    if not a.summarize_only:
        run(a)
    summarize()
