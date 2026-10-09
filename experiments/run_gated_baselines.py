# experiments/run_gated_baselines.py
# isort: skip_file
"""
Reviewer item A1 — is the efficiency gain ArchTree's, or just the proxy gate's?

Every search strategy is run with the SAME gate, the same gate rule (first WARMUP proposals always
evaluated, then a proposal is really evaluated only if its gate score is in the top-PCT% of the
scores seen so far) and the same proposal budget, so all arms spend a comparable number of real
evaluations.

Search strategies ("proposers")
    archtree   search_phase3(alpha=0) - the headline system (trie + UCB + dedup + pruning)
    random     uniform sampling, never re-proposes a really-evaluated arch
    re         regularized evolution (Real et al. 2019; baselines/regularized_evolution.py) -
               the standard strong NB201/NB101 baseline (added Oct 2026)
    ls         local search (White et al. 2021; baselines/local_search.py), added Oct 3, 2026
    enas       ENAS-style per-position policy (baselines/enas_proxy.py)
    mcts       MCTS-style (baselines/mcts_nas.py)
Gates
    zc         the SUPERVISED ensemble (fit on 1,500 true accuracies, see analyze_proxy_cost.py)
               - always supervised, so "labelled vs label-free gate" stays measurable whatever
               config zc_proxy.calibration_mode says
    zc_unsup   label-free ensemble (z-scored E,S,T, equal weights; zero labels)
    params     #parameters of the cell (conv3x3=9, conv1x1=1 per edge) - the trivial
               zero-cost baseline many papers find hard to beat (CoRA-NAS); zero labels, no GPU
    none       no gate (every proposal is a real evaluation)

Outputs (outputs/gated_baselines/results/):
    gated_runs.csv            one row per (dataset, budget, seed, proposer, gate)  [resumable]
    gated_summary.csv         mean/std of the report metric and real evals
    gated_paired_tests.csv    archtree vs each proposer under the same gate (Wilcoxon, Holm, CI)

Usage:
    python experiments/run_gated_baselines.py
    python experiments/run_gated_baselines.py --gates zc params --budgets 100 200
    python experiments/run_gated_baselines.py --summarize-only
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

PHASE = "gated_baselines"
RUNS = "gated_runs.csv"
PROPOSERS = ["archtree", "random", "re", "ls", "enas", "mcts"]
GATES = ["zc", "zc_unsup", "params", "none"]

logger.remove()
LOG_DIR = PROJECT_ROOT / "outputs" / PHASE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>")
logger.add(LOG_DIR / "gated_run.log", level="DEBUG", rotation="10 MB", retention=5)

PARAM_W = {"none": 0, "skip_connect": 0, "avg_pool_3x3": 0, "conv_1x1": 1, "conv_3x3": 9}


class ParamsGate:
    """Zero-label, zero-GPU gate: number of conv weights in the cell."""
    labels_used = 0

    def score(self, tokens):
        return float(sum(PARAM_W[t] for t in tokens))


def make_gate(name, bench, dataset):
    from archtree.zc_proxy import ZCProxyEnsemble
    if name == "none":
        return None
    if name == "params":
        return ParamsGate()
    if name == "zc":
        return ZCProxyEnsemble.from_saved("phase2", bench.name, dataset, mode="supervised")
    if name == "zc_unsup":
        return ZCProxyEnsemble.unsupervised(bench, dataset)
    raise ValueError(name)


# ── proposers for the non-trie strategies ────────────────────────────────────
class RandomProposer:
    def __init__(self, bench, seed):
        self.bench, self.rng = bench, random.Random(seed)

    def propose(self, evaluated):
        while True:
            i = self.rng.randrange(self.bench.total_archs)
            if i not in evaluated:
                return self.bench.index_to_tokens(i)

    def update(self, tokens, acc):
        pass


class ENASProposer:
    def __init__(self, bench, dataset, seed):
        from baselines.enas_proxy import ENASProxy
        self.m, self.bench = ENASProxy(bench, dataset, seed), bench

    def propose(self, evaluated):
        from archtree.encoder import index_from_tokens_nb201
        for _ in range(500):
            t = self.m._sample_tokens()
            if index_from_tokens_nb201(t) not in evaluated:
                return t
        return RandomProposer(self.bench, self.m.rng.random()).propose(evaluated)

    def update(self, tokens, acc):
        self.m._update(tokens, acc)


class MCTSProposer:
    def __init__(self, bench, dataset, seed):
        from baselines.mcts_nas import MCTSNASProxy
        random.seed(seed)                        # MCTSNASProxy uses the global RNG
        self.m, self.bench, self._last = MCTSNASProxy(bench, dataset, seed), bench, None

    def propose(self, evaluated):
        from archtree.encoder import index_from_tokens_nb201
        for _ in range(200):
            path = self.m._select_and_expand()
            self.m._path_to_index(path)          # pads `path` in place to 6 tokens
            if index_from_tokens_nb201(path) not in evaluated:
                self._last = path
                return list(path)
        t = RandomProposer(self.bench, random.random()).propose(evaluated)
        self._last = t
        return t

    def update(self, tokens, acc):
        self.m._backpropagate(tokens, acc / 100.0)


def run_flat(bench, dataset, budget, seed, proposer_name, gate, pct, warmup):
    """Gated search with a non-trie proposer; budget = number of proposals (as in ArchTree)."""
    from archtree.encoder import index_from_tokens_nb201
    from benchmarks.nb201 import best_from_history
    from baselines.regularized_evolution import RegularizedEvolution
    from baselines.local_search import LocalSearch
    prop = {"random": lambda: RandomProposer(bench, seed),
            "ls": lambda: LocalSearch(bench, seed),
            "enas": lambda: ENASProposer(bench, dataset, seed),
            "mcts": lambda: MCTSProposer(bench, dataset, seed),
            "re": lambda: RegularizedEvolution(bench, seed, budget)}[proposer_name]()
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
        idx = index_from_tokens_nb201(toks)
        acc = bench.query(idx, dataset)
        evaluated.add(idx)
        prop.update(toks, acc)
        history.append({"iter": i + 1, "arch": "|".join(toks), "acc": acc})
    rep = best_from_history(bench, history, dataset)
    from archtree.search import _train_hours
    return {"best_acc": rep["best_acc"], "best_test_acc": rep["best_test_acc"],
            "real_eval_count": len(history), "proxy_eval_count": proxy_evals,
            "train_hours": _train_hours(bench, history, dataset)}


def _llm_free_backend_kwargs():
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
    return set(zip(d["dataset"], d["budget"], d["seed"], d["proposer"], d["gate"]))


def run(args):
    from benchmarks.nb201 import NB201
    from archtree.search import search_phase3
    from archtree.zc_proxy import get_mini_batch
    bench = NB201(cfg)
    bench._load()
    pct, warmup = settings.PROXY_THRESHOLD_PERCENTILE, settings.WARMUP_REAL_EVALS
    if any(g.startswith("zc") for g in args.gates):
        get_mini_batch()
    backend_kw = _llm_free_backend_kwargs() if "archtree" in args.proposers else {}
    done = _done()
    for dataset in args.datasets:
        gates = {g: make_gate(g, bench, dataset) for g in args.gates}
        for budget in args.budgets:
            for seed in args.seeds:
                for gname, gate in gates.items():
                    for pname in args.proposers:
                        if (dataset, budget, seed, pname, gname) in done:
                            continue
                        t0 = time.time()
                        try:
                            if pname == "archtree":
                                r = search_phase3(
                                    bench, budget, dataset, seed, phase=PHASE, alpha=0.0, use_llm=False,
                                    use_proxy=gate is not None, zc_override=gate,
                                    ablation_id=f"G_{gname}", proxy_threshold_percentile=pct,
                                    backup_mode=settings.BACKUP_MODE_P34,
                                    results_filename="G_raw_archtree_runs.csv", tcr_filename="G_raw_tcr.csv",
                                    cache_stats_filename="G_raw_cache_stats.csv",
                                    dedup_filename="G_dedup_stats.csv", **backend_kw)
                            else:
                                random.seed(seed)
                                r = run_flat(bench, dataset, budget, seed, pname, gate, pct, warmup)
                        except KeyboardInterrupt:
                            logger.warning("Interrupted — append-only CSV, rerun to resume.")
                            sys.exit(0)
                        except Exception:
                            logger.exception(f"ERROR {dataset} b={budget} s={seed} {pname}/{gname}")
                            continue
                        storage.append_result_row(PHASE, RUNS, {
                            "dataset": dataset, "budget": budget, "seed": seed,
                            "proposer": pname, "gate": gname,
                            "best_acc": r["best_acc"], "best_test_acc": r.get("best_test_acc"),
                            "real_eval_count": r["real_eval_count"],
                            # number of gate SCORINGS (archtree's own proxy_eval_count counts only the
                            # skipped proposals, so it is recomputed here on the same footing)
                            "proxy_eval_count": (max(0, budget - warmup) if gate is not None else 0)
                            if pname == "archtree" else r.get("proxy_eval_count", 0),
                            "gate_labels_used": getattr(gate, "labels_used", 0) if gate is not None else 0,
                            "train_hours": r.get("train_hours", np.nan),
                            "wall_time_sec": time.time() - t0})
                        logger.info(f"[G] {dataset} b={budget} s={seed} {pname:8s} gate={gname:8s} "
                                    f"best={r['best_acc']:.2f} real={r['real_eval_count']}")
                        gc.collect()


def _holm(p):
    p = np.asarray(p, float)
    o, m, out, run_ = np.argsort(p), len(p), np.empty(len(p)), 0.0
    for k, i in enumerate(o):
        run_ = max(run_, (m - k) * p[i])
        out[i] = min(1.0, run_)
    return out


def summarize():
    p = PROJECT_ROOT / "outputs" / PHASE / "results" / RUNS
    if not p.exists():
        logger.error("no runs yet")
        return
    df = storage.to_report_metric(pd.read_csv(p)).drop_duplicates(
        ["dataset", "budget", "seed", "proposer", "gate"], keep="last")
    res = storage.get_output_dir(PHASE, "results")
    summ = (df.groupby(["dataset", "budget", "gate", "proposer"])
              .agg(n=("seed", "nunique"), acc_mean=("best_acc", "mean"), acc_std=("best_acc", "std"),
                   real_evals=("real_eval_count", "mean"), proxy_evals=("proxy_eval_count", "mean"),
                   labels=("gate_labels_used", "first"),
                   train_hours=("train_hours", "mean") if "train_hours" in df else ("seed", "size"))
              .reset_index())
    summ.to_csv(res / "gated_summary.csv", index=False)

    rows = []
    rng = np.random.default_rng(0)
    for (ds, b, g), grp in df.groupby(["dataset", "budget", "gate"]):
        piv = grp.pivot(index="seed", columns="proposer", values="best_acc")
        if "archtree" not in piv:
            continue
        for other in [c for c in piv.columns if c != "archtree"]:
            both = piv[["archtree", other]].dropna()
            if len(both) < 2:
                continue
            d = (both["archtree"] - both[other]).values
            try:
                pv = wilcoxon(both["archtree"], both[other]).pvalue if np.any(d != 0) else 1.0
            except ValueError:
                pv = 1.0
            boots = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
            rows.append({"dataset": ds, "budget": b, "gate": g, "other": other, "n": len(both),
                         "mean_diff_archtree_minus_other": float(d.mean()),
                         "ci95_lo": float(np.percentile(boots, 2.5)), "ci95_hi": float(np.percentile(boots, 97.5)),
                         "p_value": pv})
    t = pd.DataFrame(rows)
    if len(t):
        # Holm within each comparison family (one gate x one other proposer = 3 datasets x budgets).
        # Oct 3: a single Holm over all 192 tests was over-conservative and hid that regularized
        # evolution beats v1 ArchTree at budgets >= 100.
        t["p_holm"] = t.groupby(["gate", "other"])["p_value"].transform(lambda s: _holm(s.values))
        t["outcome_holm"] = np.where(t["p_holm"] < 0.05,
                                     np.where(t["mean_diff_archtree_minus_other"] > 0, "win", "loss"), "tie")
        t["wilcoxon_p"] = t["p_value"]
        t["n_paired_seeds"] = t["n"]
        t["cohens_d"] = np.nan
        t["significant"] = t["p_holm"] < 0.05
    t.to_csv(res / "gated_paired_tests.csv", index=False)

    logger.info("=" * 78)
    logger.info("Does the trie add anything once every method gets the same gate? (report metric)")
    for (ds, g), grp in t.groupby(["dataset", "gate"]) if len(t) else []:
        c = grp["outcome_holm"].value_counts()
        logger.info(f"  {ds:15s} gate={g:9s} archtree vs flat proposers: wins={c.get('win', 0)} "
                    f"ties={c.get('tie', 0)} losses={c.get('loss', 0)} (of {len(grp)}, Holm per family)")
    logger.success("gated_summary.csv / gated_paired_tests.csv written.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--proposers", nargs="+", default=PROPOSERS, choices=PROPOSERS)
    ap.add_argument("--gates", nargs="+", default=GATES, choices=GATES)
    ap.add_argument("--datasets", nargs="+", default=cfg["benchmarks"]["datasets"])
    ap.add_argument("--budgets", nargs="+", type=int, default=cfg["search"]["budgets"])
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg["search"]["seeds"])
    a = ap.parse_args()
    if not a.summarize_only:
        run(a)
    summarize()
