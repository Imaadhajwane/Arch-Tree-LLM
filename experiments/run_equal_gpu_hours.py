# experiments/run_equal_gpu_hours.py
# isort: skip_file
"""
Equal-GPU-hour comparison, proxy cost analysis and the gate-cost model (Oct 3, 2026; v2).

Why: every other experiment compares methods at an equal number of PROPOSALS / real evaluations.
But a zero-cost gate prefers larger networks (zero-cost proxies are largely cost detectors), so one
gated evaluation costs more training time than an ungated one. The fair currency is GPU-hours
(the benchmark's recorded training time of the architectures that were really evaluated).

Protocol
    Each (benchmark, dataset, seed, arm) is run ONCE with a large proposal budget, and its
    evaluation history is cut at GPU-hour budgets  H_k = k x (mean training time of one
    architecture of that benchmark/dataset),  k in K (default 25, 50, 100). The reported number is
    the TEST accuracy of the best-VALIDATION architecture evaluated within H_k.
    Cutting a long run is exactly equivalent to running with budget H_k: no arm uses the budget
    to plan (ArchTree with alpha=0, random and local search ignore it; RE's population is fixed at
    20 for any budget >= 80). A row is flagged `under_budget` if a run ran out of proposals before
    spending H_k (should not happen with the default proposal budgets).

Benchmarks: nb201 (3 datasets), nb101 (cifar10), natssss (NATS-Bench size space, 3 datasets;
skipped with a warning until data/natssss_cache.json exists).

Arms (gate rule = the usual one: 10 warm-up proposals, then top-35% of the scores seen so far)
    archtree_zc_unsup     ArchTree v2 + label-free ZC gate                    <- headline, reference
    archtree_none         ArchTree v2, no gate
    archtree_params       ArchTree v2 + #params gate (proxies ~ cost detectors: is cost alone enough?)
    archtree_costpen      ArchTree v2 + COST-PENALISED gate z(ens) - 1.00 z(log(1 + #params))
    archtree_costpen025   same with lambda = 0.25
    re_none, re_zc_unsup  regularized evolution without / with the same gate
    re_zcwarm             RE + zero-cost warm-up (Abdelfattah et al. 2021): initial population =
                          top-20 of N label-free-scored random archs (closest prior work)
    ls_none, ls_zc_unsup  local search (White et al. 2021) without / with the same gate
    random_none, random_zc_unsup, random_costpen, random_costpen025
                          random search; the costpen arms test the gate-cost model's predictions
    (nb101 runs a reduced set - every new NB101 proposal needs a GPU proxy computation)

Cost analysis (cost_correlation.csv): for each proxy and #params, Spearman rho with training time,
Kendall tau with validation accuracy, cost of the top-35% by score vs the truly best 35%.
Gate-cost model (tools/gate_cost_model.py, gate_cost_model.csv): Gaussian-copula fit of
(valid acc, test acc, proxy, cost, #params) on the same sample; closed-form criterion
r_PA r_PD - r_DA (> 0: a small cost penalty helps) and simulated delta of the cost-penalised gate.
gate_cost_validation.csv: predicted vs measured delta (random_costpen* and archtree_costpen* vs
their plain-gate arm) - predict-then-confirm.
Samples: NB201 all 15,625 archs (disk cache); NB101 / NATS-SSS a fixed random sample of 2,000 archs
(scored on the GPU the first time, cached afterwards; also the warm-up pool of re_zcwarm there).

Outputs (outputs/equal_gpu_hours/results/)
    egh_runs.csv              one row per (benchmark, dataset, seed, arm, k)      [resumable per run]
    egh_summary.csv           mean/std test accuracy, real evals, GPU-hours used
    egh_paired_tests.csv      every arm vs archtree_zc_unsup, Wilcoxon, Holm within each arm (family)
    cost_correlation.csv      proxy / cost analysis
    gate_cost_model.csv       criterion + predictions per (benchmark, dataset, lambda, k)
    gate_cost_validation.csv  predicted vs measured delta of the cost-penalised gate

Speed (Oct 4, 2026): --workers N runs N searches in parallel processes (default config
runtime.workers); every run seeds its own RNGs, so results are identical to --workers 1. The
per-iteration TCR curve, the per-run JSON log and the weight-bank file are not written here (not
used by this study); only the main process writes egh_runs.csv. Pre-compute the NATS-SSS proxies
first with tools/precompute_proxies.py (pipeline step 25) so the workers only read the cache.

Usage
    python experiments/run_equal_gpu_hours.py                      # all benchmarks (GPU for NB101/SSS proxies)
    python experiments/run_equal_gpu_hours.py --workers 4
    python experiments/run_equal_gpu_hours.py --benchmarks nb201
    python experiments/run_equal_gpu_hours.py --summarize-only
"""
import argparse
import gc
import itertools
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
from scipy.stats import kendalltau, spearmanr, wilcoxon  # noqa: E402
from dotenv import load_dotenv    # noqa: E402
load_dotenv()

import archtree.storage as storage   # noqa: E402
from archtree import settings        # noqa: E402

with open(PROJECT_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

PHASE = "equal_gpu_hours"
RUNS = "egh_runs.csv"
REF = "archtree_zc_unsup"
K_DEFAULT = [25, 50, 100]
PROPOSALS = {"nb201": 500, "nb101": 400, "natssss": 500}   # >= 2x what k=100 needs (under_budget flags it)
SAMPLE_N = {"nb101": 2000, "natssss": 2000}                # cost analysis / model / warm-up pool
WARMUP_N = 1000                                            # re_zcwarm: archs scored per run (<= sample)
LAMBDAS = [0.0, 0.25, 1.0]
ARMS = {   # name -> (proposer, gate, lambda)
    "archtree_zc_unsup":   ("archtree", "zc_unsup", 0.0),
    "archtree_none":       ("archtree", "none", 0.0),
    "archtree_params":     ("archtree", "params", 0.0),
    "archtree_costpen":    ("archtree", "costpen", 1.0),
    "archtree_costpen025": ("archtree", "costpen", 0.25),
    "re_none":             ("re", "none", 0.0),
    "re_zc_unsup":         ("re", "zc_unsup", 0.0),
    "re_zcwarm":           ("re_zcwarm", "none", 0.0),
    "ls_none":             ("ls", "none", 0.0),
    "ls_zc_unsup":         ("ls", "zc_unsup", 0.0),
    "random_none":         ("random", "none", 0.0),
    "random_zc_unsup":     ("random", "zc_unsup", 0.0),
    "random_costpen":      ("random", "costpen", 1.0),
    "random_costpen025":   ("random", "costpen", 0.25),
}
ARMSETS = {
    "nb201": list(ARMS),
    "natssss": list(ARMS),
    "nb101": ["archtree_zc_unsup", "archtree_none", "archtree_params", "archtree_costpen",
              "re_none", "re_zc_unsup", "re_zcwarm", "ls_none", "ls_zc_unsup", "random_none"],
}
NB201_PARAM_W = {"none": 0, "skip_connect": 0, "avg_pool_3x3": 0, "conv_1x1": 1, "conv_3x3": 9}

logger.remove()
LOG_DIR = PROJECT_ROOT / "outputs" / PHASE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>")
logger.add(LOG_DIR / "egh_run.log", level="DEBUG", rotation="10 MB", retention=5)


# ── benchmark helpers (one interface for NB201, NB101, NATS-SSS) ─────────────
class Bench:
    def __init__(self, name):
        self.name = name
        if name == "nb201":
            from benchmarks.nb201 import NB201
            self.b = NB201(cfg)
        elif name == "nb101":
            from benchmarks.nb101 import NB101
            self.b = NB101(cfg)
        else:
            from benchmarks.natssss import NATSSSS
            self.b = NATSSSS(cfg)
        self.datasets = list(self.b.datasets)
        self.b._load()
        self._mean_h, self._sample = {}, None

    @property
    def zc_phase(self):
        return {"nb201": "phase2", "nb101": "nb101", "natssss": "natssss"}[self.name]

    def idx(self, arch):
        """history `arch` field (token string or index string) -> index."""
        if self.name == "nb201":
            from benchmarks.nb201 import arch_key_to_index
            return arch_key_to_index(arch)
        return self.b.arch_key_to_index(arch)

    def hours(self, idx, ds):
        t = self.b.train_sec(idx, ds)
        return float(t or 0.0) / 3600.0

    def params(self, tokens):
        if self.name == "nb201":
            return float(sum(NB201_PARAM_W[t] for t in tokens))
        return float(self.b.params(self.b.tokens_to_index(tokens)))

    def mean_hours(self, ds):
        """Mean training time of one architecture (whole space; 20k-sample for NB101)."""
        if ds not in self._mean_h:
            if self.name == "nb101":
                ids = np.random.default_rng(0).choice(self.b.total_archs, 20000, replace=False)
            else:
                ids = range(self.b.total_archs)
            self._mean_h[ds] = float(np.mean([self.hours(int(i), ds) for i in ids]))
        return self._mean_h[ds]

    def sample_ids(self):
        """Architectures for the cost analysis / gate model / warm-up pool."""
        if self._sample is None:
            if self.name == "nb201":
                self._sample = list(range(self.b.total_archs))
            else:
                self._sample = [int(i) for i in np.random.default_rng(7).choice(
                    self.b.total_archs, SAMPLE_N[self.name], replace=False)]
        return self._sample

    def best(self, history, ds):
        if not history:
            return np.nan, np.nan
        e = max(history, key=lambda x: x["acc"])
        return float(e["acc"]), float(self.b.query_test(self.idx(e["arch"]), ds))


# ── gates ────────────────────────────────────────────────────────────────────
class ParamsGate:
    labels_used = 0

    def __init__(self, bench):
        self.bench = bench

    def score(self, tokens):
        return self.bench.params(tokens)


class CostPenalisedGate:
    """z(label-free ensemble) - lam * z(log(1 + #params)); the z of the cost term comes from a
    label-free sample of the space (all NB201 / NATS-SSS archs, 20k random NB101 archs)."""
    labels_used = 0

    def __init__(self, base, bench, lam):
        self.base, self.bench, self.lam = base, bench, lam
        if bench.name == "nb201":
            c = [sum(NB201_PARAM_W[o] for o in a) for a in itertools.product(list(NB201_PARAM_W), repeat=6)]
        elif bench.name == "natssss":
            c = [bench.b.params(i) for i in range(bench.b.total_archs)]
        else:
            ids = np.random.default_rng(1).choice(bench.b.total_archs, 20000, replace=False)
            c = [bench.b.params(int(i)) for i in ids]
        lc = np.log1p(np.asarray(c, float))
        self.mu, self.sd = float(lc.mean()), float(lc.std() + 1e-8)

    def score(self, tokens):
        return self.base.score(tokens) - self.lam * (np.log1p(self.bench.params(tokens)) - self.mu) / self.sd


def zc_ensemble(bench, ds, cache):
    from archtree.zc_proxy import ZCProxyEnsemble
    key = (bench.name, ds)
    if key not in cache:
        cache[key] = ZCProxyEnsemble.unsupervised(bench.b, ds, phase=bench.zc_phase)
    return cache[key]


def make_gate(kind, lam, bench, ds, cache):
    if kind == "none":
        return None
    if kind == "params":
        return ParamsGate(bench)
    if kind == "zc_unsup":
        return zc_ensemble(bench, ds, cache)
    if kind == "costpen":
        return CostPenalisedGate(zc_ensemble(bench, ds, cache), bench, lam)
    raise ValueError(kind)


# ── searches that return the full evaluation history ─────────────────────────
_CAPTURED = {}
_orig_save_run_log = storage.save_run_log
# This study writes only egh_runs.csv (main process). The searches' own side files (raw run rows,
# dedup / cache stats, TCR curves, weight banks) are not used here and would be written
# concurrently by parallel workers, so they are switched off for this script.
storage.append_result_row = lambda *a, **k: None
storage.append_result_row_buffered = lambda *a, **k: None
storage.save_weight_bank = lambda *a, **k: None
storage.load_weight_bank = lambda *a, **k: {}
# The (unused, LLM-off) LLM prior also checkpoints its call counter per (benchmark, dataset, seed);
# two arms of the same seed in parallel workers collided on that file on Windows (WinError 32,
# 1 of 1128 runs on Oct 4). Not needed here -> switched off as well.
from utils.checkpoint import Checkpoint  # noqa: E402
Checkpoint.save = lambda self, *a, **k: None


def _capture_save_run_log(phase, method, bench, dataset, seed, log, budget=None):
    _CAPTURED["log"] = log                      # the JSON log is not written (thousands of runs)


def _llm_free_backend_kwargs():
    return dict(backend="heuristic")            # LLMPrior is built but never asked (alpha = 0)


def run_archtree(bench, ds, seed, gate, proposals):
    from archtree.search import search_phase3
    storage.save_run_log = _capture_save_run_log
    try:
        search_phase3(bench.b, proposals, ds, seed, phase=PHASE, alpha=0.0, use_llm=False,
                      use_proxy=gate is not None, zc_override=gate, ablation_id="EGH",
                      proxy_threshold_percentile=settings.PROXY_THRESHOLD_PERCENTILE,
                      backup_mode=settings.BACKUP_MODE_P34,
                      results_filename="EGH_raw_runs.csv", tcr_filename=None,
                      cache_stats_filename="EGH_raw_cache_stats.csv",
                      dedup_filename="EGH_dedup_stats.csv", **_llm_free_backend_kwargs())
    finally:
        storage.save_run_log = _orig_save_run_log
    return _CAPTURED.pop("log")["eval_history"]


class _Random:
    def __init__(self, bench, seed):
        self.bench, self.rng = bench, random.Random(seed)

    def propose(self, evaluated):
        b = self.bench.b
        if hasattr(b, "random_unevaluated_index"):
            return b.index_to_tokens(b.random_unevaluated_index(evaluated, self.rng))
        while True:
            i = self.rng.randrange(b.total_archs)
            if i not in evaluated:
                return b.index_to_tokens(i)

    def update(self, tokens, acc):
        pass


def _warmup_pool(bench, ds, seed, cache):
    """re_zcwarm: top-20 of WARMUP_N label-free-scored random archs (per seed)."""
    from baselines.regularized_evolution import zc_warmup_pool, default_sizes
    ens = zc_ensemble(bench, ds, cache)
    pop, _ = default_sizes(PROPOSALS[bench.name])
    if bench.name == "nb201":
        pool, _ = zc_warmup_pool(bench.b, ds, seed, pop, n=WARMUP_N, gate=ens)
        return pool
    cand = random.Random(10_000 + seed).sample(bench.sample_ids(), min(WARMUP_N, len(bench.sample_ids())))
    toks = [bench.b.index_to_tokens(i) for i in cand]
    return sorted(toks, key=ens.score, reverse=True)[:pop]


def run_flat(bench, ds, seed, proposer, gate, proposals, cache):
    from baselines.regularized_evolution import RegularizedEvolution
    from baselines.local_search import LocalSearch
    random.seed(seed)
    if proposer == "random":
        prop = _Random(bench, seed)
    elif proposer == "ls":
        prop = LocalSearch(bench.b, seed)
    elif proposer == "re_zcwarm":
        prop = RegularizedEvolution(bench.b, seed, proposals, init_pool=_warmup_pool(bench, ds, seed, cache))
    else:
        prop = RegularizedEvolution(bench.b, seed, proposals)
    evaluated, scores, hist = set(), [], []
    pct, warmup = settings.PROXY_THRESHOLD_PERCENTILE, settings.WARMUP_REAL_EVALS
    for i in range(proposals):
        toks = prop.propose(evaluated)
        if gate is not None and i >= warmup:
            s = gate.score(toks)
            scores.append(s)
            if s < np.percentile(scores, 100 - pct):
                continue
        idx = bench.b.tokens_to_index(toks)
        acc = bench.b.query(idx, ds)
        evaluated.add(idx)
        prop.update(toks, acc)
        hist.append({"iter": i + 1, "arch": "|".join(toks) if bench.name != "nb101" else str(idx), "acc": acc})
    return hist


def truncate(bench, hist, ds, budget_h):
    used, out = 0.0, []
    for e in hist:
        c = bench.hours(bench.idx(e["arch"]), ds)
        if used + c > budget_h:
            return out, used, False
        used += c
        out.append(e)
    return out, used, True                     # ran out of history before the budget


# ── main loop ────────────────────────────────────────────────────────────────
def _done():
    p = PROJECT_ROOT / "outputs" / PHASE / "results" / RUNS
    if not p.exists():
        return set()
    d = pd.read_csv(p)
    return set(zip(d["benchmark"], d["dataset"], d["seed"], d["arm"]))


def _benches(names):
    for n in names:
        try:
            yield Bench(n)
        except FileNotFoundError as e:
            logger.warning(f"[{n}] skipped: {e}")


_W = {}   # per-process state: benchmarks and zero-cost ensembles (built once per worker)


def _worker_init():
    import torch
    torch.set_num_threads(1)                    # N processes: no CPU oversubscription


def _bench_cached(name):
    if name not in _W:
        _W[name] = Bench(name)
    return _W[name]


def _run_job(job):
    """One (benchmark, dataset, seed, arm) run -> (job, rows, message). Pure function of the job."""
    bname, ds, seed, arm, ks = job
    bench = _bench_cached(bname)
    zc_cache = _W.setdefault("_zc", {})
    proposals = PROPOSALS[bname]
    mean_h = bench.mean_hours(ds)
    proposer, gkind, lam = ARMS[arm]
    t0 = time.time()
    try:
        from archtree.zc_proxy import get_mini_batch
        get_mini_batch()
        gate = make_gate(gkind, lam, bench, ds, zc_cache)
        hist = (run_archtree(bench, ds, seed, gate, proposals) if proposer == "archtree"
                else run_flat(bench, ds, seed, proposer, gate, proposals, zc_cache))
    except Exception as e:                      # noqa: BLE001
        import traceback
        return job, [], f"ERROR {bname}/{ds} s={seed} {arm}: {e}\n{traceback.format_exc()}"
    rows, msg = [], []
    for k in ks:
        tr, used, short = truncate(bench, hist, ds, k * mean_h)
        val, test = bench.best(tr, ds)
        rows.append({"benchmark": bname, "dataset": ds, "seed": seed, "arm": arm, "k": k,
                     "gpu_hours_budget": k * mean_h, "gpu_hours_used": used,
                     "real_eval_count": len(tr), "best_acc": val, "best_test_acc": test,
                     "under_budget": bool(short), "proposals": proposals,
                     "wall_time_sec": time.time() - t0})
        msg.append(f"k={k}:{test:.2f}/{len(tr)}{' (UNDER BUDGET)' if short else ''}")
    return job, rows, f"[EGH] {bname}/{ds} s={seed} {arm:19s} " + "  ".join(msg)


def run(args):
    done = _done()
    jobs = []
    for bench in _benches(args.benchmarks):
        _W[bench.name] = bench
        for ds in bench.datasets:
            mean_h = bench.mean_hours(ds)
            logger.info(f"[{bench.name}/{ds}] mean training time of one architecture = {mean_h:.4f} h; "
                        f"budgets = {[f'{k * mean_h:.1f} h' for k in args.k]}")
            for seed in args.seeds:
                for arm in ARMSETS[bench.name]:
                    if arm in args.arms and (bench.name, ds, seed, arm) not in done:
                        jobs.append((bench.name, ds, seed, arm, tuple(args.k)))
    logger.info(f"{len(jobs)} runs to do ({len(done)} done), {args.workers} worker(s)")
    if not jobs:
        return
    t0, n = time.time(), 0

    def _collect(result):
        nonlocal n
        job, rows, msg = result
        n += 1
        if rows:
            storage_rows(rows)
            logger.info(msg)
        else:
            logger.error(msg)
        if n % 50 == 0:
            el = time.time() - t0
            logger.info(f"progress {n}/{len(jobs)} runs, {el / 60:.1f} min, ~{el / n * (len(jobs) - n) / 60:.0f} min left")

    try:
        if args.workers <= 1:
            for job in jobs:
                _collect(_run_job(job))
        else:
            from multiprocessing import get_context
            with get_context("spawn").Pool(args.workers, initializer=_worker_init) as pool:
                for result in pool.imap_unordered(_run_job, jobs, chunksize=2):
                    _collect(result)
    except KeyboardInterrupt:
        logger.warning("Interrupted - finished runs are saved, rerun to resume.")
        sys.exit(0)


def storage_rows(rows):
    """The only writer of egh_runs.csv (main process)."""
    from archtree.storage import append_result_rows
    append_result_rows(PHASE, RUNS, rows)


# ── cost analysis + gate-cost model ──────────────────────────────────────────
def cost_analysis(args):
    from archtree.zc_proxy import get_mini_batch
    from tools.gate_cost_model import fit_copula, criterion, predict
    get_mini_batch()
    rows, model_rows = [], []
    zc_cache = {}
    for bench in _benches(args.benchmarks):
        ids = bench.sample_ids()
        toks = [bench.b.index_to_tokens(i) for i in ids]
        logger.info(f"[cost] {bench.name}: proxy scores for {len(ids)} archs (disk cache; GPU for new ones)")
        raw = None
        for ds in bench.datasets:
            ens = zc_ensemble(bench, ds, zc_cache)
            if raw is None:
                raw = [ens.raw(t) for t in toks]      # E, S, T, C do not depend on the dataset
            cost = np.array([bench.hours(i, ds) for i in ids])
            val = np.array([bench.b.query(i, ds) for i in ids])
            test = np.array([bench.b.query_test(i, ds) for i in ids])
            scores = {k: np.array([r[k] for r in raw]) for k in ("E", "S", "T", "C")}
            scores["ensemble_unsup"] = np.array([ens._aggregate(r) for r in raw])
            scores["params"] = np.array([bench.params(t) for t in toks])
            orc = val >= np.percentile(val, 65)
            orc_ratio = cost[orc].mean() / cost.mean()
            for name, s in scores.items():
                top = s >= np.percentile(s, 65)
                rows.append({"benchmark": bench.name, "dataset": ds, "score": name, "n_archs": len(ids),
                             "spearman_with_train_time": spearmanr(s, cost)[0],
                             "kendall_with_valid_acc": kendalltau(s, val)[0],
                             "top35_cost_ratio": cost[top].mean() / cost.mean(),
                             "oracle_top35_cost_ratio": orc_ratio,
                             "excess_cost_vs_oracle": cost[top].mean() / cost[orc].mean() - 1.0,
                             "top35_mean_valid_acc": val[top].mean()})
            rows.append({"benchmark": bench.name, "dataset": ds, "score": "valid_acc (oracle)", "n_archs": len(ids),
                         "spearman_with_train_time": spearmanr(val, cost)[0], "kendall_with_valid_acc": 1.0,
                         "top35_cost_ratio": orc_ratio, "oracle_top35_cost_ratio": orc_ratio,
                         "excess_cost_vs_oracle": 0.0, "top35_mean_valid_acc": val[orc].mean()})
            # gate-cost model on the same sample
            mdl = fit_copula(val, test, scores["ensemble_unsup"], cost, np.log1p(scores["params"]))
            cr = criterion(mdl)
            for o in predict(mdl, LAMBDAS, args.k, q=settings.PROXY_THRESHOLD_PERCENTILE / 100,
                             warmup=settings.WARMUP_REAL_EVALS):
                model_rows.append({"benchmark": bench.name, "dataset": ds, **cr, **o})
            logger.info(f"[model] {bench.name}/{ds}: r_PA={cr['r_PA']:.2f} r_PD={cr['r_PD']:.2f} "
                        f"r_DA={cr['r_DA']:.2f} -> slope {cr['slope_at_0']:+.3f} "
                        f"({'a small cost penalty should HELP' if cr['penalty_helps_selection'] else 'cost penalty should HURT'})")
    res = storage.get_output_dir(PHASE, "results")
    t = pd.DataFrame(rows)
    t.to_csv(res / "cost_correlation.csv", index=False)
    pd.DataFrame(model_rows).to_csv(res / "gate_cost_model.csv", index=False)
    logger.info("Proxy vs cost (rho with training time | tau with valid acc | top-35% cost ratio):")
    for r in t.itertuples():
        logger.info(f"  {r.benchmark}/{r.dataset:15s} {r.score:18s} rho={r.spearman_with_train_time:+.2f} "
                    f"tau={r.kendall_with_valid_acc:+.2f} cost x{r.top35_cost_ratio:.2f} "
                    f"(oracle x{r.oracle_top35_cost_ratio:.2f})")


# ── statistics ───────────────────────────────────────────────────────────────
def _holm(p):
    p = np.asarray(p, float)
    o, m, out, run_ = np.argsort(p), len(p), np.empty(len(p)), 0.0
    for k, i in enumerate(o):
        run_ = max(run_, (m - k) * p[i])
        out[i] = min(1.0, run_)
    return out


def _validate_model(df, res):
    """Predicted vs measured delta of the cost-penalised gate (vs the plain gate, same proposer)."""
    mp = res / "gate_cost_model.csv"
    if not mp.exists():
        return
    model = pd.read_csv(mp)
    rows = []
    for prop in ("random", "archtree"):
        for lam, arm in ((1.0, f"{prop}_costpen"), (0.25, f"{prop}_costpen025")):
            base = f"{prop}_zc_unsup"
            for (bn, ds, k), g in df.groupby(["benchmark", "dataset", "k"]):
                piv = g.pivot(index="seed", columns="arm", values="best_test_acc")
                if arm not in piv or base not in piv:
                    continue
                both = piv[[arm, base]].dropna()
                if len(both) < 2:
                    continue
                d = both[arm] - both[base]
                pred = model[(model.benchmark == bn) & (model.dataset == ds) & (model["lambda"] == lam) &
                             (model.k == k)]
                rows.append({"benchmark": bn, "dataset": ds, "k": k, "proposer": prop, "lambda": lam,
                             "measured_delta": float(d.mean()),
                             "measured_p": wilcoxon(both[arm], both[base]).pvalue if np.any(d != 0) else 1.0,
                             "predicted_delta": float(pred["pred_delta_vs_lambda0"].iloc[0]) if len(pred) else np.nan,
                             "criterion_slope": float(pred["slope_at_0"].iloc[0]) if len(pred) else np.nan})
    v = pd.DataFrame(rows)
    v.to_csv(res / "gate_cost_validation.csv", index=False)
    if len(v):
        for prop, g in v.groupby("proposer"):
            g = g.dropna(subset=["predicted_delta"])
            if len(g) < 3:
                continue
            agree = np.mean(np.sign(g.measured_delta) == np.sign(g.predicted_delta))
            rho = spearmanr(g.predicted_delta, g.measured_delta)[0]
            logger.info(f"[model check] {prop}: predicted vs measured delta of the cost penalty: sign agreement "
                        f"{agree:.0%}, Spearman {rho:+.2f} over {len(g)} (benchmark, dataset, k, lambda) cells")


def summarize():
    p = PROJECT_ROOT / "outputs" / PHASE / "results" / RUNS
    if not p.exists():
        logger.error("no runs yet")
        return
    df = pd.read_csv(p).drop_duplicates(["benchmark", "dataset", "seed", "arm", "k"], keep="last")
    res = storage.get_output_dir(PHASE, "results")
    (df.groupby(["benchmark", "dataset", "k", "arm"])
       .agg(n=("seed", "nunique"), test_mean=("best_test_acc", "mean"), test_std=("best_test_acc", "std"),
            real_evals=("real_eval_count", "mean"), gpu_hours_used=("gpu_hours_used", "mean"),
            gpu_hours_budget=("gpu_hours_budget", "mean"), under_budget=("under_budget", "sum"))
       .reset_index().to_csv(res / "egh_summary.csv", index=False))
    rows = []
    for (bn, ds, k), g in df.groupby(["benchmark", "dataset", "k"]):
        piv = g.pivot(index="seed", columns="arm", values="best_test_acc")
        if REF not in piv:
            continue
        for arm in [a for a in piv.columns if a != REF]:
            both = piv[[arm, REF]].dropna()
            if len(both) < 2:
                continue
            d = (both[REF] - both[arm]).values
            try:
                pv = wilcoxon(both[REF], both[arm]).pvalue if np.any(d != 0) else 1.0
            except ValueError:
                pv = 1.0
            sd = d.std(ddof=1)
            rows.append({"benchmark": bn, "dataset": ds, "budget": k, "k_mean_arch_costs": k,
                         "other": arm, "n_paired_seeds": len(both),
                         "mean_diff_archtree_minus_other": float(d.mean()),
                         "wilcoxon_p": pv, "cohens_d": float(d.mean() / sd) if sd > 0 else 0.0})
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm"] = t.groupby("other")["wilcoxon_p"].transform(lambda s: _holm(s.values))
        t["significant"] = t["p_holm"] < 0.05
        t["outcome_holm"] = np.where(t["significant"], np.where(
            t["mean_diff_archtree_minus_other"] > 0, "win", "loss"), "tie")
    t.to_csv(res / "egh_paired_tests.csv", index=False)
    logger.info("=" * 78)
    logger.info(f"Equal GPU-hours: {REF} vs each arm (Holm within arm, all benchmarks):")
    for arm, g in (t.groupby("other") if len(t) else []):
        c = g["outcome_holm"].value_counts()
        logger.info(f"  vs {arm:19s} mean {g['mean_diff_archtree_minus_other'].mean():+.3f} pts  "
                    f"wins={c.get('win', 0)} ties={c.get('tie', 0)} losses={c.get('loss', 0)} (of {len(g)})")
    if df["under_budget"].any():
        logger.warning(f"{int(df['under_budget'].sum())} rows ran out of proposals before their budget.")
    _validate_model(df, res)
    logger.success("egh_summary.csv / egh_paired_tests.csv / gate_cost_validation.csv written.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--skip-cost-analysis", action="store_true")
    ap.add_argument("--benchmarks", nargs="+", default=["nb201", "nb101", "natssss"],
                    choices=["nb201", "nb101", "natssss"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--k", nargs="+", type=int, default=K_DEFAULT)
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg["search"]["seeds"])
    ap.add_argument("--workers", type=int, default=int(cfg.get("runtime", {}).get("workers", 1)))
    a = ap.parse_args()
    if not a.summarize_only:
        if not a.skip_cost_analysis:
            cost_analysis(a)          # first: also builds the NB101 / SSS calibration + proxy cache
        run(a)
    summarize()
