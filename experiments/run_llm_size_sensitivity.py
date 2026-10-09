# experiments/run_llm_size_sensitivity.py
# isort: skip_file
"""
Weaker-LLM sensitivity (e.g. a 3B model via Ollama).

Two measurements per model, because the prior's effect on search results is
small and noisy (it only orders never-visited siblings), so search results
alone can hide a large difference in prior quality.

1. PRIOR-QUALITY PROBE (LLM only, no search)
   For sampled prefixes of depth 0-3, the model scores the 5 next ops. Scores
   are compared with the ground truth: the mean accuracy of all leaves under
   prefix+op, computed from the benchmark table.
     spearman          rank agreement with the truth (chance = 0)
     top1_hit          model's best op == truly best op (chance = 0.2)
     uniform_fallback  unparseable/empty reply -> LLMPrior returned uniform scores
     name_mismatch_ops returned keys that were not the requested op names
     latency_sec       wall time of the call

2. SEARCH
   search_phase3(alpha=ALPHA, use_llm=True) per seed vs an LLM-off arm
   (alpha=0) with identical settings. Paired Wilcoxon + Cohen's d per model,
   plus best@25 (cold start) from each run's log.

ALPHA: any alpha >= 0.2 keeps the LLM gate open for the whole run, so 0.3
means "prior fully on".

Model spec syntax  backend:model, e.g.
    groq:openai/gpt-oss-20b    ollama:llama3.1:8b    ollama:llama3.2:3b
Ollama models must be pulled first. A model whose server/model check fails is
skipped with a message.

Cost: probe = up to ~4*N calls per model per dataset; search = up to
max_calls per run per seed per model, from a cold cache (this study's
namespace is wiped before each run so timings/calls are real).

Outputs (outputs/llm_size/results/):
  llm_size_probe.csv, llm_size_search.csv, llm_size_summary.csv,
  llm_size_paired_tests.csv

Usage:
    python experiments/run_llm_size_sensitivity.py
    python experiments/run_llm_size_sensitivity.py --models ollama:llama3.1:8b ollama:llama3.2:3b --probe-only
    python experiments/run_llm_size_sensitivity.py --summarize-only
"""
import argparse
import gc
import itertools
import json
import os
import random
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402
import yaml                                     # noqa: E402
from loguru import logger                       # noqa: E402
from scipy.stats import spearmanr, wilcoxon     # noqa: E402
from dotenv import load_dotenv                  # noqa: E402
load_dotenv()

import archtree.storage as storage  # noqa: E402

with open(PROJECT_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

PHASE = "llm_size"
ALPHA = 0.3
OPS = ["none", "skip_connect", "conv_1x1", "conv_3x3", "avg_pool_3x3"]
L, K = 6, 5   # L = edges per cell (tree depth), K = number of ops (branching factor)
DEFAULT_MODELS = [f"groq:{cfg['llm']['groq']['model']}", "ollama:llama3.1:8b", "ollama:llama3.2:3b"]

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
LOG_DIR = PROJECT_ROOT / "outputs" / PHASE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>")
logger.add(LOG_DIR / "llm_size_run.log", level="DEBUG", rotation="10 MB", retention=5)


# ── Helpers ──────────────────────────────────────────────────────────────────
def parse_spec(spec: str):
    """'backend:model' -> (backend, model). Model may itself contain ':'."""
    backend, model = spec.split(":", 1)
    assert backend in ("groq", "ollama"), spec
    return backend, model


def tag_of(spec: str) -> str:
    """Filesystem/namespace-safe version of a model spec."""
    return re.sub(r"[^A-Za-z0-9.\-]+", "-", spec)


def prior_kwargs(backend, model) -> dict:
    """kwargs accepted by BOTH LLMPrior(...) and search_phase3(...)."""
    g = cfg["llm"]["groq"]
    if backend == "groq":
        pool = bool(os.environ.get("GROQ_API_KEYS_FILE", "").strip()
                    or os.environ.get("GROQ_API_KEYS", "").strip())
        if not (pool or os.environ.get("GROQ_API_KEY", "").strip()):
            raise RuntimeError("no GROQ_API_KEY / GROQ_API_KEYS / GROQ_API_KEYS_FILE in environment")
        return dict(backend="groq", groq_model=model,
                    groq_daily_token_budget=g.get("daily_token_budget", 190_000),
                    groq_daily_request_budget=g.get("daily_request_budget", 950),
                    groq_reasoning_effort=g.get("reasoning_effort", "low"), use_key_pool=pool)
    return dict(backend="ollama", model=model, host=cfg["llm"].get("host", "http://localhost:11434"))


def wipe_namespace(ns: str) -> None:
    """Delete this study's cached LLM answers / call counters / checkpoints for one
    namespace (all datasets and seeds), so every run starts with a cold cache."""
    for f in LOG_DIR.glob(f"*__{ns}.json"):
        if any(f.name.startswith(p) for p in ("llm_cache", "groq_calls", "llm_prior_ckpt")):
            f.unlink()


# ── Ground truth ─────────────────────────────────────────────────────────────
def subtree_means(acc: np.ndarray, prefix):
    """Mean accuracy of all leaves under prefix+[op], for each op.

    Assumes architecture index = base-K encoding of the L ops (OPS order),
    with token 0 most significant, so each subtree is a contiguous slice.
    """
    out = {}
    for oi, op in enumerate(OPS):
        digits = [OPS.index(t) for t in prefix] + [oi]
        base = 0
        for d in digits:
            base = base * K + d
        span = K ** (L - len(digits))   # leaves under this node
        lo = base * span
        out[op] = float(acc[lo:lo + span].mean())
    return out


_BENCH = None
_ACC_CACHE: dict = {}   # dataset -> accuracy table


def _acc_table(dataset):
    """Accuracy of every architecture on `dataset`; benchmark is loaded once."""
    global _BENCH
    if _BENCH is None:
        from benchmarks.nb201 import NB201
        _BENCH = NB201(cfg)
        _BENCH._load()
    if dataset not in _ACC_CACHE:
        _ACC_CACHE[dataset] = np.array(
            [_BENCH.query(i, dataset) for i in range(_BENCH.total_archs)], dtype=float)
    return _ACC_CACHE[dataset]


# ── 1. Probe ─────────────────────────────────────────────────────────────────
def probe(spec, datasets, n_per_depth, seed):
    """Score sampled prefixes with one model; return per-prefix quality rows."""
    from archtree.llm_prior import LLMPrior
    backend, model = parse_spec(spec)
    tag = tag_of(spec)
    kw = prior_kwargs(backend, model)
    rows = []

    for dataset in datasets:
        n_start = len(rows)
        acc = _acc_table(dataset)
        ns = f"probe_{tag}"
        wipe_namespace(ns)   # cold cache -> real latencies

        pk = dict(kw)
        if backend == "ollama":
            pk["base_url"] = pk.pop("host")   # LLMPrior calls it base_url
        prior = LLMPrior(phase=PHASE, bench="nb201", dataset=dataset, seed=seed, namespace=ns,
                         max_calls=4 * n_per_depth + 5, **pk)
        if not prior.verify_server():
            logger.warning(f"[probe] {spec}: server/model check failed — skipping.")
            return rows

        rng = random.Random(seed)
        for depth in range(4):
            # Distinct prefixes only: repeats would be cache hits (fake ~0 latency)
            # and would inflate n. Depth 0 has one prefix, depth 1 has five, etc.
            all_prefixes = list(itertools.product(OPS, repeat=depth))
            for prefix in rng.sample(all_prefixes, min(n_per_depth, len(all_prefixes))):
                prefix = list(prefix)
                truth = subtree_means(acc, prefix)
                mm0 = prior.name_mismatch_count
                t0 = time.time()
                fb0 = prior.fallback_count
                scores = prior.score_branches(prefix, list(OPS), alpha_t=1.0)   # alpha_t=1 keeps the gate open
                dt = time.time() - t0
                if prior.fallback_count > fb0:
                    n_fb = sum(r.get("uniform_fallback", False) for r in rows[n_start:]) + 1
                    if n_fb >= 3 and n_fb >= 0.5 * (len(rows) - n_start + 1):
                        logger.error(f"[probe] {spec} {dataset}: most calls are falling back to uniform "
                                     f"(keys exhausted / model error) - probe aborted, nothing written for it.")
                        return rows[:n_start]

                s = np.array([scores[o] for o in OPS])
                t = np.array([truth[o] for o in OPS])
                uniform = bool(np.ptp(s) < 1e-9)   # all-equal scores = parse failure fallback
                rho = np.nan if uniform else spearmanr(s, t)[0]
                rows.append({"model": spec, "dataset": dataset, "depth": depth,
                             "prefix": "|".join(prefix),
                             "spearman": rho,
                             "top1_hit": bool(np.argmax(s) == np.argmax(t)) if not uniform else False,
                             "uniform_fallback": uniform,
                             "name_mismatch_ops": prior.name_mismatch_count - mm0,
                             "latency_sec": dt})
        logger.info(f"[probe] {spec} {dataset}: {len(rows) - n_start} prefixes")
    return rows


# ── 2. Search ────────────────────────────────────────────────────────────────
def _best_at(path, k=25):
    """Best accuracy within the first k real evals of a run log (NaN if unreadable)."""
    try:
        with open(path) as fh:
            h = json.load(fh)["eval_history"]
        return float(max(e["acc"] for e in h[:k]))
    except Exception:
        return np.nan


def _done():
    """Return {(arm, dataset, budget, seed)} already in the search CSV."""
    p = PROJECT_ROOT / "outputs" / PHASE / "results" / "llm_size_search.csv"
    if not p.exists():
        return set()
    d = pd.read_csv(p)
    return set(zip(d["arm"], d["dataset"], d["budget"], d["seed"]))


def _llm_free_kwargs():
    """Backend kwargs for the LLM-off arm. The backend is never called, but
    search_phase3 still expects one; prefer ollama since it needs no API key."""
    try:
        import ollama  # noqa: F401
        return dict(backend="ollama")
    except ImportError:
        return prior_kwargs("groq", cfg["llm"]["groq"]["model"])


def search_runs(args):
    from archtree.search import search_phase3
    from benchmarks.nb201 import NB201
    bench = NB201(cfg)
    bench._load()
    pct = cfg.get("zc_proxy", {}).get("proxy_threshold_percentile", 35)
    done = _done()
    arms = [("off", None)] + [(tag_of(s), s) for s in args.models]   # (arm name, model spec)
    failed = set()   # models that produced LLM fallbacks this session - skipped, the others go on

    for dataset in args.datasets:
        for seed in args.seeds:
            for arm, spec in arms:
                if (arm, dataset, args.budget, seed) in done or spec in failed:
                    continue

                aid = f"LS_{arm}"   # per-arm cache namespace
                logger.info(f"[LS] {dataset} seed={seed} arm={arm}")
                try:
                    if spec is None:
                        kw = dict(alpha=0.0, use_llm=False, **_llm_free_kwargs())
                    else:
                        wipe_namespace(aid)   # cold cache so calls/timings are real
                        b, m = parse_spec(spec)
                        kw = dict(alpha=ALPHA, use_llm=True, **prior_kwargs(b, m))

                    t0 = time.time()
                    r = search_phase3(bench, args.budget, dataset, seed, phase=PHASE, ablation_id=aid,
                                      proxy_threshold_percentile=pct, max_llm_calls=cfg["llm"]["max_calls"],
                                      results_filename="LS_raw_runs.csv", tcr_filename="LS_raw_tcr.csv",
                                      cache_stats_filename="LS_raw_cache_stats.csv",
                                      dedup_filename="LS_dedup_stats.csv", **kw)

                    if spec is not None and r.get("llm_fallbacks", 0) > 0:
                        logger.error(f"[LS] {spec}: {r['llm_fallbacks']} LLM fallbacks - run discarded "
                                     f"(not valid LLM data). Skipping this model for the rest of this run "
                                     f"(rerun later to fill it); the other models continue.")
                        failed.add(spec)
                        continue
                    # Run log is only needed for best@25; needs the patched filename scheme.
                    logp = LOG_DIR / f"archtree_llm__{aid}__nb201__{dataset}__seed{seed}__budget{args.budget}.json"
                    storage.append_result_row(PHASE, "llm_size_search.csv", {
                        "arm": arm, "model": spec or "off", "dataset": dataset,
                        "budget": args.budget, "seed": seed,
                        "alpha": kw["alpha"], "best_acc": r.get("best_acc"),
                        "best_test_acc": r.get("best_test_acc"),
                        "llm_fallbacks": r.get("llm_fallbacks", 0),
                        "best_at_25": _best_at(logp),
                        "real_eval_count": r.get("real_eval_count"),
                        "llm_calls": r.get("llm_calls"), "cache_hits": r.get("cache_hits"),
                        "llm_sec": r.get("llm_sec"),
                        "name_mismatch_ops": r.get("name_mismatch_ops", 0),
                        "wall_time_sec": time.time() - t0})
                except KeyboardInterrupt:
                    # CSV is append-only, so rerunning resumes from here.
                    logger.warning("Interrupted — progress saved, rerun to resume.")
                    sys.exit(0)
                except Exception:
                    # Failed runs write no row, so they are retried next launch.
                    logger.exception("    ERROR:")
                finally:
                    gc.collect()


# ── Summary ──────────────────────────────────────────────────────────────────
def _wt(a, b):
    """Paired Wilcoxon p-value and Cohen's d (mean diff / sd of diffs)."""
    d = a - b
    if len(d) < 2 or np.all(d == 0):
        return np.nan, np.nan
    try:
        p = wilcoxon(a, b).pvalue
    except ValueError:
        p = np.nan
    sd = d.std(ddof=1)
    return p, (d.mean() / sd if sd > 0 else np.nan)


def summarize():
    res = storage.get_output_dir(PHASE, "results")
    pp, sp = res / "llm_size_probe.csv", res / "llm_size_search.csv"
    logger.info("=" * 78)

    # Probe summary per (model, dataset).
    if pp.exists():
        pr = pd.read_csv(pp)
        pr["spearman_fb0"] = pr["spearman"].fillna(0.0)   # count fallbacks as chance-level
        g = (pr.groupby(["model", "dataset"])
               .agg(n=("spearman", "size"),
                    spearman_mean=("spearman", "mean"),                       # skips fallbacks (NaN)
                    spearman_mean_fallback_as_0=("spearman_fb0", "mean"),
                    top1_hit_rate=("top1_hit", "mean"),
                    uniform_fallback_rate=("uniform_fallback", "mean"),
                    name_mismatch_per_call=("name_mismatch_ops", "mean"),
                    latency_mean_sec=("latency_sec", "mean")).reset_index())
        g.to_csv(res / "llm_size_summary.csv", index=False)

        logger.info("PROBE (chance: spearman 0.00, top1 0.20). Rows = model, dataset")
        for r in g.itertuples():
            logger.info(f"  {r.model:34s} {r.dataset:12s} rho={r.spearman_mean:+.2f}  top1={r.top1_hit_rate:.2f}  "
                        f"uniform_fallback={r.uniform_fallback_rate:.0%}  "
                        f"key_mismatch/call={r.name_mismatch_per_call:.2f}  {r.latency_mean_sec:.1f}s/call")
        by_depth = pr.groupby(["model", "depth"])["spearman"].mean().unstack()
        logger.info("  mean rho by prefix depth:\n" + by_depth.round(2).to_string())

    # Search summary: each LLM arm vs the LLM-off arm, paired over seeds.
    if sp.exists():
        s = pd.read_csv(sp).drop_duplicates(["arm", "dataset", "budget", "seed"], keep="last")
        tests = []
        for (ds, b), g in s.groupby(["dataset", "budget"]):
            for metric in ("best_acc", "best_at_25"):
                piv = g.pivot(index="seed", columns="arm", values=metric)
                if "off" not in piv:
                    continue
                for arm in piv.columns:
                    if arm == "off":
                        continue
                    both = piv[[arm, "off"]].dropna()   # seeds present in both arms
                    if len(both) < 2:
                        continue
                    p, d = _wt(both[arm].values, both["off"].values)
                    tests.append({"dataset": ds, "budget": b, "metric": metric, "arm": arm,
                                  "n_paired_seeds": len(both),
                                  "mean_on": both[arm].mean(), "mean_off": both["off"].mean(),
                                  "mean_diff": both[arm].mean() - both["off"].mean(),
                                  "wilcoxon_p": p, "cohens_d": d,
                                  "significant": bool(p < 0.05) if not np.isnan(p) else False})
        t = pd.DataFrame(tests)
        t.to_csv(res / "llm_size_paired_tests.csv", index=False)

        logger.info("SEARCH: LLM-on minus LLM-off (paired over seeds)")
        for r in t.itertuples():
            logger.info(f"  {r.dataset:12s} {r.metric:10s} {r.arm:34s} diff={r.mean_diff:+.2f}pp  "
                        f"p={r.wilcoxon_p:.3f}  d={r.cohens_d:+.2f}  n={r.n_paired_seeds}")
        lc = s[s.arm != "off"].groupby("arm").agg(calls=("llm_calls", "mean"), sec=("llm_sec", "mean"),
                                                  mismatch=("name_mismatch_ops", "mean"))
        logger.info("  mean LLM cost per run:\n" + lc.round(1).to_string())

    logger.success("llm_size_* CSVs written.")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="backend:model specs")
    ap.add_argument("--datasets", nargs="+", default=["cifar10"])
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg["search"]["seeds"])
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--probe-prefixes-per-depth", type=int, default=10)
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--search-only", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()

    if not a.summarize_only:
        if not a.search_only:
            allrows = []
            for spec in a.models:
                try:
                    allrows += probe(spec, a.datasets, a.probe_prefixes_per_depth, seed=0)
                except Exception:
                    logger.exception(f"[probe] {spec} failed:")   # one bad model shouldn't stop the rest

            if allrows:
                pth = storage.get_output_dir(PHASE, "results") / "llm_size_probe.csv"
                new = pd.DataFrame(allrows)
                old = pd.read_csv(pth) if pth.exists() else pd.DataFrame()
                if len(old):
                    # Replace only the (model, dataset) pairs re-probed now; keep the rest.
                    rerun = set(zip(new["model"], new["dataset"]))
                    keep = [(m, d) not in rerun for m, d in zip(old["model"], old["dataset"])]
                    old = old[keep]
                pd.concat([old, new]).to_csv(pth, index=False)

        if not a.probe_only:
            search_runs(a)

    summarize()