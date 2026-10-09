# experiments/run_convergence_study.py
# isort: skip_file
"""
Convergence study: does the LLM prior speed up convergence / cold start,
even though it does not raise final best_acc?

Paired design: for every (dataset, budget, seed), run the same search with each arm:
  llm_off        search_phase3(alpha=0.0, use_llm=False)
  llm_on         search_phase3(alpha=ALPHA_ON, use_llm=True)                   LLM, standard prompt
  size_prior     same, backend="heuristic": knowledge-free prior, op score ~ #params (NO LLM)
  llm_described  same LLM, prompt without the benchmark name / NB201 op names (labels +
                 functional descriptions) - contamination control (Oct 3, 2026)
Same proxy gate, seeds and trie settings; only the prior differs.
Pairs tested (Wilcoxon, Holm within each pair x metric over datasets x budgets):
  every arm vs llm_off;  llm_on vs size_prior (does the LLM beat "bigger is better"?);
  llm_on vs llm_described (does the benefit depend on recalling NAS-Bench-201 by name?).

Metrics (from per-run JSON logs; requires the patched search.py):
  best@k         best accuracy after the first k real evals (k in K_COLD)
  auc30          mean best-so-far over the first 30 real evals
  evals_to_gap   real evals until best >= (dataset ceiling - gap) for each gap
                 in GAPS; censored at n_real_evals + 1 if never reached
  llm_sec        seconds spent inside LLMPrior.score_branches
  simulated time k * S + llm_sec_at_k for an assumed eval cost of S seconds
                 (--sim-eval-seconds). NB201 is a lookup table, so this asks
                 whether the LLM's latency pays off if a real eval were costly.
                 A break-even S is reported when defined.

llm_on runs first delete their own cached LLM answers (namespace "conv_on_*"),
otherwise a rerun would hit the cache and report ~0 LLM seconds.
Nothing outside that namespace is touched.

Outputs (outputs/convergence/results/):
  convergence_runs.csv                one row per (arm, dataset, budget, seed); resumable
  convergence_per_run_metrics.csv     metrics above, per run
  convergence_paired_tests.csv        arm_a - arm_b for the pairs above, Wilcoxon + Holm + Cohen's d
  outputs/convergence/plots/figures/convergence_*.png

Usage:
    python experiments/run_convergence_study.py                  # run, then analyse
    python experiments/run_convergence_study.py --analyze-only   # no GPU / no LLM
    python experiments/run_convergence_study.py --budgets 100 --datasets cifar10 --sim-eval-seconds 60 600
"""
import argparse
import gc
import glob
import json
import os
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _cal_path(ds):
    """Phase-2 calibration file the gate will load (respects zc_proxy.calibration_mode)."""
    from archtree import storage as _st, settings as _se
    return _st._zc_cal_path("phase2", "nb201", ds, _se.CALIBRATION_MODE)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402
import yaml                         # noqa: E402
from loguru import logger           # noqa: E402
from scipy.stats import wilcoxon    # noqa: E402
from dotenv import load_dotenv      # noqa: E402
load_dotenv()

import archtree.storage as storage  # noqa: E402

with open(PROJECT_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

PHASE = "convergence"
RUNS_FILE = "convergence_runs.csv"

# The prior only orders never-visited siblings, and the LLM gate stays open
# for the whole run once alpha >= 0.2 (all such alphas behave identically).
# So use any alpha >= 0.2 to measure the prior's best case.
ALPHA_ON = 0.3
# arm -> (ablation-id tag, uses an LLM, backend override, prompt mode)
ARMS = {
    "llm_off":       ("off",  False, None,        "standard"),
    "llm_on":        ("on",   True,  None,        "standard"),
    "size_prior":    ("size", False, "heuristic", "standard"),
    "llm_described": ("desc", True,  None,        "described"),
}
PAIRS = [("llm_on", "llm_off"), ("size_prior", "llm_off"), ("llm_described", "llm_off"),
         ("llm_on", "size_prior"), ("llm_on", "llm_described")]
GAPS = (2.0, 1.0, 0.5)   # distance below dataset ceiling, in percentage points
K_COLD = (10, 25, 50)    # cold-start checkpoints (real evals)
AUC_K = 30               # window for auc30

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
LOG_DIR = PROJECT_ROOT / "outputs" / PHASE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>")
logger.add(LOG_DIR / "convergence_run.log", level="DEBUG", rotation="10 MB", retention=5)


# ── Backend picker (same env / prompt convention as run_phase3.py) ───────────
def pick_backend(args) -> dict:
    """Return the backend kwargs for search_phase3 (groq or ollama)."""
    llm_cfg = cfg["llm"]
    env = os.environ.get("ARCHTREE_LLM_BACKEND", "").strip().lower()
    if env in ("groq", "ollama"):
        choice = env
    else:
        print("\nPick LLM backend for the llm_on arm:\n  1) Ollama (local)\n  2) Groq (cloud)")
        choice = "groq" if input("Enter 1 or 2: ").strip() == "2" else "ollama"

    if choice == "groq":
        pool_file = bool(os.environ.get("GROQ_API_KEYS_FILE", "").strip())
        pool_csv = bool(os.environ.get("GROQ_API_KEYS", "").strip())
        single = bool(os.environ.get("GROQ_API_KEY", "").strip())
        if not (pool_file or pool_csv or single):
            logger.error("groq selected but no GROQ_API_KEY / GROQ_API_KEYS / GROQ_API_KEYS_FILE set.")
            sys.exit(1)
        g = llm_cfg["groq"]
        return dict(backend="groq", groq_model=args.llm_model or g["model"],
                    groq_daily_token_budget=g.get("daily_token_budget", 190_000),
                    groq_daily_request_budget=g.get("daily_request_budget", 950),
                    groq_reasoning_effort=g.get("reasoning_effort", "low"),
                    use_key_pool=pool_file or pool_csv)
    return dict(backend="ollama", model=args.llm_model or llm_cfg["model"],
                host=llm_cfg.get("host", "http://localhost:11434"))


# ── Run helpers ──────────────────────────────────────────────────────────────
def _cold_start(dataset: str, seed: int, aid: str) -> None:
    """Delete this arm's own LLM cache / call counter / checkpoint (namespace `aid` only)."""
    for prefix in ("llm_cache", "groq_calls", "llm_prior_ckpt"):
        p = LOG_DIR / f"{prefix}__nb201__{dataset}__seed{seed}__{aid}.json"
        if p.exists():
            p.unlink()


def _discard_logs(dataset, budget, seed, aid) -> None:
    """Remove the run log of a discarded run so analyze() never reads it."""
    p = LOG_DIR / f"archtree_llm__{aid}__nb201__{dataset}__seed{seed}__budget{budget}.json"
    if p.exists():
        p.unlink()
    _cold_start(dataset, seed, aid)


def _done() -> set:
    """Return {(dataset, budget, seed, arm)} already recorded in the runs CSV."""
    p = PROJECT_ROOT / "outputs" / PHASE / "results" / RUNS_FILE
    if not p.exists():
        return set()
    d = pd.read_csv(p)
    return set(zip(d["dataset"], d["budget"], d["seed"], d["arm"]))


def _fmt(x, spec: str = ".2f") -> str:
    """Format a number, tolerating None."""
    return format(x, spec) if x is not None else "n/a"


def run(args) -> None:
    # Fail fast: llm_on/llm_off both use the ZC proxy, so calibration must exist.
    missing = [ds for ds in args.datasets
               if not _cal_path(ds).exists()]
    if missing:
        logger.error(f"Missing Phase 2 calibration for {missing}.")
        sys.exit(1)

    from benchmarks.nb201 import NB201
    from archtree.search import search_phase3

    llm_arms = [a for a in args.arms if ARMS[a][1]]
    backend_kw = pick_backend(args) if llm_arms else {}
    bench = NB201(cfg)
    bench._load()
    pct = cfg.get("zc_proxy", {}).get("proxy_threshold_percentile", 35)
    done = _done()
    n_run = n_skip = n_err = 0
    logger.info(f"convergence study: alpha_on={args.alpha_on} datasets={args.datasets} "
                f"budgets={args.budgets} seeds={len(args.seeds)} backend={backend_kw['backend']}")

    for dataset in args.datasets:
        for budget in args.budgets:
            for seed in args.seeds:
                for arm in args.arms:
                    if (dataset, budget, seed, arm) in done:
                        n_skip += 1
                        continue
                    tag, uses_llm, backend_override, prompt_mode = ARMS[arm]
                    prior_on = arm != "llm_off"

                    # Per-arm cache namespace; LLM arms start with a cold cache.
                    aid = f"conv_{tag}_b{budget}"
                    if uses_llm:
                        _cold_start(dataset, seed, aid)
                    kw = dict(backend_kw) if uses_llm else {}
                    if backend_override:
                        kw = dict(backend=backend_override)
                    elif not uses_llm:
                        kw = dict(backend="heuristic")     # llm_off: no prior used, no model needed

                    logger.info(f"[conv] {dataset} b={budget} seed={seed} {arm}")
                    try:
                        t0 = time.time()
                        r = search_phase3(
                            bench, budget, dataset, seed, phase=PHASE,
                            alpha=args.alpha_on if prior_on else 0.0,
                            use_llm=prior_on, ablation_id=aid,
                            proxy_threshold_percentile=pct,
                            max_llm_calls=cfg["llm"]["max_calls"],
                            results_filename="conv_raw_runs.csv", tcr_filename="conv_raw_tcr.csv",
                            cache_stats_filename="conv_raw_cache_stats.csv",
                            dedup_filename="conv_dedup_stats.csv",
                            llm_prompt_mode=prompt_mode, **kw)
                        if uses_llm and r.get("llm_fallbacks", 0) > 0:
                            # Oct 2026: 64 of 72 Sep-2026 llm_on runs were silently uniform. Never again:
                            # an llm_on run with ANY fallback is not recorded, and the study stops so
                            # it can be resumed after the Groq budget resets.
                            logger.error(f"    {arm} had {r['llm_fallbacks']} LLM fallbacks - run discarded. "
                                         f"Stopping; rerun this script after the key budget resets.")
                            _discard_logs(dataset, budget, seed, aid)
                            logger.info(f"runs done: ran={n_run} skipped={n_skip} errors={n_err}")
                            return
                        storage.append_result_row(PHASE, RUNS_FILE, {
                            "arm": arm, "dataset": dataset, "budget": budget, "seed": seed,
                            "best_test_acc": r.get("best_test_acc"),
                            "llm_fallbacks": r.get("llm_fallbacks", 0),
                            "alpha": args.alpha_on if prior_on else 0.0,
                            "best_acc": r.get("best_acc"),
                            "real_eval_count": r.get("real_eval_count"),
                            "llm_calls": r.get("llm_calls"), "cache_hits": r.get("cache_hits"),
                            "llm_sec": r.get("llm_sec"), "wall_time_sec": time.time() - t0,
                            "ablation_id": aid,
                        })
                        logger.success(f"    best_acc={_fmt(r.get('best_acc'))}  "
                                       f"llm_calls={r.get('llm_calls')}  "
                                       f"llm_sec={_fmt(r.get('llm_sec'), '.1f')}")
                        n_run += 1
                    except KeyboardInterrupt:
                        # CSV is append-only, so rerunning resumes from here.
                        logger.warning("Interrupted — progress saved, rerun to resume.")
                        sys.exit(0)
                    except Exception:
                        # Failed runs write no row, so they are retried next launch.
                        logger.exception("    ERROR:")
                        n_err += 1
                    finally:
                        gc.collect()
    logger.info(f"runs done: ran={n_run} skipped={n_skip} errors={n_err}")


# ── Analysis ─────────────────────────────────────────────────────────────────
_CEIL_CACHE: dict = {}   # dataset -> best accuracy in the benchmark
_BENCH = None            # lazily loaded, shared across datasets


def _ceiling(dataset: str) -> float:
    """Best achievable accuracy on `dataset` (exhaustive lookup over the benchmark)."""
    global _BENCH
    if dataset not in _CEIL_CACHE:
        if _BENCH is None:
            from benchmarks.nb201 import NB201
            _BENCH = NB201(cfg)
            _BENCH._load()
        _CEIL_CACHE[dataset] = max(_BENCH.query(i, dataset) for i in range(_BENCH.total_archs))
    return _CEIL_CACHE[dataset]


def _load_run_logs() -> pd.DataFrame:
    """Read per-run JSON logs written by the patched search_phase3."""
    rows = []
    for path in glob.glob(str(LOG_DIR / "archtree_llm__conv_*__nb201__*__seed*__budget*.json")):
        with open(path) as fh:
            j = json.load(fh)
        h = j.get("eval_history") or []
        if not h:
            continue
        if j.get("use_llm") and j.get("llm_fallbacks", 0) > 0:
            logger.warning(f"skipping {Path(path).name}: {j['llm_fallbacks']} LLM fallbacks (not a valid LLM-on run)")
            continue
        if "llm_fallbacks" not in j:
            logger.warning(f"skipping {Path(path).name}: pre-Oct-2026 log (prior-bias bug, unknown fallbacks)")
            continue
        aid = j.get("ablation_id", "")
        arm = next((a for a, v in ARMS.items() if aid.startswith(f"conv_{v[0]}_b")), None)
        if arm is None:
            continue
        rows.append({"arm": arm, "dataset": j["dataset"], "budget": j["budget"], "seed": j["seed"],
                     "acc": [e["acc"] for e in h],
                     "llm_sec_at": [e.get("llm_sec", 0.0) for e in h],   # cumulative LLM seconds at each eval
                     "t_sec": [e.get("t_sec", np.nan) for e in h],
                     "llm_sec": j.get("llm_sec", np.nan), "proxy_sec": j.get("proxy_sec", np.nan),
                     "wall": j.get("wall_time_sec", np.nan)})
    return pd.DataFrame(rows)


def _metrics(r, sim_costs) -> dict:
    """Compute all per-run convergence metrics from one run's eval history."""
    acc = np.asarray(r["acc"], float)
    bsf = np.maximum.accumulate(acc)   # best-so-far curve
    n = len(acc)
    m = {"arm": r["arm"], "dataset": r["dataset"], "budget": r["budget"], "seed": r["seed"],
         "n_real_evals": n, "final_best": bsf[-1], "llm_sec": r["llm_sec"], "proxy_sec": r["proxy_sec"],
         "search_sec_excl_proxy": r["wall"] - r["proxy_sec"] if not np.isnan(r["proxy_sec"]) else np.nan}

    # Cold start: best accuracy after k evals (clamped if the run is shorter).
    for k in K_COLD:
        m[f"best_at_{k}"] = bsf[min(k, n) - 1]

    # AUC over the first AUC_K evals; short runs are padded with their final best.
    pad = np.concatenate([bsf, np.full(max(0, AUC_K - n), bsf[-1])])
    m["auc30"] = pad[:AUC_K].mean()

    # Evals (and simulated time) to get within `gap` pp of the dataset ceiling.
    ceil = _ceiling(r["dataset"])
    for gap in GAPS:
        hit = np.where(bsf >= ceil - gap)[0]
        k = int(hit[0]) + 1 if len(hit) else n + 1   # n+1 = censored (never reached)
        m[f"evals_to_{gap}pp"] = k
        m[f"reached_{gap}pp"] = bool(len(hit))
        for S in sim_costs:
            llm_k = r["llm_sec_at"][min(k, n) - 1]
            m[f"time_to_{gap}pp_S{S:g}"] = k * S + llm_k
    return m


def _test_note(a, b):
    """Is there seed-to-seed variation to test? A deterministic prior (LLM / size prior) makes the first
    proposals identical for every seed, so a metric can take ONE value across all 12 seeds (Oct 4, 2026).
      both_deterministic: the paired difference is the same for every seed -> effectively n = 1, no test
      one_deterministic : one arm is constant -> one-sample test of the other arm against that value
    """
    ua, ub = len(np.unique(np.round(a, 6))), len(np.unique(np.round(b, 6)))
    if len(np.unique(np.round(a - b, 6))) == 1:
        return "constant_difference"
    if ua == 1 and ub == 1:
        return "both_deterministic"
    if ua == 1 or ub == 1:
        return "one_deterministic"
    return ""


def _wt(a, b):
    """Paired Wilcoxon p-value and Cohen's d (mean diff / sd of diffs). No test when the paired
    difference does not vary across seeds (pseudo-replication: 12 copies of one comparison)."""
    d = a - b
    if len(d) < 2 or np.all(d == 0) or len(np.unique(np.round(d, 6))) == 1:
        return np.nan, np.nan
    try:
        p = wilcoxon(a, b).pvalue
    except ValueError:
        p = np.nan
    sd = d.std(ddof=1)
    return p, (d.mean() / sd if sd > 0 else np.nan)


def _holm(p):
    p = np.asarray(p, float)
    o, m, res, run_ = np.argsort(p), len(p), np.empty(len(p)), 0.0
    for k, i in enumerate(o):
        run_ = max(run_, (m - k) * p[i])
        res[i] = min(1.0, run_)
    return res


def analyze(args) -> None:
    runs = _load_run_logs()
    if runs.empty:
        logger.error(f"No convergence logs under {LOG_DIR}. Run the study first.")
        return

    res = storage.get_output_dir(PHASE, "results")
    met = pd.DataFrame([_metrics(r, args.sim_eval_seconds) for _, r in runs.iterrows()])
    met.to_csv(res / "convergence_per_run_metrics.csv", index=False)

    metric_cols = ([f"best_at_{k}" for k in K_COLD] + ["auc30", "final_best", "llm_sec"] +
                   [f"evals_to_{g}pp" for g in GAPS] +
                   [f"time_to_1.0pp_S{S:g}" for S in args.sim_eval_seconds])
    out = []
    logger.info("=" * 78)
    logger.info("paired arm_a - arm_b  (positive = arm_a better for accuracy metrics; "
                "negative = better for evals_to_/time_to_)")

    for (ds, b), g in met.groupby(["dataset", "budget"]):
        # One column per arm, indexed by seed, so tests are paired per seed.
        piv = {c: g.pivot(index="seed", columns="arm", values=c) for c in metric_cols}
        for arm_a, arm_b in PAIRS:
            for c in metric_cols:
                pv = piv[c]
                if arm_a not in pv or arm_b not in pv:
                    continue
                pv = pv[[arm_a, arm_b]].dropna()
                if len(pv) < 2:
                    continue
                p, d = _wt(pv[arm_a].values, pv[arm_b].values)
                note = _test_note(pv[arm_a].values, pv[arm_b].values)
                diff = float((pv[arm_a] - pv[arm_b]).mean())
                out.append({"dataset": ds, "budget": b, "metric": c, "arm_a": arm_a, "arm_b": arm_b,
                            "n_paired_seeds": len(pv),
                            # mean_on / mean_off kept for backward compatibility (= arm_a / arm_b)
                            "mean_on": pv[arm_a].mean(), "mean_off": pv[arm_b].mean(),
                            "mean_diff": diff, "wilcoxon_p": p, "cohens_d": d,
                            "n_unique_a": int(pv[arm_a].round(6).nunique()),
                            "n_unique_b": int(pv[arm_b].round(6).nunique()), "test_note": note})
                if (arm_a, arm_b) == ("llm_on", "llm_off") or c in ("evals_to_1.0pp", "best_at_10"):
                    logger.info(f"  {ds:14s} b={b:<4d} {arm_a:>13s} vs {arm_b:<13s} {c:16s} "
                                f"{pv[arm_a].mean():8.2f} vs {pv[arm_b].mean():8.2f} diff={diff:+7.2f} p={p:.3f}"
                                f"{'  [' + note + ']' if note else ''}")

        # Break-even eval cost: extra LLM seconds divided by real evals saved.
        on, off = g[g.arm == "llm_on"], g[g.arm == "llm_off"]
        if len(on) and len(off):
            dk = off["evals_to_1.0pp"].median() - on["evals_to_1.0pp"].median()   # evals saved
            dl = on["llm_sec"].median() - off["llm_sec"].median()                 # extra LLM seconds
            if dk > 0:
                logger.info(f"  {ds:14s} b={b:<4d} LLM reaches ceiling-1pp {dk:.0f} real evals sooner (median) at "
                            f"{dl:.1f}s extra LLM time -> pays for itself if one real eval costs > {dl / dk:.1f}s")
            else:
                logger.info(f"  {ds:14s} b={b:<4d} LLM does NOT reach ceiling-1pp sooner in median real evals "
                            f"(Δ={dk:+.0f}) -> no eval cost makes it pay for itself on this metric")

    t = pd.DataFrame(out)
    if len(t):
        t["p_holm"] = np.nan
        ok = t["wilcoxon_p"].notna()
        t.loc[ok, "p_holm"] = (t[ok].groupby(["arm_a", "arm_b", "metric"])["wilcoxon_p"]
                                .transform(lambda x: _holm(x.values)))
        t["significant"] = t["p_holm"] < 0.05
        logger.info("Holm-significant (within pair x metric over datasets x budgets):")
        for r in t[t["significant"] & t["metric"].isin(["evals_to_1.0pp", "best_at_10", "auc30",
                                                       "final_best"])].itertuples():
            logger.info(f"  {r.arm_a} vs {r.arm_b} {r.dataset} b={r.budget} {r.metric}: "
                        f"{r.mean_diff:+.2f} (p_holm={r.p_holm:.3f})"
                        f"{'  [' + r.test_note + ']' if r.test_note else ''}")
        nd = t[t["test_note"] == "constant_difference"]
        if len(nd):
            logger.warning(f"{len(nd)} comparisons not tested: the paired difference is identical for every seed "
                           f"(deterministic prior -> the same first proposals for all seeds; n = 1 in effect).")
    out = t
    out.to_csv(res / "convergence_paired_tests.csv", index=False)
    logger.success("convergence_per_run_metrics.csv / convergence_paired_tests.csv written.")
    _plot(runs, args)


def _plot(runs: pd.DataFrame, args) -> None:
    """Per (dataset, budget): best-so-far vs real evals, and vs simulated wall-clock."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from plots.figure_export import install as _gallery; _gallery()   # copy + data CSV to plots/figures
    except ImportError:
        logger.warning("matplotlib missing — skipping figures.")
        return

    color = {"llm_off": "#1f3a5f", "llm_on": "#c0392b", "size_prior": "#7f8c8d", "llm_described": "#e67e22"}
    for (ds, b), g in runs.groupby(["dataset", "budget"]):
        S_list = list(args.sim_eval_seconds[:2]) or [0]   # at most two simulated-cost panels
        fig, axes = plt.subplots(1, 1 + len(S_list), figsize=(4.4 * (1 + len(S_list)), 3.6))
        axes = np.atleast_1d(axes)

        for arm in ARMS:
            ga = g[g["arm"] == arm]
            if ga.empty:
                continue

            # Panel 0: mean best-so-far vs real evals (runs padded to equal length).
            K = int(max(len(a) for a in ga["acc"]))
            curves = np.array([np.maximum.accumulate(np.concatenate([a, np.full(K - len(a), max(a))]))
                               for a in ga["acc"]])
            mu = curves.mean(0)
            if len(curves) > 1:
                se = curves.std(0, ddof=1) / np.sqrt(len(curves))
            else:
                se = np.zeros_like(mu)
            x = np.arange(1, K + 1)
            axes[0].plot(x, mu, color=color[arm], label=arm)
            axes[0].fill_between(x, mu - se, mu + se, color=color[arm], alpha=0.2)

            # Other panels: best-so-far vs simulated time (k * S + cumulative LLM seconds).
            for ax, S in zip(axes[1:], S_list):
                tmax = max(len(a) * S + ls[-1] for a, ls in zip(ga["acc"], ga["llm_sec_at"]))
                grid = np.linspace(0, tmax, 200)
                cs = []
                for a, ls in zip(ga["acc"], ga["llm_sec_at"]):
                    tk = np.arange(1, len(a) + 1) * S + np.asarray(ls)   # time of each eval
                    bsf = np.maximum.accumulate(a)
                    idx = np.searchsorted(tk, grid, side="right") - 1    # last eval finished by each grid time
                    cs.append(np.where(idx >= 0, bsf[np.clip(idx, 0, None)], np.nan))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN before the first eval finishes
                    ax.plot(grid, np.nanmean(np.array(cs), 0), color=color[arm], label=arm)

        axes[0].set(title=f"{ds} b={b}: vs real evals", xlabel="real evals", ylabel="mean best-so-far (%)")
        for ax, S in zip(axes[1:], S_list):
            ax.set(title=f"vs simulated wall-clock (S={S:g}s/eval + LLM)", xlabel="seconds")
        for ax in axes:
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        out = storage.get_figure_path(PHASE, f"convergence_{ds}_b{b}")
        fig.savefig(out, dpi=180)
        plt.close(fig)
        logger.info(f"figure -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", action="store_true", help="skip runs, analyse existing logs")
    ap.add_argument("--datasets", nargs="+", default=cfg["benchmarks"]["datasets"])
    ap.add_argument("--budgets", nargs="+", type=int, default=[100, 200])
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg["search"]["seeds"])
    ap.add_argument("--alpha-on", type=float, default=ALPHA_ON)
    ap.add_argument("--llm-model", default=None, help="override model for the chosen backend")
    ap.add_argument("--sim-eval-seconds", nargs="+", type=float, default=[10.0, 60.0, 600.0])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    a = ap.parse_args()
    if not a.analyze_only:
        run(a)
    analyze(a)