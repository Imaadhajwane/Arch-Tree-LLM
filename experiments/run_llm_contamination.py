# experiments/run_llm_contamination.py
# isort: skip_file
"""
Does the LLM prior's knowledge come from recalling NAS-Bench-201?  (Oct 3, 2026)

A reviewer can argue that gpt-oss simply remembers NAS-Bench-201 (its paper / tables are public),
so the "LLM prior" is benchmark recall, not reasoning. This probe asks the same model the same
question in three prompt modes and measures how well its op ranking matches the truth:

    standard   the C7 prompt: names NAS-Bench-201 and its op names (conv_3x3, ...)
    described  no benchmark name, no NB201 op names: ops are labels op_A..op_E (scrambled order)
               with a plain functional description ("3x3 convolution with batch norm and ReLU")
    anonymous  labels only, no description - NEGATIVE CONTROL: the model has nothing to go on,
               so it must score ~chance (if it does not, the probe itself is broken)
    size_prior NO LLM - op score ~ #params: the "bigger is better" rule as a reference point

Reading:  described ~ standard  -> the benefit does not depend on recalling the benchmark by name
          described << standard -> part of the benefit is benchmark recall (report it)
          size_prior ~ standard -> the LLM adds nothing beyond "bigger is better"
The SEARCH-level counterpart (do described / size_prior speed up the search like the standard
prompt?) is the convergence study, arms llm_described and size_prior (pipeline step 15).

Ground truth for a prefix = mean validation accuracy of all architectures below each child
(exact, from the benchmark table). The LLM answers do not depend on the dataset (the prompt
never names it), so each prompt is asked once and scored against all three datasets.

Outputs (outputs/llm_contamination/results/):
    contamination_probe.csv      one row per (mode, dataset, prefix)
    contamination_summary.csv    mean Spearman rho / top-1 hit rate per (mode, dataset)
    contamination_tests.csv      paired Wilcoxon over prefixes: standard vs each other mode,
                                 every mode vs chance (rho = 0); Holm within each comparison
                                 (n_paired_seeds = number of prefixes, for the Phase 6 loader)

Usage
    python experiments/run_llm_contamination.py                 # ~60 Groq calls per LLM mode
    python experiments/run_llm_contamination.py --summarize-only
"""
import argparse
import itertools
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np                               # noqa: E402
import pandas as pd                              # noqa: E402
import yaml                                      # noqa: E402
from loguru import logger                        # noqa: E402
from scipy.stats import spearmanr, wilcoxon      # noqa: E402
from dotenv import load_dotenv                   # noqa: E402
load_dotenv()

import archtree.storage as storage               # noqa: E402

with open(PROJECT_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

PHASE = "llm_contamination"
OPS = ["none", "skip_connect", "conv_1x1", "conv_3x3", "avg_pool_3x3"]
L, K = 6, 5
MODES = ["standard", "described", "anonymous", "size_prior"]

logger.remove()
LOG_DIR = PROJECT_ROOT / "outputs" / PHASE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>")
logger.add(LOG_DIR / "contamination_run.log", level="DEBUG", rotation="10 MB", retention=5)


def subtree_means(acc, prefix):
    """Mean accuracy of all leaves under prefix+[op] (index = base-5 code, token 0 most significant)."""
    out = {}
    for oi, op in enumerate(OPS):
        base = 0
        for d in [OPS.index(t) for t in prefix] + [oi]:
            base = base * K + d
        span = K ** (L - len(prefix) - 1)
        out[op] = float(acc[base * span:(base + 1) * span].mean())
    return out


def prefixes(n_per_depth, seed):
    """Distinct prefixes of depth 0-3 (depth 0 and 1 are all of them)."""
    rng = random.Random(seed)
    out = []
    for depth in range(4):
        allp = list(itertools.product(OPS, repeat=depth))
        out += [list(p) for p in rng.sample(allp, min(n_per_depth, len(allp)))]
    return out


def make_prior(mode, model):
    from archtree.llm_prior import LLMPrior
    if mode == "size_prior":
        return LLMPrior(phase=PHASE, bench="nb201", dataset="all", seed=0, namespace="probe_size",
                        backend="heuristic", max_calls=0)
    g = cfg["llm"]["groq"]
    pool = bool(os.environ.get("GROQ_API_KEYS_FILE", "").strip() or os.environ.get("GROQ_API_KEYS", "").strip())
    if not (pool or os.environ.get("GROQ_API_KEY", "").strip()):
        raise RuntimeError("no GROQ_API_KEY / GROQ_API_KEYS / GROQ_API_KEYS_FILE in the environment")
    return LLMPrior(phase=PHASE, bench="nb201", dataset="all", seed=0, namespace=f"probe_{mode}",
                    backend="groq", groq_model=model, use_key_pool=pool, max_calls=10_000,
                    groq_daily_token_budget=g.get("daily_token_budget", 190_000),
                    groq_daily_request_budget=g.get("daily_request_budget", 950),
                    groq_reasoning_effort=g.get("reasoning_effort", "low"), prompt_mode=mode)


def run(args):
    from benchmarks.nb201 import NB201
    bench = NB201(cfg)
    bench._load()
    acc = {ds: np.array([bench.query(i, ds) for i in range(bench.total_archs)], float)
           for ds in bench.datasets}
    pfx = prefixes(args.prefixes_per_depth, seed=0)
    res = storage.get_output_dir(PHASE, "results")
    out_path = res / "contamination_probe.csv"
    done = set()
    if out_path.exists():
        old = pd.read_csv(out_path)
        done = set(zip(old["mode"], old["prefix"].fillna("")))
    rows = []
    for mode in args.modes:
        prior = make_prior(mode, args.model)
        n_fb = 0
        for p in pfx:
            key = "|".join(p)
            if (mode, key) in done:
                continue
            fb0, mm0 = prior.fallback_count, prior.name_mismatch_count
            t0 = time.time()
            scores = prior.score_branches(p, list(OPS), alpha_t=1.0)     # alpha_t=1: prior always asked
            dt = time.time() - t0
            if prior.fallback_count > fb0:
                n_fb += 1
                logger.warning(f"[{mode}] fallback on prefix {key or '(root)'} - not recorded")
                if n_fb >= 3:
                    logger.error(f"[{mode}] repeated fallbacks (keys exhausted / model error) - stopping this "
                                 f"mode; rerun later to fill it (finished prefixes are kept).")
                    break
                continue
            s = np.array([scores[o] for o in OPS])
            for ds in bench.datasets:
                truth = subtree_means(acc[ds], p)
                t = np.array([truth[o] for o in OPS])
                flat = bool(np.ptp(s) < 1e-12)
                rows.append({"mode": mode, "model": "size_prior" if mode == "size_prior" else args.model,
                             "dataset": ds, "depth": len(p), "prefix": key,
                             "spearman": np.nan if flat else spearmanr(s, t)[0],
                             "top1_hit": bool(np.argmax(s) == np.argmax(t)) if not flat else False,
                             "flat_scores": flat,
                             "name_mismatch_ops": prior.name_mismatch_count - mm0, "latency_sec": dt})
        logger.info(f"[{mode}] probed ({len([r for r in rows if r['mode'] == mode]) // 3} new prefixes)")
    if rows:
        new = pd.DataFrame(rows)
        old = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
        pd.concat([old, new], ignore_index=True).to_csv(out_path, index=False)


def _holm(p):
    p = np.asarray(p, float)
    o, m, out, run_ = np.argsort(p), len(p), np.empty(len(p)), 0.0
    for k, i in enumerate(o):
        run_ = max(run_, (m - k) * p[i])
        out[i] = min(1.0, run_)
    return out


def summarize():
    res = storage.get_output_dir(PHASE, "results")
    pth = res / "contamination_probe.csv"
    if not pth.exists():
        logger.error("no probe results yet")
        return
    d = pd.read_csv(pth).drop_duplicates(["mode", "dataset", "prefix"], keep="last")
    d["prefix"] = d["prefix"].fillna("")
    d["spearman_fb0"] = d["spearman"].fillna(0.0)          # flat answer = no information = 0
    summ = (d.groupby(["mode", "dataset"])
              .agg(n_prefixes=("prefix", "nunique"), spearman_mean=("spearman_fb0", "mean"),
                   top1_hit_rate=("top1_hit", "mean"), flat_rate=("flat_scores", "mean"),
                   name_mismatch_per_call=("name_mismatch_ops", "mean")).reset_index())
    summ.to_csv(res / "contamination_summary.csv", index=False)
    tests = []
    for ds, g in d.groupby("dataset"):
        piv = g.pivot(index="prefix", columns="mode", values="spearman_fb0")
        for m in piv.columns:                                   # vs chance
            x = piv[m].dropna()
            if len(x) >= 5 and np.any(x != 0):
                tests.append({"dataset": ds, "comparison": f"{m}_vs_chance", "n_paired_seeds": len(x),
                              "mean_diff": float(x.mean()), "wilcoxon_p": wilcoxon(x).pvalue,
                              "cohens_d": float(x.mean() / x.std(ddof=1)) if x.std(ddof=1) > 0 else np.nan})
        if "standard" in piv:
            for m in [c for c in piv.columns if c != "standard"]:
                both = piv[["standard", m]].dropna()
                diff = both["standard"] - both[m]
                if len(both) >= 5 and np.any(diff != 0):
                    tests.append({"dataset": ds, "comparison": f"standard_vs_{m}", "n_paired_seeds": len(both),
                                  "mean_diff": float(diff.mean()),
                                  "wilcoxon_p": wilcoxon(both["standard"], both[m]).pvalue,
                                  "cohens_d": float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else np.nan})
    t = pd.DataFrame(tests)
    if len(t):
        t["p_holm"] = t.groupby("comparison")["wilcoxon_p"].transform(lambda s: _holm(s.values))
        t["significant"] = t["p_holm"] < 0.05
    t.to_csv(res / "contamination_tests.csv", index=False)
    logger.info("=" * 78)
    logger.info("Op-ranking quality per prompt mode (chance: rho 0.00, top-1 0.20):")
    for r in summ.itertuples():
        logger.info(f"  {r.mode:11s} {r.dataset:15s} rho={r.spearman_mean:+.2f} top1={r.top1_hit_rate:.2f} "
                    f"flat={r.flat_rate:.0%} n={r.n_prefixes}")
    for r in t.itertuples():
        logger.info(f"  {r.dataset:15s} {r.comparison:24s} diff={r.mean_diff:+.3f} p_holm={r.p_holm:.3f}"
                    f"{'  *' if r.significant else ''}")
    logger.success("contamination_* CSVs written.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--model", default=cfg["llm"]["groq"]["model"])
    ap.add_argument("--prefixes-per-depth", type=int, default=25)
    a = ap.parse_args()
    if not a.summarize_only:
        run(a)
    summarize()
