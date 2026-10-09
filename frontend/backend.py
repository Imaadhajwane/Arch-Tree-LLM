"""
backend.py - the dashboard server for ArchTree-LLM (Oct 2026 edition).

The browser cannot run the search (it is Python), so this file runs it on request and hands
the results back as JSON. It draws nothing; all drawing lives in index.html.

What changed in this edition
  * The live search is the ArchTree headline system, not an approximation of v1:
    percentile-rank UCB reward, C from config.yaml, 50% evolutionary mutation after 10 real
    evaluations, label-free proxy gate (first 10 proposals always pass, then top-35% of the
    scores seen so far), value pruning every 50 proposals, max backup, dedup through the trie.
    It searches on VALIDATION accuracy and reports the TEST accuracy of the best-validation
    architecture, exactly like the offline pipeline.
  * Every evaluation carries its benchmark training time, so the live comparison is done at
    equal GPU-hours (the paper's main framing) instead of equal proposal counts.
  * /api/story reads every result file the Oct 2026 pipeline produced (equal GPU-hours,
    gate-cost model, NB101, NATS-SSS, design ablation, LLM knowledge controls, Friedman / CD /
    TOST / P(improvement), Phase 6, hardware, regret) and returns one JSON for the Evidence pages.
  * /api/figures + /figures/<name> serve the 50 paper figures from plots/figures/.

Endpoints
  GET  /                  index.html
  GET  /api/config        datasets, devices, seeds, live constants
  POST /api/run           one live v2 search + equal-GPU-hour comparison
  POST /api/hardware      Pareto front of the run's architectures on a device (HW-NAS-Bench)
  GET  /api/story         every offline result, aggregated
  GET  /api/status        which result files exist (rows, last modified)
  GET  /api/replay        short real v2 run for the landing-page animation
  GET  /api/figures       list of paper figures (png + csv of plotted points)
  GET  /figures/<file>    one figure / its csv

Install once:  pip install flask flask-cors pandas pyyaml python-dotenv scipy numpy
Run:           python frontend/backend.py      then open http://localhost:5000
"""
from __future__ import annotations

import bisect
import json
import math
import os
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from flask import Flask, abort, jsonify, request, send_from_directory
from flask_cors import CORS

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next((p for p in [APP_DIR, *APP_DIR.parents]
                     if (p / "config.yaml").exists() and (p / "archtree").is_dir()), APP_DIR)
os.chdir(PROJECT_ROOT)   # archtree code writes a few cwd-relative files
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

app = Flask(__name__, static_folder=None)
CORS(app)

OUT = PROJECT_ROOT / "outputs"
FIG_DIR = PROJECT_ROOT / "plots" / "figures"
LIVE_DIR = OUT / "live"
BACKEND_VERSION = "2026.10.05-1"   # index.html checks this and asks for a restart when it differs
LLM_ALPHA = 0.3          # "prior on" alpha used by the convergence study
LLM_MAX_CALLS = 12       # live cap on Groq consults per run
CKPT_EVERY = 10


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f) or {}


def _settings():
    from archtree import settings
    return settings


def _atomic_json(path: Path, obj) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    tmp = path.with_suffix(path.suffix + ".tmp")
    for _ in range(3):
        try:
            with open(tmp, "w") as f:
                json.dump(obj, f)
            os.replace(tmp, path)
            return True
        except OSError:
            time.sleep(0.05)
    return False


def _groq_pool_available() -> bool:
    return bool(os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEYS_FILE"))


def _pool_key_count() -> int:
    try:
        raw = os.environ.get("GROQ_API_KEYS", "")
        if raw.strip():
            return len([k for k in raw.split(",") if k.strip()])
        f = os.environ.get("GROQ_API_KEYS_FILE")
        if f and Path(f).exists():
            return len([l for l in Path(f).read_text().splitlines()
                        if l.strip() and not l.strip().startswith("#")])
    except Exception:
        pass
    return 0


def _clean(v):
    """numpy / NaN -> plain JSON values."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(f"{v:.3g}") if 0 < abs(v) < 1e-3 else round(v, 5)   # keep tiny p-values
    if hasattr(v, "item"):
        return _clean(v.item())
    return v


def _recs(df, cols=None, limit=None):
    if df is None or not len(df):
        return []
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    if limit:
        df = df.head(limit)
    return [{str(k): _clean(v) for k, v in r.items()} for r in df.to_dict("records")]


# ─────────────────────────────────────────────────────────────── pages
@app.route("/")
def index():
    # dashboard.html is the page Claude updates; index.html is the fallback copy (an editor with an old
    # index.html open kept saving it back over newer versions)
    page = "dashboard.html" if (APP_DIR / "dashboard.html").exists() else "index.html"
    r = send_from_directory(APP_DIR, page, max_age=0)
    r.headers["Cache-Control"] = "no-store"   # always serve the file on disk, never a cached copy
    return r


@app.route("/vendor/<path:name>")
def vendor_file(name):
    """Bundled Plotly and fonts, so the dashboard needs no internet during a demo."""
    if "/" in name or "\\" in name or ".." in name:
        abort(404)
    return send_from_directory(APP_DIR / "vendor", name, max_age=86400)


@app.route("/api/config")
def api_config():
    cfg = load_config()
    st = _settings()
    return jsonify({
        "datasets": cfg["benchmarks"]["datasets"],
        "hw_targets": cfg["benchmarks"]["hw_targets"],
        "seeds": cfg["search"]["seeds"],
        "alpha_init": st.ALPHA_INIT,
        "ucb_c": st.C, "reward": st.REWARD_MODE, "mutation_prob": st.MUTATION_PROB,
        "proxy_pct": st.PROXY_THRESHOLD_PERCENTILE, "warmup": st.WARMUP_REAL_EVALS,
        "calibration_mode": st.CALIBRATION_MODE,
        "prune_percentile": st.PRUNE_PERCENTILE, "prune_every": st.PRUNE_EVERY_N,
        "gate_ready": {d: _cal_path(d).exists() for d in cfg["benchmarks"]["datasets"]},
        "groq_pool": _groq_pool_available(), "llm_max_calls": LLM_MAX_CALLS,
        "version": BACKEND_VERSION,
    })


def _cal_path(ds):
    from archtree import storage as _st
    return _st._zc_cal_path("phase2", "nb201", ds, _settings().CALIBRATION_MODE)


# ─────────────────────────────────────────────────────────────── live search (v2)
_CACHE, _CACHE_LOCK = {}, threading.Lock()
_SEARCH_LOCK = threading.Lock()   # ucb.py keeps the rank-reward state in a module global


def _get_bench(kind, cls, cfg):
    with _CACHE_LOCK:
        if kind not in _CACHE:
            obj = cls(cfg)
            obj._load()
            _CACHE[kind] = obj
        return _CACHE[kind]


def _get_zc(dataset):
    """-> (ensemble or None, note). Never raises."""
    key = ("zc", dataset)
    try:
        cal = _cal_path(dataset)
        if not cal.exists():
            return None, f"Gate off: no saved calibration {cal.name} (run run_phase2.py for '{dataset}')."
        with _CACHE_LOCK:
            if key not in _CACHE:
                from archtree.zc_proxy import ZCProxyEnsemble
                _CACHE[key] = ZCProxyEnsemble.from_saved("phase2", "nb201", dataset)
            return _CACHE[key], None
    except Exception as e:
        return None, f"Gate off: {type(e).__name__}: {e}"


def _mean_cost(bench, dataset) -> float:
    """Mean training seconds of one NB201 architecture (sets the equal-GPU-hour budget)."""
    key = ("cost", dataset)
    with _CACHE_LOCK:
        if key not in _CACHE:
            vals = [bench.train_sec(i, dataset) for i in range(bench.total_archs)]
            vals = [float(v) for v in vals if v is not None]
            _CACHE[key] = sum(vals) / len(vals) if vals else 1.0
        return _CACHE[key]


def _pruned_prefixes(trie) -> list:
    out = []

    def walk(n, p):
        for op, c in n.children.items():
            q = p + [op]
            if c.is_pruned:
                out.append("|".join(q))
            else:
                walk(c, q)
    walk(trie.root, [])
    return out


def _search(bench, ops, dataset, rng_seed, *, mode="v2", gate=None, pct=35,
            budget=None, hour_budget=None, max_props=None, prior=None, prune=True,
            record=True):
    """One search.

    mode   "v2"     rank reward + mutation (the headline system)
           "v1"     raw accuracy/100 reward, C=1.414, no mutation
           "random" uniform sampling without replacement
           "re"     regularized evolution (population 10, tournament 5, aging)
    gate   ZC ensemble or None.
    Stops after `budget` proposals, or once `hour_budget` GPU-hours are spent (equal-cost mode).
    History entries hold REAL evaluations only (valid acc = search signal, test acc = report).
    """
    from archtree.trie import ArchTrie, TrieNode
    from archtree import ucb as U
    from archtree.encoder import index_from_tokens_nb201
    st = _settings()

    rng = random.Random(rng_seed)
    random.seed(rng_seed)                    # ucb.traverse uses the module RNG for tie-breaks
    reward = "rank" if mode == "v2" else "raw"
    C = st.C if mode == "v2" else 1.414
    mut_p = st.MUTATION_PROB if mode == "v2" else 0.0
    U.reward_reset(reward)
    warm = st.WARMUP_REAL_EVALS

    trie = ArchTrie(max_depth=6)
    history, gate_log, kinds = [], [], []
    seen_scores, evaluated, skipped = [], set(), set()
    prune_events, dup = [], 0
    hours = 0.0
    llm = {"calls": 0, "sec": 0.0, "first": None, "error": None}
    pop = []                                 # RE population (tokens, acc)
    max_props = max_props or (budget if budget else 2000)

    def bump(path):
        trie.root.visit_count += 1
        node = trie.root
        for tok in path:
            if tok not in node.children:
                node.children[tok] = TrieNode()
            node = node.children[tok]
            node.visit_count += 1

    def random_new():
        for _ in range(400):
            p = [rng.choice(ops) for _ in range(6)]
            if index_from_tokens_nb201(p) not in evaluated:
                return p
        return [rng.choice(ops) for _ in range(6)]

    i = 0
    while i < max_props:
        if hour_budget is not None and hours >= hour_budget:
            break
        kind = "ucb"
        if mode == "random":
            path, kind = random_new(), "random"
        elif mode == "re":
            if len(pop) < 10:
                path, kind = random_new(), "random"
            else:
                parent = max(rng.sample(pop, 5), key=lambda x: x[1])[0]
                path, kind = bench.mutate_tokens(parent, rng), "mutation"
                if index_from_tokens_nb201(path) in evaluated:
                    path, kind = random_new(), "redirect"
        else:
            if prior is not None:
                t0 = time.time()
                try:
                    path = U.traverse_with_llm(trie, ops, prior, C=C, alpha=LLM_ALPHA, t=i + 1,
                                               budget=budget or max_props)
                    llm["calls"] = getattr(prior, "call_count", llm["calls"])
                    llm["hits"] = getattr(prior, "hit_count", 0)            # answered from the saved cache
                    llm["fallbacks"] = getattr(prior, "fallback_count", 0)  # no usable model answer
                except Exception as e:
                    llm["error"] = f"{type(e).__name__}: {e}"
                    prior = None
                    path = U.traverse(trie, ops, C=C)
                llm["sec"] += time.time() - t0
            else:
                path = U.traverse(trie, ops, C=C)
            if mut_p > 0 and len(history) >= 10 and rng.random() < mut_p:
                pool = rng.sample(history, min(st.MUTATION_TOURNAMENT, len(history)))
                parent = max(pool, key=lambda e: e["acc"])["arch"].split("|")
                for _ in range(20):
                    cand = bench.mutate_tokens(parent, rng)
                    if not trie.is_visited(cand):
                        path, kind = cand, "mutation"
                        break
            if trie.is_visited(path):
                dup += 1
                path, kind = U.random_unvisited(trie, ops), "redirect"
        key = "|".join(path)
        idx = index_from_tokens_nb201(path)

        passed = True
        if gate is not None and mode in ("v2", "v1") and i >= warm:
            s = float(gate.score(path))
            seen_scores.append(s)
            passed = s >= float(np.percentile(seen_scores, 100 - pct))   # same rule as search.py
            if record:
                gate_log.append({"iter": i + 1, "arch": key, "score": round(s, 5), "passed": passed})
            if not passed:
                skipped.add(key)
                bump(path)
        elif gate is not None and mode in ("v2", "v1") and record:
            gate_log.append({"iter": i + 1, "arch": key, "score": None, "passed": True, "warmup": True})

        if passed:
            acc = float(bench.query(idx, dataset))
            sec = float(bench.train_sec(idx, dataset) or 0.0)
            if hour_budget is not None and hours + sec / 3600 > hour_budget * 1.0 and history:
                break
            hours += sec / 3600
            evaluated.add(idx)
            U.reward_register(acc)
            if mode in ("v2", "v1"):
                node = trie.insert(path)
                node.accuracy = acc
                U.backpropagate(trie, path, acc, backup_mode="max")
            if mode == "re":
                pop.append((path, acc))
                if len(pop) > 10:
                    pop.pop(0)
            history.append({"iter": i + 1, "arch": key, "acc": round(acc, 3),
                            "test": round(float(bench.query_test(idx, dataset)), 3),
                            "h": round(hours, 4), "kind": kind})
        if prune and mode in ("v2", "v1") and (i + 1) % st.PRUNE_EVERY_N == 0:
            before = set(_pruned_prefixes(trie)) if record else set()
            n = trie.prune(st.PRUNE_PERCENTILE)
            if n and record:
                after = set(_pruned_prefixes(trie))
                prune_events.append({"at_proposal": i + 1, "at_real_eval": len(history),
                                     "pruned": sorted(after - before)})
        kinds.append(kind)
        i += 1
    return {"history": history, "gate_log": gate_log, "dup": dup, "trie": trie,
            "proposals": i, "hours": hours, "prune_events": prune_events,
            "pruned": _pruned_prefixes(trie) if record else [], "llm": llm,
            "kinds": {k: kinds.count(k) for k in set(kinds)}}


def _best_curve(hist):
    """Best-validation-so-far -> (hours, test of that arch) step points."""
    pts, best = [], None
    for h in hist:
        if best is None or h["acc"] > best["acc"]:
            best = h
        pts.append((h["h"], best["test"]))
    return pts


def _wilcoxon(a, b):
    n = len(a)
    d = [x - y for x, y in zip(a, b)]
    md = sum(d) / n if n else 0.0
    p = None
    try:
        if n >= 5 and any(abs(x) > 1e-12 for x in d):
            from scipy.stats import wilcoxon
            p = float(wilcoxon(a, b).pvalue)
    except Exception:
        p = None
    return md, p


LIVE_ARMS = [
    ("v2_gate", "ArchTree + ZC gate", dict(mode="v2", gated=True)),
    ("v2", "ArchTree, no gate", dict(mode="v2", gated=False)),
    ("v1", "Trie-UCB, raw reward (earlier design)", dict(mode="v1", gated=False)),
    ("re", "Regularized Evolution", dict(mode="re", gated=False)),
    ("random", "Random Search", dict(mode="random", gated=False)),
]


def _compare(bench, ops, dataset, k, base, zc, pct, n_seeds=8, max_sec=25.0):
    """Every arm gets the same GPU-hour budget H = k x mean training time of one architecture
    (the step-22 protocol). Reported: test accuracy of the best-validation architecture found
    within H, mean +/- SD over seeds, paired Wilcoxon vs ArchTree + ZC gate."""
    H = k * _mean_cost(bench, dataset) / 3600
    arms = [a for a in LIVE_ARMS if not (a[2]["gated"] and zc is None)]
    t0, res = time.time(), {a[0]: [] for a in arms}
    seeds_done = 0
    for s in range(n_seeds):
        if s >= 3 and time.time() - t0 > max_sec:
            break
        for key, _, spec in arms:
            r = _search(bench, ops, dataset, base + 7919 * (s + 1), mode=spec["mode"],
                        gate=zc if spec["gated"] else None, pct=pct, hour_budget=H,
                        max_props=12 * k, record=False)
            res[key].append({"curve": _best_curve(r["history"]), "evals": len(r["history"]),
                             "props": r["proposals"]})
        seeds_done += 1
    grid = [H * j / 40 for j in range(1, 41)]
    out = []
    for key, label, spec in arms:
        runs = res[key]
        if not runs:
            continue
        mat = []
        for r in runs:
            row, ci, cur = [], 0, None
            for g in grid:
                while ci < len(r["curve"]) and r["curve"][ci][0] <= g + 1e-9:
                    cur = r["curve"][ci][1]
                    ci += 1
                row.append(cur)
            mat.append(row)
        mean = []
        for j in range(len(grid)):
            col = [m[j] for m in mat if m[j] is not None]
            mean.append(round(sum(col) / len(col), 3) if len(col) == len(mat) else None)
        finals = [r["curve"][-1][1] if r["curve"] else None for r in runs]
        fin = [f for f in finals if f is not None]
        mu = sum(fin) / len(fin) if fin else None
        sd = (sum((x - mu) ** 2 for x in fin) / max(1, len(fin) - 1)) ** .5 if len(fin) > 1 else 0.0
        out.append({"key": key, "label": label, "gated": spec["gated"], "mode": spec["mode"],
                    "grid": [round(g, 3) for g in grid], "mean": mean, "finals": finals,
                    "final_mean": None if mu is None else round(mu, 3), "final_sd": round(sd, 3),
                    "evals_mean": round(sum(r["evals"] for r in runs) / len(runs), 1),
                    "props_mean": round(sum(r["props"] for r in runs) / len(runs), 1)})
    ref = next((o for o in out if o["key"] == "v2_gate"), None) or next((o for o in out if o["key"] == "v2"), None)
    stats = []
    if ref:
        for o in out:
            if o is ref:
                continue
            pairs = [(a, b) for a, b in zip(ref["finals"], o["finals"]) if a is not None and b is not None]
            if len(pairs) >= 2:
                md, p = _wilcoxon([a for a, _ in pairs], [b for _, b in pairs])
                stats.append({"other": o["label"], "key": o["key"], "n": len(pairs),
                              "mean_diff": round(md, 3), "p": None if p is None else round(p, 4)})
    return {"arms": out, "stats": stats, "n_seeds": seeds_done, "hour_budget": round(H, 3), "k": k,
            "ref": ref["label"] if ref else None}


def _make_prior(cfg, dataset, seed):
    from archtree.llm_prior import LLMPrior
    g = cfg["llm"]["groq"]
    kw = dict(phase="live_demo", bench="nb201", dataset=dataset, seed=seed,
              namespace=f"live_{int(time.time())}", backend="groq", groq_model=g["model"],
              groq_daily_token_budget=g.get("daily_token_budget", 190_000),
              groq_daily_request_budget=g.get("daily_request_budget", 950),
              groq_reasoning_effort=g.get("reasoning_effort", "low"),
              max_calls=LLM_MAX_CALLS)
    if _groq_pool_available():
        try:
            return LLMPrior(**kw, use_key_pool=True), True
        except TypeError:
            pass
    return LLMPrior(**kw), False


def _kendall(x, y):
    try:
        from scipy.stats import kendalltau
        t = float(kendalltau(x, y)[0])
        return None if t != t else round(t, 3)
    except Exception:
        return None


@app.route("/api/run", methods=["POST"])
def api_run():
    body = request.get_json(force=True)
    dataset = body["dataset"]
    budget = max(10, min(400, int(body["budget"])))
    seed = int(body["seed"])
    use_llm = bool(body.get("use_llm", False))
    use_gate = bool(body.get("use_gate", True))
    use_prune = bool(body.get("use_prune", True))
    run_count = int(body.get("run_count", 0))

    if body.get("presentation") and not use_llm:
        hit = _present_load(dataset, budget, seed, use_gate, use_prune)
        if hit is not None:
            return jsonify(hit)
    _SEARCH_LOCK.acquire()
    try:
        if body.get("presentation") and not use_llm:   # the start-up warm-up may have just saved it
            hit = _present_load(dataset, budget, seed, use_gate, use_prune)
            if hit is not None:
                return jsonify(hit)
        resp = _api_run_locked(body, dataset, budget, seed, use_llm, use_gate, use_prune, run_count)
    finally:
        _SEARCH_LOCK.release()
    if body.get("presentation") and not use_llm:
        _present_save(dataset, budget, seed, use_gate, use_prune, resp)
    return jsonify(resp)


# ─────────────────────────────────────────────────────────────── presentation run (saved)
# The "Presentation run" button replays one fixed, known-good search. It is computed once (at
# start-up, in the background) and saved to outputs/live/presentation_run.json, so in front of an
# audience it appears instantly and identically every time. "Run search" always computes live.
PRESENT = {"dataset": "cifar10", "budget": 100, "seed": 42, "use_gate": True, "use_prune": True}
PRESENT_FILE = LIVE_DIR / "presentation_run.json"


def _present_key(dataset, budget, seed, use_gate, use_prune):
    return f"{dataset}|b{budget}|s{seed}|g{int(use_gate)}|p{int(use_prune)}|{BACKEND_VERSION}"


def _present_load(dataset, budget, seed, use_gate, use_prune):
    try:
        d = json.loads(PRESENT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
    if d.get("key") != _present_key(dataset, budget, seed, use_gate, use_prune):
        return None
    out = dict(d["result"])
    out["cached"] = {"saved": d.get("saved"), "wall_sec": out.get("wall_sec")}
    return out


def _present_save(dataset, budget, seed, use_gate, use_prune, resp):
    # never save a run whose gate silently failed to load: the presentation must show the full method
    if use_gate and not resp.get("gate_on"):
        return False
    if resp.get("comparison", {}).get("error"):
        return False
    return _atomic_json(PRESENT_FILE, {"key": _present_key(dataset, budget, seed, use_gate, use_prune),
                                       "saved": time.strftime("%Y-%m-%d %H:%M"), "result": resp})


def _warm_up():
    """Background start-up job: prepare the landing replay and the saved presentation run."""
    try:
        _replay("cifar10")
        print("[warm-up] landing-page replay ready")
    except Exception as e:
        print(f"[warm-up] replay skipped: {type(e).__name__}: {e}")
    p = PRESENT
    if _present_load(p["dataset"], p["budget"], p["seed"], p["use_gate"], p["use_prune"]) is not None:
        print(f"[warm-up] presentation run already saved ({PRESENT_FILE.name})")
        return
    try:
        print("[warm-up] computing the presentation run once (cifar10, 100 proposals, seed 42)...")
        with _SEARCH_LOCK:
            resp = _api_run_locked(dict(p, run_count=0), p["dataset"], p["budget"], p["seed"], False,
                                   p["use_gate"], p["use_prune"], 0)
        ok = _present_save(p["dataset"], p["budget"], p["seed"], p["use_gate"], p["use_prune"], resp)
        print("[warm-up] presentation run saved" if ok else
              "[warm-up] presentation run NOT saved (gate unavailable or comparison failed); the button will compute live")
    except Exception as e:
        print(f"[warm-up] presentation run skipped: {type(e).__name__}: {e}")


def _api_run_locked(body, dataset, budget, seed, use_llm, use_gate, use_prune, run_count):
    from archtree.encoder import OPS_NB201
    from archtree.tcr import tcr as tcr_fn
    from benchmarks.nb201 import NB201
    cfg = load_config()
    st = _settings()
    ops = OPS_NB201
    bench = _get_bench("nb201", NB201, cfg)
    pct = st.PROXY_THRESHOLD_PERCENTILE
    zc, gate_note = _get_zc(dataset) if use_gate else (None, "Gate switched off for this run.")
    base = seed * 100_000 + run_count
    prior, pool_used, llm_err = None, False, None
    if use_llm:
        try:
            prior, pool_used = _make_prior(cfg, dataset, seed)
        except Exception as e:
            llm_err = f"{type(e).__name__}: {e}"

    t0 = time.time()
    main = _search(bench, ops, dataset, base, mode="v2", gate=zc, pct=pct, budget=budget,
                   prior=prior, prune=use_prune)
    wall = time.time() - t0
    trie, hist = main["trie"], main["history"]
    best = max(hist, key=lambda h: h["acc"]) if hist else None
    oracle = {"cifar10": 94.37, "cifar100": 73.51, "ImageNet16-120": 46.20}.get(dataset)
    mean_cost = _mean_cost(bench, dataset)
    # what the same proposals would have cost without the gate: every proposal a real eval
    ungated_h = None
    if main["gate_log"]:
        from archtree.encoder import index_from_tokens_nb201
        ungated_h = sum(float(bench.train_sec(index_from_tokens_nb201(g["arch"].split("|")), dataset) or 0)
                        for g in main["gate_log"]) / 3600

    tau = None
    scored = [g for g in main["gate_log"] if g.get("score") is not None]
    if len(scored) >= 5:
        from archtree.encoder import index_from_tokens_nb201
        real = [bench.query(index_from_tokens_nb201(g["arch"].split("|")), dataset) for g in scored]
        tau = {"value": _kendall([g["score"] for g in scored], real), "n": len(scored)}

    try:
        comparison = _compare(bench, ops, dataset, max(10, budget // 2), base, zc, pct)
    except Exception as e:
        comparison = {"arms": [], "stats": [], "error": f"{type(e).__name__}: {e}"}

    llm_out = None
    if use_llm:
        err = llm_err or main["llm"]["error"]
        llm_out = {"error": err} if err else {
            "calls": main["llm"]["calls"], "hits": main["llm"].get("hits", 0), "fallbacks": main["llm"].get("fallbacks", 0),
            "sec": round(main["llm"]["sec"], 2),
            "model": cfg["llm"]["groq"]["model"], "alpha": LLM_ALPHA, "key_pool": pool_used}

    root_children = {op: {"n": c.visit_count, "val": None if c.accuracy is None else float(c.accuracy)}
                     for op, c in trie.root.children.items()}
    resp = {
        "dataset": dataset, "budget": budget, "seed": seed,
        "history": hist, "gate_log": main["gate_log"], "gate_on": zc is not None, "gate_note": None if zc else gate_note,
        "skipped": sum(1 for g in main["gate_log"] if not g["passed"]),
        "dup_redirected": main["dup"], "kinds": main["kinds"],
        "real_evals": len(hist), "proposals": main["proposals"],
        "tcr": float(tcr_fn(trie, bench.total_archs)),
        "gpu_hours": round(main["hours"], 3), "gpu_hours_ungated": None if ungated_h is None else round(ungated_h, 3),
        "mean_arch_hours": round(mean_cost / 3600, 4),
        "best": None if not best else {**best, "oracle_test": oracle},
        "pruned": main["pruned"], "prune_events": main["prune_events"],
        "root_children": root_children, "ops": ops, "tau": tau,
        "settings": {"reward": "rank", "C": st.C, "mutation_prob": st.MUTATION_PROB, "pct": pct,
                     "warmup": st.WARMUP_REAL_EVALS, "prune_every": st.PRUNE_EVERY_N,
                     "prune_pct": st.PRUNE_PERCENTILE, "prune": use_prune},
        "comparison": comparison, "llm": llm_out, "wall_sec": round(wall, 2),
        "pool": {"env": _groq_pool_available(), "keys": _pool_key_count(), "used": pool_used},
    }
    run_id = f"live__{dataset}__s{seed}__b{budget}__r{run_count}__g{int(zc is not None)}l{int(prior is not None)}"
    stored = _atomic_json(LIVE_DIR / "runs" / f"{run_id}.json", {"payload": body, "result": resp})
    resp["run_log"] = f"outputs/live/runs/{run_id}.json" if stored else None
    return resp


@app.route("/api/replay")
def api_replay():
    """A short real v2 search for the landing-page animation (cached per dataset)."""
    return jsonify(_replay(request.args.get("dataset", "cifar10")))


def _replay(dataset):
    key = ("replay", dataset)
    if key in _CACHE:
        return _CACHE[key]
    from archtree.encoder import OPS_NB201
    from benchmarks.nb201 import NB201
    cfg = load_config()
    bench = _get_bench("nb201", NB201, cfg)
    zc, _ = _get_zc(dataset)
    with _SEARCH_LOCK:
        r = _search(bench, OPS_NB201, dataset, 42, mode="v2", gate=zc,
                    pct=_settings().PROXY_THRESHOLD_PERCENTILE, budget=160)
    out = {"dataset": dataset, "gate_on": zc is not None, "history": r["history"],
           "gate_log": [g for g in r["gate_log"] if not g["passed"]], "prune_events": r["prune_events"],
           "proposals": r["proposals"], "dup": r["dup"], "hours": round(r["hours"], 2),
           "mean_arch_hours": round(_mean_cost(bench, dataset) / 3600, 4)}
    _CACHE[key] = out
    return out


# ─────────────────────────────────────────────────────────────── /api/hardware
def _pareto(points):
    if not points:
        return [], 0.0
    order = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    front, best = [], -1e9
    for i in order:
        if points[i][1] > best:
            front.append(i)
            best = points[i][1]
    lat_ref = 1.05 * max(p[0] for p in points)
    area, prev = 0.0, 0.0
    for i in front:
        lat, acc = points[i]
        area += (lat_ref - lat) * (acc - prev)
        prev = acc
    return front, area / (lat_ref * 100.0) if lat_ref > 0 else 0.0


@app.route("/api/hardware", methods=["POST"])
def api_hardware():
    body = request.get_json(force=True)
    hardware = body["hardware"]
    from archtree.encoder import index_from_tokens_nb201, OPS_NB201
    from benchmarks.hwnb import HWNASBench, HWMetricUnavailable
    cfg = load_config()
    try:
        hw = _get_bench("hw", HWNASBench, cfg)

        def query(paths):
            rows = []
            for p in paths:
                idx = index_from_tokens_nb201(p)
                try:
                    acc, lat = hw.hw_query(idx, hardware=hardware, dataset="cifar10")
                    rows.append({"arch": "|".join(p), "accuracy": round(float(acc), 2),
                                 "latency_ms": round(float(lat), 3)})
                except HWMetricUnavailable:
                    rows.append({"arch": "|".join(p), "accuracy": None, "latency_ms": None})
            return rows

        found = [a.split("|") for a in (body.get("archs") or [])][:60]
        sample = found or [random.choices(OPS_NB201, k=6) for _ in range(12)]
        rows = query(sample)
        ok = [r for r in rows if r["latency_ms"] is not None]
        front, hv = _pareto([(r["latency_ms"], r["accuracy"]) for r in ok])
        for j, r in enumerate(ok):
            r["on_front"] = j in front
        return jsonify({"rows": rows, "from_search": bool(found), "n_supported": len(ok),
                        "front_size": len(front), "hv": round(hv, 4), "unsupported": len(rows) - len(ok)})
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


# ─────────────────────────────────────────────────────────────── /api/story
_CSV = {}


def _csv(rel: str):
    """outputs/<rel> as a DataFrame (cached on mtime) or None."""
    p = OUT / rel
    if not p.exists():
        return None
    mt = p.stat().st_mtime
    hit = _CSV.get(rel)
    if hit and hit[0] == mt:
        return hit[1]
    try:
        df = pd.read_csv(p)
        df = df if len(df) else None
    except Exception:
        df = None
    _CSV[rel] = (mt, df)
    return df


DS_ORDER = ["cifar10", "cifar100", "ImageNet16-120"]


def _wtl(series):
    vc = series.value_counts()
    return {"win": int(vc.get("win", 0)), "tie": int(vc.get("tie", 0)), "loss": int(vc.get("loss", 0))}


def _sec_headline():
    p, s = _csv("equal_gpu_hours/results/egh_paired_tests.csv"), _csv("equal_gpu_hours/results/egh_summary.csv")
    ranks, fr = _csv("final/results/stats_avg_ranks.csv"), _csv("final/results/stats_friedman.csv")
    out = {}
    if p is not None:
        rows = []
        for other, g in p.groupby("other"):
            rows.append({"other": other, **_wtl(g["outcome_holm"]), "n": int(len(g)),
                         "mean_diff": _clean(float(g["mean_diff_archtree_minus_other"].mean()))})
        out["wtl"] = sorted(rows, key=lambda r: -r["mean_diff"])
        out["cells"] = _recs(p, ["benchmark", "dataset", "budget", "other", "n_paired_seeds", "mean_diff_archtree_minus_other",
                                 "wilcoxon_p", "cohens_d", "p_holm", "outcome_holm"])
        out["totals"] = _wtl(p["outcome_holm"])
        out["n_cells"] = int(p.groupby(["benchmark", "dataset", "budget"]).ngroups)
    if s is not None:
        out["summary"] = _recs(s, ["benchmark", "dataset", "k", "arm", "n", "test_mean", "test_std",
                                   "real_evals", "gpu_hours_used", "gpu_hours_budget", "under_budget"])
        out["n_runs"] = int(s["n"].sum())
    if ranks is not None:
        out["ranks"] = _recs(ranks)
    if fr is not None:
        out["friedman"] = _recs(fr)
    return out


def _sec_gate():
    out = {}
    cal = _csv("phase2/results/phase2_proxy_calibration.csv")
    if cal is not None:
        g = cal.groupby(["dataset", "proxy_name"])["kendall_tau"].mean().reset_index()
        out["tau"] = _recs(g)
    gs = _csv("gated_baselines/results/gated_summary.csv")
    if gs is not None:
        a = gs[gs["proposer"] == "archtree"]
        out["savings"] = _recs(a, ["dataset", "budget", "gate", "acc_mean", "acc_std", "real_evals",
                                   "proxy_evals", "labels", "train_hours"])
    cc = _csv("equal_gpu_hours/results/cost_correlation.csv")
    if cc is not None:
        out["cost_corr"] = _recs(cc)
    gm = _csv("equal_gpu_hours/results/gate_cost_model.csv")
    if gm is not None:
        sl = gm.drop_duplicates(["benchmark", "dataset"])[["benchmark", "dataset", "r_PA", "r_PD", "r_DA", "slope_at_0",
                                                           "penalty_helps_selection"]]
        out["slopes"] = _recs(sl)
    gv = _csv("equal_gpu_hours/results/gate_cost_validation.csv")
    if gv is not None:
        out["validation"] = _recs(gv)
        try:
            from scipy.stats import spearmanr
            agg = []
            for prop, g in gv.groupby("proposer"):
                rho, rho_p = (float(x) for x in spearmanr(g["predicted_delta"], g["measured_delta"]))
                sign = float((np.sign(g["predicted_delta"]) == np.sign(g["measured_delta"])).mean())   # same rule as run_equal_gpu_hours.py
                agg.append({"proposer": prop, "spearman": round(rho, 3), "spearman_p": rho_p, "sign_agree": round(sign, 3), "n": int(len(g))})
            out["validation_agg"] = agg
        except Exception:
            pass
    pc = _csv("final/results/proxy_cost_summary.csv")
    if pc is not None:
        out["proxy_cost"] = _recs(pc)
    return out


def _sec_strategy():
    out = {}
    gp = _csv("gated_baselines/results/gated_paired_tests.csv")
    if gp is not None:
        m = []
        for (gate, other), g in gp.groupby(["gate", "other"]):
            m.append({"gate": gate, "other": other, **_wtl(g["outcome_holm"]),
                      "mean_diff": _clean(float(g["mean_diff_archtree_minus_other"].mean()))})
        out["same_gate"] = m
    gs = _csv("gated_baselines/results/gated_summary.csv")
    if gs is not None:
        out["same_gate_summary"] = _recs(gs, ["dataset", "budget", "gate", "proposer", "acc_mean", "real_evals", "train_hours"])
    ds, dt = _csv("design_ablation/results/design_summary.csv"), _csv("design_ablation/results/design_paired_tests.csv")
    if ds is not None:
        out["design"] = _recs(ds)
    if dt is not None:
        rows = []
        for arm, g in dt.groupby("arm"):
            vc = g["outcome_vs_v2"].value_counts()
            rows.append({"arm": arm, "mean_diff": _clean(float(g["mean_diff_arm_minus_v2"].mean())),
                         "v2_better": int(vc.get("v2 better", 0)),
                         "arm_better": int(sum(v for k, v in vc.items() if k not in ("tie", "v2 better"))),
                         "n": int(len(g))})
        out["design_tests"] = rows
    a7s, a7t = _csv("ablations/results/A7_summary.csv"), _csv("ablations/results/A7_paired_tests.csv")
    if a7s is not None:
        out["a7"] = _recs(a7s)
    if a7t is not None:
        out["a7_tests"] = _recs(a7t)
    us, ut = _csv("ucb_sensitivity/results/ucb_summary.csv"), _csv("ucb_sensitivity/results/ucb_paired_tests.csv")
    if us is not None:
        out["ucb"] = _recs(us)
    if ut is not None:
        out["ucb_tests"] = {"n": int(len(ut)), "sig": int(ut["significant_holm"].astype(bool).sum())}
    pa, pt = _csv("proxy_ablation/results/proxy_ablation_summary.csv"), _csv("proxy_ablation/results/proxy_ablation_paired_tests.csv")
    if pa is not None:
        out["proxy_ablation"] = _recs(pa)
    if pt is not None:
        out["proxy_ablation_tests"] = {"n": int(len(pt)), "sig": int(pt["significant_holm"].astype(bool).sum())}
    return out


def _sec_general():
    out = {}
    ns, npt, ne = (_csv("nb101/results/nb101_summary.csv"), _csv("nb101/results/nb101_paired_tests.csv"),
                   _csv("nb101/results/nb101_equal_compute_tests.csv"))
    if ns is not None:
        out["nb101"] = _recs(ns)
    if npt is not None:
        out["nb101_tests"] = [{"other": o, **_wtl(g["outcome_holm"])} for o, g in npt.groupby("other")]
    if ne is not None:
        out["nb101_eqc"] = _recs(ne, ["gate", "archtree_budget", "archtree_real_evals_max", "other",
                                      "mean_diff_archtree_minus_other", "p_holm", "outcome_holm"])
    nc = _csv("nb101/results/phase2_proxy_calibration.csv")
    if nc is not None:
        out["nb101_tau"] = _recs(nc.groupby("proxy_name")["kendall_tau"].mean().reset_index()) if "proxy_name" in nc.columns else _recs(nc)
    return out


def _sec_llm():
    out = {}
    cm = _csv("convergence/results/convergence_per_run_metrics.csv")
    if cm is not None:
        out["conv_runs"] = _recs(cm, ["arm", "dataset", "budget", "seed", "evals_to_1.0pp", "best_at_10",
                                      "best_at_25", "auc30", "final_best"])
    ct = _csv("convergence/results/convergence_paired_tests.csv")
    if ct is not None:
        out["conv_tests"] = _recs(ct[ct["metric"].isin(["evals_to_1.0pp", "best_at_10", "auc30", "final_best"])],
                                  ["dataset", "budget", "metric", "arm_a", "arm_b", "n_paired_seeds", "mean_on", "mean_off",
                                   "mean_diff", "wilcoxon_p", "cohens_d", "p_holm", "significant", "test_note"])
    cs, cts = _csv("llm_contamination/results/contamination_summary.csv"), _csv("llm_contamination/results/contamination_tests.csv")
    if cs is not None:
        out["contamination"] = _recs(cs)
    if cts is not None:
        out["contamination_tests"] = _recs(cts)
    ls, lp = _csv("llm_size/results/llm_size_summary.csv"), _csv("llm_size/results/llm_size_paired_tests.csv")
    if ls is not None:
        out["size"] = _recs(ls)
    if lp is not None:
        out["size_tests"] = _recs(lp)
    av = _csv("alpha_sweep/results/alpha_vs_baseline_stats.csv")
    if av is not None:
        u = av.drop_duplicates(["dataset", "budget", "same_runs_as_alpha"]) if "same_runs_as_alpha" in av.columns else av
        out["alpha"] = _recs(av, ["alpha", "dataset", "budget", "mean_diff_vs_alpha0", "p_value", "p_holm", "significant"])
        out["alpha_counts"] = {"distinct": int(len(u)), "positive": int((u["mean_diff_vs_alpha0"] > 0).sum()),
                               "sig": int(u["significant"].astype(bool).sum())}
    return out


def _sec_hardware():
    out = {}
    hp = _csv("phase4/results/phase4_hw_paired_tests.csv")
    if hp is not None:
        out["tests"] = {"n": int(len(hp)), **_wtl(hp["outcome_holm"])}
        out["by_device"] = [{"hardware": h, **_wtl(g["outcome_holm"])} for h, g in hp.groupby("hardware")]
    hv = _csv("phase4/results/phase4_hv_igd_summary.csv")
    if hv is not None:
        g = hv.groupby(["hardware", "budget"]).agg(hv=("hypervolume", "mean"), igd=("igd", "mean"),
                                                   front=("pareto_front_size", "mean")).reset_index()
        out["hv"] = _recs(g)
    ov = _csv("final/results/cross_device_truth_overlap.csv")
    if ov is not None:
        out["overlap"] = _recs(ov)
    tr = _csv("final/results/cross_device_truth_transfer.csv")
    if tr is not None:
        out["transfer"] = _recs(tr)
    return out


def _sec_rigor():
    out = {}
    eq = _csv("final/results/equal_compute_statistical_tests_strict.csv")
    if eq is not None:
        ext = eq[eq["comparator_type"] != "internal"] if "comparator_type" in eq.columns else eq
        out["phase6"] = [{"baseline": b, "holm": _wtl(g["outcome_holm"]), "raw": _wtl(g["outcome_raw"]),
                          "mean_diff": _clean(float(g["mean_diff_archtree_minus_baseline"].mean()))}
                         for b, g in ext.groupby("baseline")]
        out["phase6_totals"] = {"holm": _wtl(ext["outcome_holm"]), "raw": _wtl(ext["outcome_raw"]), "n": int(len(ext))}
    pi = _csv("final/results/stats_prob_improvement.csv")
    if pi is not None:
        out["prob"] = _recs(pi)
    to = _csv("final/results/stats_tost.csv")
    if to is not None:
        g = to.groupby(["source", "other", "delta"]).agg(n=("equivalent", "size"),
                                                         eq=("equivalent", lambda x: int(x.astype(bool).sum()))).reset_index()
        out["tost"] = _recs(g)
    fr = _csv("final/results/stats_friedman.csv")
    if fr is not None:
        out["friedman"] = _recs(fr)
    rc = _csv("final/results/regret_curves.csv")
    if rc is not None:
        keep = ["archtree_llm", "re", "local_search", "re_zc_warmup", "random_search", "zico", "random_exact"]
        sub = rc[rc["method"].isin(keep) & (rc["budget"] == rc["budget"].max())]
        sub = sub[(sub["t"] <= 40) | (sub["t"] % 5 == 0)]
        out["regret"] = _recs(sub, ["method", "dataset", "budget", "t", "mean_regret", "sem"])
    rf = _csv("final/results/regret_fit.csv")
    if rf is not None:
        out["regret_fit"] = _recs(rf[rf["budget"] == rf["budget"].max()],
                                  ["method", "dataset", "b_exponent", "b_ci_lo", "b_ci_hi", "simple_regret_at_T"])
    return out


def _sec_leaderboard():
    out = {}
    m = _csv("final/results/main_results_table.csv")
    if m is not None:
        std = m[m["dataset"].isin(DS_ORDER)]
        out["main"] = _recs(std, ["method", "dataset", "budget", "best_acc_mean", "best_acc_std"])
        hw = m[~m["dataset"].isin(DS_ORDER)]
        out["main_hw"] = _recs(hw, ["method", "dataset", "budget", "best_acc_mean", "best_acc_std"])
    d = _csv("final/results/duplicate_evals_summary.csv")
    if d is not None:
        out["evals"] = _recs(d, ["method", "dataset", "budget", "real_evals_mean", "dup_real_mean"])
    e = _csv("final/results/eval_efficiency_summary.csv")
    if e is not None:
        out["efficiency"] = _recs(e)
    return out


def _sec_repro():
    out = {}
    try:
        p = OUT / "final" / "repro" / "environment.json"
        if p.exists():
            env = json.loads(p.read_text())
            out["environment"] = {k: env[k] for k in list(env)[:12] if not isinstance(env[k], (dict, list))}
    except Exception:
        pass
    z = PROJECT_ROOT / "release" / "archtree_repro_bundle.zip"
    out["bundle"] = {"exists": z.exists(), "mb": round(z.stat().st_size / 1e6, 2) if z.exists() else None}
    out["n_figures"] = len(list(FIG_DIR.glob("*.png"))) if FIG_DIR.exists() else 0
    return out


SECTIONS = {"headline": _sec_headline, "gate": _sec_gate, "strategy": _sec_strategy,
            "general": _sec_general, "llm": _sec_llm, "hardware": _sec_hardware,
            "rigor": _sec_rigor, "leaderboard": _sec_leaderboard, "repro": _sec_repro}


# every result file the dashboard reads, by section (also listed by /api/status)
SOURCES = {
    "Equal GPU-hours": ["equal_gpu_hours/results/egh_summary.csv", "equal_gpu_hours/results/egh_paired_tests.csv"],
    "Gate and cost": ["phase2/results/phase2_proxy_calibration.csv", "equal_gpu_hours/results/cost_correlation.csv",
                      "equal_gpu_hours/results/gate_cost_model.csv", "equal_gpu_hours/results/gate_cost_validation.csv",
                      "final/results/proxy_cost_summary.csv"],
    "Search strategy": ["gated_baselines/results/gated_summary.csv", "gated_baselines/results/gated_paired_tests.csv",
                        "design_ablation/results/design_summary.csv", "design_ablation/results/design_paired_tests.csv",
                        "ablations/results/A7_summary.csv", "ablations/results/A7_paired_tests.csv",
                        "ucb_sensitivity/results/ucb_summary.csv", "ucb_sensitivity/results/ucb_paired_tests.csv",
                        "proxy_ablation/results/proxy_ablation_summary.csv", "proxy_ablation/results/proxy_ablation_paired_tests.csv"],
    "NAS-Bench-101": ["nb101/results/nb101_summary.csv", "nb101/results/nb101_paired_tests.csv",
                      "nb101/results/nb101_equal_compute_tests.csv"],
    "LLM prior": ["convergence/results/convergence_per_run_metrics.csv", "convergence/results/convergence_paired_tests.csv",
                  "llm_contamination/results/contamination_summary.csv", "llm_contamination/results/contamination_tests.csv",
                  "llm_size/results/llm_size_summary.csv", "llm_size/results/llm_size_paired_tests.csv",
                  "alpha_sweep/results/alpha_vs_baseline_stats.csv"],
    "Hardware": ["phase4/results/phase4_hw_paired_tests.csv", "phase4/results/phase4_hv_igd_summary.csv",
                 "final/results/cross_device_truth_overlap.csv", "final/results/cross_device_truth_transfer.csv"],
    "Statistics": ["final/results/equal_compute_statistical_tests_strict.csv", "final/results/stats_friedman.csv",
                   "final/results/stats_avg_ranks.csv", "final/results/stats_prob_improvement.csv",
                   "final/results/stats_tost.csv", "final/results/regret_curves.csv", "final/results/regret_fit.csv"],
    "Leaderboard": ["final/results/main_results_table.csv", "final/results/duplicate_evals_summary.csv",
                    "final/results/eval_efficiency_summary.csv"],
}
_STORY = {"sig": None, "out": None}


def _sources_sig():
    sig = []
    for files in SOURCES.values():
        for rel in files:
            p = OUT / rel
            sig.append(p.stat().st_mtime if p.exists() else 0)
    return tuple(sig)


@app.route("/api/story")
def api_story():
    """All offline results, aggregated. Rebuilt only when a source file changes."""
    sig = _sources_sig()
    if _STORY["sig"] == sig and _STORY["out"] is not None:
        return jsonify(_STORY["out"])
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "root": PROJECT_ROOT.name, "version": BACKEND_VERSION}
    for k, fn in SECTIONS.items():
        try:
            out[k] = fn()
        except Exception as e:
            out[k] = {"error": f"{type(e).__name__}: {e}"}
    _STORY.update(sig=sig, out=out)
    return jsonify(out)


@app.route("/api/status")
def api_status():
    """Which result files exist, how many rows they have and when they were written."""
    rows = []
    for section, files in SOURCES.items():
        for rel in files:
            p = OUT / rel
            n = None
            if p.exists():
                try:
                    with open(p, "rb") as f:
                        n = max(0, sum(1 for _ in f) - 1)
                except OSError:
                    n = None
            rows.append({"section": section, "file": rel, "exists": p.exists(), "rows": n,
                         "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)) if p.exists() else None})
    return jsonify({"version": BACKEND_VERSION, "files": rows,
                    "found": sum(r["exists"] for r in rows), "total": len(rows)})


# ─────────────────────────────────────────────────────────────── figures
@app.route("/api/figures")
def api_figures():
    if not FIG_DIR.exists():
        return jsonify({"figures": [], "missing": "plots/figures/ not found - run python plots/plot_results.py all"})
    figs = []
    for p in sorted(FIG_DIR.glob("*.png")):
        stem = p.stem
        group, _, name = stem.partition("__")
        figs.append({"file": p.name, "group": group if name else "other", "name": name or stem,
                     "csv": (p.with_suffix(".csv").name if p.with_suffix(".csv").exists() else None),
                     "kb": round(p.stat().st_size / 1024)})
    return jsonify({"figures": figs})


@app.route("/figures/<path:name>")
def figure_file(name):
    if "/" in name or "\\" in name or ".." in name or not name.lower().endswith((".png", ".csv")):
        abort(404)
    if not (FIG_DIR / name).exists():
        abort(404)
    return send_from_directory(FIG_DIR, name, max_age=300)


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    _port = int(os.environ.get("ARCHTREE_PORT", 5000))
    print(f"Open http://localhost:{_port}")
    if os.environ.get("ARCHTREE_NO_WARMUP") != "1":
        threading.Thread(target=_warm_up, daemon=True).start()
    # no debug reloader: the search touches cache files and the reloader would restart mid-request
    app.run(host=os.environ.get("ARCHTREE_HOST", "127.0.0.1"), port=_port,
            debug=False, use_reloader=False, threaded=True)
