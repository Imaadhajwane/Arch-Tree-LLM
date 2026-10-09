# archtree/search.py
"""
Search loops for phases 1-4.

  search_phase1   trie + UCB1 + weight bank
  search_phase2   phase 1 + ZC-proxy gating
  search_phase3   phase 2 + LLM prior (C7)
  search_flat_ucb ablation: per-position UCB1, no trie
  search_phase4   hardware-aware (accuracy + latency Pareto), cache-read-only LLM

Oct 2026 changes (see ReadMe/FIXES_OCT2026.md):
- Constants come from config.yaml via archtree/settings.py (C, pruning, gate, latency penalty).
- One UCB scale (accuracy/100) everywhere, including search_flat_ucb.
- Each result row/log carries best_acc (search metric = validation accuracy with the v2 cache)
  AND best_test_acc (test accuracy of that same architecture) - report best_test_acc.
- search_phase3 reports llm_fallbacks (LLM calls that silently returned uniform scores).
- Pruning uses the value rule (trie.prune), see trie.py.

Rules worth remembering:
- New metrics go to NEW csv files or the per-run JSON log. append_result_row()
  uses the first row's keys as the header, so adding a key to an existing
  results file raises ValueError.
- When UCB proposes an already-visited leaf, random_unvisited() forces a new
  one. Without it, backup_mode="max" locks onto one branch and burns the
  budget on no-op revisits.
- Ablation runs (ablation_id != "") suffix the run-log name so arms at the same
  (seed, budget) don't overwrite each other. ablation_id == "" keeps the old
  filename, so run_phase3.py's resume check still works.
- Every eval_history entry carries t_sec. Phase 3 also logs llm_sec and
  proxy_sec (JSON only).
- Phase 4: archs with no valid HW latency (HWMetricUnavailable) still count as
  real evals and trie visits, but never enter the Pareto archive, HV or IGD.
"""
import json
import math
import random
import time
from typing import Dict, List

import numpy as np
import torch

import archtree.storage as storage
from archtree import settings
from archtree.encoder import OPS_NB201, index_from_tokens_nb201
from archtree.llm_prior import LLMPrior
from archtree.tcr import tcr, tcr_h, update_pareto_archive, hypervolume, igd
from archtree.trie import ArchTrie, TrieNode
from archtree.ucb import traverse_with_llm, backpropagate, bump_visit_only, random_unvisited
from archtree import ucb as _ucb
from archtree.weight_bank import WeightBank
from archtree.zc_proxy import ZCProxyEnsemble, get_mini_batch
from benchmarks.hwnb import HWMetricUnavailable
from benchmarks.nb201 import best_from_history


PRUNE_EVERY_N = settings.PRUNE_EVERY_N
PRUNE_PERCENTILE = settings.PRUNE_PERCENTILE
WARMUP_REAL_EVALS = settings.WARMUP_REAL_EVALS          # first N iterations always get a real eval
PROXY_THRESHOLD_PERCENTILE = settings.PROXY_THRESHOLD_PERCENTILE   # 35 (was a silent 50 default)
LATENCY_PENALTY = settings.LATENCY_PENALTY
LATENCY_NORM_MS = settings.LATENCY_NORM_MS
C_DEFAULT = settings.C


def _bump_visit_only(trie: ArchTrie, path: List[str]) -> None:
    """Count a proxy-skipped visit along the path (no eval, no accuracy update).
    Also bumps root, so depth-0 UCB sees the true total trials."""
    trie.root.visit_count += 1
    node = trie.root
    for tok in path:
        if tok not in node.children:
            node.children[tok] = TrieNode()
        node = node.children[tok]
        node.visit_count += 1


# ── benchmark-generic helpers (Oct 2026, NB101 support) ──────────────────────
# NB201 (no `token_vocab` attribute) goes through exactly the old code path, so NB201 results are
# unchanged bit for bit. Benchmarks with a per-position vocabulary (NB101) can propose INVALID token
# strings (disconnected cell, > 9 edges) and token strings that are a different spelling of an
# already-evaluated graph; both are redirected to a random not-yet-evaluated graph (no real eval
# is spent on them), invalid leaves are marked dead in the trie so UCB stops proposing them.

def _is_generic(bench) -> bool:
    return hasattr(bench, "token_vocab")


def _space(bench, ops):
    """(ops / per-position vocab, trie depth) of this benchmark."""
    if ops is None:
        ops = bench.token_vocab() if _is_generic(bench) else OPS_NB201
    return ops, int(getattr(bench, "depth", 6))


def _fix_proposal(bench, trie, path, ops, evaluated: set, counters: dict):
    """-> (path, idx) to use this iteration. NB201: the old `is_visited -> random_unvisited` rule."""
    if not _is_generic(bench):
        if trie.is_visited(path):
            counters["dup"] += 1
            path = random_unvisited(trie, ops)
        return path, index_from_tokens_nb201(path)
    idx = bench.tokens_to_index(path)
    if idx is None:
        counters["invalid"] += 1
        trie.mark_dead(path)
    elif idx in evaluated or trie.is_visited(path):
        counters["dup"] += 1
    else:
        return path, idx
    idx = bench.random_unevaluated_index(evaluated)
    return bench.index_to_tokens(idx), idx


def _mutation_proposal(bench, history, is_new, tournament: int = None, tries: int = 20):
    """ArchTree v2 evolutionary expansion: tournament-select a parent among the really evaluated
    archs (best of `tournament` random ones, as in regularized evolution) and mutate it with the
    benchmark's own mutation. Returns tokens, or None if no new architecture was found."""
    tournament = tournament or settings.MUTATION_TOURNAMENT
    pool = random.sample(history, min(tournament, len(history)))
    parent = max(pool, key=lambda e: e["acc"])["arch"].split("|")
    for _ in range(tries):
        cand = bench.mutate_tokens(parent, random)
        if is_new(cand):
            return cand
    return None


def _train_hours(bench, history, dataset) -> float:
    """Training time (hours) of the really evaluated archs = the real cost of the search."""
    if not hasattr(bench, "train_sec"):
        return float("nan")
    tot = 0.0
    for e in history:
        idx = (bench.arch_key_to_index(e["arch"]) if hasattr(bench, "arch_key_to_index")
               else index_from_tokens_nb201(e["arch"].split("|")))
        t = bench.train_sec(idx, dataset)
        tot += float(t) if t is not None else 0.0
    return tot / 3600.0


def _report(bench, history, dataset):
    """best_from_history for any benchmark (selection by search metric, report by test)."""
    if hasattr(bench, "best_from_history"):
        return bench.best_from_history(history, dataset)
    return best_from_history(bench, history, dataset)


def _log_dedup_stats(phase: str, filename: str, row: dict) -> None:
    """Per-run duplicate-proposal row. Separate file per phase to avoid schema clashes."""
    storage.append_result_row(phase, filename, row)


def _random_unvisited_set(visited: set, ops: List[str], depth: int,
                           max_tries: int = 500) -> List[str]:
    """Set-backed version of ucb.random_unvisited(). If nothing new is found
    in max_tries, accept a revisit rather than loop forever."""
    for _ in range(max_tries):
        cand = [random.choice(ops) for _ in range(depth)]
        if tuple(cand) not in visited:
            return cand
    return [random.choice(ops) for _ in range(depth)]


def search_phase1(bench, budget: int, dataset: str, seed: int,
                   phase: str = "phase1", ops=None, C: float = C_DEFAULT,
                   backup_mode: str = "mean") -> Dict:
    """ArchTree v1: trie + UCB1 + weight bank. No LLM, no ZC proxy."""
    from archtree.ucb import traverse  # plain UCB1, no LLM prior
    random.seed(seed)
    torch.manual_seed(seed)
    _ucb.reward_reset()
    ops, depth = _space(bench, ops)
    t0 = time.time()

    trie = ArchTrie(max_depth=depth)
    weight_bank = WeightBank(phase, bench.name, dataset, seed)

    log = {"eval_history": []}
    real_count = 0
    cnt = {"dup": 0, "invalid": 0}   # dup = UCB proposed an already-evaluated leaf/graph
    evaluated: set = set()
    tcr_val = 0.0

    for i in range(budget):
        path = traverse(trie, ops, C=C)
        path, idx = _fix_proposal(bench, trie, path, ops, evaluated, cnt)

        acc = bench.query(idx, dataset)
        real_count += 1
        evaluated.add(idx)
        _ucb.reward_register(acc)

        anc_state = weight_bank.load_ancestor(trie, path)
        new_state = weight_bank.simulate_train_new_layers(path, anc_state)
        weight_bank.store(path, new_state)

        node = trie.insert(path)
        node.accuracy = acc
        backpropagate(trie, path, acc, backup_mode=backup_mode)

        log["eval_history"].append({"iter": i + 1, "arch": "|".join(path), "acc": acc,
                                    "t_sec": round(time.time() - t0, 3)})

        tcr_val = tcr(trie, bench.total_archs)
        best_so_far = max((e["acc"] for e in log["eval_history"]), default=0.0)
        storage.append_result_row_buffered(phase, "phase1_tcr_per_iter.csv", {
            "method": "archtree_v1", "benchmark": bench.name, "dataset": dataset,
            "seed": seed, "budget": budget,
            "iteration": i + 1, "tcr": tcr_val,
            "best_acc_so_far": best_so_far,
        })

        if (i + 1) % PRUNE_EVERY_N == 0:
            trie.prune(PRUNE_PERCENTILE)

    weight_bank.flush()
    elapsed = time.time() - t0
    best = max(log["eval_history"], key=lambda x: x["acc"], default={"acc": 0.0})
    rep = _report(bench, log["eval_history"], dataset)

    log.update({
        "best_test_acc": rep["best_test_acc"], "best_arch": rep["best_arch"],
        "method": "archtree_v1", "benchmark": bench.name, "dataset": dataset,
        "seed": seed, "budget": budget, "best_acc": best["acc"],
        "eval_count": budget, "real_eval_count": real_count,
        "dup_proposals": cnt["dup"], "invalid_proposals": cnt["invalid"], "wall_time_sec": elapsed,
    })
    storage.save_run_log(phase, "archtree_v1", bench.name, dataset, seed, log, budget=budget)

    result_row = {
        "method": "archtree_v1", "benchmark": bench.name, "dataset": dataset,
        "seed": seed, "budget": budget, "best_acc": best["acc"],
        "best_test_acc": rep["best_test_acc"],
        "eval_count": budget, "real_eval_count": real_count,
        "tcr": tcr_val, "train_hours": _train_hours(bench, log["eval_history"], dataset),
        "wall_time_sec": elapsed,
    }
    storage.flush_row_buffers()
    storage.append_result_row(phase, "phase1_main_results.csv", result_row)
    _log_dedup_stats(phase, "phase1_dedup_stats.csv", {
        "method": "archtree_v1", "benchmark": bench.name, "dataset": dataset,
        "seed": seed, "budget": budget, "ablation_id": "",
        "real_eval_count": real_count, "dup_proposals": cnt["dup"],
        "repeat_proxy_skips": 0,
    })

    return {"best_acc": best["acc"], "real_eval_count": real_count, **result_row}


def search_phase2(bench, budget: int, dataset: str, seed: int, zc,
                   phase: str = "phase2", ops=None, C: float = C_DEFAULT,
                   backup_mode: str = "mean",
                   proxy_threshold_percentile: int = PROXY_THRESHOLD_PERCENTILE) -> Dict:
    """Phase 1 + ZC-proxy gating (C8). No LLM."""
    from archtree.ucb import traverse
    random.seed(seed)
    torch.manual_seed(seed)
    _ucb.reward_reset()
    ops, depth = _space(bench, ops)
    t0 = time.time()

    trie = ArchTrie(max_depth=depth)
    weight_bank = WeightBank(phase, bench.name, dataset, seed)
    get_mini_batch()

    log = {"eval_history": []}
    real_count = proxy_count = 0
    repeat_proxy_skips = 0
    cnt = {"dup": 0, "invalid": 0}
    evaluated: set = set()
    proxy_skipped_paths: set = set()   # paths the proxy already rejected once
    proxy_scores_seen: List[float] = []
    tcr_val = 0.0

    for i in range(budget):
        path = traverse(trie, ops, C=C)
        path, idx = _fix_proposal(bench, trie, path, ops, evaluated, cnt)

        # Proxy gate: after warm-up, real-eval only if the proxy score is in the
        # top proxy_threshold_percentile% of scores seen so far
        take_real_path = True
        if i >= WARMUP_REAL_EVALS:
            proxy_val = zc.score(path)
            proxy_scores_seen.append(proxy_val)
            thresh = np.percentile(proxy_scores_seen, 100 - proxy_threshold_percentile)
            take_real_path = proxy_val >= thresh
            if not take_real_path:
                if tuple(path) in proxy_skipped_paths:
                    repeat_proxy_skips += 1
                proxy_skipped_paths.add(tuple(path))
                _bump_visit_only(trie, path)
                proxy_count += 1

        if take_real_path:
            acc = bench.query(idx, dataset)
            real_count += 1
            evaluated.add(idx)
            _ucb.reward_register(acc)

            anc_state = weight_bank.load_ancestor(trie, path)
            new_state = weight_bank.simulate_train_new_layers(path, anc_state)
            weight_bank.store(path, new_state)

            node = trie.insert(path)
            node.accuracy = acc
            backpropagate(trie, path, acc, backup_mode=backup_mode)
            log["eval_history"].append({"iter": i + 1, "arch": "|".join(path), "acc": acc,
                                        "t_sec": round(time.time() - t0, 3)})

        tcr_val = tcr(trie, bench.total_archs)
        best_so_far = max((e["acc"] for e in log["eval_history"]), default=0.0)
        storage.append_result_row_buffered(phase, "phase2_tcr_per_iter.csv", {
            "method": "archtree_zc", "benchmark": bench.name, "dataset": dataset,
            "seed": seed, "budget": budget,
            "iteration": i + 1, "tcr": tcr_val,
            "best_acc_so_far": best_so_far,
        })

        if (i + 1) % PRUNE_EVERY_N == 0:
            trie.prune(PRUNE_PERCENTILE)

    weight_bank.flush()
    elapsed = time.time() - t0
    best = max(log["eval_history"], key=lambda x: x["acc"], default={"acc": 0.0})
    rep = _report(bench, log["eval_history"], dataset)

    log.update({
        "best_test_acc": rep["best_test_acc"], "best_arch": rep["best_arch"],
        "method": "archtree_zc", "benchmark": bench.name, "dataset": dataset,
        "seed": seed, "budget": budget, "best_acc": best["acc"],
        "eval_count": budget, "real_eval_count": real_count,
        "proxy_eval_count": proxy_count, "dup_proposals": cnt["dup"],
        "invalid_proposals": cnt["invalid"],
        "repeat_proxy_skips": repeat_proxy_skips, "wall_time_sec": elapsed,
    })
    storage.save_run_log(phase, "archtree_zc", bench.name, dataset, seed, log, budget=budget)

    result_row = {
        "method": "archtree_zc", "benchmark": bench.name, "dataset": dataset,
        "seed": seed, "budget": budget, "best_acc": best["acc"],
        "best_test_acc": rep["best_test_acc"],
        "eval_count": budget, "real_eval_count": real_count,
        "proxy_eval_count": proxy_count, "tcr": tcr_val,
        "train_hours": _train_hours(bench, log["eval_history"], dataset), "wall_time_sec": elapsed,
    }
    storage.flush_row_buffers()
    storage.append_result_row(phase, "phase2_main_results.csv", result_row)
    _log_dedup_stats(phase, "phase2_dedup_stats.csv", {
        "method": "archtree_zc", "benchmark": bench.name, "dataset": dataset,
        "seed": seed, "budget": budget, "ablation_id": "",
        "real_eval_count": real_count, "dup_proposals": cnt["dup"],
        "repeat_proxy_skips": repeat_proxy_skips,
    })

    # Compare real evals against the matching phase-1 run, if it exists
    try:
        p1 = storage.load_results_csv("phase1", "phase1_main_results.csv")
        p1_row = p1[(p1["seed"] == seed) & (p1["budget"] == budget) & (p1["dataset"] == dataset)]
        if not p1_row.empty:
            p1_real = p1_row["real_eval_count"].iloc[0]
            reduction_pct = 100.0 * (1 - real_count / max(p1_real, 1))
            storage.append_result_row(phase, "phase2_eval_reduction.csv", {
                "benchmark": bench.name, "dataset": dataset, "seed": seed, "budget": budget,
                "phase1_real_evals": p1_real, "phase2_real_evals": real_count,
                "reduction_pct": reduction_pct,
            })
    except FileNotFoundError:
        pass

    return {"best_acc": best["acc"], "real_eval_count": real_count, **result_row}


def search_phase3(
    bench,
    budget: int,
    dataset: str,
    seed: int,
    phase: str = "phase3",
    ops=None,
    alpha: float = 0.3,
    alpha_decay: str = "inv_sqrt",
    C: float = C_DEFAULT,
    backup_mode: str = "max",
    use_llm: bool = True,
    use_proxy: bool = True,
    model: str = "llama3.1",
    host: str = "http://localhost:11434",
    max_llm_calls: int = 500,
    prune_percentile: int = PRUNE_PERCENTILE,
    use_weight_inherit: bool = True,
    ablation_id: str = "",           # also namespaces WeightBank / LLM cache per ablation arm
    backend: str = "ollama",         # "ollama" | "groq"
    groq_model: str = "openai/gpt-oss-20b",
    groq_daily_token_budget: int = 190_000,
    groq_daily_request_budget: int = 950,
    groq_reasoning_effort: str = "low",
    use_key_pool: bool = False,
    proxy_threshold_percentile: int = PROXY_THRESHOLD_PERCENTILE,
    results_filename: str = "phase3_main_results.csv",
    tcr_filename: str = "phase3_tcr_per_iter.csv",
    cache_stats_filename: str = "phase3_llm_cache_stats.csv",
    dedup_filename: str = "phase3_dedup_stats.csv",
    zc_override=None,                # pre-built ZCProxyEnsemble (proxy-composition ablation)
    reward_mode: str = None,         # None -> config ucb.reward ("rank" v2 | "raw" v1)
    mutation_prob: float = None,     # None -> config search.mutation_prob (v2: 0.5, v1: 0)
    llm_prompt_mode: str = "standard",   # standard | described | anonymous (contamination test)
) -> Dict:
    """Phase 2 + LLM prior (C7) guiding UCB traversal (+ v2 evolutionary expansion).
    backend="heuristic" gives the knowledge-free size prior (no LLM) for the C7 control."""
    if _is_generic(bench) and use_llm and alpha > 0:
        raise NotImplementedError("the LLM prior prompt is NB201-specific; run NB101 with use_llm=False")
    random.seed(seed)
    torch.manual_seed(seed)
    _ucb.reward_reset(reward_mode)
    reward_mode = reward_mode or settings.REWARD_MODE
    mutation_prob = settings.MUTATION_PROB if mutation_prob is None else float(mutation_prob)
    n_mutations = 0
    ops, depth = _space(bench, ops)
    t0 = time.time()

    trie = ArchTrie(max_depth=depth)
    weight_bank = WeightBank(phase, bench.name, dataset, seed, namespace=ablation_id)
    llm_prior = LLMPrior(
        phase=phase, bench=bench.name, dataset=dataset, seed=seed,  # dataset scopes the LLM cache file
        model=model, base_url=host,
        max_calls=max_llm_calls if use_llm else 0,
        namespace=ablation_id,
        backend=backend,
        groq_model=groq_model,
        groq_daily_token_budget=groq_daily_token_budget,
        groq_daily_request_budget=groq_daily_request_budget,
        groq_reasoning_effort=groq_reasoning_effort,
        use_key_pool=use_key_pool,
        prompt_mode=llm_prompt_mode,
    )

    # Time LLM calls by wrapping the instance method; traverse_with_llm()
    # looks score_branches up on the instance, so llm_prior.py stays untouched.
    llm_sec = [0.0]
    _orig_score_branches = llm_prior.score_branches

    def _timed_score_branches(*a, **k):
        _t = time.perf_counter()
        try:
            return _orig_score_branches(*a, **k)
        finally:
            llm_sec[0] += time.perf_counter() - _t

    llm_prior.score_branches = _timed_score_branches

    zc = None
    proxy_scores_seen: List[float] = []
    if use_proxy:
        zc = zc_override if zc_override is not None else ZCProxyEnsemble.from_saved(
            phase="phase2", bench_name=bench.name, dataset=dataset)
        get_mini_batch()

    log = {"eval_history": []}
    real_count = proxy_count = 0
    repeat_proxy_skips = 0
    cnt = {"dup": 0, "invalid": 0}
    evaluated: set = set()
    proxy_skipped_paths: set = set()
    proxy_sec = 0.0   # timed at the call site so a shared zc_override is never mutated
    tcr_val = 0.0

    for i in range(budget):
        eff_alpha = alpha if use_llm else 0.0
        path = traverse_with_llm(trie, ops, llm_prior, C=C, alpha=eff_alpha, t=i + 1, budget=budget)
        # v2: part of the proposals mutate a good evaluated arch instead (RNG untouched when p == 0)
        if (mutation_prob > 0 and len(log["eval_history"]) >= settings.MUTATION_WARMUP
                and random.random() < mutation_prob):
            cand = _mutation_proposal(bench, log["eval_history"], lambda t: not trie.is_visited(t))
            if cand is not None:
                path = cand
                n_mutations += 1
        path, idx = _fix_proposal(bench, trie, path, ops, evaluated, cnt)

        take_real_path = True
        if use_proxy and i >= WARMUP_REAL_EVALS:
            _tp = time.perf_counter()
            proxy_val = zc.score(path)
            proxy_sec += time.perf_counter() - _tp
            proxy_scores_seen.append(proxy_val)
            thresh = np.percentile(proxy_scores_seen, 100 - proxy_threshold_percentile)
            take_real_path = proxy_val >= thresh
            if not take_real_path:
                if tuple(path) in proxy_skipped_paths:
                    repeat_proxy_skips += 1
                proxy_skipped_paths.add(tuple(path))
                _bump_visit_only(trie, path)
                proxy_count += 1

        if take_real_path:
            acc = bench.query(idx, dataset)
            real_count += 1
            evaluated.add(idx)
            _ucb.reward_register(acc)

            anc_state = weight_bank.load_ancestor(trie, path) if use_weight_inherit else None
            new_state = weight_bank.simulate_train_new_layers(path, anc_state)
            weight_bank.store(path, new_state)

            node = trie.insert(path)
            node.accuracy = acc
            backpropagate(trie, path, acc, backup_mode=backup_mode)

            log["eval_history"].append({"iter": i + 1, "arch": "|".join(path), "acc": acc,
                                        "t_sec": round(time.time() - t0, 3),
                                        "llm_sec": round(llm_sec[0], 3),
                                        "proxy_sec": round(proxy_sec, 3)})

        if (i + 1) % PRUNE_EVERY_N == 0:
            trie.prune(prune_percentile)

        if tcr_filename:                 # None: no per-iteration curve (equal-GPU-hour study)
            tcr_val = tcr(trie, bench.total_archs)
            best_so_far = max((e["acc"] for e in log["eval_history"]), default=0.0)
            storage.append_result_row_buffered(phase, tcr_filename, {
                "method": "archtree_llm", "benchmark": bench.name, "dataset": dataset,
                "seed": seed, "budget": budget,
                "iteration": i + 1, "tcr": tcr_val, "tcr_h": "",
                "best_acc_so_far": best_so_far,
            })

    weight_bank.flush()
    elapsed = time.time() - t0
    best = max(log["eval_history"], key=lambda x: x["acc"], default={"acc": 0.0})
    rep = _report(bench, log["eval_history"], dataset)
    llm_fallbacks = getattr(llm_prior, "fallback_count", 0)
    train_hours = _train_hours(bench, log["eval_history"], dataset)

    # Extra fields go in the JSON log only, never in a CSV row
    log.update({
        "best_test_acc": rep["best_test_acc"], "best_arch": rep["best_arch"],
        "llm_fallbacks": llm_fallbacks, "C": C, "backup_mode": backup_mode,
        "proxy_threshold_percentile": proxy_threshold_percentile,
        "prune_percentile": prune_percentile, "prune_rule": settings.PRUNE_RULE,
        "method": "archtree_llm", "benchmark": bench.name, "dataset": dataset,
        "seed": seed, "budget": budget,
        "best_acc": best["acc"], "eval_count": budget,
        "real_eval_count": real_count, "proxy_eval_count": proxy_count,
        "wall_time_sec": elapsed,
        "alpha": alpha if use_llm else 0.0, "use_llm": use_llm, "use_proxy": use_proxy,
        "ablation_id": ablation_id, "llm_model": llm_prior.model,
        "llm_backend": backend, "llm_sec": round(llm_sec[0], 3),
        "proxy_sec": round(proxy_sec, 3),
        "dup_proposals": cnt["dup"], "repeat_proxy_skips": repeat_proxy_skips,
        "invalid_proposals": cnt["invalid"],
        "reward_mode": reward_mode, "mutation_prob": mutation_prob, "mutation_proposals": n_mutations,
        "train_hours": train_hours,
    })
    _method_tag = "archtree_llm" + (f"__{ablation_id}" if ablation_id else "")
    storage.save_run_log(phase, _method_tag, bench.name, dataset, seed, log, budget=budget)

    llm_prior.log_cache_stats(dataset, filename=cache_stats_filename)

    result_row = {
        "method": "archtree_llm", "benchmark": bench.name, "dataset": dataset,
        "seed": seed, "budget": budget, "best_acc": best["acc"],
        "best_test_acc": rep["best_test_acc"],
        "eval_count": budget, "proxy_eval_count": proxy_count,
        "real_eval_count": real_count, "tcr": tcr_val, "tcr_h": "",
        "train_hours": train_hours, "wall_time_sec": elapsed,
    }
    storage.flush_row_buffers()
    storage.append_result_row(phase, results_filename, result_row)
    _log_dedup_stats(phase, dedup_filename, {
        "method": "archtree_llm", "benchmark": bench.name, "dataset": dataset,
        "seed": seed, "budget": budget, "ablation_id": ablation_id,
        "real_eval_count": real_count, "dup_proposals": cnt["dup"],
        "repeat_proxy_skips": repeat_proxy_skips,
    })

    return {
        "best_acc": best["acc"], "eta": best["acc"],
        "llm_calls": llm_prior.call_count, "cache_hits": llm_prior.hit_count,
        "llm_sec": llm_sec[0], "dup_proposals": cnt["dup"],
        "name_mismatch_ops": llm_prior.name_mismatch_count,
        "llm_fallbacks": llm_fallbacks,
        **result_row,
    }


def search_flat_ucb(bench, budget: int, dataset: str, seed: int,
                    dedup: str = "set",                       # "set" | "none"
                    phase: str = "ablations",
                    ops=None, C: float = C_DEFAULT, backup_mode: str = "max",
                    use_proxy: bool = True,
                    proxy_threshold_percentile: int = PROXY_THRESHOLD_PERCENTILE,
                    zc_override=None,
                    ablation_id: str = "",
                    results_filename: str = "flat_ucb_main_results.csv",
                    dedup_filename: str = "flat_ucb_dedup_stats.csv",
                    reward_mode: str = None, mutation_prob: float = None) -> Dict:
    """
    Ablation: flat UCB1, NO trie. Answers "does the trie do something flat
    search can't?".

    State: one independent UCB1 bandit per position (depth x ops arms), stats
    NOT conditioned on the prefix. No pruning, no weight-bank ancestry.

    Mirrors search_phase3(alpha=0, use_llm=False) so the only difference is the
    state representation: same 1e9 score for unvisited arms, same accuracy/100
    scale (Oct 2026: was 0-100, which matched the OLD phase-3 scale), same proxy
    gate and budget accounting. Visit counts include
    proxy-skipped iterations; the running mean uses real evals only.

    dedup="set":  duplicate proposals are redirected to a random unvisited leaf
                  (plain set). Isolates trie structure from bare dedup.
    dedup="none": duplicates are re-evaluated and cost a real eval each
                  (counted in dup_real_evals).

    v2 (Oct 3): same reward scale (ucb.node_value) and the same mutation share as search_phase3,
    so A7 still differs ONLY in the state representation.

    UCB1 over the 15,625 leaves directly is not implemented: every unvisited
    leaf scores +inf, so it degenerates to random search without replacement
    (the phase-0 random_search baseline).
    """
    assert dedup in ("set", "none"), dedup
    if _is_generic(bench):
        raise NotImplementedError("search_flat_ucb is NB201-only (A7 ablation)")
    random.seed(seed)
    torch.manual_seed(seed)
    _ucb.reward_reset(reward_mode)
    mutation_prob = settings.MUTATION_PROB if mutation_prob is None else float(mutation_prob)
    ops = ops or OPS_NB201
    depth = 26 if bench.name == "nb101" else 6
    t0 = time.time()

    zc = None
    if use_proxy:
        zc = zc_override if zc_override is not None else ZCProxyEnsemble.from_saved(
            phase="phase2", bench_name=bench.name, dataset=dataset)
        get_mini_batch()

    n_all = [dict() for _ in range(depth)]    # slot -> op -> visits (real + proxy-skipped)
    n_real = [dict() for _ in range(depth)]   # slot -> op -> real evals
    val = [dict() for _ in range(depth)]      # slot -> op -> backed-up accuracy (0-100)
    total_visits = 0                          # flat analogue of root.visit_count
    visited: set = set()

    log = {"eval_history": []}
    real_count = proxy_count = dup_real = dup_proposals = 0
    proxy_scores_seen: List[float] = []

    for i in range(budget):
        # Select: independent UCB1 per slot, random tie-break via shuffle
        path: List[str] = []
        for d in range(depth):
            cands = list(ops)
            random.shuffle(cands)
            best_op, best_s = None, float("-inf")
            for op in cands:
                v = val[d].get(op)
                n = n_all[d].get(op, 0)
                if n == 0 or v is None:
                    s = 1e9   # unvisited (or only proxy-skipped): try first
                else:
                    s = _ucb.node_value(v) + C * math.sqrt(math.log(max(total_visits, 1)) / n)
                if s > best_s:
                    best_op, best_s = op, s
            path.append(best_op)

        if (mutation_prob > 0 and len(log["eval_history"]) >= settings.MUTATION_WARMUP
                and random.random() < mutation_prob):
            cand = _mutation_proposal(bench, log["eval_history"], lambda t: tuple(t) not in visited)
            if cand is not None:
                path = cand
        key = tuple(path)
        if key in visited:
            dup_proposals += 1
            if dedup == "set":
                path = _random_unvisited_set(visited, ops, depth)
                key = tuple(path)

        take_real_path = True
        if use_proxy and i >= WARMUP_REAL_EVALS:
            proxy_val = zc.score(path)
            proxy_scores_seen.append(proxy_val)
            thresh = np.percentile(proxy_scores_seen, 100 - proxy_threshold_percentile)
            take_real_path = proxy_val >= thresh
            if not take_real_path:
                # Proxy-skipped: count visits only, no accuracy update
                total_visits += 1
                for d, op in enumerate(path):
                    n_all[d][op] = n_all[d].get(op, 0) + 1
                proxy_count += 1

        if take_real_path:
            idx = index_from_tokens_nb201(path)
            acc = bench.query(idx, dataset)
            real_count += 1
            _ucb.reward_register(acc)
            if key in visited:
                dup_real += 1
            visited.add(key)

            total_visits += 1
            for d, op in enumerate(path):
                n_all[d][op] = n_all[d].get(op, 0) + 1
                nr = n_real[d].get(op, 0) + 1
                n_real[d][op] = nr
                old = val[d].get(op)
                if old is None:
                    val[d][op] = acc
                elif backup_mode == "max":
                    val[d][op] = max(old, acc)
                else:
                    val[d][op] = (old * (nr - 1) + acc) / nr   # running mean

            log["eval_history"].append({"iter": i + 1, "arch": "|".join(path), "acc": acc,
                                        "t_sec": round(time.time() - t0, 3)})

    elapsed = time.time() - t0
    best = max(log["eval_history"], key=lambda x: x["acc"], default={"acc": 0.0})
    rep = best_from_history(bench, log["eval_history"], dataset)
    log["best_test_acc"] = rep["best_test_acc"]
    method = f"flat_ucb_{dedup}"
    unique_real = len(visited)

    log.update({
        "method": method, "benchmark": bench.name, "dataset": dataset,
        "seed": seed, "budget": budget, "best_acc": best["acc"],
        "eval_count": budget, "real_eval_count": real_count,
        "proxy_eval_count": proxy_count, "unique_real_evals": unique_real,
        "dup_real_evals": dup_real, "dup_proposals": dup_proposals,
        "use_proxy": use_proxy, "backup_mode": backup_mode,
        "ablation_id": ablation_id, "wall_time_sec": elapsed,
    })
    tag = method + (f"__{ablation_id}" if ablation_id else "")
    storage.save_run_log(phase, tag, bench.name, dataset, seed, log, budget=budget)

    result_row = {
        "method": method, "benchmark": bench.name, "dataset": dataset,
        "seed": seed, "budget": budget, "best_acc": best["acc"],
        "best_test_acc": rep["best_test_acc"],
        "eval_count": budget, "proxy_eval_count": proxy_count,
        "real_eval_count": real_count, "unique_real_evals": unique_real,
        "dup_real_evals": dup_real, "tcr": unique_real / bench.total_archs,
        "wall_time_sec": elapsed,
    }
    storage.flush_row_buffers()
    storage.append_result_row(phase, results_filename, result_row)
    _log_dedup_stats(phase, dedup_filename, {
        "method": method, "benchmark": bench.name, "dataset": dataset,
        "seed": seed, "budget": budget, "ablation_id": ablation_id,
        "real_eval_count": real_count, "dup_proposals": dup_proposals,
        "dup_real_evals": dup_real,
    })
    return {"best_acc": best["acc"], "dup_proposals": dup_proposals, **result_row}


def search_phase4(bench, budget: int, dataset: str, seed: int, hardware: str,
                   llm_cache_phase: str = "phase3", C: float = C_DEFAULT,
                   backup_mode: str = "max", prune_percentile: int = PRUNE_PERCENTILE,
                   backend: str = "ollama",                     # "ollama" | "groq"
                   groq_model: str = "openai/gpt-oss-20b",
                   groq_daily_token_budget: int = 190_000,
                   model: str = "llama3.1",
                   host: str = "http://localhost:11434",
                   groq_daily_request_budget: int = 950,
                   groq_reasoning_effort: str = "low",
                   use_key_pool: bool = False) -> Dict:
    """
    Hardware-aware search: real-evals unvisited leaves via bench.hw_query() and
    keeps a Pareto archive over (accuracy, latency).

    LLM prior is cache-read-only (max_calls=0, alpha=0.0): no backend call is
    ever fired. The backend args exist only so LLMPrior can be constructed
    (e.g. groq-only machine with no ollama installed, or pool-only env).
    No ZC-proxy gate here.

    Unsupported archs: HW-NAS-Bench marks some (arch, hardware) pairs as
    unsupported with a negative latency (seen on edgetpu, ~5% of archs).
    hw_query() raises HWMetricUnavailable for these. The arch still gets a real
    eval, trie insert and accuracy-only backprop (no latency to penalize), but
    is kept out of the Pareto archive, pareto rows and node.latency so it can't
    corrupt HV/IGD. Count is logged as unsupported_hw_skipped.
    """
    if _is_generic(bench):
        raise NotImplementedError("search_phase4 is NB201-only (HW-NAS-Bench has no NB101 latencies)")
    random.seed(seed)
    _ucb.reward_reset()

    trie = ArchTrie(max_depth=6)
    wb = WeightBank("phase4", bench.name, f"{dataset}__{hardware}", seed)

    llm_prior = LLMPrior(phase=llm_cache_phase, bench=bench.name, dataset=dataset,
                          seed=seed, max_calls=0,
                          model=model, base_url=host,
                          backend=backend,
                          groq_model=groq_model,
                          groq_daily_token_budget=groq_daily_token_budget,
                          groq_daily_request_budget=groq_daily_request_budget,
                          groq_reasoning_effort=groq_reasoning_effort,
                          use_key_pool=use_key_pool)

    pareto_archive = []
    history = []
    tcr_h_per_iter = []
    best_acc, real_eval_count = 0.0, 0
    unsupported_count = 0

    t0 = time.time()
    # Buffered and written once at the end of the run
    pareto_rows_buffer = []
    tcrh_rows_buffer = []

    for i in range(budget):
        path = traverse_with_llm(trie, OPS_NB201, llm_prior, C=C, alpha=0.0, t=i + 1, budget=budget)

        # Already visited: try to force a new leaf first. bump_visit_only() is
        # the fallback only when random_unvisited() also fails (space ~exhausted).
        if trie.is_visited(path):
            path = random_unvisited(trie, OPS_NB201)

        if trie.is_visited(path):
            bump_visit_only(trie, path)
        else:
            idx = _path_to_index(bench, path)
            try:
                acc, latency = bench.hw_query(idx, hardware=hardware, dataset=dataset)
            except HWMetricUnavailable:
                # Accuracy is real, latency isn't: count the eval, mark the leaf
                # visited, backprop on accuracy alone, skip the Pareto archive.
                unsupported_count += 1
                acc = bench.query(idx, dataset)

                node = trie.insert(path)
                node.accuracy = acc
                real_eval_count += 1

                ancestor = wb.load_ancestor(trie, path)
                wb.store(path, wb.simulate_train_new_layers(path, ancestor))

                backpropagate(trie, path, acc, backup_mode=backup_mode)
                _ucb.reward_register(acc)
                best_acc = max(best_acc, acc)
                history.append({"iter": i + 1, "arch": "|".join(path), "acc": acc, "latency_ms": None})
                continue

            node = trie.insert(path)
            node.accuracy = acc
            node.latency = latency
            real_eval_count += 1

            ancestor = wb.load_ancestor(trie, path)
            wb.store(path, wb.simulate_train_new_layers(path, ancestor))

            # Reward = accuracy minus a latency penalty (latency capped at LATENCY_NORM_MS)
            norm_lat = min(latency / LATENCY_NORM_MS, 1.0)
            scalar_reward = acc - LATENCY_PENALTY * norm_lat * 100.0
            backpropagate(trie, path, scalar_reward, backup_mode=backup_mode)
            _ucb.reward_register(scalar_reward)

            pareto_archive = update_pareto_archive(pareto_archive, node)
            best_acc = max(best_acc, acc)
            history.append({"iter": i + 1, "arch": "|".join(path), "acc": acc, "latency_ms": latency})

            pareto_rows_buffer.append({
                "benchmark": bench.name, "hardware": hardware, "dataset": dataset,
                "seed": seed, "budget": budget, "iter": i + 1,
                "arch_tokens": "|".join(path), "accuracy": acc, "latency_ms": latency,
                "on_pareto_front": node in pareto_archive,
            })

        if (i + 1) % 10 == 0 or i == budget - 1:
            th = tcr_h(trie, pareto_archive)
            tcr_h_per_iter.append(th)
            tcrh_rows_buffer.append({
                "benchmark": bench.name, "hardware": hardware, "dataset": dataset,
                "seed": seed, "budget": budget, "iter": i + 1, "tcr_h": th,
            })

        if (i + 1) % PRUNE_EVERY_N == 0:
            trie.prune(prune_percentile)

    wb.flush()
    elapsed = time.time() - t0
    hv = hypervolume(pareto_archive)

    storage.append_result_rows("phase4", "phase4_pareto_results.csv", pareto_rows_buffer)
    storage.append_result_rows("phase4", "phase4_tcr_h_per_iter.csv", tcrh_rows_buffer)

    # IGD needs a precomputed reference front for this hardware target
    true_front_path = storage.PROJECT_ROOT / "data" / f"true_pareto_front__{hardware}.json"
    if true_front_path.exists():
        with open(true_front_path) as f:
            true_front = [tuple(p) for p in json.load(f)]
        igd_val = igd(pareto_archive, true_front)
    else:
        igd_val = float("nan")
        print(f"[phase4] WARNING: {true_front_path} missing — igd=NaN. "
              f"Run tools/build_igd_reference_front.py first.")

    if unsupported_count:
        print(f"[phase4] {hardware}/{dataset}/seed{seed}/budget{budget}: "
              f"{unsupported_count} arch(es) skipped from Pareto/HV/IGD due to "
              f"HW-NAS-Bench 'unsupported' (negative-latency sentinel) — "
              f"still counted toward real_eval_count and trie exploration.")

    storage.append_result_row("phase4", "phase4_hv_igd_summary.csv", {
        "benchmark": bench.name, "hardware": hardware, "dataset": dataset,
        "seed": seed, "budget": budget, "hypervolume": hv, "igd": igd_val,
        "pareto_front_size": len(pareto_archive),
        "real_eval_count": real_eval_count, "wall_time_sec": elapsed,
        "unsupported_hw_skipped": unsupported_count,
    })

    storage.save_run_log("phase4", "archtree_llm", bench.name, f"{dataset}__{hardware}", seed,
                          {"history": history, "eval_history": history, "best_acc": best_acc,
                           "igd": igd_val, "method": "archtree_hw",
                           "hypervolume": hv, "pareto_front_size": len(pareto_archive),
                           "unsupported_hw_skipped": unsupported_count},
                          budget=budget)

    return {
        "best_acc": best_acc, "real_eval_count": real_eval_count,
        "hypervolume": hv, "pareto_front_size": len(pareto_archive),
        "wall_time_sec": elapsed,
        "unsupported_hw_skipped": unsupported_count,
    }


def _path_to_index(bench, path: List[str]) -> int:
    return index_from_tokens_nb201(path)