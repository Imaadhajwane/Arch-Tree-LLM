# baselines/local_search.py
"""
Local search (White, Nolen, Savani, "Exploring the loss landscape in neural architecture search",
UAI 2021 - "local search is state of the art for NAS benchmarks"). Added Oct 3, 2026.

Algorithm (first-improvement hill climbing with random restarts):
    1. start from a random architecture and evaluate it
    2. go through its neighbours (every architecture that differs in exactly one position) in
       random order; as soon as a neighbour is better than the current one, move there and start
       again from step 2 with the new neighbourhood
    3. if every neighbour has been tried without improvement (local optimum), restart at step 1
Already evaluated architectures are never re-evaluated (dedup), as for every method here.

Proposer interface (same as baselines/regularized_evolution.py and the proposers in
experiments/run_gated_baselines.py):
    propose(evaluated) -> tokens of a not-yet-evaluated architecture
    update(tokens, acc)    called only after a REAL evaluation. With a zero-cost gate, a
                           neighbour that the gate rejects is simply skipped (never evaluated).
Works on NB201 (flat op list), NB101 and NATS-Bench SSS (per-position vocabulary via
bench.token_vocab(); NB101 neighbours that are invalid cells are skipped).
"""
import random
from typing import List, Optional


def neighbours(bench, tokens: List[str]) -> List[List[str]]:
    """All valid architectures that differ from `tokens` in exactly one position."""
    vocab = bench.token_vocab() if hasattr(bench, "token_vocab") else [list(bench.OPS)] * len(tokens)
    out, seen = [], set()
    for i, choices in enumerate(vocab):
        for tok in choices:
            if tok == tokens[i]:
                continue
            cand = list(tokens)
            cand[i] = tok
            idx = bench.tokens_to_index(cand)
            if idx is None or idx in seen:          # invalid cell / another spelling of the same one
                continue
            seen.add(idx)
            out.append(cand)
    return out


class LocalSearch:
    def __init__(self, bench, seed: int):
        self.bench = bench
        self.rng = random.Random(seed)
        self.cur: Optional[List[str]] = None
        self.cur_acc = float("-inf")
        self.queue: List[List[str]] = []
        self._pending_start: Optional[List[str]] = None
        self.n_restarts = 0

    def _random_tokens(self, evaluated) -> List[str]:
        for _ in range(10000):
            i = self.rng.randrange(self.bench.total_archs)
            if i not in evaluated:
                return self.bench.index_to_tokens(i)
        raise RuntimeError("search space exhausted")

    def _set_current(self, tokens, acc) -> None:
        self.cur, self.cur_acc = list(tokens), acc
        self.queue = neighbours(self.bench, self.cur)
        self.rng.shuffle(self.queue)

    def propose(self, evaluated) -> List[str]:
        while self.queue:
            cand = self.queue.pop()
            if self.bench.tokens_to_index(cand) not in evaluated:
                return cand
        # no current point yet, or local optimum reached: (re)start from a random architecture
        if self.cur is not None:
            self.n_restarts += 1
        self._pending_start = self._random_tokens(evaluated)
        return list(self._pending_start)

    def update(self, tokens, acc: float) -> None:
        if self._pending_start is not None and list(tokens) == self._pending_start:
            self._pending_start = None
            self._set_current(tokens, acc)           # a (re)start always becomes the current point
        elif acc > self.cur_acc:
            self._set_current(tokens, acc)           # first improvement: move


class LocalSearchSearch:
    """Stand-alone local-search baseline with the NASMethod interface of experiments/run_phase0.py
    (every proposal is a real evaluation)."""

    def __init__(self, bench, dataset: str, seed: int):
        self.bench, self.dataset, self.seed = bench, dataset, seed
        self._metrics = {}

    def search(self, budget: int):
        random.seed(self.seed)
        prop = LocalSearch(self.bench, self.seed)
        evaluated, history = set(), []
        for i in range(budget):
            toks = prop.propose(evaluated)
            idx = self.bench.tokens_to_index(toks)
            acc = self.bench.query(idx, self.dataset)
            evaluated.add(idx)
            prop.update(toks, acc)
            history.append({"iter": i + 1, "arch": "|".join(toks), "acc": acc})
        self._metrics = {"eval_history": history, "best_acc": max(e["acc"] for e in history),
                         "eval_count": budget, "restarts": prop.n_restarts}
        return self._metrics

    def get_metrics(self):
        return self._metrics
