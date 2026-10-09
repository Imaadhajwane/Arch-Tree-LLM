# baselines/regularized_evolution.py
"""
Regularized evolution (aging evolution), Real et al. AAAI 2019 - the standard strong baseline on
NAS-Bench-101 and NAS-Bench-201 (both benchmark papers report it). Added Oct 2026.

Proposer interface (same as the proposers in experiments/run_gated_baselines.py):
    propose(evaluated) -> tokens of a not-yet-evaluated architecture
    update(tokens, acc)   called only after a REAL evaluation (gated-out proposals are not added)

    1. the first `population` proposals are random architectures
    2. afterwards: sample `sample_size` members of the population, mutate the best one
       (NB101: the official nasbench mutation; NB201: change the op of one random edge)
    3. the new member is appended, the OLDEST member is removed (aging)

Population size defaults scale with the budget so small budgets still get some evolution:
population = clamp(budget // 4, 5, 20), sample_size = max(2, population // 4).

Zero-cost warm-up (Abdelfattah et al., "Zero-Cost Proxies for Lightweight NAS", ICLR 2021, the
"warmup" variant - the closest prior work to ArchTree's proxy gate), added Oct 3, 2026:
`init_pool` = the architectures the initial population is drawn from, in order. ZCWarmupRESearch
scores N_WARMUP random architectures with the label-free zero-cost ensemble and puts the top
`population` of them first; the rest of the search is plain regularized evolution.
"""
import random
from collections import deque
from typing import List, Optional


def default_sizes(budget: int):
    p = max(5, min(20, budget // 4))
    return p, max(2, p // 4)


class RegularizedEvolution:
    def __init__(self, bench, seed: int, budget: int = 100,
                 population: Optional[int] = None, sample_size: Optional[int] = None,
                 init_pool: Optional[List[List[str]]] = None):
        self.bench = bench
        self.rng = random.Random(seed)
        p, s = default_sizes(budget)
        self.P = population or p
        self.S = sample_size or s
        self.pop = deque()
        self.n_initial = 0
        self.init_pool = [list(t) for t in (init_pool or [])]

    def _random_tokens(self, evaluated) -> List[str]:
        for _ in range(10000):
            i = self.rng.randrange(self.bench.total_archs)
            if i not in evaluated:
                return self.bench.index_to_tokens(i)
        raise RuntimeError("search space exhausted")

    def propose(self, evaluated) -> List[str]:
        if self.n_initial < self.P or len(self.pop) < 2:
            self.n_initial += 1
            while self.init_pool:                       # zero-cost warm-up candidates first
                cand = self.init_pool.pop(0)
                if self.bench.tokens_to_index(cand) not in evaluated:
                    return cand
            return self._random_tokens(evaluated)
        cands = self.rng.sample(list(self.pop), min(self.S, len(self.pop)))
        parent = max(cands, key=lambda m: m[1])[0]
        for _ in range(50):
            child = self.bench.mutate_tokens(parent, self.rng)
            idx = self.bench.tokens_to_index(child)
            if idx is not None and idx not in evaluated:
                return child
        return self._random_tokens(evaluated)

    def update(self, tokens, acc: float) -> None:
        self.pop.append((list(tokens), acc))
        if len(self.pop) > self.P:
            self.pop.popleft()


class RegularizedEvolutionSearch:
    """Stand-alone RE baseline with the NASMethod interface used by experiments/run_phase0.py
    (every proposal is a real evaluation; never re-evaluates an architecture)."""

    def __init__(self, bench, dataset: str, seed: int):
        self.bench, self.dataset, self.seed = bench, dataset, seed
        self._metrics = {}

    def search(self, budget: int):
        random.seed(self.seed)
        prop = RegularizedEvolution(self.bench, self.seed, budget)
        evaluated, history = set(), []
        for i in range(budget):
            toks = prop.propose(evaluated)
            idx = self.bench.tokens_to_index(toks)
            acc = self.bench.query(idx, self.dataset)
            evaluated.add(idx)
            prop.update(toks, acc)
            history.append({"iter": i + 1, "arch": "|".join(toks), "acc": acc})
        self._metrics = {"eval_history": history, "best_acc": max(e["acc"] for e in history),
                         "eval_count": budget}
        return self._metrics

    def get_metrics(self):
        return self._metrics


N_WARMUP = 1000      # architectures scored by the zero-cost ensemble before the search (no labels)


def zc_warmup_pool(bench, dataset: str, seed: int, population: int, n: int = N_WARMUP, gate=None):
    """Top-`population` of `n` random architectures by the label-free zero-cost ensemble."""
    if gate is None:
        from archtree.zc_proxy import ZCProxyEnsemble
        gate = ZCProxyEnsemble.unsupervised(bench, dataset)
    rng = random.Random(10_000 + seed)
    idxs = rng.sample(range(bench.total_archs), min(n, bench.total_archs))
    toks = [bench.index_to_tokens(i) for i in idxs]
    scored = sorted(toks, key=gate.score, reverse=True)
    return scored[:population], len(toks)


class ZCWarmupRESearch:
    """Regularized evolution with zero-cost warm-up (Abdelfattah et al. 2021), NASMethod interface
    of experiments/run_phase0.py. The N_WARMUP proxy scorings are reported as proxy_evals."""

    def __init__(self, bench, dataset: str, seed: int):
        self.bench, self.dataset, self.seed = bench, dataset, seed
        self._metrics = {}

    def search(self, budget: int):
        random.seed(self.seed)
        p, _ = default_sizes(budget)
        pool, n_scored = zc_warmup_pool(self.bench, self.dataset, self.seed, p)
        prop = RegularizedEvolution(self.bench, self.seed, budget, init_pool=pool)
        evaluated, history = set(), []
        for i in range(budget):
            toks = prop.propose(evaluated)
            idx = self.bench.tokens_to_index(toks)
            acc = self.bench.query(idx, self.dataset)
            evaluated.add(idx)
            prop.update(toks, acc)
            history.append({"iter": i + 1, "arch": "|".join(toks), "acc": acc})
        self._metrics = {"eval_history": history, "best_acc": max(e["acc"] for e in history),
                         "eval_count": budget, "proxy_evals": n_scored}
        return self._metrics

    def get_metrics(self):
        return self._metrics
