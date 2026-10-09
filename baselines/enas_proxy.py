# baselines/enas_proxy.py
"""
ENAS-style controller baseline ("ENAS-style": no supernet, no weight sharing, no RL controller
network; call it that in the paper).

A per-position categorical policy over ops, updated from observed accuracy: each position keeps a
running-mean reward per op, and architectures are sampled position by position from
softmax(reward / temperature), with epsilon-greedy uniform exploration. Duplicates are skipped.

Oct 2026 audit fix: the Sep-2026 "biased" sampler shifted a uniformly random INDEX by a constant
derived from op scores, which is still uniform random sampling - the baseline was random search
under another name. This version actually samples from the learned per-position policy.
"""
import math
import random
from typing import Dict, List

from baselines.base import NASMethod

OPS = ["none", "skip_connect", "conv_1x1", "conv_3x3", "avg_pool_3x3"]


class ENASProxy(NASMethod):
    def __init__(self, bench, dataset: str, seed: int, epsilon: float = 0.2,
                 temperature: float = 0.05, depth: int = 6):
        self.bench, self.dataset, self.seed = bench, dataset, seed
        self.rng = random.Random(seed)
        self.eps, self.temp, self.depth = epsilon, temperature, depth
        self.score = [[0.5] * len(OPS) for _ in range(depth)]    # reward on 0-1 scale
        self.count = [[0] * len(OPS) for _ in range(depth)]
        self._metrics = {}

    def _sample_tokens(self) -> List[str]:
        if self.rng.random() < self.eps:
            return [self.rng.choice(OPS) for _ in range(self.depth)]
        toks = []
        for d in range(self.depth):
            m = max(self.score[d])
            w = [math.exp((s - m) / self.temp) for s in self.score[d]]
            toks.append(self.rng.choices(OPS, weights=w)[0])
        return toks

    def _update(self, tokens: List[str], acc: float) -> None:
        r = acc / 100.0
        for d, op in enumerate(tokens):
            j = OPS.index(op)
            n = self.count[d][j]
            self.score[d][j] = (self.score[d][j] * n + r) / (n + 1)
            self.count[d][j] = n + 1

    def search(self, budget: int) -> Dict:
        from archtree.encoder import index_from_tokens_nb201
        visited, history, best_acc = set(), [], 0.0
        for i in range(budget):
            for _ in range(500):
                toks = self._sample_tokens()
                idx = index_from_tokens_nb201(toks)
                if idx not in visited:
                    break
            else:   # policy collapsed onto visited archs: uniform unvisited
                while idx in visited:
                    idx = self.rng.randrange(self.bench.total_archs)
                toks = self.bench.index_to_tokens(idx)
            visited.add(idx)
            acc = self.bench.query(idx, self.dataset)
            self._update(toks, acc)
            best_acc = max(best_acc, acc)
            history.append({"iter": i + 1, "arch": "|".join(toks), "acc": acc,
                            "tcr": len(visited) / self.bench.total_archs})
        self._metrics = {"eval_history": history, "best_acc": best_acc, "eval_count": budget}
        return self._metrics

    def get_metrics(self) -> Dict:
        return self._metrics
