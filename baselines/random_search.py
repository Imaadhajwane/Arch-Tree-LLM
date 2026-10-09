# baselines/random_search.py
"""
Random Search baseline.
Uniformly samples architectures at random. No learning, no memory.
Serves as the weakest baseline — everything should beat this.
"""

import random
import time
from typing import Dict

from baselines.base import NASMethod


class RandomSearch(NASMethod):

    def __init__(self, bench, dataset: str, seed: int):
        self.bench = bench
        self.dataset = dataset
        self.seed = seed
        random.seed(seed)
        self._metrics = {}

    def search(self, budget: int) -> Dict:
        visited = set()
        history = []
        best_acc = 0.0

        for i in range(budget):
            # sample unique arch
            while True:
                idx = random.randint(0, self.bench.total_archs - 1)
                if idx not in visited:
                    visited.add(idx)
                    break

            acc = self.bench.query(idx, self.dataset)
            if acc > best_acc:
                best_acc = acc

            tcr = len(visited) / self.bench.total_archs
            history.append({
                "iter": i + 1,
                "arch": str(idx),
                "acc":  acc,
                "tcr":  tcr,
            })

        self._metrics = {
            "eval_history": history,
            "best_acc":     best_acc,
            "eval_count":   budget,
        }
        return self._metrics

    def get_metrics(self) -> Dict:
        return self._metrics
