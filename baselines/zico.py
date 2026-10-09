# baselines/zico.py
"""
Zero-shot sample-and-rank baseline ("ZiCo-style"). No tree, no LLM, no weight sharing.

NOTE ON THE NAME: this does NOT compute the ZiCo score (inverse coefficient of variation of
gradients, Li et al., ICLR 2023). It ranks a random pool with THIS project's zero-cost ensemble.
Call it "zero-shot sample-and-rank (ZiCo-style)" in the paper.

Oct 2026 audit fixes
--------------------
* Pool is sampled WITHOUT replacement (was with replacement -> duplicate real evals, ~6%).
* Default proxy = label-free (unsupervised, z-scored, E=S=T=1/3) ensemble instead of the raw
  un-normalised sum. Pass proxy=<ZCProxyEnsemble> for the supervised calibration.
* The pool's proxy evaluations are counted (proxy_evals) so proxy cost can be compared.
"""
import random
from typing import Dict

from baselines.base import NASMethod


class ZiCoBaseline(NASMethod):
    POOL_FACTOR = 10

    def __init__(self, bench, dataset: str, seed: int, proxy=None):
        from archtree.zc_proxy import ZCProxyEnsemble, get_mini_batch
        self.bench, self.dataset, self.seed = bench, dataset, seed
        self.rng = random.Random(seed)
        self.zc = proxy if proxy is not None else ZCProxyEnsemble.unsupervised(bench, dataset)
        get_mini_batch()
        self._metrics = {}

    def search(self, budget: int) -> Dict:
        pool_size = min(budget * self.POOL_FACTOR, self.bench.total_archs)
        candidates = self.rng.sample(range(self.bench.total_archs), pool_size)
        scored = [(idx, self.zc.score(self.bench.index_to_tokens(idx))) for idx in candidates]
        scored.sort(key=lambda x: -x[1])

        history, best_acc = [], 0.0
        for i, (idx, _) in enumerate(scored[:budget]):
            acc = self.bench.query(idx, self.dataset)
            best_acc = max(best_acc, acc)
            history.append({"iter": i + 1, "arch": str(idx), "acc": acc,
                            "tcr": (i + 1) / self.bench.total_archs})

        self._metrics = {"eval_history": history, "best_acc": best_acc, "eval_count": budget,
                         "proxy_evals": pool_size}
        return self._metrics

    def get_metrics(self) -> Dict:
        return self._metrics
