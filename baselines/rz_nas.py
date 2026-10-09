# baselines/rz_nas.py
"""
ZC-guided mutation walk ("rz_nas-style" baseline). No trie, no LLM.

NOTE ON THE NAME: the published RZ-NAS (ICML 2025) is an LLM-guided method with a reflective
zero-cost strategy. This file is NOT that method; it is a zero-cost-proxy-guided mutation walk.
Call it "ZC mutation walk (rz-style)" in the paper and cite RZ-NAS only as inspiration.

Oct 2026 audit fixes
--------------------
* The Sep-2026 version paid a REAL evaluation for its current architecture on every iteration,
  including every rejected mutation (82% duplicate real evals at budget 200 -> a strawman).
  Now: `budget` counts REAL evaluations of DISTINCT architectures. A mutation is scored with the
  proxy; only an accepted, never-evaluated candidate is really evaluated.
* The proxy is the label-free (unsupervised) ensemble by default, not the raw un-normalised sum
  (whose scale was dominated by one component). Pass proxy=<ZCProxyEnsemble> to give it the
  same supervised calibration ArchTree uses (run_gated_baselines.py does both).
"""
import random
from typing import Dict, List

from baselines.base import NASMethod
from archtree.encoder import index_from_tokens_nb201

MAX_PROPOSALS_PER_EVAL = 50   # safety cap so a stuck walk cannot loop forever


class RZNASBaseline(NASMethod):
    OPS = ["none", "skip_connect", "conv_1x1", "conv_3x3", "avg_pool_3x3"]

    def __init__(self, bench, dataset: str, seed: int, max_depth: int = 6, proxy=None,
                 accept_worse_prob: float = 0.1):
        from archtree.zc_proxy import ZCProxyEnsemble, get_mini_batch
        self.bench, self.dataset, self.seed, self.max_depth = bench, dataset, seed, max_depth
        self.rng = random.Random(seed)
        self.zc = proxy if proxy is not None else ZCProxyEnsemble.unsupervised(bench, dataset)
        get_mini_batch()
        self.accept_worse_prob = accept_worse_prob
        self._metrics = {}

    def _mutate(self, tokens: List[str]) -> List[str]:
        t = list(tokens)
        pos = self.rng.randrange(len(t))
        t[pos] = self.rng.choice([o for o in self.OPS if o != t[pos]])   # always a real change
        return t

    def search(self, budget: int) -> Dict:
        history, best_acc, visited = [], 0.0, set()
        proposals = proxy_evals = 0

        def evaluate(tokens):
            idx = index_from_tokens_nb201(tokens)
            acc = self.bench.query(idx, self.dataset)
            visited.add(idx)
            history.append({"iter": len(history) + 1, "arch": "|".join(tokens), "acc": acc,
                            "tcr": len(visited) / self.bench.total_archs})
            return acc

        cur = self.bench.index_to_tokens(self.rng.randrange(self.bench.total_archs))
        cur_score = self.zc.score(cur)
        proxy_evals += 1
        best_acc = evaluate(cur)

        while len(history) < budget:
            proposals += 1
            if proposals > MAX_PROPOSALS_PER_EVAL * budget:
                # stuck: jump to a random unvisited architecture
                while True:
                    cand = self.bench.index_to_tokens(self.rng.randrange(self.bench.total_archs))
                    if index_from_tokens_nb201(cand) not in visited:
                        break
                cur, cur_score = cand, self.zc.score(cand)
                best_acc = max(best_acc, evaluate(cur))
                continue
            cand = self._mutate(cur)
            cand_score = self.zc.score(cand)
            proxy_evals += 1
            if cand_score >= cur_score or self.rng.random() < self.accept_worse_prob:
                cur, cur_score = cand, cand_score
                if index_from_tokens_nb201(cand) not in visited:
                    best_acc = max(best_acc, evaluate(cand))

        self._metrics = {"eval_history": history, "best_acc": best_acc, "eval_count": budget,
                         "proposals": proposals, "proxy_evals": proxy_evals}
        return self._metrics

    def get_metrics(self) -> Dict:
        return self._metrics
