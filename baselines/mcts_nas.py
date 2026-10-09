# baselines/mcts_nas.py
"""
MCTS-NAS-style baseline (our simplified re-implementation; call it "MCTS-style" and cite
AlphaX / LaNAS-type MCTS NAS as the originals).
Monte Carlo Tree Search over the architecture space.
No trie deduplication, no weight inheritance — flat MCTS.
This is the key baseline ArchTree improves upon:
  ArchTree adds: trie dedup + formal regret proof + ZC proxy + LLM prior.
"""

import math
import random
from typing import Dict, List, Optional

from baselines.base import NASMethod


class MCTSNode:
    def __init__(self, op: Optional[str] = None, parent=None):
        self.op = op        # operation at this node
        self.parent = parent
        self.children:    Dict[str, "MCTSNode"] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb(self, C: float = 1.414) -> float:
        if self.visit_count == 0:
            return float("inf")
        parent_visits = self.parent.visit_count if self.parent else 1
        return self.mean_value + C * math.sqrt(
            math.log(parent_visits) / self.visit_count
        )


class MCTSNASProxy(NASMethod):

    OPS = ["none", "skip_connect", "conv_1x1", "conv_3x3", "avg_pool_3x3"]

    def __init__(self, bench, dataset: str, seed: int, max_depth: int = 6, C: float = 1.414):
        self.bench = bench
        self.dataset = dataset
        self.seed = seed
        self.max_depth = max_depth
        self.C = C
        random.seed(seed)

        self.root = MCTSNode()
        self.visited = set()   # flat set — no dedup guarantee (unlike ArchTrie)
        self._metrics = {}

    def search(self, budget: int) -> Dict:
        history = []
        best_acc = 0.0

        for i in range(budget):
            # MCTS: select → expand → simulate → backpropagate
            path = self._select_and_expand()
            idx = self._path_to_index(path)

            # no dedup — MCTS may re-visit (unlike ArchTree trie)
            acc = self.bench.query(idx, self.dataset)
            self.visited.add(idx)

            self._backpropagate(path, acc / 100.0)   # normalise for UCB

            if acc > best_acc:
                best_acc = acc

            tcr = len(self.visited) / self.bench.total_archs
            history.append({
                "iter": i + 1,
                "arch": "|".join(path),
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

    # ── MCTS phases ───────────────────────────────────────────────────────────

    def _select_and_expand(self) -> List[str]:
        """Select path via UCB, expand one new node if not fully expanded."""
        node = self.root
        path = []

        for depth in range(self.max_depth):
            unvisited_ops = [op for op in self.OPS if op not in node.children]

            if unvisited_ops:
                # expand: add one new child (random unvisited op)
                op = random.choice(unvisited_ops)
                node.children[op] = MCTSNode(op=op, parent=node)
                path.append(op)
                return path

            # select: pick child with best UCB
            best_op = max(node.children.keys(),
                          key=lambda op: node.children[op].ucb(self.C))
            path.append(best_op)
            node = node.children[best_op]

        return path

    def _backpropagate(self, path: List[str], value: float) -> None:
        """Update visit counts and values bottom-up."""
        node = self.root
        node.visit_count += 1
        node.total_value += value

        for op in path:
            if op in node.children:
                node = node.children[op]
                node.visit_count += 1
                node.total_value += value

    def _path_to_index(self, path: List[str]) -> int:
        """
        Convert op path to architecture index (deterministic hash).
        Pads short paths with random ops to reach max_depth.
        """
        while len(path) < self.max_depth:
            path.append(random.choice(self.OPS))

        # encode as base-5 index (5 ops per position)
        idx = 0
        for op in path[:self.max_depth]:
            op_idx = self.OPS.index(op) if op in self.OPS else 0
            idx = idx * len(self.OPS) + op_idx

        return idx % self.bench.total_archs
