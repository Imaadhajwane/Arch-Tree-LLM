# archtree/trie.py
"""
Prefix trie over architecture tokens.

Each path from root to a leaf is one architecture (NB201: depth 6, one op per
position). A node is "visited" only if it was inserted as a terminal, i.e.
really evaluated. Nodes created by proxy-skipped visits exist in the trie and
carry visit counts, but are not terminal.
"""
from typing import Dict, List, Optional


class TrieNode:
    # __slots__ keeps per-node memory small; the trie can hold thousands of nodes
    __slots__ = ("children", "visit_count", "ucb_score", "weight_bank_ref",
                 "accuracy", "proxy_score", "latency", "is_terminal", "is_pruned")

    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.visit_count: int = 0                     # real evals + proxy-skipped visits
        self.ucb_score: float = float("inf")          # inf = unexplored, tried first
        self.weight_bank_ref: Optional[tuple] = None
        self.accuracy: Optional[float] = None
        self.proxy_score: Optional[float] = None
        self.latency: Optional[float] = None          # phase 4 only
        self.is_terminal: bool = False                # True once the arch was really evaluated
        self.is_pruned: bool = False                  # UCB traversal skips pruned nodes


class ArchTrie:
    def __init__(self, max_depth: int = 6):
        self.root = TrieNode()
        self.max_depth = max_depth
        self._n_terminals = 0   # running count, so coverage queries are O(1)

    def insert(self, tokens: List[str]) -> TrieNode:
        """Create the path if absent and mark the leaf terminal. O(L)."""
        node = self.root
        for tok in tokens:
            if tok not in node.children:
                node.children[tok] = TrieNode()
            node = node.children[tok]
        if not node.is_terminal:   # re-inserting must not double count
            node.is_terminal = True
            self._n_terminals += 1
        return node

    def is_visited(self, tokens: List[str]) -> bool:
        """True only if the full path exists and its leaf is terminal. O(L)."""
        node = self.root
        for tok in tokens:
            node = node.children.get(tok)
            if node is None:
                return False
        return node.is_terminal

    def get_node(self, tokens: List[str]) -> Optional[TrieNode]:
        """Node at the end of the path, or None if the path doesn't exist."""
        node = self.root
        for tok in tokens:
            node = node.children.get(tok)
            if node is None:
                return None
        return node

    def mark_dead(self, tokens: List[str]) -> int:
        """Mark an INVALID leaf (e.g. a disconnected NB101 cell) as pruned so UCB never proposes
        it again, and propagate upwards: a node whose created children are all pruned is pruned
        too (traversal creates every child of a node it passes, so "all created" = "all").
        Returns the number of nodes marked. Unused on NB201 (every token string is valid)."""
        chain, node = [self.root], self.root
        for tok in tokens:
            node = node.children.get(tok)
            if node is None:
                return 0
            chain.append(node)
        marked = 0
        if not chain[-1].is_pruned:
            chain[-1].is_pruned = True
            marked += 1
        for parent in reversed(chain[1:-1]):
            if parent.children and all(c.is_pruned for c in parent.children.values()):
                if not parent.is_pruned:
                    parent.is_pruned = True
                    marked += 1
            else:
                break
        return marked

    def count_visited_terminals(self) -> int:
        return self._n_terminals

    def all_visited_terminals(self) -> List[TrieNode]:
        """All terminal nodes via DFS. Recursion depth = trie depth (6 or 26), so it is safe."""
        out = []

        def dfs(n):
            if n.is_terminal:
                out.append(n)
            for c in n.children.values():
                dfs(c)

        dfs(self.root)
        return out

    def prune(self, threshold_percentile: int = 25, rule: str = None, min_visits: int = None) -> int:
        """
        Mark a fraction of internal (non-terminal) nodes as pruned so UCB traversal skips them.
        Returns how many were pruned. threshold_percentile <= 0 prunes nothing.

        rule="value" (default, Oct 2026): among internal nodes with a real backed-up accuracy and
            at least `min_visits` visits, prune the bottom `threshold_percentile`% by accuracy.
            A node is never pruned if that would leave its parent without an unpruned child.
            Unexplored subtrees are never pruned (they have no accuracy yet).
        rule="visits" (legacy, Sep 2026): prune the LEAST-VISITED nodes. This removes exactly the
            unexplored subtrees, so it can cut the optimum; kept only to reproduce old runs.
            (It also pruned at least one node even at 0%.)

        Neither rule guarantees the optimum survives (that would need confidence bounds), so
        theorem T4 must be stated as a heuristic, not a guarantee.
        """
        from archtree import settings
        rule = rule or settings.PRUNE_RULE
        min_visits = settings.PRUNE_MIN_VISITS if min_visits is None else min_visits
        if threshold_percentile <= 0:
            return 0

        if rule == "visits":
            candidates = []

            def collect(n):
                for c in n.children.values():
                    if not c.is_terminal and c.visit_count > 0 and not c.is_pruned:
                        candidates.append(c)
                    collect(c)

            collect(self.root)
            if not candidates:
                return 0
            candidates.sort(key=lambda n: n.visit_count)
            k = max(1, int(len(candidates) * threshold_percentile / 100))
            for n in candidates[:k]:
                n.is_pruned = True
            return k

        if rule != "value":
            raise ValueError(f"unknown prune rule {rule!r}")

        cands = []  # (accuracy, node, parent)

        def collect_v(parent):
            for c in parent.children.values():
                if (not c.is_terminal and not c.is_pruned and c.accuracy is not None
                        and c.visit_count >= min_visits):
                    cands.append((c.accuracy, c, parent))
                collect_v(c)

        collect_v(self.root)
        k = int(len(cands) * threshold_percentile / 100)
        if k == 0:
            return 0
        cands.sort(key=lambda x: x[0])
        pruned = 0
        for _, node, parent in cands[:k]:
            alive = sum(1 for ch in parent.children.values() if not ch.is_pruned)
            if alive <= 1:
                continue
            node.is_pruned = True
            pruned += 1
        return pruned
