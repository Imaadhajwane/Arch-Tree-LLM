# archtree/tcr.py
"""
Coverage and Pareto metrics.

- tcr:   visited terminals / search-space size.
- tcr_h: Pareto-weighted coverage (phase 4). Each visited terminal adds
         1/pareto_rank, so front points add 1.0 and dominated points add less.
         Divided by the search-space size.
- Pareto archive: non-dominated set under (max accuracy, min latency).
- hypervolume / igd: front-quality summaries.
"""
import math
from typing import List

from archtree.trie import ArchTrie, TrieNode


def tcr(trie: ArchTrie, total_archs: int) -> float:
    """Trie Coverage Ratio: visited terminals / search-space size."""
    return trie.count_visited_terminals() / total_archs


def update_pareto_archive(archive: List[TrieNode], node: TrieNode) -> List[TrieNode]:
    """Insert `node` into the non-dominated archive (max accuracy, min latency).
    Removes members that `node` dominates; adds `node` unless a member dominates it.
    Nodes without both accuracy and latency are ignored."""
    if node.accuracy is None or node.latency is None:
        return archive

    # Drop members dominated by the new node
    archive = [a for a in archive
               if not (node.accuracy >= a.accuracy and node.latency <= a.latency
                       and (node.accuracy > a.accuracy or node.latency < a.latency))]

    dominated = any(a.accuracy >= node.accuracy and a.latency <= node.latency
                    and (a.accuracy > node.accuracy or a.latency < node.latency)
                    for a in archive)
    if not dominated:
        archive.append(node)
    return archive


def pareto_rank(node: TrieNode, pareto_archive: List[TrieNode], max_rank_scan: int = 10) -> int:
    """Cheap rank against the current archive: 1 + number of archive members
    that are at least as good on both objectives. Capped at max_rank_scan.
    This is not a true non-dominated-sort layer, but it avoids a full sort per
    iteration."""
    if node in pareto_archive:
        return 1
    if node.accuracy is None or node.latency is None:
        return max_rank_scan   # no valid metrics: worst rank

    rank = 1
    for a in pareto_archive:
        if a.accuracy >= node.accuracy and a.latency <= node.latency:
            rank += 1
    return min(rank, max_rank_scan)


def count_all_terminals(trie: ArchTrie) -> int:
    """Search-space size, assuming 5 ops per position (NB201: 5^6 = 15625)."""
    return 5 ** trie.max_depth


def tcr_h(trie: ArchTrie, pareto_archive: List[TrieNode]) -> float:
    """Pareto-weighted coverage ratio (see module docstring)."""
    total = count_all_terminals(trie)
    if total == 0:
        return 0.0
    terminals = trie.all_visited_terminals()
    if not terminals:
        return 0.0
    weighted = sum(1.0 / pareto_rank(n, pareto_archive) for n in terminals)
    return weighted / total


def hypervolume(pareto_archive: List[TrieNode],
                 ref_point: tuple = (0.0, 1000.0)) -> float:
    """2-D hypervolume of the front against ref_point = (worst accuracy,
    worst latency in ms). Rectangle sweep, O(n log n). Assumes the archive is
    non-dominated."""
    pts = [(n.accuracy, n.latency) for n in pareto_archive
           if n.accuracy is not None and n.latency is not None]
    if not pts:
        return 0.0
    ref_acc, ref_lat = ref_point
    hv = 0.0
    prev_lat = ref_lat
    # Sweep from highest latency down; each point adds a slab up to the previous latency
    for acc, lat in sorted(pts, key=lambda p: -p[1]):
        width = max(prev_lat - lat, 0.0)
        height = max(acc - ref_acc, 0.0)
        hv += width * height
        prev_lat = min(prev_lat, lat)
    return hv


def igd(pareto_archive: List[TrieNode], true_front: List[tuple]) -> float:
    """Inverted Generational Distance: mean distance from each true-front point
    (acc, latency) to its nearest found point. Lower is better.
    Returns inf if either front is empty."""
    found = [(n.accuracy, n.latency) for n in pareto_archive if n.accuracy is not None]
    if not found or not true_front:
        return float("inf")
    dists = []
    for tf in true_front:
        d = min(math.sqrt((tf[0] - f[0]) ** 2 + (tf[1] - f[1]) ** 2) for f in found)
        dists.append(d)
    return sum(dists) / len(dists)