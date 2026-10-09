# archtree/ucb.py
"""
UCB1 traversal, backpropagation and LLM-prior helpers for the ArchTree trie.

Oct 2026 fixes
--------------
1. ONE accuracy scale. Every traversal divides the 0-100 accuracy by 100 before adding the
   exploration bonus. (Sep-2026 results were produced with ucb_with_llm_prior on the raw 0-100
   scale, i.e. a ~100x smaller relative bonus in phases 3/4. Those runs are NOT reproducible with
   this file; rerun them.)
2. ONE notion of "cold" child. A child is cold only if it has never been visited
   (visit_count == 0). A child visited only through proxy-skipped proposals (visits > 0,
   accuracy None) is scored like in plain UCB1: value 0 + exploration bonus.
   Before, such children were treated as cold and received a default LLM prior of 0.5 while truly
   unvisited siblings received ~1/k, so ANY alpha > 0 made the search re-walk proxy-rejected
   branches. A zero-information LLM reproduced the reported "LLM slows convergence" effect; after
   this fix it is identical to LLM-off (verified on 24/24 seeds).
   Consequence: traverse_with_llm(alpha=0) now makes exactly the same choices as traverse().
3. Reward scale (ArchTree v2, Oct 3). With accuracy/100 the exploration bonus C*sqrt(ln N / n)
   (~1-2.5 at C=1.414) dwarfs the accuracy differences between branches (~0.01-0.05), so UCB only
   balanced visit counts and behaved like random search (and C had no measurable effect).
   reward="rank" scores a node by the percentile rank of its backed-up accuracy among all
   accuracies evaluated so far in the run (0-1, spread over the whole range). Each search calls
   reward_reset() at the start and reward_register(value) after every real evaluation.
   reward="raw" keeps the v1 behaviour bit for bit.
"""
import math
import random
from typing import List, Optional

from archtree.trie import ArchTrie, TrieNode


# --------------------------------------------------------------------------
# Core UCB1
# --------------------------------------------------------------------------

from archtree import settings

DEFAULT_C = settings.C

# ── reward scale (per search run) ────────────────────────────────────────────
import bisect as _bisect

_REWARD_MODE = settings.REWARD_MODE
_SEEN: List[float] = []          # sorted values evaluated in the current run (rank mode)


def reward_reset(mode: Optional[str] = None) -> None:
    """Start of a search run: forget the previous run's values; optionally switch mode."""
    global _REWARD_MODE
    _REWARD_MODE = mode or settings.REWARD_MODE
    if _REWARD_MODE not in ("raw", "rank"):
        raise ValueError(f"unknown reward mode {_REWARD_MODE!r}")
    _SEEN.clear()


def reward_register(value: float) -> None:
    """Record one real evaluation's backed-up value (accuracy, or HW-penalised reward)."""
    _bisect.insort(_SEEN, float(value))


def node_value(acc: Optional[float]) -> float:
    """Value of a node's backed-up accuracy on the 0-1 UCB scale."""
    if acc is None:
        return 0.0
    if _REWARD_MODE == "rank" and _SEEN:
        return _bisect.bisect_right(_SEEN, acc) / len(_SEEN)
    return acc / 100.0


def vocab_at(ops, depth: int) -> List[str]:
    """Tokens allowed at trie depth `depth`. `ops` is either one flat list shared by every position
    (NB201) or a list of per-position lists (NB101: 5 op positions + 21 adjacency bits)."""
    if ops and isinstance(ops[0], (list, tuple)):
        return list(ops[depth])
    return ops


def _score(node: TrieNode, parent_visits: int, C: float) -> float:
    """UCB1 = mean reward + exploration bonus. Accuracy is stored on a 0-100 scale."""
    if node.visit_count == 0:
        return float("inf")  # always try unvisited children first
    mean = node_value(node.accuracy)
    # parent_visits is clamped to >= 1 so log() is never negative
    return mean + C * math.sqrt(math.log(max(parent_visits, 1)) / node.visit_count)


def ucb_score(node: TrieNode, parent_visits: int, C: float = DEFAULT_C) -> float:
    """Public wrapper around _score()."""
    return _score(node, parent_visits, C)


def traverse(trie: ArchTrie, ops: List[str], C: float = DEFAULT_C) -> List[str]:
    """Walk the trie root -> leaf using UCB1 and return the chosen op path."""
    node, path = trie.root, []
    for d in range(trie.max_depth):
        # Lazily create a child for every op at this node
        for op in vocab_at(ops, d):
            if op not in node.children:
                node.children[op] = TrieNode()

        # Skip pruned children unless everything is pruned
        candidates = [(op, c) for op, c in node.children.items() if not c.is_pruned]
        if not candidates:
            candidates = list(node.children.items())
        random.shuffle(candidates)  # break score ties randomly, not by dict order

        best_op, _ = max(
            candidates,
            key=lambda kv: _score(kv[1], max(node.visit_count, 1), C),
        )
        path.append(best_op)
        node = node.children[best_op]
    return path


def random_unvisited(trie: ArchTrie, ops: List[str], max_tries: int = 500) -> List[str]:
    """Sample a random path that has not been visited yet."""
    for _ in range(max_tries):
        path = [random.choice(vocab_at(ops, d)) for d in range(trie.max_depth)]
        if not trie.is_visited(path):
            return path
    # Search space (nearly) exhausted: accept a revisit instead of looping forever
    return [random.choice(vocab_at(ops, d)) for d in range(trie.max_depth)]


# --------------------------------------------------------------------------
# Backpropagation
# --------------------------------------------------------------------------

def backpropagate(trie: ArchTrie, path: List[str], value_acc: float,
                  backup_mode: str = "mean", C: float = DEFAULT_C) -> None:
    """
    Update visit_count, accuracy and ucb_score along `path` (value_acc is 0-100).

    backup_mode:
        "mean" - running mean of accuracy (plain UCB1 baseline).
        "max"  - keep the best accuracy seen (use with LLM prior so promising
                 subtrees are not averaged away).

    The LLM prior is NOT applied here; it is blended in at traversal time
    via ucb_with_llm_prior().
    """
    # Collect nodes along the path: chain[0] is root, chain[i] is the node after path[i-1]
    node = trie.root
    chain = [node]
    for tok in path:
        node = node.children[tok]
        chain.append(node)

    # Root is part of every trial. Without this, root.visit_count stays 0 and the
    # exploration term for first-token decisions collapses to log(1) = 0.
    trie.root.visit_count += 1

    for depth in range(1, len(chain)):
        n = chain[depth]
        n.visit_count += 1

        if n.accuracy is None:
            n.accuracy = value_acc
        elif backup_mode == "max":
            n.accuracy = max(n.accuracy, value_acc)
        else:
            # Incremental running mean
            n.accuracy = (n.accuracy * (n.visit_count - 1) + value_acc) / n.visit_count

        # Parent was already incremented (root above, others in the previous
        # iteration), so its count is always > 0 here.
        n.ucb_score = _score(n, chain[depth - 1].visit_count, C)


def bump_visit_only(trie: ArchTrie, path: List[str]) -> None:
    """
    Increment visit counts along `path` without touching accuracy.
    Used for soft signals (proxy-only eval or LLM-prior-only hops) where no
    real accuracy is available.
    """
    # Same reason as in backpropagate(): root counts as visited on every trial
    trie.root.visit_count += 1
    for depth in range(len(path), 0, -1):
        trie.get_node(path[:depth]).visit_count += 1


# --------------------------------------------------------------------------
# LLM-prior-aware traversal
# --------------------------------------------------------------------------

def alpha_at(alpha_init: float, t: int, schedule: str = "inv_sqrt",
             budget: Optional[int] = None, reference_budget: int = 100) -> float:
    """
    Decay weight of the LLM prior at trial t.

    When `budget` is given, t is rescaled to a fraction of the budget
    (relative to reference_budget) so the prior fades out proportionally
    for any budget, not after a fixed number of iterations.
    """
    t = max(t, 1)
    if budget is not None and budget > 0:
        t = max(t / budget * reference_budget, 1e-9)

    if schedule == "inv_sqrt":
        return alpha_init / math.sqrt(t)
    if schedule == "constant":
        return alpha_init
    raise ValueError(f"unknown alpha_decay schedule: {schedule}")


def ucb_with_llm_prior(node: TrieNode, parent_visits: int, llm_prior_score: float,
                       C: float = DEFAULT_C, alpha: float = 0.3, t: int = 1,
                       budget: Optional[int] = None, reference_budget: int = 100) -> float:
    """
    Never-visited child: huge constant + decayed LLM prior, so unvisited siblings are tried
    first and ordered by the prior.
    Visited child: plain UCB1 on the 0-1 scale (accuracy None -> value 0), identical to _score().
    """
    if node.visit_count == 0:
        alpha_t = alpha_at(alpha, t, budget=budget, reference_budget=reference_budget)
        return 1e9 + alpha_t * llm_prior_score
    mean = node_value(node.accuracy)
    return mean + C * math.sqrt(math.log(max(parent_visits, 1)) / node.visit_count)


def traverse_with_llm(trie: ArchTrie, ops: List[str], llm_prior, C: float = DEFAULT_C,
                      alpha: float = 0.3, t: int = 1, budget: Optional[int] = None,
                      reference_budget: int = 100) -> List[str]:
    """Same as traverse(), but never-visited children are ordered by an LLM prior."""
    node, path = trie.root, []
    alpha_t = alpha_at(alpha, t, budget=budget, reference_budget=reference_budget)

    for d in range(trie.max_depth):
        for op in vocab_at(ops, d):
            if op not in node.children:
                node.children[op] = TrieNode()

        candidates = [(op, ch) for op, ch in node.children.items() if not ch.is_pruned]
        if not candidates:
            candidates = list(node.children.items())
        random.shuffle(candidates)  # random tie-breaking

        # Ask the LLM only about branches that have never been visited. These are exactly the
        # children ucb_with_llm_prior() treats as cold, so every cold child gets a real prior.
        unvisited_ops = [op for op, ch in candidates if ch.visit_count == 0]
        prior_scores = (
            llm_prior.score_branches(path, unvisited_ops, alpha_t=alpha_t)
            if unvisited_ops else {}
        )
        neutral = 1.0 / len(unvisited_ops) if unvisited_ops else 0.0

        best_op, best_score = None, float("-inf")
        for op, ch in candidates:
            prior = prior_scores.get(op, neutral)
            s = ucb_with_llm_prior(ch, node.visit_count, prior, C=C, alpha=alpha,
                                   t=t, budget=budget, reference_budget=reference_budget)
            if s > best_score:
                best_op, best_score = op, s

        path.append(best_op)
        node = node.children[best_op]
    return path
