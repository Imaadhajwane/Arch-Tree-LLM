# tools/test_nb101.py
"""
Unit tests for the TensorFlow-free NAS-Bench-101 layer (benchmarks/nb101.py) and the generic
search pieces it needs (per-position vocabulary, dead invalid leaves). No GPU needed.

Part A runs without any data. Part B runs only if data/nb101_cache.npz exists (round trip of
every 50th graph, oracle numbers, a short ArchTree / RE / random run).

    python tools/test_nb101.py          -> must end with "ALL NB101 TESTS PASSED"
"""
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import benchmarks.nb101 as nb  # noqa: E402

failures = []


def check(name, cond):
    print(("  ok    " if cond else "  FAIL  ") + name)
    if not cond:
        failures.append(name)


def rand_spec(rng, p=0.35):
    m = np.zeros((7, 7), dtype=np.int64)
    for i, j in nb.EDGE_POS:
        m[i, j] = rng.random() < p
    return m, ["input"] + [rng.choice(nb.OPS) for _ in range(5)] + ["output"]


print("A. pure functions")
rng = random.Random(0)
# known invariants of the official hash: relabelling interior vertices (topological order kept)
# does not change it; changing an op does
ok_perm = ok_tok = ok_canon = ok_mut = True
n_valid = 0
for _ in range(3000):
    m, o = rand_spec(rng)
    h = nb.spec_hash(m, o)
    if h is None:
        continue
    n_valid += 1
    pm, po = nb.prune(m, o)
    k = len(po)
    # swap two interior vertices that are not connected to each other -> isomorphic graph
    for a in range(1, k - 1):
        for b in range(a + 1, k - 1):
            if pm[a, b] == 0 and all(pm[a, x] == 0 for x in range(a + 1, b)) \
                    and all(pm[x, b] == 0 for x in range(a + 1, b)):
                perm = list(range(k))
                perm[a], perm[b] = perm[b], perm[a]
                q = pm[np.ix_(perm, perm)]
                if np.all(np.triu(q, 1) == q):
                    ok_perm &= nb.hash_module(q, [-1] + [nb.OPS.index(po[i]) for i in perm[1:-1]] + [-2]) == h
                break
    t = nb.spec_to_tokens(pm, po)
    ok_tok &= nb.spec_hash(*nb.tokens_to_spec(t)) == h
    full = list(o[1:-1]) + [str(int(m[i, j])) for i, j in nb.EDGE_POS]
    ok_canon &= nb.canonical_tokens(full) == t
    child = nb.mutate_tokens(full, rng)
    hc = nb.spec_hash(*nb.tokens_to_spec(child))
    ok_mut &= hc is not None and hc != h
check(f"valid random specs found ({n_valid})", n_valid > 1000)
check("hash invariant under isomorphic relabelling", ok_perm)
check("pruned spec -> tokens -> same hash", ok_tok)
check("canonical_tokens == tokens of the pruned graph", ok_canon)
check("mutation gives a valid, different graph", ok_mut)
m = np.zeros((7, 7), dtype=np.int64)
m[0, 1] = 1                                         # output unreachable
check("disconnected cell is invalid", nb.spec_hash(m, ["input"] + ["maxpool3x3"] * 5 + ["output"]) is None)
m = np.triu(np.ones((7, 7), dtype=np.int64), 1)     # 21 edges > 9
check("more than 9 edges is invalid", nb.spec_hash(m, ["input"] + ["maxpool3x3"] * 5 + ["output"]) is None)
check("vocabulary = 5 op positions + 21 bit positions", len(nb.VOCAB) == 26 and nb.VOCAB[5] == ["0", "1"])

from archtree.trie import ArchTrie  # noqa: E402
from archtree.ucb import traverse, vocab_at  # noqa: E402
tr = ArchTrie(max_depth=26)
p = traverse(tr, nb.VOCAB)
check("traverse respects per-position vocabulary",
      all(p[d] in vocab_at(nb.VOCAB, d) for d in range(26)))
leaf_parent = tr.get_node(p[:-1])
for tok in leaf_parent.children:
    tr.mark_dead(p[:-1] + [tok])
check("all children dead -> parent marked dead", leaf_parent.is_pruned)

print("B. cache")
from archtree import settings  # noqa: E402
import yaml  # noqa: E402
with open(settings.PROJECT_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)
bench = nb.NB101(cfg)
if not bench._cache_path.exists():
    print(f"  skip  {bench._cache_path} not built yet (python tools/build_nb101_cache.py)")
else:
    bench._load()
    check(f"423,624 graphs ({bench.total_archs})", bench.total_archs == 423624)
    idxs = range(0, bench.total_archs, 50)
    check("index -> tokens -> index round trip (every 50th graph)",
          all(bench.tokens_to_index(bench.index_to_tokens(i)) == i for i in idxs))
    o = bench.oracle()
    print(f"        oracle: best mean valid {o['best_valid']:.2f} (test {o['test_of_best_valid']:.2f}), "
          f"best mean test {o['best_test']:.2f}")
    check("best mean test accuracy ~94.3 (NB101 paper)", 93.9 < o["best_test"] < 94.6)
    from baselines.regularized_evolution import RegularizedEvolution
    re_ = RegularizedEvolution(bench, 0, 100)
    ev, best = set(), 0.0
    for _ in range(100):
        t = re_.propose(ev)
        i = bench.tokens_to_index(t)
        ev.add(i)
        a = bench.query(i)
        re_.update(t, a)
        best = max(best, a)
    check(f"regularized evolution runs, 100 unique evals, best valid {best:.2f}", len(ev) == 100)

if failures:
    print(f"\n{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("\nALL NB101 TESTS PASSED")
