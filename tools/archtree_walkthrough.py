#!/usr/bin/env python3
"""
ArchTree-LLM - live, data-driven pipeline walkthrough
=====================================================
Runs every stage of the pipeline on YOUR dataset files and, for each stage, produces
  (1) a "step card"   : purpose / how it works / what it produced / what it feeds,
  (2) live figures    : computed from the data, nothing hard-coded,
  (3) one PDF report  : demo_output/ArchTree_walkthrough.pdf  (+ every figure as PNG).

Nothing here is a canned result. The only inputs are your files:
  data/nb201_cifar10_cache.json           {"<base-5 index>": {"cifar10": acc, "cifar100": acc, "ImageNet16-120": acc}}
  data/true_pareto_front__<device>.json   [[accuracy, latency_ms], ...]     (optional, Step 6)
  <project root>/benchmarks/hwnb.py       per-architecture latency lookup   (optional, Step 6 live search)
  llm_cache__*.json                        your cached LLM priors            (optional, Step 5)

A SMALL SUBSET of the cache works: the trie is restricted to prefixes that exist in your file,
so the search never proposes an architecture you did not supply.

Run (from the archtree-llm project root, config.yaml is auto-detected):
    python archtree_walkthrough.py
    python archtree_walkthrough.py --dataset cifar100 --budget 100 --seeds 20
    python archtree_walkthrough.py --cache my_subset.json --pareto-dir my_fronts/ --hw edgegpu raspi4
    python archtree_walkthrough.py --llm-cache outputs/phase3/logs/llm_cache__nb201__cifar10__seed42.json
Or open in VS Code and run cell by cell (`# %%` markers).

Dependencies: numpy, scipy, matplotlib (pyyaml optional). No GPU needed.
"""
# %% ---------------------------------------------------------------- setup
import argparse
import json
import math
import os
import random
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _parse():
    ap = argparse.ArgumentParser(description="ArchTree-LLM live walkthrough")
    ap.add_argument("--cache", default=None, help="NB201 JSON cache (default: <root>/data/nb201_cifar10_cache.json)")
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "ImageNet16-120"])
    ap.add_argument("--pareto-dir", default=None, help="folder with true_pareto_front__<device>.json")
    ap.add_argument("--project-root", default=None, help="archtree-llm root (for live HW-NAS-Bench latency)")
    ap.add_argument("--hw", nargs="+", default=None, help="devices for the live hardware search (default: first two found)")
    ap.add_argument("--llm-cache", default=None, help="optional llm_cache__*.json to probe real prior quality")
    ap.add_argument("--budget", type=int, default=100, help="proposal budget of the demo searches")
    ap.add_argument("--hw-budget", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=20, help="seeds for the convergence curves")
    ap.add_argument("--seed", type=int, default=42, help="seed of the single narrated run")
    ap.add_argument("--proxy-pct", type=int, default=None, help="proxy gate percentile (default: config.yaml or 35)")
    ap.add_argument("--real-proxy", action="store_true", help="use the project's calibrated GPU ZC ensemble")
    ap.add_argument("--out", default="demo_output")
    ap.add_argument("--show", action="store_true", help="open figures interactively as well")
    return ap.parse_known_args()[0]


ARGS = _parse()

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

if not ARGS.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import kendalltau, spearmanr  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "legend.frameon": False,
})
C_MAIN, C_ALT, C_GREY, C_GOOD, C_WARN = "#1f3a5f", "#c0392b", "#8a8f98", "#2e8b57", "#d98e04"

# ---------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def _find_root(explicit):
    if explicit:
        return Path(explicit).resolve()
    for c in (Path.cwd(), HERE, HERE.parent):
        if (c / "config.yaml").exists():
            return c.resolve()
    return None


ROOT = _find_root(ARGS.project_root)
CFG = {}
if ROOT is not None:
    try:
        import yaml
        CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8")) or {}
    except Exception:
        CFG = {}


def _resolve(p, default):
    q = Path(p) if p else default
    return q.resolve()


CACHE_PATH = _resolve(ARGS.cache, (ROOT or Path.cwd()) / CFG.get("paths", {}).get("nb201_cache", "data/nb201_cifar10_cache.json"))
DATA_DIR = _resolve(ARGS.pareto_dir, (ROOT or Path.cwd()) / "data")
OUT = Path(ARGS.out).resolve()
OUT.mkdir(parents=True, exist_ok=True)
LLM_CACHE = Path(ARGS.llm_cache).resolve() if ARGS.llm_cache else None
PROXY_PCT = ARGS.proxy_pct or CFG.get("zc_proxy", {}).get("proxy_threshold_percentile", 35)

# ---------------------------------------------------------------- NB201 constants (identical to archtree/encoder.py)
OPS = ["none", "skip_connect", "conv_1x1", "conv_3x3", "avg_pool_3x3"]
SHORT = {"none": "none", "skip_connect": "skip", "conv_1x1": "c1x1", "conv_3x3": "c3x3", "avg_pool_3x3": "pool"}
EDGE_NAMES = ["n1<-n0", "n2<-n0", "n2<-n1", "n3<-n0", "n3<-n1", "n3<-n2"]
L, K = 6, 5
TOTAL_SPACE = K ** L
DATASETS = ["cifar10", "cifar100", "ImageNet16-120"]
HW_ALL = ["edgegpu", "raspi4", "edgetpu", "pixel3", "eyeriss", "fpga"]


def idx_to_tokens(i):
    """Base-5 decode, token 0 = most significant digit (same bijection as NB201.index_to_tokens)."""
    out, i = [], int(i)
    for _ in range(L):
        out.append(OPS[i % K])
        i //= K
    return out[::-1]


def tokens_to_idx(tokens):
    i = 0
    for t in tokens:
        i = i * K + OPS.index(t)
    return i


def arch_string(t):
    return f"|{t[0]}~0|+|{t[1]}~0|{t[2]}~1|+|{t[3]}~0|{t[4]}~1|{t[5]}~2|"


# ---------------------------------------------------------------- report machinery
STEPS, METRICS = [], {}
CUR = None


def step(no, title, purpose, how, produced, feeds):
    global CUR
    CUR = dict(no=no, title=title, purpose=purpose, how=how, produced=produced, feeds=feeds, live=[], figs=[])
    STEPS.append(CUR)
    bar = "=" * 96
    print(f"\n{bar}\nSTEP {no} - {title}\n{bar}")
    for head, txt in (("PURPOSE", purpose), ("HOW IT WORKS", how), ("WHAT IT PRODUCES", produced)):
        print(f"\n{head}\n" + textwrap.fill(txt, 96, initial_indent="  ", subsequent_indent="  "))
    print("\nLIVE OBSERVATIONS (computed from your data)")
    return CUR


def note(text):
    CUR["live"].append(text)
    print(textwrap.fill(text, 96, initial_indent="  - ", subsequent_indent="    "))


def close_step():
    print("\nFEEDS INTO THE NEXT STEP\n" + textwrap.fill(CUR["feeds"], 96, initial_indent="  ", subsequent_indent="  "))


def save(fig, name):
    p = OUT / f"{name}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    CUR["figs"].append(p)
    print(f"  [figure] {p.name}")
    if ARGS.show:
        plt.show()
    plt.close(fig)


# %% ---------------------------------------------------------------- load + validate the dataset
print(f"Loading {CACHE_PATH}")
if not CACHE_PATH.exists():
    sys.exit(f"Cache not found: {CACHE_PATH}\nPass --cache <file> or run from the project root.")
with open(CACHE_PATH, encoding="utf-8") as f:
    _raw = json.load(f)
_raw.pop("_meta", None)
# v2 cache (Oct 2026): {"cifar10": {"valid": .., "test": .., "train_sec": ..}} -> use the search metric
CACHE = {int(k): {d: (x.get(CFG.get("benchmarks", {}).get("nb201_search_split", "valid")) if isinstance(x, dict) else x)
                  for d, x in v.items()} for k, v in _raw.items()}
_sample = list(CACHE.values())[:50]
AVAIL = [d for d in DATASETS if all(isinstance(v, dict) and d in v for v in _sample)]
if ARGS.dataset not in AVAIL:
    sys.exit(f"Dataset '{ARGS.dataset}' not in cache (found: {AVAIL})")
DS = ARGS.dataset

ACC = {i: float(v[DS]) for i, v in CACHE.items() if float(v.get(DS, 0) or 0) > 0}
N_DROPPED = len(CACHE) - len(ACC)
IDX_LIST = sorted(ACC)
IDX = np.array(IDX_LIST)
Y = np.array([ACC[i] for i in IDX_LIST])
N = len(IDX_LIST)
TOK = {i: tuple(idx_to_tokens(i)) for i in IDX_LIST}
D = np.stack([(IDX // K ** (L - 1 - d)) % K for d in range(L)], axis=1)          # N x 6 digit matrix
ACC10 = {i: float(v["cifar10"]) for i, v in CACHE.items() if "cifar10" in v and float(v["cifar10"]) > 0}
FSTAR = Y.max()

PREFIX_OPS = defaultdict(set)     # which ops exist under each prefix in YOUR file (makes subsets safe)
for _t in TOK.values():
    for _d in range(L):
        PREFIX_OPS[_t[:_d]].add(_t[_d])

print(f"  {len(CACHE)} entries, {N} usable for '{DS}' ({N_DROPPED} dropped as zero/missing), datasets in file: {AVAIL}")
print(f"  coverage of the full NB201 space: {100 * N / TOTAL_SPACE:.1f}%  (best {DS} accuracy in file f* = {FSTAR:.2f}%)")
METRICS.update(n_archs=N, fstar=float(FSTAR), dataset=DS)


# %% ---------------------------------------------------------------- core algorithm (mirrors archtree/trie.py, ucb.py, search.py)
class Node:
    __slots__ = ("ch", "n", "val", "term")

    def __init__(self):
        self.ch, self.n, self.val, self.term = {}, 0, None, False


class Trie:
    def __init__(self):
        self.root, self.n_term = Node(), 0

    def get(self, path):
        node = self.root
        for t in path:
            node = node.ch.get(t)
            if node is None:
                return None
        return node

    def insert(self, path):
        node = self.root
        for t in path:
            node = node.ch.setdefault(t, Node())
        if not node.term:
            node.term = True
            self.n_term += 1
        return node

    def visited(self, path):
        n = self.get(path)
        return n is not None and n.term

    def n_nodes(self):
        c, st = 0, [self.root]
        while st:
            n = st.pop()
            c += len(n.ch)
            st.extend(n.ch.values())
        return c


def ucb(child, parent_n, C, scale):
    """Unvisited / never-evaluated child -> +inf (forces exploration). Else mean + C*sqrt(ln N / n).
    scale='unit'  : accuracy/100 (Phase 1/2 traverse)   scale='raw' : accuracy in % (Phase 3/4)."""
    if child.n == 0 or child.val is None:
        return float("inf")
    mean = child.val / 100.0 if scale == "unit" else child.val
    return mean + C * math.sqrt(math.log(max(parent_n, 1)) / child.n)


def traverse(trie, rng, C=1.414, scale="unit"):
    node, path = trie.root, []
    for _ in range(L):
        allowed = PREFIX_OPS[tuple(path)]
        for op in allowed:
            node.ch.setdefault(op, Node())
        cands = [(op, node.ch[op]) for op in OPS if op in allowed]
        rng.shuffle(cands)                                   # random tie-break (all-inf ties are common)
        op, child = max(cands, key=lambda kv: ucb(kv[1], max(node.n, 1), C, scale))
        path.append(op)
        node = child
    return path


def backprop(trie, path, value, mode="mean"):
    trie.root.n += 1                                         # root counts every trial (bug fixed in ucb.py)
    node = trie.root
    for tok in path:
        node = node.ch[tok]
        node.n += 1
        if node.val is None:
            node.val = value
        elif mode == "max":
            node.val = max(node.val, value)
        else:
            node.val = (node.val * (node.n - 1) + value) / node.n


def bump(trie, path):
    """Visit-count-only update for proxy-skipped proposals: never writes into accuracy."""
    trie.root.n += 1
    node = trie.root
    for tok in path:
        node = node.ch.setdefault(tok, Node())
        node.n += 1


def random_unvisited(trie, rng):
    for _ in range(500):
        i = rng.choice(IDX_LIST)
        if not trie.visited(TOK[i]):
            return list(TOK[i])
    return list(TOK[rng.choice(IDX_LIST)])


def run_search(budget, seed, gate=None, warmup=10, pct=35, C=1.414, scale="unit", mode="mean"):
    rng, trie = random.Random(seed), Trie()
    hist, seen, dup, skips, real_after = [], [], 0, 0, []
    for i in range(budget):
        path = traverse(trie, rng, C, scale)
        if trie.visited(path):                               # trie dedup: redirect, never re-evaluate
            dup += 1
            path = random_unvisited(trie, rng)
        real = True
        if gate is not None and i >= warmup:
            s = gate(tuple(path))
            seen.append(s)
            real = s >= np.percentile(seen, 100 - pct)
        if real:
            idx = tokens_to_idx(path)
            acc = ACC[idx]
            trie.insert(path).val = acc
            backprop(trie, path, acc, mode)
            hist.append({"iter": i + 1, "idx": idx, "acc": acc, "path": tuple(path)})
        else:
            skips += 1
            bump(trie, path)
        real_after.append(len(hist))
    return dict(hist=hist, trie=trie, dup=dup, skips=skips, real_after=real_after, budget=budget)


def best_curve(accs, K_):
    a = np.asarray(accs, float)[:K_]
    c = np.maximum.accumulate(a)
    return np.concatenate([c, np.full(K_ - len(c), c[-1])]) if len(c) < K_ else c


def random_run(seed, n):
    rng = random.Random(seed)
    return [ACC[i] for i in rng.sample(IDX_LIST, min(n, N))]


def band(ax, x, curves, color, label, ls="-"):
    curves = np.asarray(curves)
    mu = curves.mean(0)
    se = curves.std(0, ddof=1) / math.sqrt(len(curves)) if len(curves) > 1 else 0 * mu
    ax.plot(x, mu, color=color, label=label, ls=ls, lw=1.8)
    ax.fill_between(x, mu - se, mu + se, color=color, alpha=0.18)


def draw_trie(ax, hist, title):
    """Draw the visited prefix tree: x = leaf order, y = depth, colour = accuracy."""
    tree = {}
    for h in hist:
        n = tree
        for t in h["path"]:
            n = n.setdefault(t, {})
    leaf_acc = {h["path"]: h["acc"] for h in hist}
    pos, cnt = {}, [0]

    def lay(node, path, depth):
        if not node:
            pos[path] = (cnt[0], depth)
            cnt[0] += 1
            return pos[path][0]
        xs = [lay(node[op], path + (op,), depth + 1) for op in sorted(node, key=OPS.index)]
        pos[path] = (float(np.mean(xs)), depth)
        return pos[path][0]

    lay(tree, (), 0)
    for p, (x, y) in pos.items():
        if p:
            px, py = pos[p[:-1]]
            ax.plot([px, x], [py, y], color=C_GREY, lw=0.9, zorder=1)
    accs = list(leaf_acc.values())
    vmin, vmax = (min(accs), max(accs)) if accs else (0, 1)
    for p, (x, y) in pos.items():
        if p and p in leaf_acc:
            sc = ax.scatter([x], [y], s=95, c=[leaf_acc[p]], cmap="viridis", vmin=vmin, vmax=vmax, zorder=3,
                            edgecolor="k", lw=0.5)
            ax.text(x, y + 0.28, f"{leaf_acc[p]:.1f}", ha="center", fontsize=6.5)
        elif p:
            ax.scatter([x], [y], s=26, color="white", edgecolor=C_MAIN, zorder=3)
        else:
            ax.scatter([x], [y], s=70, color=C_MAIN, zorder=3)
        if p:
            ax.text(x, y - 0.28, SHORT[p[-1]], ha="center", fontsize=6, color=C_MAIN)
    ax.set_ylim(L + 0.9, -0.9)
    ax.set_yticks(range(L + 1))
    ax.set_yticklabels(["root"] + [f"pos {d}" for d in range(1, L + 1)])
    ax.set_xticks([])
    ax.grid(False)
    ax.set_title(title)
    return sc if accs else None


# %% ================================================================ STEP 1 - dataset & encoding
s = step(
    1, "Dataset and architecture encoding",
    "Establish the ground truth every later stage relies on: what one 'evaluation' is, how an architecture becomes "
    "a token sequence, and what the accuracy landscape looks like.",
    "A NAS-Bench-201 cell has 4 nodes and 6 edges; each edge picks 1 of 5 operations, giving 5^6 = 15,625 "
    "architectures. An architecture is the sequence of 6 operation tokens, and its integer index is that sequence "
    "read as a base-5 number (token 0 is the most significant digit). Your JSON cache maps index -> accuracy per "
    "dataset, so a 'real evaluation' in this project is a dictionary lookup instead of a multi-hour training run.",
    "A validated accuracy table, decoded token sequences, the ceiling accuracy f*, the accuracy distribution of each "
    "dataset, cross-dataset agreement, and a position x operation effect map.",
    "Accuracy becomes the reward signal of the search; the 6-token sequence becomes the root-to-leaf path in the trie "
    "built in Step 2; f* is the reference for measuring regret and gap-to-optimum in every later step.")

rng0 = random.Random(ARGS.seed)
demo = rng0.sample(IDX_LIST, 3)
print("\n  index -> tokens -> NB201 architecture string -> accuracies (looked up, not computed)")
for i in demo:
    t = idx_to_tokens(i)
    digs = "".join(str(OPS.index(x)) for x in t)
    accs = ", ".join(f"{d}={CACHE[i][d]:.2f}%" for d in AVAIL)
    print(f"  idx {i:5d}  base5={digs}  {t}\n              {arch_string(t)}\n              {accs}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for d, col in zip(AVAIL, (C_MAIN, C_ALT, C_GOOD)):
    v = np.array([CACHE[i][d] for i in IDX_LIST if CACHE[i].get(d, 0) > 0])
    ax[0].hist(v, bins=60, alpha=0.55, color=col, label=f"{d} (max {v.max():.1f})", density=True)
ax[0].set(title="Accuracy distribution per dataset", xlabel="accuracy (%)", ylabel="density")
ax[0].legend()
others = [d for d in AVAIL if d != DS]
sub = np.random.default_rng(0).choice(N, min(3000, N), replace=False)
rho_txt = []
for d, col in zip(others, (C_ALT, C_GOOD)):
    ok = np.array([CACHE[IDX_LIST[j]].get(d, 0) > 0 for j in sub])
    xs = Y[sub][ok]
    ys = np.array([CACHE[IDX_LIST[j]][d] for j in sub])[ok]
    rho = spearmanr(xs, ys)[0]
    rho_txt.append((d, rho))
    ax[1].scatter(xs, ys, s=5, alpha=0.35, color=col, label=f"{d} (rho={rho:.2f})")
ax[1].set(title=f"Does a good {DS} architecture stay good elsewhere?", xlabel=f"{DS} accuracy (%)", ylabel="other dataset (%)")
if others:
    ax[1].legend(markerscale=3)
H = np.full((L, K), np.nan)
for d_ in range(L):
    for k in range(K):
        m = D[:, d_] == k
        if m.any():
            H[d_, k] = Y[m].mean()
im = ax[2].imshow(H, cmap="RdYlGn", aspect="auto")
ax[2].set_xticks(range(K))
ax[2].set_xticklabels([SHORT[o] for o in OPS])
ax[2].set_yticks(range(L))
ax[2].set_yticklabels([f"tok{d_}  {EDGE_NAMES[d_]}" for d_ in range(L)])
ax[2].grid(False)
for a_ in range(L):
    for b_ in range(K):
        if not np.isnan(H[a_, b_]):
            ax[2].text(b_, a_, f"{H[a_, b_]:.0f}", ha="center", va="center", fontsize=8)
ax[2].set_title(f"Mean {DS} accuracy by position x operation")
fig.colorbar(im, ax=ax[2], fraction=0.046)
fig.tight_layout()
save(fig, "step1_dataset_overview")

lowfrac = float((Y < 20).mean()) if DS == "cifar10" else float((Y < np.percentile(Y, 10)).mean())
top1 = np.percentile(Y, 99)
best_pos = int(np.nanargmax(np.nanmax(H, 1) - np.nanmin(H, 1)))
note(f"{N} architectures loaded ({100 * N / TOTAL_SPACE:.1f}% of NB201); f* on {DS} = {FSTAR:.2f}%, median = {np.median(Y):.2f}%, "
     f"top-1% cut-off = {top1:.2f}%.")
if DS == "cifar10":
    note(f"{100 * lowfrac:.1f}% of architectures score below 20%: these are cells whose input never reaches the output "
         f"(all paths 'none'), so the distribution is heavily left-skewed with a thin high-accuracy head.")
if rho_txt:
    note("Cross-dataset rank agreement (Spearman): " + ", ".join(f"{d} rho={r:.2f}" for d, r in rho_txt) +
         ". High but not perfect agreement is why the proxy is calibrated separately per dataset.")
note(f"The position with the largest operation effect is token {best_pos} ({EDGE_NAMES[best_pos]}): its best and worst "
     f"operation differ by {np.nanmax(H[best_pos]) - np.nanmin(H[best_pos]):.1f} accuracy points on average.")
close_step()

# %% ================================================================ STEP 2 - trie and prefix structure
s = step(
    2, "Compressed trie and prefix structure",
    "Show why a prefix tree is a sensible search state, and demonstrate the deduplication guarantee.",
    "Every architecture is a root-to-leaf path of length 6. Insert and is_visited are O(L) = O(6). Architectures "
    "that share a prefix share nodes, so a node's statistics summarise a whole family of architectures. Whether "
    "that summary is informative is measured by eta^2 = the fraction of accuracy variance explained by the first "
    "k tokens, and by the number of depth-k prefixes that still contain a near-optimal leaf.",
    "The trie size profile per depth, the eta^2 curve (prefix locality), the count of near-optimal cells, and a "
    "live demonstration that a visited architecture cannot be evaluated twice.",
    "UCB in Step 3 reads and updates exactly these node statistics. If eta^2 is near zero at shallow depth, "
    "early decisions are close to coin flips; if it rises steadily, prefix statistics carry usable signal.")

rows = []
for k in range(1, L + 1):
    pid = IDX // K ** (L - k)
    u, inv, cnt = np.unique(pid, return_inverse=True, return_counts=True)
    mean = np.bincount(inv, weights=Y) / cnt
    between = np.sum(cnt * (mean - Y.mean()) ** 2) / N
    sq = np.bincount(inv, weights=Y ** 2) / cnt - mean ** 2
    submax = np.full(len(u), -np.inf)
    np.maximum.at(submax, inv, Y)
    rows.append(dict(k=k, nodes=len(u), eta2=between / Y.var(), within=float(np.sqrt(np.clip(sq, 0, None)).mean()),
                     near=int((submax >= FSTAR - 1.0).sum())))
fig, ax = plt.subplots(1, 3, figsize=(15, 3.9))
ks = [r["k"] for r in rows]
ax[0].bar(np.array(ks) - 0.2, [K ** k for k in ks], 0.4, color=C_GREY, label="full NB201 (5^k)")
ax[0].bar(np.array(ks) + 0.2, [r["nodes"] for r in rows], 0.4, color=C_MAIN, label="in your file")
ax[0].set(yscale="log", title="Trie nodes per depth", xlabel="depth k", ylabel="# prefixes")
ax[0].legend()
ax[1].plot(ks, [r["eta2"] for r in rows], "o-", color=C_ALT)
ax[1].set(title="Prefix locality: variance explained by first k tokens", xlabel="prefix length k", ylabel="eta^2", ylim=(0, 1.02))
ax[2].plot(ks, [r["near"] for r in rows], "o-", color=C_GOOD)
ax[2].set(title="Depth-k cells containing a leaf within 1pp of optimum", xlabel="prefix length k", ylabel="# cells", yscale="log")
fig.tight_layout()
save(fig, "step2_prefix_structure")

t_demo = Trie()
pth = list(TOK[demo[0]])
t_demo.insert(pth)
before = (t_demo.n_term, t_demo.visited(pth))
t_demo.insert(pth)
note(f"Trie depth profile: {rows[0]['nodes']} first-level nodes growing to {rows[-1]['nodes']} leaves; the shared prefixes "
     f"mean {sum(r['nodes'] for r in rows)} nodes represent {N} architectures instead of {N * L} raw tokens "
     f"({100 * (1 - sum(r['nodes'] for r in rows) / (N * L)):.0f}% smaller).")
note(f"Dedup demo: inserting architecture {demo[0]} twice leaves terminals = {t_demo.n_term} (was {before[0]}), "
     f"is_visited = {t_demo.visited(pth)} - the trie makes re-evaluation structurally impossible (Theorem T2).")
note("eta^2 by depth: " + ", ".join(f"k={r['k']}: {r['eta2']:.2f}" for r in rows) +
     (". Signal is weak at the root and grows with depth - the reason UCB behaves close to random early on."
      if rows[0]["eta2"] < 0.3 else ". A large share of variance is already explained by the first tokens."))
METRICS["eta2"] = [r["eta2"] for r in rows]
close_step()

# %% ================================================================ STEP 3 - UCB traversal (Phase 1)
B = ARGS.budget
SEEDS = [1000 + i for i in range(ARGS.seeds)]
s = step(
    3, "UCB1 traversal, backpropagation and the memory it builds",
    "Decide which architecture to evaluate next, balancing exploitation of promising prefixes with exploration of "
    "rarely visited ones, while never repeating an evaluation.",
    "At each of the 6 depths the search scores every child with mean_accuracy + C*sqrt(ln N_parent / n_child) "
    "(C = 1.414). Children never visited score infinity, so each is tried once, with ties broken at random. The "
    "reached leaf is looked up in the benchmark; its accuracy is then written back along the whole path (visit "
    "counts and running mean at every ancestor, including the root). If the chosen leaf was already evaluated, the "
    "trie redirects to a random unvisited leaf.",
    "The visited-prefix tree itself (the search's memory), a per-branch decision table showing exploitation versus "
    "exploration, and a convergence curve compared with random search.",
    "The proposals produced here are exactly what the zero-cost proxy gate in Step 4 will filter, and the trie's "
    "statistics are what the LLM prior in Step 5 would extend for never-visited siblings.")

narr = run_search(30, ARGS.seed)
fig, ax = plt.subplots(1, 1, figsize=(13, 4.6))
sc = draw_trie(ax, narr["hist"][:14], f"Trie after the first 14 UCB evaluations ({DS}, seed {ARGS.seed}) - colour = accuracy")
if sc is not None:
    fig.colorbar(sc, ax=ax, fraction=0.02, label="accuracy (%)")
save(fig, "step3a_trie_snapshot")

root = narr["trie"].root
tab = []
for op in OPS:
    c = root.ch.get(op)
    if c is None or c.n == 0 or c.val is None:
        tab.append((op, 0, 0.0, 0.0, float("inf")))
    else:
        bonus = 1.414 * math.sqrt(math.log(root.n) / c.n)
        tab.append((op, c.n, c.val / 100, bonus, c.val / 100 + bonus))
print(f"\n  Decision table at the root after 30 iterations (root visits = {root.n}):")
print(f"  {'first token':<14}{'visits':>7}{'mean acc/100':>14}{'explore bonus':>15}{'UCB score':>12}")
for op, n_, m_, b_, s_ in tab:
    print(f"  {op:<14}{n_:>7}{m_:>14.3f}{b_:>15.3f}{s_:>12.3f}")

ucb_curves, rnd_curves, dups = [], [], []
for sd in SEEDS:
    r = run_search(B, sd)
    ucb_curves.append(best_curve([h["acc"] for h in r["hist"]], B))
    rnd_curves.append(best_curve(random_run(sd, B), B))
    dups.append(r["dup"])
fig, ax = plt.subplots(1, 2, figsize=(13, 4.1))
vis = [t[1] for t in tab]
ex = [t[2] if t[1] else 0 for t in tab]
bo = [t[3] if t[1] else 0 for t in tab]
ax[0].bar([SHORT[o] for o in OPS], ex, color=C_MAIN, label="exploitation (mean acc / 100)")
ax[0].bar([SHORT[o] for o in OPS], bo, bottom=ex, color=C_WARN, label="exploration bonus")
for i_, v in enumerate(vis):
    ax[0].text(i_, ex[i_] + bo[i_] + 0.01, f"n={v}" if v else "unvisited\n(score = inf)", ha="center", fontsize=8)
ax[0].set(title="Root decision: what UCB weighs for each first token", ylabel="score component")
ax[0].legend(loc="lower right")
x = np.arange(1, B + 1)
band(ax[1], x, rnd_curves, C_GREY, "random search", "--")
band(ax[1], x, ucb_curves, C_MAIN, "ArchTree UCB1 (trie)")
ax[1].axhline(FSTAR, color=C_GOOD, ls=":", label=f"optimum f*={FSTAR:.2f}")
ax[1].set(title=f"Best-so-far accuracy ({len(SEEDS)} seeds, mean +/- s.e.m.)", xlabel="real evaluations", ylabel="accuracy (%)",
          ylim=(np.mean(rnd_curves, 0)[0] - 1, FSTAR + 0.6))
ax[1].legend(loc="lower right")
fig.tight_layout()
save(fig, "step3b_ucb_decision_and_convergence")

u_end, r_end = np.mean([c[-1] for c in ucb_curves]), np.mean([c[-1] for c in rnd_curves])
u_25, r_25 = np.mean([c[min(24, B - 1)] for c in ucb_curves]), np.mean([c[min(24, B - 1)] for c in rnd_curves])
note(f"At the root, first tokens with more visits also have higher mean accuracy - exploitation is working - while unvisited "
     f"branches would still score infinity (exploration guarantee).")
note(f"After {B} evaluations UCB reached {u_end:.2f}% vs {r_end:.2f}% for random search (gap to optimum "
     f"{FSTAR - u_end:.2f} vs {FSTAR - r_end:.2f} pp); after 25 evaluations {u_25:.2f}% vs {r_25:.2f}%.")
note(f"UCB proposed an already-evaluated leaf {np.mean(dups):.1f} times per run on average; the trie redirected every one, "
     f"so {B} proposals became {B} distinct evaluations.")
note("Honest reading: on NB201 the accuracy landscape saturates quickly, so the advantage over random search is modest "
     "and grows mainly at small budgets. (Pruning every 50 evaluations is omitted in this demo for clarity.)")
METRICS.update(ucb_best=float(u_end), rnd_best=float(r_end))
close_step()

# %% ================================================================ STEP 4 - zero-cost proxy gate (Phase 2)
s = step(
    4, "Zero-cost proxy ensemble and the evaluation gate",
    "Avoid spending a real evaluation on architectures that are unlikely to be good, using a score that needs no training.",
    "A proxy scores an architecture from its structure alone (the project's GPU ensemble uses expressivity, SynFlow, "
    "NASWOT and complexity on an untrained network). Its component scores are z-normalised and combined by "
    "least-squares weights fitted on a calibration sample against that dataset's true accuracy; quality is judged on "
    "a disjoint held-out sample by Kendall tau. In the search, a proposal only triggers a real evaluation if its proxy "
    "score is in the top X% of scores seen so far (after 10 warm-up real evaluations); skipped proposals only bump "
    "visit counts and never write into the accuracy statistic.",
    "A calibrated proxy with a measured ranking quality, the recall curve of the gate (how much of the true top "
    "architectures survive at each threshold), and the saving in real evaluations at equal accuracy.",
    "The gate is what turns 'N proposals' into far fewer real evaluations - the efficiency result of the project. "
    "It also defines which proposals reach the LLM/UCB machinery of Step 5 and the hardware step.")


def structural_features(Dm):
    """Label-free, GPU-free features computed from the 6 tokens only (stand-in for E/S/T/C, see --real-proxy)."""
    e = [Dm[:, j] for j in range(L)]                        # ops: 0 none 1 skip 2 c1 3 c3 4 pool
    nz = [x != 0 for x in e]
    n1 = nz[0]
    n2 = nz[1] | (nz[2] & n1)
    connected = (nz[3] | (nz[4] & n1) | (nz[5] & n2)).astype(float)
    w = np.array([0, 0, 2, 6, 1])
    flops = sum(w[x] for x in e).astype(float)
    cnt = lambda k: sum((x == k).astype(float) for x in e)
    return np.stack([connected, flops, cnt(3), cnt(2), cnt(1), cnt(4), cnt(0), connected * flops], 1)


def make_proxy():
    if ARGS.real_proxy and ROOT is not None:
        try:
            os.chdir(ROOT)
            sys.path.insert(0, str(ROOT))
            from archtree.zc_proxy import ZCProxyEnsemble
            zc = ZCProxyEnsemble.from_saved("phase2", "nb201", DS)
            memo = {}

            def f(tok):
                if tok not in memo:
                    memo[tok] = zc.score(list(tok))
                return memo[tok]
            return f, "project ZC ensemble (E,S,T,C) - real calibrated proxy"
        except Exception as e:
            print(f"  [proxy] real proxy unavailable ({e!r}); using the structural proxy instead.")
    return None, "structural zero-cost proxy (label-free features of the token sequence)"


real_f, proxy_name = make_proxy()
rs = np.random.default_rng(7)
perm = rs.permutation(N)
ncal = min(1500, N // 3)
cal, tst = perm[:ncal], perm[ncal:ncal + min(1500, N - ncal)]
if real_f is None:
    F = structural_features(D)
    mu, sd = F[cal].mean(0), F[cal].std(0) + 1e-8
    Z = (F - mu) / sd
    coef, *_ = np.linalg.lstsq(np.hstack([Z[cal], np.ones((ncal, 1))]), Y[cal], rcond=None)
    w = coef[:-1]
    SC = Z @ w
    PROXY_SCORE = {IDX_LIST[j]: float(SC[j]) for j in range(N)}
    gate = lambda tok: PROXY_SCORE[tokens_to_idx(tok)]
    sc_test = SC[tst]
else:
    gate = real_f
    sc_test = np.array([gate(TOK[IDX_LIST[j]]) for j in tst[:300]])
    tst = tst[:300]
tau = kendalltau(sc_test, Y[tst])[0]
rho_p = spearmanr(sc_test, Y[tst])[0]

fracs = np.linspace(0.05, 1.0, 20)
order = np.argsort(-sc_test)
top5 = Y[tst] >= np.percentile(Y[tst], 95)
top1 = Y[tst] >= np.percentile(Y[tst], 99)
rec5 = [top5[order[:max(1, int(f_ * len(tst)))]].sum() / max(top5.sum(), 1) for f_ in fracs]
rec1 = [top1[order[:max(1, int(f_ * len(tst)))]].sum() / max(top1.sum(), 1) for f_ in fracs]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].scatter(sc_test, Y[tst], s=5, alpha=0.4, color=C_MAIN)
ax[0].set(title=f"Held-out ranking quality: Kendall tau = {tau:+.3f}", xlabel="proxy score", ylabel=f"true {DS} accuracy (%)")
ax[1].plot(fracs, rec5, "o-", color=C_MAIN, label="true top-5% retained")
ax[1].plot(fracs, rec1, "s-", color=C_ALT, label="true top-1% retained")
ax[1].plot(fracs, fracs, ":", color=C_GREY, label="random gate")
ax[1].axvline(PROXY_PCT / 100, color=C_WARN, ls="--", label=f"gate keeps top {PROXY_PCT}%")
ax[1].set(title="Gate recall: what survives at each threshold", xlabel="fraction of proposals sent to real evaluation",
          ylabel="fraction of true top architectures retained")
ax[1].legend()
fig.tight_layout()
save(fig, "step4a_proxy_quality_and_gate")

rec_at = float(np.interp(PROXY_PCT / 100, fracs, rec5))
G_runs, U_runs, R_runs = [], [], []
for sd in SEEDS:
    G_runs.append(run_search(B, sd, gate=gate, pct=PROXY_PCT))
    U_runs.append(run_search(B, sd))
Rmin = max(10, int(min(len(g["hist"]) for g in G_runs)))
gc_ = [best_curve([h["acc"] for h in g["hist"]], Rmin) for g in G_runs]
uc_ = [best_curve([h["acc"] for h in u["hist"]], Rmin) for u in U_runs]
rc_ = [best_curve(random_run(sd, Rmin), Rmin) for sd in SEEDS]
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))
xp = np.arange(1, B + 1)
band(ax[0], xp, [g["real_after"] for g in G_runs], C_MAIN, "proxy-gated search (real evals so far)")
ax[0].plot(xp, xp, ":", color=C_GREY, label="ungated (1 proposal = 1 evaluation)")
ax[0].set(title="Proposals vs real evaluations", xlabel="proposals", ylabel="real evaluations spent")
ax[0].legend()
xr = np.arange(1, Rmin + 1)
band(ax[1], xr, rc_, C_GREY, "random search", "--")
band(ax[1], xr, uc_, C_ALT, "UCB, ungated")
band(ax[1], xr, gc_, C_MAIN, "UCB + proxy gate")
ax[1].set(title=f"Equal real-evaluation cost (first {Rmin} real evals)", xlabel="real evaluations", ylabel="best accuracy (%)",
          ylim=(np.mean(rc_, 0)[0] - 1, FSTAR + 0.6))
ax[1].legend(loc="lower right")
fig.tight_layout()
save(fig, "step4b_gate_efficiency")

real_mean = np.mean([len(g["hist"]) for g in G_runs])
note(f"Proxy used: {proxy_name}. Held-out Kendall tau = {tau:+.3f}, Spearman = {rho_p:+.3f} "
     f"(calibrated on {ncal} architectures with {DS} labels, tested on {len(tst)} disjoint ones).")
note(f"With the gate keeping the top {PROXY_PCT}% of proposals, {100 * rec_at:.0f}% of the true top-5% architectures still pass "
     f"(random gating would keep {PROXY_PCT}%): the proxy is a filter, not a predictor.")
note(f"Out of {B} proposals the gated search spent {real_mean:.1f} real evaluations on average "
     f"({100 * (1 - real_mean / B):.0f}% saved). At the same number of real evaluations ({Rmin}) it reached "
     f"{np.mean([c[-1] for c in gc_]):.2f}% vs {np.mean([c[-1] for c in uc_]):.2f}% (ungated) and {np.mean([c[-1] for c in rc_]):.2f}% (random).")
note("Why calibration is per dataset: the proxy is fitted against one dataset's labels, so a cifar10 fit is not assumed valid "
     "for cifar100 or ImageNet16-120 - run with --dataset to see the weights and tau change.")
METRICS.update(proxy_tau=float(tau), gate_recall=float(rec_at), real_evals=float(real_mean), gated_best=float(np.mean([c[-1] for c in gc_])))
close_step()

# %% ================================================================ STEP 5 - LLM warm-start prior (Phase 3)
s = step(
    5, "LLM warm-start prior and what it can and cannot influence",
    "Give never-visited sibling branches an informed ordering instead of a random one, using a language model's "
    "semantic knowledge of operations.",
    "For unvisited children the UCB score is 1e9 + alpha_t * prior, for visited children it is the ordinary UCB1 score. "
    "The huge offset means the prior can only change the ORDER in which unvisited siblings are first expanded - it never "
    "overrides observed accuracy. alpha_t = alpha / sqrt(t/budget*100) decays, and once it drops below a floor of 0.02 the "
    "prior is switched off and no LLM call is made, so alpha really controls for how long the LLM is active. Answers are "
    "cached per trie prefix.",
    "The alpha schedule and its activity window, a live scoring table showing the two score regimes, and a measurement "
    "of how much signal a prior could possibly exploit in your data (plus real prior quality if you pass --llm-cache).",
    "Whatever ordering the prior gives feeds the same traversal as Step 3; its usefulness is bounded by how different "
    "sibling subtrees really are, which is measured here and connects to the convergence findings of the project.")


def alpha_at(alpha, t, budget=100, ref=100):
    t = max(t, 1)
    t = max(t / budget * ref, 1e-9)
    return alpha / math.sqrt(t)


alphas = [0.05, 0.1, 0.15, 0.2, 0.3, 1.0]
tt = np.arange(1, 101)
tab5 = run_search(4, ARGS.seed)
r5 = tab5["trie"].root
prior_root = None
llm_raw = None
if LLM_CACHE is not None and LLM_CACHE.exists():
    try:
        llm_raw = json.load(open(LLM_CACHE, encoding="utf-8"))
        prior_root = llm_raw.get("", None)
    except Exception as e:
        print(f"  [llm] could not read {LLM_CACHE}: {e!r}")
unv = [op for op in OPS if op in PREFIX_OPS[()] and (op not in r5.ch or r5.ch[op].n == 0)]
rows5 = []
for op in OPS:
    if op not in PREFIX_OPS[()]:
        continue
    c = r5.ch.get(op)
    if c is not None and c.n > 0 and c.val is not None:
        sc_ = c.val + 1.414 * math.sqrt(math.log(max(r5.n, 1)) / c.n)
        rows5.append([op, c.n, f"{c.val:.1f}", "-", f"{sc_:.2f}"])
    else:
        pr = (prior_root or {}).get(op, 1 / max(len(unv), 1))
        rows5.append([op, 0, "-", f"{pr:.2f}", f"1e9 + 0.3*{pr:.2f}"])

pref_spread = []
for d_ in range(L):
    pid = IDX // K ** (L - d_)
    cid = IDX // K ** (L - d_ - 1)
    n_c = K ** (d_ + 1)
    cnt = np.bincount(cid, minlength=n_c).astype(float)
    sm = np.bincount(cid, weights=Y, minlength=n_c)
    M = np.where(cnt > 0, sm / np.maximum(cnt, 1), np.nan).reshape(-1, K)
    pref_spread.append(M)
spread = []
for M in pref_spread:
    ok = (~np.isnan(M)).sum(1) >= 2
    spread.append(float(np.nanmax(M[ok], 1).mean() - np.nanmin(M[ok], 1).mean()) if ok.any() else 0.0)

rho_by_depth = None
if llm_raw:
    acc_rho = defaultdict(list)
    for key, val in llm_raw.items():
        toks = key.split("|") if key else []
        if len(toks) >= L or any(t not in OPS for t in toks):
            continue
        pid_ = 0
        for t in toks:
            pid_ = pid_ * K + OPS.index(t)
        M = pref_spread[len(toks)]
        if pid_ >= M.shape[0]:
            continue
        truth = M[pid_]
        ops_ok = [(OPS.index(o), p) for o, p in val.items() if o in OPS and not np.isnan(truth[OPS.index(o)])]
        if len(ops_ok) >= 3 and np.ptp([p for _, p in ops_ok]) > 1e-9:
            acc_rho[len(toks)].append(spearmanr([p for _, p in ops_ok], [truth[i] for i, _ in ops_ok])[0])
    rho_by_depth = {d_: float(np.nanmean(v)) for d_, v in acc_rho.items() if len(v)}

fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
for a in alphas:
    ax[0].plot(tt, [alpha_at(a, t) for t in tt], label=f"alpha={a}")
ax[0].axhline(0.02, color="k", ls="--", lw=1)
ax[0].text(60, 0.023, "alpha_floor = 0.02 -> prior OFF below", fontsize=7.5)
ax[0].set(title="alpha_t schedule (budget 100)", xlabel="iteration t", ylabel="alpha_t", yscale="log")
ax[0].legend(fontsize=7)
ax[1].axis("off")
tb = ax[1].table(cellText=rows5, colLabels=["first token", "visits", "mean acc", "prior", "score"], loc="center", cellLoc="center")
tb.auto_set_font_size(False)
tb.set_fontsize(8.5)
tb.scale(1, 1.5)
ax[1].set_title("Root scores after 4 evaluations (two regimes)")
ax[2].bar(range(L), spread, color=C_MAIN, alpha=0.85, label="headroom: best-vs-worst sibling subtree mean (pp)")
ax[2].set(title="How much can any prior exploit?", xlabel="depth of the decision (0 = first token)", ylabel="accuracy points")
if rho_by_depth:
    a2 = ax[2].twinx()
    a2.plot(list(rho_by_depth), list(rho_by_depth.values()), "o-", color=C_ALT, label="your LLM cache: Spearman vs truth")
    a2.axhline(0, color=C_ALT, ls=":", lw=0.8)
    a2.set_ylabel("Spearman rho")
    a2.grid(False)
ax[2].legend(loc="upper right", fontsize=7.5)
fig.tight_layout()
save(fig, "step5_llm_prior")

first_off = {a: next((t for t in tt if alpha_at(a, t) < 0.02), None) for a in alphas}
note("LLM-active window by alpha (budget 100): " + ", ".join(
    f"alpha={a}: {'whole run' if first_off[a] is None else f'first {first_off[a] - 1} iterations'}" for a in alphas) +
    ". Every alpha >= 0.2 is identical - alpha is an 'active fraction' knob, not a mixing weight.")
note("Two regimes in the table: visited branches are ranked by observed accuracy + exploration bonus; unvisited ones sit "
     "at ~1e9 and only their tiny prior term differs. Observed data always wins once a branch has been tried.")
note("Headroom by depth (accuracy points between best and worst sibling subtree, averaged over all prefixes): " +
     ", ".join(f"depth {i}: {v:.1f}" for i, v in enumerate(spread)) +
     ". A prior cannot gain more than this, and only for the first visit to each branch.")
if rho_by_depth:
    note("Real prior quality from your cache (Spearman with true subtree means): " +
         ", ".join(f"depth {d_}: {v:+.2f}" for d_, v in sorted(rho_by_depth.items())) + " (0 = no better than chance).")
else:
    note("No --llm-cache supplied, so the table uses a uniform prior. Pass one of your llm_cache__*.json files to measure "
         "the real prior's agreement with the data.")
note("Project finding for context: across the accuracy, alpha-sweep and convergence experiments the prior did not improve "
     "final accuracy and slowed early convergence - consistent with a component that can only reorder first visits.")
close_step()

# %% ================================================================ STEP 6 - hardware-aware search (Phase 4)
s = step(
    6, "Hardware-aware search, Pareto front and TCR-H",
    "Search for architectures that are simultaneously accurate and fast on a chosen edge device, and measure how "
    "much of the accuracy/latency trade-off frontier has been recovered.",
    "Each evaluation now returns (accuracy, latency_ms). The reward is accuracy minus a latency penalty "
    "(0.05 * min(latency/50ms, 1) * 100). A running Pareto archive keeps only non-dominated points. Quality is "
    "measured by hypervolume (area dominated vs a reference corner), IGD to the exact front, and TCR-H = sum of "
    "1/pareto_rank over visited leaves divided by the space size, which only rewards visiting good trade-offs. "
    "Architectures the benchmark marks 'unsupported' (negative latency) count as evaluations but never enter the archive.",
    "The exact per-device Pareto fronts, cross-device overlap, and - if the project's hardware benchmark is reachable - "
    "a live device-specific search with its archive, HV/IGD trajectories and TCR-H curve.",
    "This is the final stage of the pipeline. Its outputs (fronts, HV, IGD, TCR-H) are what the report uses to show that "
    "the same trie machinery extends from single-objective accuracy to multi-objective, device-specific search.")

fronts = {}
for hw in HW_ALL:
    p = DATA_DIR / f"true_pareto_front__{hw}.json"
    if p.exists():
        try:
            pts = sorted((float(a), float(b)) for a, b in json.load(open(p, encoding="utf-8")))
            fronts[hw] = pts
        except Exception as e:
            print(f"  [hw] could not read {p.name}: {e!r}")
print(f"  Pareto front files found in {DATA_DIR}: {list(fronts)}")


def hv(points, ref=(0.0, 1000.0)):
    """2-D hypervolume (max acc, min latency) vs ref = (0%, 1000 ms) - same sweep as archtree/tcr.py."""
    pts = [p for p in points if p[0] is not None]
    if not pts:
        return 0.0
    tot, prev = 0.0, ref[1]
    for a, l in sorted(pts, key=lambda p: -p[1]):
        tot += max(prev - l, 0.0) * max(a - ref[0], 0.0)
        prev = min(prev, l)
    return tot


def nd(points):
    keep = []
    for p in sorted(points, key=lambda p: (p[1], -p[0])):
        if not keep or p[0] > keep[-1][0]:
            keep.append(p)
    return keep


def igd_norm(found, truth):
    if not found:
        return float("inf")
    t, f = np.array(truth, float), np.array(found, float)
    lo, hi = t.min(0), t.max(0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    t, f = (t - lo) / span, (f - lo) / span
    return float(np.sqrt(((t[:, None, :] - f[None, :, :]) ** 2).sum(-1)).min(1).mean())


if fronts:
    ncol = min(3, len(fronts))
    nrow = math.ceil(len(fronts) / ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.5 * nrow), squeeze=False)
    for ax_, (hw, pts) in zip(axs.ravel(), fronts.items()):
        a_ = [p[0] for p in pts]
        l_ = [p[1] for p in pts]
        ax_.step(l_, a_, where="post", color=C_MAIN, lw=1.6)
        ax_.scatter(l_, a_, s=10, color=C_ALT)
        ax_.set(title=f"{hw}: exact front ({len(pts)} pts)", xlabel="latency (ms)", ylabel="cifar10 acc (%)", xscale="log")
    for ax_ in axs.ravel()[len(fronts):]:
        ax_.axis("off")
    fig.tight_layout()
    save(fig, "step6a_true_fronts")

    # cross-device overlap: front points are matched back to architectures through their (unique) cifar10 accuracy
    acc_lookup = defaultdict(list)
    for i, a in ACC10.items():
        acc_lookup[a].append(i)
    members = {}
    for hw, pts in fronts.items():
        m = set()
        for a, _ in pts:
            c = acc_lookup.get(a, [])
            if len(c) == 1:
                m.add(c[0])
        members[hw] = m
    if len(fronts) >= 2 and any(members.values()):
        hws = list(fronts)
        J = np.eye(len(hws))
        for i_, a in enumerate(hws):
            for j_, b in enumerate(hws):
                if i_ != j_ and (members[a] | members[b]):
                    J[i_, j_] = len(members[a] & members[b]) / len(members[a] | members[b])
        allsets = [members[h] for h in hws]
        universal = set.intersection(*allsets) if allsets else set()
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        im = ax[0].imshow(J, vmin=0, vmax=1, cmap="Blues")
        ax[0].set_xticks(range(len(hws)))
        ax[0].set_xticklabels(hws, rotation=40, ha="right")
        ax[0].set_yticks(range(len(hws)))
        ax[0].set_yticklabels(hws)
        ax[0].grid(False)
        for i_ in range(len(hws)):
            for j_ in range(len(hws)):
                ax[0].text(j_, i_, f"{J[i_, j_]:.2f}", ha="center", va="center", fontsize=8,
                           color="white" if J[i_, j_] > 0.6 else "black")
        fig.colorbar(im, ax=ax[0], fraction=0.046)
        ax[0].set_title("Same architectures Pareto-optimal on two devices? (Jaccard)")
        cover = np.zeros(len(hws) + 1)
        cnt_ = defaultdict(int)
        for m in allsets:
            for i in m:
                cnt_[i] += 1
        for v in cnt_.values():
            cover[v] += 1
        ax[1].bar(range(1, len(hws) + 1), cover[1:], color=C_MAIN)
        ax[1].set(title="Architectures on the front of exactly k devices", xlabel="k devices", ylabel="# architectures")
        fig.tight_layout()
        save(fig, "step6b_cross_device_overlap")
        offd = J[~np.eye(len(hws), dtype=bool)]
        note(f"Exact fronts loaded for {len(fronts)} devices ({', '.join(f'{h}: {len(p)} pts' for h, p in fronts.items())}).")
        note(f"Mean pairwise Jaccard of front membership = {offd.mean():.2f} (min {offd.min():.2f}); only {len(universal)} "
             f"architecture(s) are Pareto-optimal on every device - a front found for one device does not transfer, "
             f"which is the case for searching per device.")
    else:
        note(f"Exact fronts loaded for {len(fronts)} device(s): " + ", ".join(f"{h}: {len(p)} points" for h, p in fronts.items()) + ".")
else:
    note(f"No true_pareto_front__<device>.json files found in {DATA_DIR}; pass --pareto-dir to enable the front analysis.")

# ---- live device-specific search (needs the project's hardware benchmark for per-architecture latency)
BACKEND, ERR = None, None
if ROOT is None:
    ERR = "no project root (config.yaml) found - pass --project-root"
else:
    try:
        os.chdir(ROOT)
        sys.path.insert(0, str(ROOT))
        from benchmarks.hwnb import HWNASBench, HWMetricUnavailable
        _b = HWNASBench(CFG)
        _b._load()
        BACKEND = (_b, HWMetricUnavailable)
    except Exception as e:
        ERR = repr(e)


def make_lat(hw):
    memo = {}

    def f(idx):
        if idx not in memo:
            try:
                memo[idx] = float(BACKEND[0].hw_query(idx, hardware=hw, dataset="cifar10")[1])
            except BACKEND[1]:
                memo[idx] = None
        return memo[idx]
    return f


def run_hw(budget, seed, lat_fn, truth):
    rng, trie = random.Random(seed), Trie()
    entries, archive, trace, unsup = [], [], [], 0
    for i in range(budget):
        path = traverse(trie, rng, 1.414, "raw")
        if trie.visited(path):
            path = random_unvisited(trie, rng)
        if trie.visited(path):
            bump(trie, path)
            continue
        idx = tokens_to_idx(path)
        acc = ACC10.get(idx)
        if acc is None:
            bump(trie, path)
            continue
        lat = lat_fn(idx)
        trie.insert(path).val = acc
        if lat is None:
            unsup += 1
            reward = acc
        else:
            reward = acc - 0.05 * min(lat / 50.0, 1.0) * 100.0
            new = (acc, lat, idx)
            archive = [a for a in archive if not (new[0] >= a[0] and new[1] <= a[1] and (new[0] > a[0] or new[1] < a[1]))]
            if not any(a[0] >= new[0] and a[1] <= new[1] and (a[0] > new[0] or a[1] < new[1]) for a in archive):
                archive.append(new)
        backprop(trie, path, reward, "max")
        entries.append((idx, acc, lat))
        if len(entries) % 5 == 0 or i == budget - 1:
            in_arch = {a[2] for a in archive}
            w = 0.0
            for idx_e, a_e, l_e in entries:
                if idx_e in in_arch:
                    w += 1.0
                elif l_e is None:
                    w += 0.1
                else:
                    r = min(1 + sum(1 for a in archive if a[0] >= a_e and a[1] <= l_e), 10)
                    w += 1.0 / r
            pf = [(a[0], a[1]) for a in archive]
            trace.append(dict(n=len(entries), hv_ratio=hv(pf) / hv(truth) if truth else np.nan,
                              igd=igd_norm(pf, truth) if truth else np.nan, front=len(archive),
                              tcr_h=w / TOTAL_SPACE, tcr=len(entries) / TOTAL_SPACE))
    return dict(entries=entries, archive=archive, trace=trace, unsup=unsup)


HWRES = {}
if BACKEND is None:
    note(f"Live hardware search skipped: per-architecture latency needs the project's hwnb backend ({ERR}). "
         f"Run from the project root (or --project-root) with HW-NAS-Bench installed to enable it.")
else:
    want = ARGS.hw or [h for h in fronts][:2] or ["edgegpu", "raspi4"]
    for hw in want:
        lat_fn = make_lat(hw)
        truth = fronts.get(hw)
        print(f"\n  live search on {hw} (budget {ARGS.hw_budget}, seed {ARGS.seed}) ...")
        res = run_hw(ARGS.hw_budget, ARGS.seed, lat_fn, truth)
        HWRES[hw] = (res, lat_fn, truth)
        tr = res["trace"]
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
        e_ok = [(e[1], e[2]) for e in res["entries"] if e[2] is not None]
        ax[0].scatter([l for _, l in e_ok], [a for a, _ in e_ok], s=8, color=C_GREY, alpha=0.5, label="evaluated")
        if truth:
            ax[0].step([p[1] for p in truth], [p[0] for p in truth], where="post", color="k", lw=1.2, label="exact front")
        fa = sorted(nd([(a[0], a[1]) for a in res["archive"]]), key=lambda p: p[1])
        ax[0].step([p[1] for p in fa], [p[0] for p in fa], where="post", color=C_ALT, lw=1.8, label="found front")
        ax[0].scatter([p[1] for p in fa], [p[0] for p in fa], s=22, color=C_ALT, zorder=3)
        ax[0].set(title=f"{hw}: objective space", xlabel="latency (ms)", ylabel="cifar10 acc (%)", xscale="log")
        ax[0].legend(fontsize=7.5)
        xs = [t["n"] for t in tr]
        ax[1].plot(xs, [t["hv_ratio"] for t in tr], "o-", color=C_MAIN, label="HV / HV(exact front)")
        ax[1].set(title="Convergence toward the exact front", xlabel="real evaluations", ylabel="hypervolume ratio", ylim=(0, 1.02))
        a2 = ax[1].twinx()
        a2.plot(xs, [t["igd"] for t in tr], "s--", color=C_ALT, label="normalised IGD (lower better)")
        a2.set_ylabel("IGD")
        a2.grid(False)
        ax[1].legend(loc="lower left", fontsize=7.5)
        a2.legend(loc="lower right", fontsize=7.5)
        ax[2].plot(xs, [t["tcr_h"] for t in tr], "o-", color=C_GOOD, label="TCR-H (Pareto-weighted)")
        ax[2].plot(xs, [t["tcr"] for t in tr], "s--", color=C_GREY, label="TCR (plain coverage)")
        ax[2].set(title="Coverage: plain vs hardware-aware", xlabel="real evaluations", ylabel="ratio of 15,625")
        ax[2].legend()
        fig.tight_layout()
        save(fig, f"step6c_live_search_{hw}")
        last = tr[-1]
        note(f"[{hw}] {len(res['entries'])} real evaluations -> found front of {last['front']} points, "
             f"HV recovery {100 * last['hv_ratio']:.0f}% of the exact front, normalised IGD {last['igd']:.3f}, "
             f"TCR-H {last['tcr_h']:.5f} vs plain TCR {last['tcr']:.5f}; {res['unsup']} 'unsupported' architecture(s) kept out of the archive.")
    if len(HWRES) >= 2:
        hws = list(HWRES)
        native, transfer, labels = [], [], []
        for a in hws:
            for b in hws:
                if a == b:
                    continue
                res_a = HWRES[a][0]
                lb = HWRES[b][1]
                pts = [(x[0], lb(x[2])) for x in res_a["archive"]]
                pts = nd([p for p in pts if p[1] is not None])
                truth_b = HWRES[b][2]
                if truth_b:
                    native.append(HWRES[b][0]["trace"][-1]["hv_ratio"])
                    transfer.append(hv(pts) / hv(truth_b))
                    labels.append(f"{a}->{b}")
        if labels:
            fig, ax = plt.subplots(figsize=(7.5, 4))
            xx = np.arange(len(labels))
            ax.bar(xx - 0.2, native, 0.4, color=C_MAIN, label="front searched ON target device")
            ax.bar(xx + 0.2, transfer, 0.4, color=C_ALT, label="front from another device, re-measured")
            ax.set_xticks(xx)
            ax.set_xticklabels(labels)
            ax.set(title="Why device-specific search matters", ylabel="HV / HV(exact front of target)", ylim=(0, 1.05))
            ax.legend(loc="lower right")
            fig.tight_layout()
            save(fig, "step6d_transfer")
            note(f"Re-measuring a front found on one device with another device's latency recovers on average "
                 f"{100 * np.mean(transfer):.0f}% of the target's exact hypervolume, versus {100 * np.mean(native):.0f}% "
                 f"when the search is run on the target itself.")
    if HWRES:
        first = next(iter(HWRES.values()))[0]["trace"][-1]
        METRICS.update(hv_ratio=float(first["hv_ratio"]), tcr_h=float(first["tcr_h"]))
close_step()

# %% ================================================================ STEP 7 - pipeline map
s = step(
    7, "End-to-end synthesis: how the stages connect",
    "Put the six stages together and show what each one contributed, using the numbers this run actually produced.",
    "Each stage consumes the previous stage's output: the accuracy table defines the reward, the encoder turns "
    "architectures into paths, the trie provides memory and deduplication, UCB chooses proposals, the proxy gate decides "
    "which proposals deserve a real evaluation, the (optional) LLM prior orders first visits, and the hardware stage "
    "extends the same loop to two objectives.",
    "A single map of the pipeline with this run's headline measurements, saved as the closing page of the PDF report.",
    "This is the end of the walkthrough; the PDF collects every step card and figure for presentation.")

boxes = [
    ("1 Dataset\n+ encoder", f"{N} archs\nf* = {METRICS['fstar']:.1f}%"),
    ("2 Trie", f"eta2(k=6)\n= {METRICS['eta2'][-1]:.2f}"),
    ("3 UCB1", f"{METRICS['ucb_best']:.1f}% vs\n{METRICS['rnd_best']:.1f}% random"),
    ("4 ZC gate", f"tau {METRICS['proxy_tau']:+.2f}\n{100 * (1 - METRICS['real_evals'] / B):.0f}% evals saved"),
    ("5 LLM prior", "reorders first\nvisits only"),
    ("6 HW Pareto\n+ TCR-H", (f"HV {100 * METRICS['hv_ratio']:.0f}% of exact" if "hv_ratio" in METRICS else "needs hwnb\nbackend")),
]
fig, ax = plt.subplots(figsize=(15, 3.6))
ax.axis("off")
ax.set_xlim(0, 6.2)
ax.set_ylim(0, 1)
for i, (t, m) in enumerate(boxes):
    x0 = i * 1.03 + 0.05
    ax.add_patch(plt.Rectangle((x0, 0.25), 0.88, 0.6, fc="#eaf0f7", ec=C_MAIN, lw=1.4))
    ax.text(x0 + 0.44, 0.7, t, ha="center", va="center", fontweight="bold", fontsize=9.5, color=C_MAIN)
    ax.text(x0 + 0.44, 0.42, m, ha="center", va="center", fontsize=8.5)
    if i < len(boxes) - 1:
        ax.annotate("", xy=(x0 + 1.0, 0.55), xytext=(x0 + 0.9, 0.55), arrowprops=dict(arrowstyle="->", color=C_MAIN))
ax.text(3.1, 0.08, f"Live results on {DS} ({N} architectures, budget {B}, {len(SEEDS)} seeds)", ha="center", fontsize=9, style="italic")
fig.tight_layout()
save(fig, "step7_pipeline_map")
note("Data flow: accuracy table -> token paths -> trie memory -> UCB proposals -> proxy-gated real evaluations -> "
     "(optional) LLM ordering of first visits -> multi-objective archive and TCR-H.")
close_step()

# %% ================================================================ PDF report
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402


def text_page(pdf, title, sections):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.05, 0.95, title, fontsize=17, fontweight="bold", color=C_MAIN, va="top")
    y = 0.885
    for head, body, bullets in sections:
        fig.text(0.05, y, head.upper(), fontsize=9.5, fontweight="bold", color=C_ALT, va="top")
        y -= 0.03
        items = [body] if body else []
        items += [f"- {b}" for b in bullets]
        for it in items:
            for ln in textwrap.wrap(it, 150, subsequent_indent="  "):
                fig.text(0.05, y, ln, fontsize=9.5, va="top")
                y -= 0.024
            y -= 0.006
        y -= 0.016
    pdf.savefig(fig)
    plt.close(fig)


pdf_path = OUT / "ArchTree_walkthrough.pdf"
with PdfPages(pdf_path) as pdf:
    text_page(pdf, "ArchTree-LLM - live pipeline walkthrough", [
        ("Dataset used", f"{CACHE_PATH.name}: {N} usable architectures for {DS} "
                         f"({100 * N / TOTAL_SPACE:.1f}% of NAS-Bench-201's 15,625); datasets in file: {', '.join(AVAIL)}.", []),
        ("Hardware fronts", f"Exact Pareto fronts found for: {', '.join(fronts) if fronts else 'none'}.", []),
        ("Reproducibility", f"Single narrated run seed = {ARGS.seed}; curves use {len(SEEDS)} seeds; demo budget = {B} proposals; "
                            f"proxy gate keeps the top {PROXY_PCT}%. Every number and figure is computed from the input files at run time.", []),
        ("Contents", "", [f"Step {s_['no']}: {s_['title']}" for s_ in STEPS]),
    ])
    for s_ in STEPS:
        text_page(pdf, f"Step {s_['no']} - {s_['title']}", [
            ("Purpose", s_["purpose"], []),
            ("How it works", s_["how"], []),
            ("What it produces", s_["produced"], []),
            ("Live observations from your data", "", s_["live"]),
            ("Feeds into the next step", s_["feeds"], []),
        ])
        for p in s_["figs"]:
            fig = plt.figure(figsize=(11.69, 8.27))
            axi = fig.add_axes([0.03, 0.05, 0.94, 0.88])
            axi.imshow(plt.imread(p))
            axi.axis("off")
            fig.text(0.5, 0.965, f"Step {s_['no']} - {p.stem}", ha="center", fontsize=10, color=C_GREY)
            pdf.savefig(fig)
            plt.close(fig)
json.dump(METRICS, open(OUT / "metrics.json", "w"), indent=2)
print(f"\nDONE. Report: {pdf_path}\n      Figures: {OUT}\n      Headline metrics: {OUT / 'metrics.json'}")