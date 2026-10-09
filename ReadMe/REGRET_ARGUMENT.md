# Why UCB1 over a trie (informal regret argument) — roadmap §3.1

**Status: informal argument with explicit assumptions. Not a proof, not a theorem.**
Results quoted from the literature below are from memory — check the exact
statements and preconditions before they go in the paper.

Companion code: `tools/analyze_regret.py` (measures the two structural
assumptions on NB201 and fits empirical regret curves). Companion ablation:
`experiments/run_trie_ablation.py` (A7, trie vs flat UCB1).

---

## 1. Setting

* Search space: leaves of a depth-`L=6` trie, branching `K=5`, `N = K^L = 15,625` leaves.
* Reward of a leaf `x`: `f(x)` = benchmark accuracy (a table lookup — deterministic
  for a given dataset). Optimum `f* = max f`.
* Budget: `T` real evaluations, `T ∈ {25,…,200} ≪ N`.
* The right currency is **simple regret** `r_T = f* − max_{t≤T} f(x_t)` (we report the
  best architecture found), not cumulative regret.
* The trie stores each evaluated leaf once (`is_visited`, `random_unvisited`). So every
  leaf is pulled **at most once**: there is no leaf-level noise to average out, and no
  budget is spent re-pulling.

## 2. Baseline: UCB1 over the leaves does nothing

UCB1 gives an unpulled arm index `+∞`. With `T < N` every proposal is an unpulled leaf,
so the algorithm is uniform sampling without replacement. Its expected simple regret is
the order-statistic curve `E[r_T] = f* − E[max of T uniform draws]`, fixed by the upper
tail of the accuracy distribution. `analyze_regret.py` computes it exactly
(`random_exact`). No learning happens, which is why this variant is not implemented
(it is the phase-0 `random_search` baseline).

## 3. What the trie changes

At every internal node the decision is a `K`-armed bandit over child subtrees, so a
descent is `L` decisions among `K` arms (30 arm-indices to fill along one root-to-leaf
path) instead of one decision among `N` arms. A subtree's value is estimated from the
leaves already evaluated under it (running mean in phase 1/2, best-seen in phase 3+).

This is only useful if a subtree's value says something about its unevaluated leaves.

**Assumption A1 — prefix locality.** Leaves sharing a depth-`k` prefix have similar
accuracy, with spread `δ_k` non-increasing in `k`.
*Measured by:* `eta2` (variance explained by the first `k` tokens) and `within_std`
in `landscape_smoothness.csv`. If `eta2` is small for `k ≤ 3`, top-level statistics
carry little information and the argument does not go through.

**Assumption A2 — few near-optimal cells.** The number of depth-`k` cells containing a
leaf within `ε` of `f*` grows slowly in `k`: `N_k(ε) ≲ C · 5^{γk}` with `γ` small
(near-optimality-dimension-like).
*Measured by:* `N_k_eps*` and `gamma_eps*` in `landscape_smoothness.csv`.
Caveat: NB201 saturates — many leaves are within 1 pp of `f*` — so `γ` may be large
for small `ε`-scales relative to the table's noise floor. Report what the script
prints, not what the argument wants.

## 4. Known results this maps onto (verify before citing)

* **UCT / tree UCB with mean backup** (phase 1/2): Kocsis & Szepesvári (2006).
  Coquelin & Munos (2007) show the worst case can be hyper-exponentially bad in depth,
  so there is no clean worst-case guarantee. Under smoothness (A1) the
  hierarchical-optimistic-optimisation analysis (HOO, Bubeck–Munos–Stoltz–Szepesvári,
  2011) gives cumulative regret `Õ(T^{(d+1)/(d+2)})` for near-optimality dimension `d`;
  `d = 0` gives `Õ(√T)`.
* **Deterministic rewards, optimistic tree search** (closer to our table lookup):
  Munos (2011), DOO/SOO. With known smoothness the simple regret decays like
  `T^{-1/d}` for `d > 0` and exponentially for `d = 0`.

## 5. Where the implementation departs from those theorems

1. **Backup mode.** Phase 3+ uses `backup_mode="max"`: the index is the best accuracy
   seen in the subtree plus a bonus. That is an optimistic estimate in the DOO spirit,
   not the mean estimate UCB1/HOO analyse.
2. **Scale of the bonus.** Since Oct 2026 every traversal (`traverse`, `traverse_with_llm`,
   `search_flat_ucb`) uses accuracy/100 + `C·√(ln N / n)`, so phases 1-4 share one
   exploration balance. (The Sep-2026 phase 3/4 runs used the raw 0-100 scale; they are
   superseded.) `experiments/run_ucb_sensitivity.py` reports the sensitivity to `C` and the
   backup rule.
3. **Smoothness is not known to the algorithm**, and DOO needs it. There is no
   adaptive-smoothness component here.
4. **Finite budget.** `T ≤ 200 ≪ N`. Asymptotic rates say little about 25–200 evals;
   the fitted exponent `b` in `regret_fit.csv` is a descriptive summary.
5. **ZC proxy gate and LLM prior are outside the analysis.**
   * The LLM prior enters only for never-visited children (index `1e9 + α_t·p`). It
     changes the *order* in which unvisited siblings are first expanded. Since the Oct-2026
     fix, children visited only through proxy-skipped proposals are scored like any visited
     child (value 0 + bonus) and never receive a prior; before the fix they received a default
     prior of 0.5, which biased every α > 0 run towards proxy-rejected branches (this, not
     the LLM, produced the Sep-2026 "slower convergence" result). With the fix, a
     zero-information prior reproduces α = 0 exactly, so the prior can only permute the
     expansion order of unvisited siblings and **the argument is unchanged**.
   * The proxy gate is a filter with unknown rank correlation to accuracy (a
     multi-fidelity setting). Not covered.

## 5b. Status of the formal statements (Oct 2026)

| Id | Statement | Status |
|---|---|---|
| T1 | trie insert / lookup is O(L) | holds |
| T2 | no architecture is really evaluated twice | holds **by construction** (`is_visited` + `random_unvisited` redirect); report it as a property of the design, not as an empirical finding |
| T3 | sub-linear regret of UCB1 over the trie | informal argument only (this document) |
| T4 | pruning never removes the optimum | **false as stated.** The Sep-2026 rule pruned the least-visited nodes (exactly the unexplored ones). The Oct-2026 value rule prunes the lowest backed-up accuracies, which is a sensible heuristic but still gives no guarantee without confidence bounds. Restate as: "pruning is a heuristic; A5 measures its effect" |
| T5 | weight inheritance is consistent | holds for the simulated weight bank; it has no measurable effect on a tabular benchmark (nothing is trained), so make no empirical claim |
| T6 | LLM prior keeps the regret bound | near-trivial once T3 is informal: the prior only reorders never-visited siblings (see 5.5) |
| T7 | TCR-H = 1 iff all Pareto-optimal archs found | definitional |

## 6. What the evidence can and cannot support

| Claim | Supported by | Status |
|---|---|---|
| The trie avoids duplicate evals that a flat, no-dedup bandit pays for | A7 `flat_none` vs `trie`, `analyze_duplicate_evals.py` | empirical |
| The trie's *structure* helps beyond plain dedup | A7 `flat_set` vs `trie` (paired Wilcoxon) | empirical, may come out null |
| NB201 is prefix-smooth enough for subtree statistics to be informative | `eta2`, `within_std` | empirical, measured exactly |
| Regret decays faster than uniform random search | `regret_fit.csv` vs `random_exact` | descriptive |
| A regret *bound* for ArchTree | — | **not established** |

## 7. Suggested wording for the paper

> We give an informal argument, not a regret bound. On a deterministic table with
> `T ≪ K^L`, flat UCB1 reduces to uniform sampling without replacement, whereas a
> trie reduces each decision to a `K`-armed choice and shares statistics across
> architectures with a common prefix. Whether that sharing helps depends on
> prefix locality of the accuracy landscape, which we measure directly on
> NAS-Bench-201 (Fig. X: variance explained by the first `k` tokens; Table Y:
> near-optimal cell counts). Under such locality, hierarchical optimistic search
> is known to achieve sub-linear regret (Bubeck et al., 2011; Munos, 2011); our
> implementation departs from these analyses in its backup rule and exploration
> scaling (Sec. Z), so we treat them as motivation rather than guarantee. The LLM
> prior acts only on the expansion order of unvisited siblings and does not
> alter this argument.

## 8. If a reviewer pushes for more

* Implement a DOO/SOO baseline on the same trie and show ArchTree's curve against it.
* Sweep `C` and `backup_mode` on both traversal functions and report the sensitivity
  (§5.2 suggests the phase 1/2 vs 3+ gap could be a real confound in the main
  comparisons).
* Replace the single deterministic accuracy with per-seed NB201 accuracies to test
  the stochastic (HOO) setting honestly.
