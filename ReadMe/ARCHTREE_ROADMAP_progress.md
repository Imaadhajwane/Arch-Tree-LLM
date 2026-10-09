# ArchTree-LLM — Strengthening Roadmap (progress-annotated)

**Project:** Hierarchical Trie-Based NAS with LLM-Guided UCB Traversal, Zero-Cost Proxy Ensemble & Hardware-Aware Coverage
**Author:** Imaad Hajwane · M.Tech Research Project, A.Y. 2026–27
**Guide:** Dr. Jayshree Aher
**Institution:** MIT-WPU, School of Computer Science and Engineering

> Legend for this pass: **[DONE]** = verified against actual code/data · **[CODE-READY]** = implemented in code but not yet executed/verified against fresh results (blocked on Phase 1–4/ablations rerun) · **[NOT STARTED]** = no code for this yet.
> Only **Phase 0** has been (re)executed since the recent fixes (gpt-oss-20b swap, NB201 cache index-desync fix, dataset-scoped LLM cache, zc_proxy v4, phase6 paired-stats rewrite). Phases 1–4 and ablations need a fresh run before anything downstream of them can be marked truly done.
> **[Sep 2026 update]** The A4 alpha ablation (only) has since been rerun twice more, post two additional bug fixes to the LLM prior pipeline itself — see new §0.1 below. This is the first real signal on whether the LLM prior (C7) is worth its place in the title, and it's a negative/neutral one. Everything else in this doc is unchanged from the prior pass; Phase 1–4 main sweeps still need a full rerun as noted throughout.
>
> **[Sep 20, 2026 update]** Code for every previously-missing segment (§2.1 all three bullets, §2.3 convergence angle, §3.1, §3.2 cross-device, §3.3 both bullets, §3.4 both bullets) has been written and is marked **[CODE-READY]** below — **tested on a synthetic stub benchmark only, not yet run on real data.** Reading the code to build it also surfaced several real bugs/config traps, one of which (**ZC score-cache contamination, §0.4(b)**) makes the proxy-gated and zico/rz_nas numbers on disk provisional. New §0.4 records all of it; §4a has the exact rerun order and what to move first. No file in this update has been executed against real outputs.
>
> Checkbox legend added this pass: `- [x]` done · `- [~]` **code-ready, not yet run** · `- [ ]` not started.
>
> **[Sep 21, 2026 update]** The full rerun (§4a) is IN PROGRESS. **Phase 0-4 are done, verified, clean** — see new §0.5 for numbers, the two operational issues hit along the way (both resolved, no data lost), and the new finding on Phase 3/4 exploration collapse. Ablations (A1-A6) running next; the new experiments (A7, proxy ablation, convergence study, LLM-size study) and analysis scripts have not started yet.
>
> **[Sep 22, 2026 update]** Ablations A1-A6 **DONE** (144/144 each, 0 errors — one mid-run crash on an empty `groq_calls__*.json` counter file, root-caused and patched in `storage.py`: atomic writes + empty/corrupt-file tolerance for the groq counter and LLM cache, same pattern `zc_proxy.py` already used for its raw cache; no data lost). **A7 (trie vs flat-UCB1) DONE — real result, see new §0.6(a): the trie's benefit is dedup, not tree structure.** **Proxy composition ablation DONE — see §0.6(b): no-proxy still wins on accuracy (expected, that's the efficiency trade), and ranking quality (tau) predicts search quality on cifar10/cifar100 but not ImageNet16-120.** Convergence study running next; LLM-size study, Phase 6 aggregate, and analysis scripts not started.
>
> **[Sep 23, 2026 update]** **Convergence study DONE, all 3 datasets — C7's strongest and most consistent negative result yet: the LLM prior significantly SLOWS early convergence (5 of 6 dataset×budget cells, p<0.05, full 12/12 paired seeds), with no compensating benefit anywhere. See new §0.7(a).** **Target changed: this is now explicitly a MASTER'S SEMINAR PAPER, not an AAAI/IJCAI/ECAI submission — §0.7(b) records the full re-scoping discussion, the honest publishability assessment, and the recommended reframing (drop "LLM-Guided" emphasis, lead with trie-dedup+ZC-proxy efficiency and TCR-H, report C7 and the A7 trie-structure finding as rigorous, well-tested negative results).** LLM-size sensitivity study (probe stage) running next.
>
> **[Sep 24, 2026 update]** **`run_phase6_aggregate.py` DONE — extended with a strict, bias-checked equal-real-compute comparison (see new §0.8). This is the project's final headline number: at matched real-eval cost, with the matching rule verified never to favor archtree_llm, archtree_llm wins-or-ties 95 of 96 baseline comparisons (99%), with exactly one significant, named exception (`llm_nas`, cifar10, nominal budget=25). All other Phase 6 outputs (main_results_table, statistical_tests, extended_findings_summary rolling up A7/proxy-ablation/convergence) also regenerated clean on the full rerun data.** Only the LLM-size sensitivity study remains, blocked on the Groq daily token-budget reset (§0.7's run exhausted it across all 13 keys).
>
> **[Sep 24, 2026, later — analysis scripts + repro bundle]** All 4 analysis scripts (`analyze_duplicate_evals.py`, `analyze_tcr_h.py`, `analyze_regret.py`, `analyze_cross_device_pareto.py`) and `build_repro_appendix.py --bundle` are now DONE — see new §0.9. Headline additions: **zero duplicate real evals across every ArchTree run at every budget (864+ logs checked)**, vs 82% for rz_nas and 41% for llm_nas — a cleaner dedup story than the MCTS-NAS comparison originally planned. **Cross-device Pareto analysis is the strongest single result of this batch**: only 1 of 15,625 architectures is Pareto-optimal on all 6 devices, mean front Jaccard 0.22, transfer precision A→B just 0.03 — strong, clean evidence hardware-aware search is necessary. TCR-H-vs-accuracy correlation is weak/inconsistent (flagged, not to be oversold). Regret-decay analysis independently confirms the A7 no-dedup finding from a different angle (flat_ucb_none shows ~zero regret decay). **The full execution plan is now complete except the LLM-size sensitivity study, blocked on the Groq daily token reset.**
>
> **[Sep 29, 2026 update]** Model naming/ownership pass + frontend `index.html` / `backend.py` updated so **every ArchTree variant is tagged "(ours)"** and highlighted (headline model gets a stronger marker). New **§0.10 at the very end of this file** has the model table and a record of what was discussed. **One correction to the Sep 24 headline, see §0.10(d):** the "95 of 96 vs 9 baselines" figure is 12 cells × **8** comparators, and 2 of those 8 are our own variants (`archtree_v1`, `archtree_zc`) — against the **6 external** baselines it is **71 of 72 (98.6%)**, same single exception (`llm_nas`, cifar10, budget 25). No new experiments were run.
>
> **[Oct 2, 2026 update]** Technical-reviewer pass (self-review as a skeptical NAS reviewer). **No code or experiments were run - this pass only edited this document.** (1) Headline corrected everywhere to **71 of 72 (98.6%) vs the 6 external baselines**; the old "95 of 96 vs 9 baselines" wording is retired (see §0.10(d)). (2) New **§0.11 at the very end of this file**: reviewer score (about 7/10 as a seminar paper, about 3/10 as a top-venue submission), the title discussion, and a prioritized 20-item master checklist. The most important new item: **the efficiency gain is not yet attributed to ArchTree vs the proxy gate alone - run random / MCTS-NAS / ENAS with the same proxy gate (§0.11(c), item A1).**

---

## 0.1 [NEW] LLM-prior (C7) investigation — two pipeline bugs found & fixed, result is still negative/neutral

Triggered by Phase 3's exit-criteria check consistently showing `best_acc` *below* Phase 2 baseline (cifar10 -0.74%, cifar100 -1.31%, ImageNet16-120 -1.39%) and the A4 alpha ablation showing `alpha=0.0` (LLM off) beating every non-zero alpha. Rather than accept "the LLM prior doesn't help" at face value, dug into *why* — found two real implementation bugs in `archtree/llm_prior.py`, fixed both, and reran A4 after each fix.

**Bug 1 — depth-cache starvation (fixed).** `score_branches()` checked a coarse fallback cache — keyed only on `(depth, frozenset(unvisited_ops))`, with **no path context at all** — *before* attempting a real LLM call. Two completely unrelated partial architectures at the same depth with the same remaining-op set would silently share the same cached score vector. Evidence: mean real `groq_call_count` was only **~13.5/run** despite `max_calls=300`, while `depth_cache_hits` averaged **~90/run** — i.e. ~90% of "LLM guidance" was this architecture-blind reuse, not real model output.
- **Fix:** real call is now attempted first whenever budget remains; the depth-cache fallback only fires once `max_calls` is genuinely exhausted, and no longer poisons the exact-path cache.
- **Verified fixed:** post-fix rerun, mean `groq_call_count` jumped to **~50/run** (near the temporarily-lowered `max_calls=60` used for this validation pass), `depth_cache_hits` dropped to **~12/run**.
- **Result after Bug 1 fix alone:** alpha=0.0 still won (92.11% mean), gap to best non-zero alpha (1.0) was -0.23 to -0.89 across alphas, closest to significant at alpha=0.3 (Wilcoxon p=0.052, n=12) — i.e. LLM guidance was trending toward *significantly worse* than no LLM, not better.

**Bug 2 — prompt/schema key mismatch (fixed).** `_build_prompt()`'s JSON example used different key spellings (`"conv3x3"`, `"skip"`) than the real op vocabulary (`conv_3x3`, `skip_connect`, `conv_1x1`, `avg_pool_3x3`, `none`). The model frequently echoed the example's naming convention back instead of the real names given earlier in the same prompt. `_parse_scores()`'s `raw.get(op, uniform_val)` then silently substituted a placeholder score for any op whose returned key didn't exact-match — no error, no log. Confirmed directly in a `llm_cache__*.json` sample: 31 of 60 cached paths produced numerically-identical (12+ decimal places) score vectors to structurally unrelated paths — reverse-engineering the softmax math showed this matches exactly what you'd get when 1–2 of 5 ops silently fall back to the same uniform placeholder.
- **Fix:** the JSON example in the prompt is now built dynamically from the actual `unvisited_ops` for that call, so the model sees (and should echo) the exact real key names. Added `name_mismatch_calls`/`name_mismatch_ops` diagnostic counters, now logged into `phase3_llm_cache_stats.csv`.
- **Verified fixed:** post-fix rerun (72 runs, ~3,586 real Groq calls total) showed only **1 total name-mismatch** — down from a systemic, majority-of-calls problem.
- **Result after Bug 2 fix (both bugs now fixed):** alpha=0.0 *still* has the highest mean (92.11%), but gaps shrank and flattened — alpha=0.7 is now essentially tied (-0.06%, p=0.43), all others -0.28 to -0.36% with p=0.16–0.23 (no longer trending toward significant harm). n=12 seeds is underpowered to distinguish "truly zero effect" from "small effect masked by noise" at this gap size.

**Honest conclusion (recorded here, not smoothed over):** both fixes were real and worth making — they removed genuine, evidenced sources of *harm* to the LLM prior's guidance quality. But neither fix, nor both together, produced evidence that the LLM prior *helps*. The trajectory across three ablation runs (pre-fix → Bug 1 fixed → both fixed) is a believable, monotonic *improvement toward neutral*, not toward a hidden positive effect — which argues against continuing to chase further bug fixes hoping for a flip to positive. Current recommendation on record (§2.3 below, updated): report this as a legitimate negative/neutral finding for the LLM-prior component specifically, rather than keep debugging. If pursued further at all, the one specific next step (not an open-ended search) would be rerunning A4 once at budget=200 rather than more seeds at budget=100, since a deeper trie gives the LLM prior more opportunity to compound any real effect — this has not been done yet.

**Files touched by this investigation:** `archtree/llm_prior.py` (both fixes), `experiments/run_phase3.py` (unrelated collision-filename fix for the A4 loop's internal CSV writes, found in the same session; also added a Ctrl-C-safe resume guard for the A4 loop, keyed on `(alpha, seed)` against `ablation_A4_alpha_sensitivity.csv`, since that loop has no dedup guard of its own). `config.yaml`'s `alpha_init` comment updated in place to record the ablation numbers as they stood at each stage — **now stale as of this section**; should be updated once more to point here.

---

## 0.2 [NEW — Sep 2026] Phase 4 verified good; ablations/plot infra had real gaps, now fixed; Phase 6 rerun exposes + fixes a dataset-blend bug; equal-compute reanalysis flips the accuracy story

Four separate findings from actually running Phase 4 → ablations → Phase 6 end-to-end for the first time since the fixes. Documented in the order they were found.

**(a) Phase 4 exploration-collapse fix confirmed working — no rectification needed.** Checked the fresh `phase4_hv_igd_summary.csv`/`phase4_pareto_results.csv`/`phase4_tcr_h_per_iter.csv` directly against what `search.py`'s own patch notes claimed:
- `real_eval_count == budget` exactly, every hardware × every budget (25/50/100/200) — the old plateau-at-~25-35 bug is gone.
- `pareto_front_size` and `hypervolume` both increase monotonically with budget, every hardware target.
- `unsupported_hw_skipped` nonzero **only** for edgetpu (4,218 total) — matches the documented negative-latency-sentinel fix; other 5 hardware targets correctly show zero.
- `latency_ms` range 0.0036–87.2, **zero negative values** in the pareto results — sentinel never leaks through.
- **[DONE]** — Phase 4 output is trustworthy as-is.

**(b) `run_ablations.py` was silently cifar10-only — fixed.** Every one of A1/A2/A3/A5/A6 hardcoded `dataset="cifar10"`, unlike phase0–4 which all loop `cfg["benchmarks"]["datasets"]`. Also: the Groq backend path in this file never wired `use_key_pool`, so a real run was observed burning 68–126s/seed on pure RPM-throttle sleep on a single key despite `GROQ_API_KEYS` (4-account pool) being available elsewhere in the pipeline.
- **Fix:** dataset loop added (outermost, matching phase1–4's nesting: `dataset → param_grid → seed`); resume key widened from `(seed, param_value)` to `(dataset, seed, param_value)` so per-dataset rows don't alias each other; A5/A6 `ablation_id` namespaces now fold in dataset too (`A5_p{pv}_{dataset}`) to stop cache/weight-bank collisions across datasets; calibration-file existence check now verifies all 3 `zc_calibration_weights__nb201__{dataset}.pt` files, not just cifar10's; Groq key-pool detection (`GROQ_API_KEYS`/`GROQ_API_KEYS_FILE`) added, matching `run_phase0_llm_nas_only.py`/`run_phase3.py`'s existing pattern.
- **[DONE]** — ablations now cover all 3 datasets; verified via a completed run (48/48/48/48/48 = 240 rows across A1/A2/A3/A5/A6 for cifar10 alone before the dataset-widened version was deployed, confirmed no double-counting on resume).

**(c) `plots/plot_results.py` did not exist at all — built from scratch.** Every phase script (`run_phase3.py`, `run_phase4.py`, `run_ablations.py`, `run_phase6_aggregate.py`) imports `plot_phase3`/`plot_phase4`/`plot_ablations`/`plot_final_all` from this module, each already wrapped in `try/except ImportError` — so this was silently degrading every run to zero figures with only a one-line warning, never surfaced as a blocker.
- **Built:** full module, all 4 functions, ~15 individual figures total. Design rule followed throughout (matching the dataset-blend fix precedent already established elsewhere in this codebase, RESULTS_REVIEW.md #1): headline baseline comparisons (fig1) stay cifar10-only since baselines never ran the other 2 datasets; every other figure (phase3 acc-vs-budget, phase4 HV/IGD/Pareto-scatter, ablation deltas) shows all 3 datasets as separate lines/colors rather than blending or dropping any. Each figure independently try/except-wrapped so one missing upstream CSV never kills the rest of the figure set.
- **[DONE]** — this closes a real, previously-invisible gap: **no figures existed anywhere in `outputs/*/plots/figures/` before this**, despite every phase script "successfully" calling its plot function for however long the pipeline has been running.

**(d) Phase 6 rerun — found the same RESULTS_REVIEW.md #1 bug class reintroduced by (b), fixed.** First `run_phase6_aggregate.py` run against real ablation data produced `A2 no_zc (proxy OFF) mean_best_acc = 62.69%` — a ~28pp drop versus every other method (~90–93%), which does not represent "proxy off" performance; it's a blended mean across cifar10 (~90%+), cifar100 (~65–70%), and ImageNet16-120 (~30–40%) once ablations started covering all 3 datasets per fix (b). Root cause: `eval_efficiency_summary`'s A2 block filtered on `budget==100` only — no dataset filter — while the `archtree_llm (proxy ON)` row two lines above it *does* filter to cifar10, so the comparison silently went apples-to-oranges the moment (b) widened A2's dataset coverage.
- **Fix:** added `& (a2["budget"] == 100) & (a2["dataset"] == "cifar10")`, matching the sibling row's existing filter.
- **[DONE]** — flagged here explicitly as a recurrence risk: any future widening of a script's dataset coverage (per this doc's own repeated §1.1/§1.3 pattern) needs an audit pass over every *downstream* consumer of that script's CSV for missing dataset filters, not just the script itself. This is the second time this exact bug class has appeared (build script → three separate driver-script summaries already patched around it per the original doc; now a fourth site, in aggregation code one phase removed from the original fix).

**(e) [Most consequential finding] Equal-nominal-budget comparison was misleading; equal-real-compute comparison tells a different, better story.** Fresh `statistical_tests.csv` (96 tests, paired-per-seed-at-fixed-budget, post all fixes above) showed archtree_llm losing to most baselines at **nominal** budget=100: significant losses vs `llm_nas` (d=−1.11, 100% of tests significant) and `zico` (d=−0.96, 83% significant); only a significant *win* vs `rz_nas` (d=+0.89, 83% significant). Overall sig_rate 45%, mean |d|=0.66 — looked like a genuinely weak result.
- **But** `phase3_main_results.csv` shows archtree_llm's ZC-proxy gating means it only spends **~28 real evals at nominal budget=100** (72% proxy-skipped) — so this was comparing a 28-eval search against baselines' full 100-eval search the whole time, not a fair fight.
- **Reran paired per-seed** against each baseline's own **budget=25** run (closest real-eval-count match, from `baseline_summary.csv`, n=12 matched seeds each):

  | baseline (budget=25) | archtree_llm (28 evals) diff | p |
  |---|---|---|
  | rz_nas | **+2.32pp** | **0.000** |
  | mcts_nas | +0.57pp | 0.380 |
  | enas | +0.49pp | 0.791 |
  | random_search | −0.18pp | 0.970 |
  | zico | −0.71pp | 0.176 |
  | llm_nas | **−1.09pp** | **0.016** |

  At matched real-compute, archtree_llm **wins or ties 5 of 6 baselines** (only `rz_nas` and `llm_nas` reach significance, one in each direction) — a fundamentally different, much more favorable picture than the nominal-budget table alone suggests.
- **[DONE — analysis]**, **[NOT YET DONE — not written back into any output file/figure]**. This lives only in this doc + chat right now; recommended next step is a dedicated `outputs/final/results/equal_compute_comparison.csv` + figure (§2.2/§4 below updated accordingly) so it's part of the actual paper artifacts, not just a one-off chat analysis.
- **[Sep 20 caveat — provisional]** this table was computed from proxy-gated runs and from `zico`/`rz_nas` baselines that used the contaminated ZC score cache (§0.4(b)). Re-derive it after the clean rerun (§4a) before it goes in the paper; the qualitative efficiency story may hold, the specific diffs/p-values may move.

---

## 0.3 [NEW — this session] Phase 6 output dupe bug found+fixed; alpha-sweep infra built (Groq/key-pool aware) to replace A4's coarse 6-point grid

Triggered by inspecting a fresh `run_phase6_aggregate.py` output: `statistical_tests.csv` had 192 rows but only 96 unique `(method_a, method_b, benchmark, dataset, budget)` tests — the whole table duplicated exactly once. Root cause: `storage.append_result_row()` is append-only by design (correct for phase1–4/ablations, which are genuinely incremental/resumable), but `run_phase6_aggregate.py` is a **full-rebuild-every-time** script (its own docstring: "touches no upstream outputs, only reads them") — so running it twice without manually clearing `outputs/final/results/statistical_tests.csv` first just stacks a second copy of every row on top. Same disease already seen once before this session in `phase3_llm_cache_stats.csv` (720 rows where 144 were expected, 5x duplication from repeated phase3 restarts) — confirms this is a recurring class of bug wherever a would-be-idempotent aggregation script is built on top of an append-only helper with no caller-side guard.

- **Fix:** added `storage.clear_result_file(phase, filename)` — deletes a results CSV if present, meant to be called once at the top of any full-rebuild script before its first `append_result_row` call. `append_result_row` itself is unchanged (zero risk to phase1–4/ablations' existing incremental behavior). Wired into `run_phase6_aggregate.py` (one line, before the stats loop) to clear `statistical_tests.csv` on every run.
- **[DONE]** — patch written, not yet re-verified against a clean rerun (Phase 6 hasn't been rerun since this fix landed; do so before trusting any `statistical_tests.csv` currently on disk — the 96-vs-192 dupe check above is how to confirm it's clean: `df.groupby([...]).size()` should be all-1s).
- **Not yet checked:** whether `main_results_table.csv`'s underlying re-run (phase1–4/ablations, all called via `append_result_row` internally e.g. `phase3_llm_cache_stats.csv`, `phase3_tcr_per_iter.csv`) has any other silent duplication left over from earlier restart history. Only `phase3_llm_cache_stats.csv` and `statistical_tests.csv` have actually been checked row-count-wise so far — worth a blanket dedupe audit (`groupby(key_cols).size().value_counts()` per results CSV) before final numbers are reported, not just spot-checks.

**Separately — the accuracy investigation deepens:** cross-checking `main_results_table.csv`/`statistical_tests.csv` from a fresh full pipeline run this session confirmed the same shape of result as §0.1's A4 finding, but now visible in the *main* (non-ablation) comparison too: archtree_llm loses significantly to `zico` (3 of 4 budgets), `enas`/`mcts_nas`/`random_search`/`archtree_v1`/`archtree_zc` (each at budget=100 specifically), and only clearly/consistently beats the two weak baselines `rz_nas` and `llm_nas`. This is consistent with — not contradicting — §0.1's A4 conclusion and §0.2(e)'s equal-real-compute reframe; it's the same underlying signal (LLM prior not adding accuracy at nominal budget) showing up in a second, independent statistical pass. Reinforces §2.3's honest framing: report LLM prior as implementation-correct-but-unproven, don't keep re-litigating it via more nominal-budget comparisons — the equal-real-compute story (§0.2(e)) is the one worth writing up.

**Alpha-sensitivity sweep built and EXECUTED this session — replaces A4's ad-hoc 6-point/single-budget/single-dataset grid with a finer, independent check. Result: alpha_init=0.0 confirmed a third time.**
- `experiments/run_alpha_sweep.py` — standalone, own `phase="alpha_sweep"` namespace (zero risk to phase3/ablations/phase4/final outputs), 9-point alpha grid `[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]` × 2 budgets `[50, 100]` × 12 seeds × cifar10 only (dataset-widen and budget=200 both deliberately skipped this pass — see decision note below), Groq key-pool aware, `max_llm_calls=25` (ranking exercise). **Ran clean: 216/216 combos, 0 skipped, 0 errors, 26.6 min wall-clock.**
- `experiments/analyze_alpha_sweep.py` — paired Wilcoxon of every non-zero alpha vs alpha=0.0, per (dataset, budget). **Ran clean.**

**Results table (mean best_acc, cifar10, n=12 seeds each):**

| alpha | budget=50 | budget=100 |
|---|---|---|
| 0.00 | 91.510 ± 1.400 | 91.684 ± 1.452 |
| 0.05 | 90.968 ± 1.467 | 91.709 ± 1.158 |
| 0.10 | 91.411 ± 1.382 | 91.215 ± 1.434 |
| 0.15 | 90.275 ± 2.072 | 91.229 ± 1.443 |
| 0.20 – 1.00 (identical, see anomaly below) | 90.007 ± 2.576 | 91.149 ± 1.418 |

**Paired-vs-alpha=0.0 significance (Wilcoxon, n=12):** every single non-zero alpha, both budgets — **`beats_alpha0_significantly = False`**. Best case was alpha=0.05 @ budget=100 (+0.025pp, p=1.0, essentially noise); worst was alpha=0.2–1.0 @ budget=50 (−1.50pp, p=0.147, trending negative but not significant). Gap-vs-alpha0 shrinks going 50→100 (−1.5pp → −0.5pp at the worst alphas), i.e. **not** growing with budget — the one open question left over from §0.1 ("would budget=200 reveal a compounding LLM-prior effect?") now has a soft "unlikely" answer from the trend direction, without needing to spend the extra budget=200 run to confirm it outright.

**Anomaly noted — ROOT-CAUSED Sep 20 (benign; alpha is only a gate schedule, see §0.4(c)):** alpha=0.2, 0.3, 0.5, 0.7, 1.0 produced byte-identical mean/std across all 12 seeds at both budgets. Two candidate explanations, not yet distinguished: (1) benign UCB saturation — `ucb_with_llm_prior()`'s `1e9 + alpha_t * prior_score` term means once one child wins the argmax at alpha=0.2, higher alpha just makes it win by more, same traversal path, same result; (2) a decay/floor bug — if `alpha_t` (post `alpha_at()` decay) drops below `alpha_floor=0.02` for all these initial values at the depths that matter, `score_branches()` falls through to uniform regardless of the nominal alpha_init, silently collapsing several "different" experiment arms into the same code path. **[NOT YET CHECKED]** — quick diagnostic: compare `llm_calls`/`cache_hits` columns in `alpha_sweep_results.csv` for alpha=0.2 vs alpha=1.0 rows; identical call counts too would support (2), differing call counts with identical best_acc would support (1). Doesn't change the alpha_init=0.0 recommendation either way, but matters for trusting the *shape* of the curve (i.e., whether "0.2 through 1.0 all equally bad" is a real finding or an artifact of only ever testing one effective alpha value across five nominal ones).

**Decision — budget=200 / 3-dataset widen: explicitly skipped, not deferred.** Considered per the pre-run discussion (denser grid costs 50–200% more Groq calls); decided against once the 50→100 trend came back shrinking rather than growing. Recorded here so this isn't silently re-opened later without a reason: **do not widen this sweep further unless something else changes** (e.g. the anomaly above turns out to be finding (2), which would warrant a rerun with a corrected alpha_floor/decay before trusting any of this table).

**Final call — three independent tests now agree: A4 (original 6-point ablation, two bug-fix reruns), the main-pipeline `statistical_tests.csv` comparison (§0.3 above), and this 9-point sweep. All three say the same thing: no tested alpha value improves best_acc over alpha=0.0 (LLM prior off) at a statistically significant level.** This closes the LLM-prior accuracy question — see updated §2.3/§5 below.

**Action items arising, in order:**
1. `config.yaml`: `llm.alpha_init: 0.1` → `0.0`. **[DONE — see updated inline comment below.]**
2. Anomaly sanity-check (llm_calls/cache_hits for alpha=0.2 vs 1.0) — **[SUPERSEDED Sep 20: root-caused by code reading, §0.4(c); the llm_calls comparison remains an optional 2-minute confirmation]**.
3. Full clean rerun of `outputs/phase3/`, `outputs/phase4/`, `outputs/ablations/`, `outputs/final/` now that `alpha_init` changed (all stale) — **[NOT YET DONE]**.
4. This doc updated to close out the question — **[DONE, this section]**.

---

## 0.4 [NEW — Sep 20, 2026] Missing segments coded; six new findings from reading the code; execution plan

### (a) What was delivered — all **[CODE-READY]**, none run on real data

| Roadmap item | File(s) | What it does |
|---|---|---|
| §2.1 flat-UCB1 no-trie ablation | `archtree/search.py` (`search_flat_ucb`), `experiments/run_trie_ablation.py` (**A7**) | 3 arms, LLM off, identical proxy gate/budget accounting/seeds: `trie` vs `flat_set` (position-wise UCB1, 30 arms, plain visited-set dedup) vs `flat_none` (no dedup, duplicates cost real evals). Paired Wilcoxon + Cohen's d. UCB1 over the 15,625 leaves is *not* implemented — every unvisited leaf scores +inf so it degenerates to random search without replacement (= the phase-0 `random_search` baseline). |
| §2.1 TCR-H in results + correlation | `tools/analyze_tcr_h.py` | **Replays finished run logs** against HW-NAS-Bench (phase 1–3 never query latency) using the same `update_pareto_archive`/`tcr_h`/`hypervolume` as `search_phase4`. Covers phase 0 baselines too. Writes TCR-H-vs-best_acc and plain-TCR-vs-best_acc correlations. No search rerun needed; the always-empty `tcr_h` column in `phase3_tcr_per_iter.csv` is left alone. |
| §2.1 duplicate evals vs MCTS-NAS | `tools/analyze_duplicate_evals.py`, `dup_proposals` / `repeat_proxy_skips` counters in `search_phase1/2/3` | Duplicate real evals per method from eval_history, plus "UCB proposed an already-evaluated leaf and the trie redirected it" counts for ArchTree (new runs only). |
| §2.3 convergence / wall-clock angle | `experiments/run_convergence_study.py`, per-eval `t_sec`/`llm_sec`/`proxy_sec` in run logs | Paired LLM-on vs LLM-off, cold LLM cache, best@10/25/50, AUC, evals-to-(ceiling−gap), simulated wall-clock (k·S + measured LLM seconds) and a break-even eval cost. |
| §3.1 regret argument | `docs/REGRET_ARGUMENT.md`, `tools/analyze_regret.py` | Informal argument with explicit assumptions and honest limits (not a bound; literature results quoted from memory — verify before citing). Script measures prefix-locality (eta²), near-optimal-cell growth on NB201 exactly, and fits empirical simple-regret exponents against an exact uniform-random reference. |
| §3.2 cross-device Pareto | `tools/analyze_cross_device_pareto.py` | True fronts per device, pairwise Jaccard, latency Spearman, A→B front transfer (HV ratio, normalised IGD); plus precision/recall/overlap of the fronts Phase 4 actually found (rebuilt from all rows). |
| §3.3 proxy composition ablation | `archtree/zc_proxy.py` v5, `experiments/run_proxy_ablation.py`, `search_phase3(zc_override=...)` | All 15 non-empty subsets of {E,S,T,C}: held-out Kendall tau, gate recall of the true top-5%/1%, search best_acc, paired tests vs the full ensemble, and Spearman(ranking quality, search quality) across subsets. |
| §3.3 smaller-LLM sensitivity | `experiments/run_llm_size_sensitivity.py` | (1) prior-quality probe: model scores vs exact subtree-mean accuracy (Spearman, top-1 hit, fallback/mismatch rates); (2) paired search vs LLM-off. Works Groq-only (`groq:openai/gpt-oss-20b`, `groq:llama-3.1-8b-instant`); Ollama only needed for a 3B model. |
| §3.4 repro appendix + release | `tools/build_repro_appendix.py` | Environment, `pip freeze`, git hashes, sha256 of benchmark files, **effective constants read from the source**, config keys no script reads, LaTeX tables; `--bundle` writes `release/archtree_repro_bundle.zip`. |
| Infra | `experiments/run_phase4.py` (rewritten, see (e)), `tools/filter_phase0_summary.py`, `run_full_pipeline.ps1` | — |

Design rules followed: **no existing CSV schema changed** (`append_result_row` uses the first row's keys as the header, so a new key on an old file raises `ValueError`) — new numbers go to new CSVs or the per-run JSON logs; `llm_prior.py` untouched (LLM time measured by wrapping the instance method).

### (b) **[NEW BUG — most consequential]** ZC score cache was contaminated across datasets and methods
`ZCProxyEnsemble.score()` persisted its cache to `outputs/_zc_score_cache.json` **keyed by tokens only, storing the already-aggregated scalar.** That scalar depends on `(weights, norm_stats)`, which differ per dataset (three dataset-scoped calibrations) and differ again for the equal-weight, un-normalised `ZCProxyEnsemble()` that `zico.py` and `rz_nas.py` build. All of them shared one file, so whichever instance scored an architecture first fixed the value every later instance received. Consequences (from code reading — **magnitude not yet measured**): a cifar100/ImageNet search could gate on a cifar10-weighted score; a calibrated search could read a raw-sum-scale value written by a baseline; the earlier dataset-scoped-calibration fix was partly defeated.
- **Fix (code):** `zc_proxy.py` v5 caches the **raw** components {E,S,T,C} (pure functions of the token sequence) in `outputs/_zc_raw_cache.json` and aggregates on read. Old `_zc_score_cache.json` is no longer read → rename/delete it.
- **What is provisional until rerun:** Phase 2, Phase 3, ablations A1/A3/A5/A6 (proxy-gated), Phase 0 `zico` + `rz_nas`, and therefore **§0.2(e)'s equal-real-compute table and the "~28% of real evals / wins-or-ties 5 of 6" headline.** Unaffected: Phase 0 random/enas/mcts/llm_nas (no proxy), A2 (proxy off), Phase 4 (no proxy), `phase2/weights` (calibration used uncached raw scores).

### (c) Alpha-sweep anomaly (§0.3) **root-caused by code reading** — benign, but it changes what the alpha experiments measured
In `ucb_with_llm_prior`, an unvisited child scores `1e9 + alpha_t·prior`, a visited child scores `accuracy + exploration`. So the prior **only orders never-visited siblings**; scaling `alpha_t` cannot change that order for any `alpha_t > 0`. `alpha_t = alpha / sqrt(t/budget·100)` matters only through the `LLMPrior` floor (`alpha_floor = 0.02`): below it the prior is uniform and no LLM call is made. The gate is open while `sqrt(t/budget·100) ≤ 50·alpha`:

| alpha | LLM active for |
|---|---|
| 0.05 | first ~6% of the run |
| 0.10 | first ~25% |
| 0.15 | first ~56% |
| ≥ 0.20 | the **whole run** — all such alphas are identical |

This exactly matches the sweep (0.15 differs slightly from 0.2–1.0; 0.2/0.3/0.5/0.7/1.0 are byte-identical). **Implications:** (1) A4 and the 9-point sweep tested "fraction of the run the LLM is active", not a mixing weight — the null still stands (nothing beat alpha=0.0), `alpha_init=0.0` unchanged; (2) the prior's structural influence is bounded to expansion order of unvisited siblings, which is also why it cannot change the regret argument (§3.1); (3) any "prior fully on" experiment should use alpha ≥ 0.2 (studies use 0.3). Cheap confirmation still available: `llm_calls` for alpha 0.2 vs 1.0 in `alpha_sweep_results.csv` should be identical.

### (d) Two different UCB scales + config values that do nothing
- `traverse()` (phase 1/2, via `_score`) divides accuracy by 100 before adding the bonus; `ucb_with_llm_prior` (phase 3/4) uses raw 0–100 accuracy, so the exploration bonus is ~100× smaller *relatively* in phase 3/4. Phase 1/2 vs 3/4 comparisons are therefore confounded by exploration balance, not only by proxy/LLM. Worth a `C`/scale sensitivity check if a reviewer asks.
- `config.yaml` `ucb.C = 2.0` is **never passed** to any search function — every phase uses the code default `C = 1.414`; phase 1/2 read only `ucb.backup_mode`. `trie.prune_*`, `ucb.lambda_lat`, `hw.latency_*` etc. are likewise unread (constants live in `search.py`). `tools/build_repro_appendix.py` lists unread keys and reports the effective constants — **report those, not the config comments.**

### (e) `run_phase4.py` in the project was a stale copy of `run_ablations.py` — rewritten
The uploaded file (310 lines, header `# experiments/run_ablations.py`, no `search_phase4` call) predates the key-pool patch. New `experiments/run_phase4.py` drives `search_phase4()` per (hardware, dataset, seed, budget), resumes on `phase4_hv_igd_summary.csv`, auto-detects a construct-only LLM backend (Phase 4 never calls the LLM), warns if `data/true_pareto_front__{hw}.json` is missing (IGD would be NaN — run `tools/build_igd_reference_front.py` first). `search_phase4` gained `use_key_pool`/`groq_daily_request_budget`/`groq_reasoning_effort` params. **Open assumption:** defaults used are `C=1.414`, `backup_mode="max"`, `prune_percentile=25` (search_phase4's signature; phases 1–3 use 15) — the lost original's values are unknown. §0.2(a)'s verification of Phase 4 output applies to the *old* run; regenerate and recheck.

### (f) Smaller findings
- **Run-log collisions:** ablation/alpha runs at the same (seed, budget) all wrote `archtree_llm__nb201__{ds}__seed{s}__budget{b}.json`, so each arm overwrote the last (only the final alpha's per-eval log survived). Fixed: non-empty `ablation_id` now suffixes the log name; main-sweep names unchanged so `run_phase3.py`'s resume check is unaffected.
- **`on_pareto_front` in `phase4_pareto_results.csv`** is membership *at insertion time* (`node in pareto_archive` right after the update), not final membership. Do not use it for final-front statistics; rebuild the front from all rows (the cross-device script does).
- **`phase0` resume** is by log file, results are appended to `baseline_summary.csv` — a partial rerun of zico/rz_nas needs the stale rows removed first (`tools/filter_phase0_summary.py`: dry-run by default, `--apply` backs up first, refuses while logs still exist).

### (g) Expectation to check before writing the duplicate-eval claim
MCTS-NAS pads each expanded path with **random** ops, so at budgets ≪ 15,625 it should almost never repeat a leaf — "no dedup guarantee" probably costs it ~nothing here, and **"the trie avoids MCTS duplicates" is likely not a usable headline.** `rz_nas` re-evaluates its current arch every iteration it rejects a mutation, so a large duplicate rate is plausible there, and A7's `flat_none` arm will show the dedup cost directly. `analyze_duplicate_evals.py` prints a warning if MCTS shows < 0.5 duplicates/run. Decide the framing from the measured numbers, not this expectation.

---

---

## 0.5 [NEW — Sep 21, 2026] Rerun progress: Phase 0-4 done, verified clean

### (a) Status

| Phase | Status | Rows | Dupes | Errors |
|---|---|---|---|---|
| 0 (baselines, zico+rz_nas only + 1 filled llm_nas gap) | **[x] DONE** | 144 per method (6 methods) | 0 | 0 |
| 1 (trie+UCB1, no proxy/LLM) | **[x] DONE** | 144 | 0 | 0 |
| 2 (ZC proxy calibration + gating) | **[x] DONE** | 144 | 0 | 0 |
| 3 (LLM warm-start; alpha=0 in main sweep, A4 sweep also completed) | **[x] DONE** | 144 | 0 | 0 |
| 4 (hardware-aware, 6 devices × cifar10) | **[x] DONE** | 288 runs | 0 | 0 |
| Ablations A1-A6 | running now | — | — | — |
| A7, proxy ablation, convergence study, LLM-size study | not started | — | — | — |
| Phase 6 aggregate, analysis scripts, repro bundle | not started | — | — | — |

### (b) Headline numbers so far (cifar10, mean best_acc %, budget 200 unless noted)

| Method | best_acc | real evals used (of 200) |
|---|---|---|
| Phase 1 (trie+UCB1) | 92.14 | 200 |
| Phase 2 (+ ZC proxy gate) | 92.30 | 78.7 |
| Phase 3 (+ LLM off, alpha=0) | 92.50 | 63.4 |
| zico (baseline) | 92.71 | 200 |
| llm_nas (baseline) | 92.98 | 200 |
| random_search | 92.19 | 200 |
| enas | 92.23 | 200 |
| mcts_nas | 92.06 | 200 |
| rz_nas | 90.58 | 200 |

**Efficiency, not accuracy, is the story.** Phase 2/3 use 60-70% fewer real evals than Phase 1 for accuracy within ±0.2pp — they don't out-accuracy zico or llm_nas, they roughly match trie-only-search accuracy at a fraction of the real-eval cost. This still needs the formal equal-real-compute comparison and significance testing from `run_phase6_aggregate.py` before it's a citable claim.

Phase 4 (hardware-aware, all 6 devices): 288 runs, all `real_evals == budget` (correct, no proxy in Phase 4). Hypervolume stable ~91-92.5k across devices; IGD falls as budget grows on all devices; raspi4 has consistently the highest IGD (harder-to-model latency), not a bug.

### (c) ZC score-cache contamination (§0.4(b)) — measured, turned out small
Compared old vs new zico/rz_nas `best_acc` after the clean cache rebuild: max difference **0.42pp**, most differences under 0.2pp, no change in method ranking (zico still clearly above rz_nas everywhere). The bug was real and is fixed, but it did not materially change the Phase-0 baseline picture. Proxy-gated ArchTree numbers (Phase 2/3 above) are the FIRST run under the fixed cache, so they carry no legacy contamination at all.

### (d) Phase 1 vs. old Phase 1 — explained, not a bug
Rerun Phase 1 differs from the pre-Sep-20 Phase 1 log by up to 5.26pp on some rows (131/144 differ, both directions, no consistent bias). Root cause, confirmed by file-modification dates: `ucb.py` (17 Sep) and `run_phase1.py` (15 Sep) were both edited after the old log was written (~9 Jul); `backup_mode` and the UCB constant `C` in the current code are unchanged from config, so the search RULE itself changed at some point between those dates, not something introduced this pass. A determinism check (`tools/check_phase1_determinism.py`) confirmed the CURRENT code is fully reproducible across two fresh processes with different hash seeds (0 divergence). **Any pre-Sep-20 Phase 1 number (including the 92.51% cifar10 figure flagged for retirement in §0's baseline notes) is from stale code and should not be used.**

### (e) cifar10 ZC calibration — checked against current proxy code, confirmed still valid
The saved `zc_calibration_weights__nb201__cifar10.pt` predates the current `zc_proxy.py` (10 Jul vs. 17 Sep for the other two datasets) and was missing from the Phase 2 calibration CSV's tau table (reporting gap only). Checked with `tools/check_cifar10_calibration.py`: held-out Kendall tau of the saved weights (0.487) is within 0.01 of a fresh fit (0.494) on 1,000 held-out architectures scored by the current proxy, and the saved z-score statistics match the current score distribution within ~9%. **Verdict: kept as-is, no recalibration needed.** Note for the proxy-composition ablation (§3.3, not yet run): on cifar10 the fitted SynFlow (S) weight is 0.0 with component tau ≈ 0 — the cifar10 ensemble is effectively {E, T, C} only.

### (f) [NEW FINDING] Phase 3/4 exploration collapse — real, not yet fixed, worth a paper paragraph
`phase3_dedup_stats.csv` (new counters this pass) show, at budget 200 on cifar10: only 63/200 proposals become real evals; 79 are duplicate proposals redirected by the trie, 52 are repeat proposals of an already proxy-skipped path. **Over 60% of the iteration budget is spent on repeats at high budget**, rising from ~0% waste at budget 25 to ~65% at budget 200, consistently across all 3 datasets. This is consistent with the previously-flagged UCB-scale issue (§0.4(d)): `ucb_with_llm_prior` (phase 3/4) uses raw 0-100 accuracy so the exploration bonus is proportionally far smaller than in `traverse()` (phase 1/2, which divides by 100), so the search converges onto a few paths and stalls. **Effect on results so far: probably UNDERSTATES Phase 3/4's real efficiency** — a properly-scaled exploration term would likely extract more real evals from the same proposal budget. Not a blocker for continuing the rerun; flagged for a sensitivity check or fix before the paper's final numbers, and for the C/scale ablation mentioned in §0.4(d).

### (g) Two operational issues hit during the rerun, both resolved, no data lost
1. **Phase 0 `llm_nas` had 143/144 rows instead of 144** (pre-existing gap, unrelated to this pass's changes) — one run (`cifar10, seed=2024, budget=100`) had a complete, valid log (100/100 evals, best_acc=92.998) but no CSV row, apparently from a prior write failure. Fixed by `tools/add_missing_phase0_row.py` (dry-run first, `.bak` made before writing) — appended the one row from the existing log, no rerun/no LLM calls needed.
### (h) Note on §1.1/§1.2/§0.2's existing `[DONE]` marks
Those were verified against the PRE-rerun data. They are not re-marked here — `run_phase6_aggregate.py` (not yet run this pass) will regenerate `statistical_tests.csv` / `main_results_table.csv` from the fresh Phase 0-4 data above, and that regeneration is what will confirm (or contradict) them for the rerun. Treat §1.1/§0.2 as "true for the old data, to be reconfirmed" until then.

2. **Phase 2's post-run summary failed** ("no results written") after the move/backup step — caused by moving `outputs/phase2/results` (which held `phase2_proxy_calibration.csv`, needed by the summary) while correctly keeping `outputs/phase2/weights`. Fixed by copying the old calibration CSV back before rerunning; all 144 Phase 2 runs were skip-resumed correctly and had already completed successfully — this only affected the summary print, not the actual results.


---

## 0.6 [NEW — Sep 22, 2026] A7 trie-ablation + proxy-composition ablation: real results

### (a) A7 — is the trie doing something a flat search couldn't? **ANSWER: dedup, not structure**
432 runs (3 arms x 3 datasets x 4 budgets x 12 seeds), 0 errors. Paired Wilcoxon + Cohen's d, trie vs each flat arm:

| Comparison | Result |
|---|---|
| **trie vs flat_set** (position-wise UCB1, plain visited-set dedup) | **No significant difference anywhere.** p ranges 0.06-0.95 across all 12 (dataset, budget) cells, \|d\| <= 0.64, sign flips between datasets. The trie's prefix-conditioned structure buys essentially nothing beyond a bare `set()`. |
| **trie vs flat_none** (no dedup at all) | **Trie wins, often significantly**, especially at higher budgets (e.g. cifar10 b=200: +2.98pp, p=0.001, d=+1.29; ImageNet16-120 b=200: +5.60pp, p=0.005, d=+1.21). |

**But read the *why* before citing this as a trie win:** `flat_none` wastes 78-98% of its real evals on duplicates at every budget (e.g. cifar10 b=200: 195.0 of 200 evals were duplicate re-evaluations of architectures already seen). It isn't losing because it lacks tree structure — it's losing because it has literally no memory of what it already tried. `flat_set` (one Python `set()`, no trie at all) recovers essentially all of that gap.

**Conclusion for the paper (honest, not spun):** the trie's measured value in this setup is **duplicate avoidance**, not the hierarchical/prefix-conditioned structure per se. A plain visited-set gets the same accuracy. This is consistent with the Phase-3 exploration-collapse finding (§0.5(f)) — the search converges onto very few paths, so what matters most is not re-visiting them, and the trie's extra machinery (prefix stats, pruning, weight-bank inheritance) isn't shown to add anything on top of that by this ablation. **§2.1's "isolate what the trie is buying you" is now answered — this is a genuine, citable (if humbling) result**, and it argues for either (i) reporting it as-is (a clean negative finding on architecture, positive finding on the general idea of state deduplication), or (ii) investigating whether fixing the exploration-scale issue (§0.4(d)) changes the picture, since a search that doesn't collapse onto a handful of paths might let prefix statistics actually matter.

Files: `outputs/ablations/results/A7_summary.csv`, `A7_paired_tests.csv`, `A7_dedup_stats.csv`, figure `A7_no_trie_ablation.png`.

### (b) Proxy ensemble composition ablation — no-proxy still wins on accuracy (expected); ranking quality predicts search quality on 2/3 datasets
All 15 non-empty {E,S,T,C} subsets x 3 datasets, LLM off, alpha=0.

- **No-proxy (`use_proxy=False`) has the highest best_acc on all 3 datasets** — cifar10 92.28% vs the saved 4-component ensemble's 92.02%; cifar100 59.57% vs 58.73-59.26% across subsets; ImageNet16-120 36.59% vs 35.87-36.53%. **This is not a problem** — the proxy trades a small amount of accuracy for the ~50-70% real-eval reduction that's the whole point of Phase 2/3 (§0.5(b)); it confirms the trade is real and roughly the size expected, it doesn't undermine the efficiency story.
- **More components is not automatically better.** Best subset per dataset: cifar10 -> **EC** (2 components, 92.22%, beats the saved 4-component ESTC's 92.19%/92.02% at different seeds — noisy but consistently >= the full ensemble); cifar100 -> ETC (3 components, 59.26%); ImageNet16-120 -> ESC (3 components, 36.53%). No dataset's best subset is the full 4-component one.
- **Ranking-quality (held-out Kendall tau) predicts search quality on cifar10 (Spearman rho=+0.61, p=0.02) and cifar100 (rho=+0.66, p=0.01), but NOT on ImageNet16-120 (rho=+0.22, p=0.44, "weak link").** This is a genuine, useful validation of the proxy design principle on 2 of 3 datasets — worth reporting — but the ImageNet16-120 null means "better tau -> better search" isn't universal and shouldn't be stated as a blanket claim.
- **Confirms §0.5(e)'s calibration-check finding:** cifar10's SynFlow (S) component contributes essentially nothing (weight 0.0 in every fitted subset that includes it, component tau ~ 0) — the cifar10-specific proxy is effectively {E,T,C}. Consider a leaner cifar10 proxy (EC or ETC) as a follow-up; the accuracy delta (92.22 vs 92.02) is small and not yet tested for significance against the saved ensemble specifically.

Files: `outputs/proxy_ablation/results/proxy_ablation_summary.csv`, `proxy_ablation_paired_tests.csv`, `proxy_ablation_size_summary.csv`, `proxy_ablation_rank_vs_search.csv`, figure `proxy_ablation.png`.

### (c) Operational: empty groq-counter-file crash, root-caused and patched
`run_ablations.py` crashed on `A5_prune_threshold` (ImageNet16-120, prune_percentile=35, seed=333) with `JSONDecodeError` reading a 0-byte `groq_calls__nb201__ImageNet16-120__seed333__A5_p35_ImageNet16-120.json`. Root cause: `storage.py`'s `load_groq_counter()` had no defense against an empty/truncated file, and `save_groq_counter()`/`save_llm_cache()` wrote directly to the target path — `zc_proxy.py`'s raw-score cache already used a temp-file-then-`os.replace()` pattern with retries for exactly this class of failure (a transient Windows file lock from antivirus/OneDrive truncating a write), but that pattern had never been applied to these two smaller side-files. Fixed in `storage.py`: both loaders now treat empty/corrupt JSON as "no data yet" (with a warning) instead of crashing, and both savers write atomically. Deleted the one empty file, reran — completed cleanly, no other occurrences. No data lost; this only affects two small per-run bookkeeping files, not any results CSV.


---

## 0.7 [NEW — Sep 23, 2026] Convergence study results (C7's final test) + project re-scoped to master's seminar paper

### (a) Convergence study — LLM prior significantly SLOWS early convergence, all 3 datasets

This was the last untested angle for C7 (§2.3): does the LLM prior help cold-start/convergence speed even though it never helped final accuracy? Run: `experiments/run_convergence_study.py`, alpha=0.3 (LLM fully active the whole run, §0.4(c)), paired LLM-on vs LLM-off, 12 seeds, budgets 100/200, all 3 datasets, cold LLM cache per arm.

**`auc30` (mean best-accuracy-so-far over the first 30 real evals) — LLM-on is WORSE, consistently:**

| Dataset | Budget | n paired seeds | mean diff (on−off) | Wilcoxon p |
|---|---|---|---|---|
| ImageNet16-120 | 100 | 12/12 | −0.39pp | **0.020** |
| ImageNet16-120 | 200 | 12/12 | −0.31pp | **0.020** |
| cifar10 | 100 | 12/12 | −0.50pp | **0.016** |
| cifar10 | 200 | 12/12 | −0.38pp | **0.014** |
| cifar100 | 100 | 12/12 | −0.51pp | **0.042** |
| cifar100 | 200 | 12/12 | −0.41pp | 0.102 (n.s., same direction) |

**5 of 6 cells significant at p<0.05, full 12/12 paired seeds in every cell (not underpowered), same negative direction in all 6.** This is the most reproducible finding in the entire project, positive or negative.

**Every other metric is a clean null (no significant difference either direction):** `final_best`, `evals_to_2.0pp/1.0pp/0.5pp`, `time_to_*pp_S{10,60,600}` (simulated wall-clock at various per-eval costs), `best_at_25`/`best_at_50` (mixed sign, mostly not significant, one exception: cifar10 b=200 `best_at_25` diff=−0.70pp p=0.049 — borderline significant, same negative direction). The "LLM pays for itself if real-eval-cost > Xs" break-even numbers printed by the script are computed from these NON-significant evals_to_*/time_to_* differences — **do not cite the break-even figures, they are noise dressed as a number.**

**Conclusion, now supported by FOUR independent tests (A4 alpha sweep, main-pipeline `statistical_tests.csv`, 9-point alpha sweep, and this convergence study):** the LLM prior never helps and in the one place it shows a significant effect (early convergence), it actively hurts. §2.3's accuracy angle was already CLOSED as negative; **the convergence angle is now closed too, and unlike the accuracy angle it isn't even neutral — it's a significant negative.** `config.yaml`'s `alpha_init=0.0` default is strongly justified by this. C7 should be written up as a rigorously-tested, well-implemented, negative-result component, not chased further or hidden.

**Plausible explanation for the slowdown** (not yet verified, worth a footnote): with alpha≥0.2 the LLM gate is open the whole run, so `LLMPrior.score_branches()` is called for every never-visited sibling at every step (§0.4(c)) — each call costs real wall-clock (`llm_sec`, up to 23s/run at budget=100 on cifar10) but only reorders which unvisited sibling gets tried next; it doesn't change the exploration/exploitation balance once visited nodes exist. If the LLM's ranking of unvisited siblings is uninformative or mildly anti-correlated with true value early on (plausible — the proxy's own held-out tau is only 0.38–0.49, §0.5(e)/§0.6(b), and the LLM has even less signal to work with at depth 0-1 with no proxy score yet), it would front-load worse choices exactly when locking in early paths matters most — consistent with `auc30` being hurt but `final_best` being unaffected once enough real evals accumulate to correct course.

Files: `outputs/convergence/results/convergence_per_run_metrics.csv`, `convergence_paired_tests.csv`; figures `outputs/convergence/plots/figures/convergence_{dataset}_b{budget}.png` (note: a `RuntimeWarning: Mean of empty slice` appeared during figure generation for at least one panel — cosmetic, does not affect the CSV numbers, not yet root-caused, low priority).

### (b) Project re-scoped: this is a MASTER'S SEMINAR PAPER, not an AAAI/IJCAI/ECAI submission

**[Sep 23, 2026]** Clarified in conversation: the target venue is a master's seminar research paper, not a top-tier AI conference as the README's "Target Venue: AAAI 2026 / IJCAI 2026 / ECAI 2026" states. This changes the recommended framing substantially. Recorded here in full since it affects how every result above should be written up.

**Honest assessment against the ORIGINAL goal (`goalREADME.md`: "first NAS framework unifying trie dedup + LLM-guided UCB + zero-cost proxy + weight inheritance + hardware-aware TCR-H, all with formal regret bounds, targeting AAAI/IJCAI/ECAI"):**

| Claimed component | Status after full rerun + new experiments |
|---|---|
| C1-C6 (trie, UCB1, dedup, weight bank) | **Solid.** Reproducible (determinism-checked, §0.5(d)), 0 errors across Phase 0-4. |
| C8 (zero-cost proxy ensemble) | **Real, working.** ~50-70% real-eval reduction (§0.5(b)); proxy-composition ablation (§0.6(b)) validates the design principle on 2/3 datasets. |
| C9 (TCR-H, hardware-aware Pareto) | **Strongest result in the project.** Clean across 6 devices × 3 datasets, 288 runs, 0 errors, sensible HV/IGD trends (§0.5(a)). |
| **C7 (LLM-guided UCB — the title's headline word)** | **Null-to-negative on every one of 4 independent tests** (§0.1's A4, §0.3's stats, §0.3's alpha sweep, §0.7(a)'s convergence study). Never once shown a benefit; significantly HURTS early convergence in 5/6 tested cells. |
| **Formal regret bound (T3/T6) — the title's "hierarchical trie" novelty** | **Undercut by A7 (§0.6(a)):** the trie's entire measured benefit is duplicate-avoidance. A plain `set()` (`flat_set`) matches the trie's accuracy with no significant difference anywhere (p 0.06-0.95, all 12 cells). The prefix-conditioned STRUCTURE that the regret proof is about isn't shown to add anything beyond what a hash set gives for free. |
| NB101 (second benchmark, part of C5) | **Still blocked** (TF1 dependency, no Windows wheel; §1.2, tracked separately). |

**Bottom line:** two of the five unifying ideas (C8, C9) are real, working, and worth reporting as genuine contributions. One (C1-C6, trie+dedup) is solid but its FORMAL/theoretical framing (regret bound) is now empirically thin — the practical benefit is dedup, which is real and provable (T2) but smaller in scope than originally framed. The other two headline ideas the TITLE is built around (LLM guidance, and the trie's structure mattering beyond dedup) are both honest, well-tested NEGATIVE results.

**Why this is still fully publishable — AS A SEMINAR PAPER:**
- A seminar paper is evaluated on rigor of investigation and depth of understanding, not on beating SOTA. Four independent tests converging on the same negative conclusion for C7, and a clean ablation isolating exactly what the trie does and doesn't buy (A7), is a STRONGER showing for this venue than an unexplained positive result would be.
- The formal proofs that still hold unconditionally (T1, T2, T4, T5 — trie correctness, no-re-evaluation guarantee, pruning correctness, weight-inheritance consistency) are real theoretical contributions independent of whether the search "wins" empirically.
- Scope (5 phases, 7 ablations, 2 new experiments beyond the original plan, full statistical testing, reproducibility appendix) is well beyond typical seminar-paper expectations.

**Recommended reframing for the actual writeup (not yet applied to code/config, this is a WRITING guidance note):**
1. **Retitle** — drop or heavily qualify "LLM-Guided" (e.g. "...with an Empirical Investigation of LLM-Guided Warm-Start" or similar), since it is not what earns the paper its results. **[Title discussion and open decision: §0.11(a).]**
2. **Lead with what's real:** trie-based deduplication + zero-cost proxy ensemble efficiency (§0.5(b)), and TCR-H (§0.5(a)/C9) as the flagship novel metric.
3. **Give C7 and A7 dedicated sections, not footnotes.** "We hypothesized X, tested it four/two ways respectively, found Y, here's our best explanation" — this is good scientific writing for this venue, not something to minimize.
4. **Narrow the formal claim.** Keep T1/T2/T4/T5 as-is (still true, still valuable). For T3/T6 (regret bound), either (a) restate the claim as being about the DEDUP guarantee specifically rather than the hierarchical structure's optimization benefit, or (b) present A7 explicitly as testing the boundary condition under which the trie's structure WOULD matter (e.g., revisit if the exploration-collapse fix in §0.4(d) changes the picture — flagged as future work either way).
5. **State the NB101 block as a limitation with reasoning given**, not a gap to apologize for.
6. **Fix the README's stated seed count** (says 5 seeds `[42,123,456,789,999]`; actual runs use 12 seeds including 111, 222, 333, 999, 1001, 2024, 27182, 31415, 27182 etc. per phase logs) — document the ACTUAL seeds used, don't leave the stale claim in.

**What would still meaningfully strengthen the paper before writing** (not required for "is this publishable", but worth doing if time allows): the LLM-size sensitivity study (running next) — even though C7 is closed on accuracy/convergence, a probe of whether a smaller/weaker LLM's branch-scoring quality (Spearman vs. true subtree-mean accuracy, §3.3) correlates with the search harm observed here would add useful mechanistic insight to the "why does the LLM prior hurt convergence" discussion in §0.7(a).


## 0. Current State (baseline, for reference)

- Pipeline built end-to-end: trie search space + UCB1 bandit controller + zero-cost
  proxy ensemble + LLM (Ollama/Groq) warm-start prior. **[DONE — since updated: Groq backend now on gpt-oss-20b, not Llama 3.1 8B (decommissioned Aug 2026); this doc's own "Llama 3.1 8B" line is now stale. See also §0.1 — the LLM prior's *implementation* is now confirmed correct (both known bugs fixed), but its *benefit* is still unconfirmed/negative on the A4 ablation.]**
- Benchmarked on NAS-Bench-201 (CIFAR-10, CIFAR-100, ImageNet16-120) and
  HW-NAS-Bench (6 edge devices). **[DONE for NB201; HW-NAS-Bench wiring exists in code (`benchmarks/hwnb.py`, `search_phase4`) but needs a fresh Phase 4 run against the corrected NB201 cache before its numbers are trustworthy.]**
- Best result: 92.51% CIFAR-10 @ budget=100. **[STALE — this number predates the cache index-desync fix in `build_nb201_cache.py`. Must be regenerated, not reused.]**
- Statistical testing done but underpowered (n=2–3 budgets, p≥0.25). **[FIXED AT CODE LEVEL — `run_phase6_aggregate.py` now does a paired test across seeds at a fixed budget, not across budgets, plus Cohen's d. Not yet re-run because Phase 6 needs Phase 1–4 + ablations outputs, which are stale/unrun for the *main* sweeps — though the A4 ablation specifically has now been re-run three times with proper paired Wilcoxon tests; see §0.1.]**
- Self-identified next step: add NAS-Bench-101. **[NOT STARTED — see §1.2.]**

---

---

## 0.8 [NEW — Sep 24, 2026] Phase 6 aggregate — final headline number, bias-checked

### (a) `run_phase6_aggregate.py` extended (additive only) and run on the full rerun data
Two things were added on top of the existing script (nothing above them changed — verified by diff): an equal-real-compute comparison (the one flagged repeatedly since §0.2(e)/§2.2 as "computed in chat, never written to disk"), and an `extended_findings_summary.csv` that rolls A7 / proxy-ablation / convergence-study results into `outputs/final/` alongside the original tables.

### (b) Nominal-budget comparison (existing script, unchanged) — confirms the known unfair-comparison picture
`statistical_tests.csv`, 96 paired-per-seed tests, archtree_llm vs 8 comparators (6 external baselines + 2 of our own variants; originally written as "9 baselines") at EQUAL NOMINAL BUDGET: only 22% significant (mean |d|=0.455). Expected and uninteresting on its own — archtree_llm spends far fewer REAL evals than its nominal budget (proxy gating), so comparing at equal nominal budget compares apples to oranges. This table exists for completeness, not as the paper's headline.

### (c) Equal-real-compute comparison — FIRST cut (nearest-neighbour) had a bias, caught before being trusted
First version matched archtree_llm's real-eval count to the NEAREST value on the {25,50,100,200} baseline-budget grid. Result: wins-or-ties 99% of 96, one significant loss (`llm_nas`, cifar10, b=25). **But nearest-neighbour rounding sometimes rounded DOWN** — in 5 of 12 (dataset, budget) cells, archtree_llm's actual real-eval usage (e.g. 32.3 at cifar10 nominal b=100) exceeded the matched baseline budget (25), meaning the baseline was quietly given LESS compute than archtree used. This inflated the win margin in exactly the cells with the strongest apparent wins (e.g. cifar10 b=100 vs rz_nas: +3.58pp). **Caught and flagged before being reported anywhere as a final number.**

### (d) Equal-real-compute comparison — STRICT (ceiling-matched) version, this IS the number for the paper
Fix: match to the SMALLEST baseline budget that is >= archtree_llm's real-eval usage (round up, never down), so the baseline is guaranteed at least as much compute as archtree used. Script self-verifies and prints `every comparison gave the baseline >= archtree's real-eval count: True` — confirmed true on the real rerun data, no cells exceeded the grid's max (200).

**Result, run on the full 12-seed / 3-dataset / 4-budget rerun: identical topline to the (now-superseded) nearest-neighbour version — wins-or-ties 95 of 96 (99%), exactly one significant loss.** *[Oct 2 correction: the 96 = 12 (dataset x budget) cells x 8 comparators, 2 of which are our own variants. External-only it is 71 of 72 (98.6%) - §0.10(d). Report wins and ties separately, per baseline.]* The bias happened to not change the bottom-line percentage here, but the STRICT version is what should be cited, since it's the one verified free of the rounding artifact — cite `equal_compute_statistical_tests_strict.csv`, not `equal_compute_statistical_tests.csv`.

**The one exception, fully specified:** `llm_nas`, cifar10, archtree nominal budget=25 (matched to baseline budget=25). mean_diff = −1.34pp, p=0.021. llm_nas is a strong low-budget baseline on cifar10 specifically — the loss does not appear at higher budgets on cifar10, nor on the other two datasets at budget=25.

**Reading the full per-baseline pattern (not just the win/tie count) — important nuance for the writeup:**
- **Large, consistently significant wins vs `rz_nas`** at every (dataset, budget): +2 to +8pp, p<0.02 in every single row (32/32). rz_nas is doing most of the "win magnitude" — it is a weak baseline in this setup.
- **Consistent wins, often significant, vs `random_search`, `mcts_nas`, `enas`, `archtree_v1`, `archtree_zc`** — smaller margins (+0.5 to +2.9pp) but the right sign almost everywhere, several individually significant (p<0.05).
- **Vs `zico` and `llm_nas` (the two strongest baselines): mostly TIES, not wins** — diffs are small and mixed-sign, almost never significant in either direction (the one llm_nas exception above being the sole significant case in either direction across both baselines, 32+32 comparisons total). **Do not write "beats zico/llm_nas" — write "matches them at a fraction of the cost," which is the actually-supported and honestly stronger claim.**

Files: `outputs/final/results/equal_compute_comparison.csv` / `_strict.csv`, `equal_compute_statistical_tests.csv` / `_strict.csv`.

### (e) Recommended headline sentence for the paper (supersedes the §5 draft below) **[CORRECTED Oct 2: 6 external baselines, 71 of 72 - see §0.10(d)]**
> "At matched real-evaluation cost — verified never to favor ArchTree-LLM, i.e. every baseline received at least as many real evaluations as ArchTree-LLM actually used — ArchTree-LLM matches or beats the 6 external NAS baselines in 71 of 72 comparisons (98.6%) across 3 datasets and 4 budgets on NAS-Bench-201, via trie deduplication and zero-cost-proxy gating alone (the LLM prior's accuracy contribution remains unproven, §0.7). The sole significant exception is `llm_nas` at the lowest budget on CIFAR-10. Wins are large and consistent against weaker baselines (random search, MCTS-NAS, ENAS, rz_nas) and are ties, not wins, against the two strongest baselines (ZiCo, LLM-NAS) — the honest framing is comparable accuracy at a fraction of the evaluation cost, not outright superiority."

### (f) extended_findings_summary.csv — consolidated cross-experiment view now exists
`outputs/final/results/extended_findings_summary.csv` (144 rows) rolls up, in one file: A7 (24 trie-vs-flat comparisons, 9 significant — see §0.6(a)), proxy-composition ablation (48 comparisons, 1 significant — see §0.6(b)), convergence study (72 comparisons, 12 significant, mostly `auc30` + the trivially-significant `llm_sec` — see §0.7(a)). LLM-size study row not yet present (not run).


---

## 0.9 [NEW — Sep 24, 2026] Analysis scripts + repro bundle — all done, full detail

### (a) Duplicate-eval analysis — cleaner result than the originally-planned MCTS-NAS comparison
`tools/analyze_duplicate_evals.py`, 864+ phase0 logs + 144 each phase1/2/3 + 432 A7 logs scanned.

**MCTS-NAS confirmed near-zero duplicates as §0.4(g) predicted** (0.06-0.92 dup evals, 0.2-0.5% at every budget) — do NOT lead with "trie avoids MCTS-NAS duplicates," there's nothing to avoid there.

**The real, much stronger result: ArchTree (v1/zc/llm, every phase, every budget) has ZERO duplicate real evaluations, with zero exceptions, across the entire rerun.** This is T2's no-re-evaluation guarantee empirically confirmed at scale, not just proven on paper.

**The interesting comparisons are rz_nas and llm_nas, not MCTS-NAS:**

| Method | Dup rate @ budget=200 |
|---|---|
| flat_ucb_none (A7 control) | 97.5% |
| **rz_nas** | **82.0%** (re-evaluates its current arch every rejected mutation) |
| **llm_nas** | **41.2%** (no structural memory — re-proposes previously-suggested archs) |
| zico | 6.0% |
| mcts_nas | 0.5% |
| **archtree (v1/zc/llm)** | **0.0% — every phase, every budget, zero exceptions** |

**Recommended paper framing:** "Zero duplicate evaluations across every ArchTree run (n=1,296+ runs), vs. 82% for rz_nas and 41% for llm_nas at budget=200" — sharper and more surprising than any MCTS-NAS comparison.

### (b) TCR-H vs. accuracy correlation — weak, inconsistent, do not oversell
`tools/analyze_tcr_h.py`, 7,776 (run x hardware) combos replayed.

Pooled across methods at fixed budget: correlation is weak and WEAKENS with budget (TCR-H rho 0.268->0.028 as budget 25->200; plain TCR goes slightly negative at budget=200). Expected, not a failure: most methods have near-zero duplicates (per (a) above), so TCR barely varies across methods at fixed budget, limiting what it can explain.

Per-method-across-seeds breakdown (holding the algorithm fixed): **inconsistent by dataset.** archtree_llm shows several genuinely significant positive correlations on ImageNet16-120 (rho 0.79-0.85, p<0.01 at budgets 50-100, multiple hardware targets) but near-zero or negative on cifar10 and cifar100 with the SAME method. rz_nas shows a real pattern on cifar100 (rho up to 0.69) but not elsewhere. No correction for multiple testing was applied (~72 tests per method) — clusters of "significant" hits are somewhat above chance but not overwhelming; treat as suggestive only.

**The "within-run rho=+0.828" the script itself flags as descriptive-only should NOT be cited as evidence TCR-H tracks quality** — both TCR-H and best-so-far are monotonically non-decreasing within a run by construction, so this correlation is largely a mechanical artifact of two increasing series, not a real finding.

**Recommended framing:** TCR-H's real validation is the Phase 4 HV/IGD trends (§0.5(a), clean and strong) and the cross-device analysis below (§0.9(c)) — not this correlation-with-single-objective-accuracy check. State plainly: "TCR-H's correlation with final accuracy was weak and dataset-inconsistent, consistent with its design intent as a multi-objective coverage metric rather than an accuracy predictor."

Files: `outputs/final/results/tcr_h_replay_per_iter.csv`, `tcr_h_replay_summary.csv`, `tcr_h_accuracy_correlation.csv`.

### (c) Regret/landscape analysis — supports §3.1's assumptions, cross-confirms A7 from a new angle
`tools/analyze_regret.py`.

**Part A (landscape, exact on NB201) supports the regret argument's core assumptions:** eta-squared (variance explained by first k tokens) jumps from ~0.1-0.2 at k=1-3 to 0.51-0.57 at k=4 on all 3 datasets — the last two of six tokens carry most of the accuracy variance (assumption A1 in `docs/REGRET_ARGUMENT.md`, now with empirical backing). N_6(1pp) (near-optimal cells) is small everywhere (5-44 of 15,625) — assumption A2 also holds.

**Part B (empirical regret decay) mostly ranks methods sensibly, cross-confirms A7:**
- zico decays fastest everywhere (b=0.89-1.36) — consistent with it being the strongest baseline throughout.
- **rz_nas and flat_ucb_none are worst (b as low as 0.00-0.28)** — matches the duplicate-eval finding (a) exactly: wasted budget on re-evaluation prevents regret decay. Independent cross-confirmation between two different analyses.
- **flat_ucb_none shows b~=0.00 on cifar100/ImageNet16-120 (literally no regret decay with budget)** — a second, independent confirmation of A7's finding (§0.6(a)) that no-dedup search is fundamentally broken, from a completely different analytical method (regret-curve shape vs. paired accuracy tests).
- archtree_llm sits mid-pack (b=0.49-0.68), neither best nor worst.

**Caveat that MUST be stated wherever `r_T` (final regret) is quoted:** archtree methods use far fewer real evals (T) than baselines at the same nominal budget (proxy gating), so `r_T` at different T is NOT an equal-cost comparison — same trap §0.8 was built to avoid for accuracy. The `b` exponent (decay rate) is the fairer comparison; raw `r_T` values at different T should not be quoted side-by-side without this caveat. See §0.8 for the proper equal-real-compute analysis on accuracy.

Files: `outputs/final/results/landscape_smoothness.csv`, `regret_fit.csv`, `regret_curves.csv`.

### (d) Cross-device Pareto analysis — STRONGEST result of this batch, lean on this for C9/§3.2
`tools/analyze_cross_device_pareto.py`, cifar10, all 6 hardware targets.

**Ground truth (Part 1) — hardware-specific optimality is real and substantial, not a rounding artifact:**
- Only **36 of 15,625 architectures** are Pareto-optimal on ANY single device; this collapses sharply with more devices required — **only 1 architecture is optimal on all 6.**
- **Mean front Jaccard = 0.22** (min 0.04, max 0.44) — true fronts share only ~22% of their architectures on average between device pairs.
- Latency Spearman = 0.70 — devices are correlated in overall ranking, but diverge sharply at the Pareto frontier itself. Useful nuance: "devices agree broadly but disagree at the efficiency extreme."
- **A->B front transfer: HV ratio 0.94 (misleadingly high — area-based, forgiving) but normalised IGD 0.174 (worst 0.46) — IGD is the metric to cite for transfer failure**, since it measures point-wise distance to the true front rather than area coverage.

**Phase 4 found-front quality (Part 2) — honestly weak on precision/recall, strong on hypervolume:**
- Precision 0.00-0.08, recall 0.00-0.03 against the TRUE front (avg ~11 architectures per device) — Phase 4 rarely lands on the exact Pareto-optimal architectures, even at budget=200 (recovers only ~3%).
- **But HV ratio 0.89-0.98** — the found front still covers most of the true front's USEFUL area even without hitting the exact optimal points. Honest framing: "rarely finds the exact optimum, reliably finds near-optimal coverage."
- **Transfer precision A->B = 0.03**, far below own-device precision — the single cleanest number for "a front found for one device is not usable on another." Directly supports the necessity of device-specific search (C9's core premise).

**Recommended paper framing:** lead the hardware-awareness argument with this analysis over the TCR-H correlation check (§0.9(b)) — it is cleaner, stronger, and more directly supports "hardware-aware search is necessary, not optional."

Files: `outputs/final/results/cross_device_truth_overlap.csv`, `cross_device_truth_transfer.csv`, `cross_device_truth_multiplicity.csv`, `cross_device_found_vs_truth.csv`, `cross_device_found_overlap.csv`; figure `cross_device_truth_jaccard.png`.

### (e) Reproducibility appendix + release bundle — done
`tools/build_repro_appendix.py --bundle`. Wrote `outputs/final/repro/REPRODUCIBILITY.md`, `repro_appendix.tex`, `environment.json`, `requirements.lock.txt`, `data_checksums.json` (sha256 of the 5GB NB201 file, cache, and HW-NAS-Bench pickle), and `release/archtree_repro_bundle.zip`.

Confirmed populated correctly (not `None`/empty): Python 3.10.9, torch 2.5.1+cu124, RTX 2050 (CC 8.6), and the effective-constants table (C=1.414, PRUNE_PERCENTILE=15, PRUNE_EVERY_N=50, WARMUP_REAL_EVALS=10, backup_mode=max, etc.) — this is what to cite in the paper's reproducibility section, not the config.yaml comments (§0.4(d)).

**Two non-blocking notes:** (1) `archtree`/`HW-NAS-Bench` commit hashes are `None` — no git repo in this project, expected, not a bug; a "files last modified" timestamp could substitute if wanted. (2) The script's own dead-config-key scan found **19 unread keys this pass** (`trie.max_depth`, `ucb.lambda_lat`, `llm.alpha_decay`, `hw.latency_penalty`, etc., full list in the script's warning output) — extends §0.4(d)'s finding (which caught `ucb.C` specifically); worth one footnote line in the reproducibility section ("several config.yaml entries are not read by the current codebase; effective values are listed in Table X") rather than fixing config.yaml itself.


## 1. MUST-FIX — Blockers

### 1.1 Statistical power
- [x] Run 10–20 independent seeds per (method, budget, dataset). **[DONE — `config.yaml` now has 12 seeds (`[42, 111, 123, 222, 333, 456, 789, 999, 1001, 2024, 31415, 27182]`), meeting the 10-seed floor. Confirmed executed for Phase 0: `baseline_summary.csv` shows 12/12 seeds for all 6 methods × all 3 datasets. Also confirmed executed for the A4 ablation specifically (§0.1): 72/72 unique (alpha, seed) combos, no duplicates, across three separate reruns.]**
- [x] Report mean ± std across seeds for every accuracy number. **[DONE — `main_results_table.csv` now exists and is populated (180 rows: phase0 baselines + phase1/2/3 + phase4-derived), per §0.2. `_grouped()`'s `best_acc_mean`/`best_acc_std` columns are live.]**
- [x] Re-run significance test as paired-across-seeds-at-fixed-budget. **[DONE — executed for real, 96 tests written to `statistical_tests.csv` (see §0.2(e)): archtree_llm vs 8 baselines × 3 datasets × 4 budgets, n=12 paired seeds throughout. sig_rate=45% overall at nominal budget — but see §0.2(e)'s equal-real-compute reanalysis, which is the number that actually matters given the ZC-proxy's ~72% real-eval reduction.]**
- [x] Report effect size (Cohen's d) alongside p-values. **[DONE — `statistical_tests.csv` has `cohens_d` for all 96 tests (mean |d|=0.66 at nominal budget). Also computed ad-hoc for the equal-real-compute reanalysis in §0.2(e) and for A4 (§0.1, p=0.16–0.43, d not yet added to that specific table — small gap, easy follow-up).]**

### 1.2 Second benchmark: NAS-Bench-101
- [ ] Integrate NAS-Bench-101 as a second search space. **[NOT STARTED (partial scaffolding only) — `benchmarks/nb101.py` exists and works standalone (query/index_to_tokens/tokens_to_index all implemented, including the hash-iterator caching fix), and `archtree/encoder.py` has NB101 encode/decode functions. But `archtree/search.py`'s own comment is explicit: "NB101 not wired for Phase 3 — encoder/index logic is NB201-only, Phase 2 zc calibration was never run for nb101. Out of scope for now." No `search_phaseN` call site actually runs NB101 through the pipeline yet.]**
- [ ] If NB101 results are weaker, report honestly. **[N/A until above is done.]**

### 1.3 Fill in missing baseline columns
- [x] Re-run each baseline (RZ-NAS, LLM-NAS, ENAS, Random Search, MCTS-NAS, ZiCo) on all 3 datasets. **[DONE — confirmed directly from the latest `baseline_summary.csv`: all 6 methods show 12/12 seeds on CIFAR-10, CIFAR-100, *and* ImageNet16-120 (863 total rows, no gaps in the method×dataset×seed grid). This closes the "table full of dashes" problem raised in the roadmap — at least for Phase 0. Phase 1–4 confirmed rerun and multi-dataset (144 rows each for phase1/2/3; phase4 864 rows across 6 hardware targets — see `run_phase6_aggregate.py`'s log). **Ablations (A1/A2/A3/A5/A6) were the one remaining cifar10-only gap — now fixed per §0.2(b), all 5 confirmed dataset-complete.**]**

---

## 2. HIGH-VALUE

### 2.1 Isolate what the trie is actually buying you
- [x] Add a flat-UCB1-no-trie ablation. **[DONE (Sep 22) — ANSWERED: the trie's measured benefit is duplicate-avoidance, not tree structure. trie vs flat_set (bare dedup): no significant difference anywhere (p 0.06-0.95). trie vs flat_none (no dedup): trie wins significantly at most budgets, but flat_none wastes 78-98% of evals on duplicates — see §0.6(a) for the full breakdown and honest framing. Prior status kept for context: NOT YET RUN (Sep 20) — `search_flat_ucb()` in `archtree/search.py` + `experiments/run_trie_ablation.py` (A7: trie vs flat+set-dedup vs flat-no-dedup; see §0.4(a)). Prior status kept for context: NOT STARTED — `run_ablations.py` currently only has A1 (no_llm), A2 (no_zc), A3 (no_weight_inherit), A5 (prune_threshold), A6 (budget_sensitivity), plus A4 (alpha) in `run_phase3.py`. None of these strip the trie itself; all still traverse `ArchTrie`. This is the single most important missing ablation per the roadmap's own §2.1/§5 framing.]**
- [x] Use TCR-H in results, correlate with accuracy. **[DONE (Sep 24) — see §0.9(b): correlation is weak and dataset-inconsistent (pooled rho falls from 0.27 to 0.03 as budget grows; per-method breakdown shows archtree_llm significant on ImageNet16-120 but not cifar10/cifar100 with the same method, no multiple-testing correction applied). The script's own "within-run rho=0.828" is a mechanical artifact (both series are monotonic) and must NOT be cited as evidence. Recommended framing: TCR-H's real validation is Phase 4's HV/IGD trends (§0.5(a)) and the cross-device analysis (§0.9(d)), not this accuracy-correlation check. Prior status kept for context: CODE-READY, NOT YET RUN (Sep 20) — `tools/analyze_tcr_h.py` replays finished run logs against HW-NAS-Bench to get TCR-H for phases 0–3 with no search rerun and writes TCR-H-vs-best_acc / plain-TCR-vs-best_acc correlations. Earlier status kept for context: CODE-READY, PARTIALLY WIRED — `tcr.py::tcr_h()` is implemented and *is* actively computed/logged in `search_phase4` (`phase4_tcr_h_per_iter.csv`, and `run_phase4.py` prints mean final TCR-H). Not yet computed for Phase 1–3 (only `tcr`, not `tcr_h`, is logged there — `phase3_tcr_per_iter.csv` has a `tcr_h` column that's written as `""` / always empty, see `search_phase3`). No correlation-with-accuracy analysis exists yet anywhere.]**
- [x] Quantify duplicate evaluations avoided vs. MCTS-NAS. **[DONE (Sep 24) — see §0.9(a): MCTS-NAS confirmed near-zero duplicates as predicted (0.2-0.5%), so this specific comparison is NOT the headline. The much stronger finding: ArchTree has ZERO duplicate real evals across every phase/budget with zero exceptions, vs. 82% for rz_nas and 41% for llm_nas — recommend leading with those instead. Prior status kept for context: A7 RUN (Sep 22) but this specific MCTS-NAS comparison NOT YET RUN — `tools/analyze_duplicate_evals.py` (which covers MCTS-NAS specifically, replaying phase-0 logs) has not been run yet; A7 measured the trie-vs-flat-bandit duplicate story instead (§0.6(a)), which is a different, now-answered question. §0.4(g)'s expectation (MCTS-NAS incurs ~0 duplicates at these budgets due to its random-op expansion) is still unverified — run `analyze_duplicate_evals.py` to check. Earlier status kept for context: NOT YET RUN (Sep 20) — `tools/analyze_duplicate_evals.py` + `dup_proposals` counters in `search_phase1/2/3` (populated only by reruns). Expect ~0 for MCTS-NAS at these budgets — see §0.4(g). Earlier status kept for context: NOT STARTED — no code computes this comparison. `ArchTrie.is_visited()` / trie dedup exists structurally, and `mcts_nas.py`'s own docstring flags it has "no dedup guarantee," but nothing tallies *how many* duplicate evals MCTS-NAS actually incurs vs. ArchTree for a head-to-head number.]**

### 2.2 Reframe headline around efficiency, not accuracy
- [x] (analysis done) Restructure narrative around speed advantage. **[DATA NOW EXISTS — §0.2(e)'s equal-real-compute reanalysis is exactly this reframe: at matched real-eval count (~28 vs baselines' 25), archtree_llm wins/ties 5 of 6 baselines (significant win vs rz_nas, significant loss only vs llm_nas), a complete flip from the nominal-budget table where it loses to most. Combined with the 72% eval-reduction number already in `eval_efficiency_summary.csv`, this is the paper's efficiency story: "72% fewer real evals, competitive or better accuracy at matched compute." Still needed: (1) write this comparison into an actual output file/figure, not just this doc (§0.2(e) flags this explicitly), (2) confirm the framing routes around the LLM prior's accuracy contribution per §0.1/§2.3 — the win here is trie+ZC-proxy, not the LLM prior. (3) **Sep 20:** re-verify the headline numbers after the clean rerun — the ZC score-cache contamination (§0.4(b)) touches exactly the proxy-gated methods and zico/rz_nas. (4) **[DONE Sep 24, §0.8]** Written to disk (`equal_compute_comparison_strict.csv` / `equal_compute_statistical_tests_strict.csv`), bias-checked (an initial nearest-neighbour version was caught quietly favoring archtree_llm in 5/12 cells and superseded by a stricter ceiling-matched version), and run on the full rerun data: **wins-or-ties 71 of 72 (98.6%) vs the 6 external baselines [originally logged as 95 of 96 incl. 2 of our own variants, §0.10(d)], one named significant exception (llm_nas, cifar10, b=25).** This is now the paper's confirmed headline number, not a projection — see §0.8 for the full breakdown including the important per-baseline nuance (ties vs zico/llm_nas, not wins).]**

### 2.3 Justify the LLM prior's cost
- [x] (accuracy angle — CLOSED) Show what LLM prior buys beyond ZC-only, accuracy dimension. **[DONE, negative result, now triple-confirmed and considered SETTLED — see §0.1 (A4, two bug-fix reruns), §0.3 (main-pipeline `statistical_tests.csv`, independent comparison), and §0.3's alpha_sweep (9-point grid, finest test run). All three agree: no alpha value beats alpha=0.0 (LLM prior off) on `best_acc` at any tested budget, with the gap-vs-alpha0 trend shrinking (not growing) from budget=50→100, arguing against a hidden compounding effect at higher budgets. `config.yaml`'s `alpha_init` is now set to `0.0` on the strength of this. Do not re-open the accuracy question without a new reason (e.g. the still-open alpha=0.2–1.0 saturation/decay anomaly in §0.3 turning out to be a real bug, not benign UCB saturation).]**
- [x] (convergence/wall-clock angle — DONE Sep 23, all 3 datasets) Whether LLM prior helps convergence speed, cold-start, or wall-clock TCR-H even though it doesn't help final `best_acc`. **[ANSWER: NO — and worse, it significantly HURTS early convergence (`auc30`) in 5 of 6 dataset×budget cells, p<0.05, full 12/12 paired seeds every time, same negative direction in all 6 (the 6th, cifar100 b=200, is p=0.102, same direction, likely underpowered not null). `final_best` and evals-to-ceiling remain clean nulls, confirming the accuracy-angle closure from §0.1/§0.3. See §0.7(a) for the full table and a candidate mechanistic explanation. This CLOSES C7 entirely — four independent tests (A4, main-pipeline stats, 9-point alpha sweep, this convergence study) now agree the LLM prior does not help and in one dimension actively hurts. Final honest writeup framing: "C7 (LLM prior) is a correctly-implemented, well-tested component that does not improve final accuracy and significantly slows early convergence; disabled by default (`alpha_init=0.0`)." Do not re-open without new evidence.]**

---

## 3. STRENGTHENING

### 3.1 Theoretical grounding
- [x] Informal regret argument for UCB1-over-trie. **[DRAFTED + RUN (Sep 24) — see §0.9(c): landscape analysis SUPPORTS both stated assumptions (A1 prefix locality: eta2 jumps to 0.5+ at token 4; A2 few near-optimal cells: 5-44 of 15,625). Empirical regret decay independently CONFIRMS the A7 no-dedup finding (flat_ucb_none: b~=0.00, no decay at all) from a different angle. Still not a formal bound — `docs/REGRET_ARGUMENT.md`'s framing as an informal argument, not a theorem, remains correct and unchanged. Caveat: r_T (final regret) at different real-eval counts T is not an equal-cost comparison, same trap as §0.8 — use the b exponent for cross-method comparison, not raw r_T.]**

### 3.2 Hardware-aware results
- [x] (partial) Per-device Pareto front vs. baselines. **[CODE-READY — `search_phase4` builds `pareto_archive` via `update_pareto_archive`, computes `hypervolume()` and `igd()` against a true reference front (`build_igd_reference_front.py`), and writes `phase4_pareto_results.csv` / `phase4_hv_igd_summary.csv` per (hardware, dataset, seed, budget). This is real infrastructure, not just a robustness range — but it's ArchTree-LLM-only; no baseline method currently runs through `search_phase4`, so there's no baseline comparison front yet, only ArchTree-LLM's own front.]**
- [x] Whether the same arch is Pareto-optimal across multiple devices. **[DONE (Sep 24) — see §0.9(d), the STRONGEST result of the whole analysis batch. Ground truth: only 1 of 15,625 architectures is Pareto-optimal on all 6 devices; mean front Jaccard 0.22; A->B transfer normalised IGD 0.174 (HV ratio 0.94 is misleadingly high, don't cite it alone). Phase 4's found fronts: precision/recall against true front are low (0.00-0.08 / 0.00-0.03) but HV ratio is high (0.89-0.98) — rarely exact-optimal, reliably near-optimal. Transfer precision A->B = 0.03. Recommend leading C9/hardware-awareness arguments with THIS analysis over the TCR-H correlation (§0.9(b), weaker).]**

### 3.3 Sensitivity / robustness checks
- [x] Proxy ensemble composition ablation (1 vs 2 vs 3+ components). **[DONE (Sep 22) — see §0.6(b): no-proxy beats every proxy subset on accuracy (expected — that's the eval-reduction trade); best subset per dataset is never the full 4-component ensemble (cifar10->EC, cifar100->ETC, ImageNet16-120->ESC); ranking-quality (tau) predicts search-quality on cifar10/cifar100 (rho=0.61/0.66, p<0.03) but not ImageNet16-120 (rho=0.22, p=0.44). Confirms cifar10's SynFlow component is ~dead, matching §0.5(e). Prior status kept for context: NOT YET RUN (Sep 20) — `experiments/run_proxy_ablation.py` (all 15 subsets × datasets × seeds; held-out ranking quality, gate recall, search best_acc, paired tests vs full ensemble, rank-vs-search correlation) on `zc_proxy.py` v5 (`fit_subset`, raw-component cache) + `search_phase3(zc_override=...)`. Prior status kept for context: NOT STARTED — `zc_proxy.py`'s `calibrate()` does report per-component tau/rho (`E`, `S`, `T`, `C` individually) and has a `DEAD_THRESHOLD` mechanism that already *excludes* near-zero components from the lstsq fit — so component-level diagnostic data exists in `phase2_proxy_calibration.csv` — but no explicit "run with only 1 proxy" / "only 2" ablation sweep exists.]**
- [~] Sensitivity to weaker/smaller LLM (e.g. 3B via Ollama). **[CODE-READY, NOT YET RUN (Sep 20) — `experiments/run_llm_size_sensitivity.py`: prior-quality probe (Spearman vs exact subtree means) + paired search vs LLM-off. Groq-only is enough (gpt-oss-20b vs llama-3.1-8b-instant; probe first — `reasoning_effort` is sent on every Groq call and may be rejected by non-reasoning models); Ollama only for a 3B model. Prior status kept for context: NOT STARTED — Ollama backend path exists and works, but no experiment varies model size and compares degradation. Lower priority now per §0.1 — worth confirming the current 20B model's guidance is at least neutral-not-harmful (done) before investing in a model-size sweep.]**

### 3.4 Reproducibility polish
- [x] Pin exact versions/hyperparameters in a repro appendix. **[DONE (Sep 24) — see §0.9(e): `tools/build_repro_appendix.py --bundle` ran successfully, confirmed populated with real values (Python 3.10.9, torch 2.5.1+cu124, RTX 2050, effective constants C=1.414/PRUNE_PERCENTILE=15/etc — not config.yaml's stale comments). Also surfaced 19 dead config keys this pass, extending §0.4(d). Prior status kept for context: CODE-READY, NOT YET RUN (Sep 20) — `tools/build_repro_appendix.py` writes REPRODUCIBILITY.md, LaTeX tables, environment.json, requirements.lock.txt, sha256 of benchmark files, effective constants read from the source, and the list of config keys no script reads (§0.4(d)). Earlier status kept for context: PARTIAL — `config.yaml` is a genuinely well-documented single source of truth (seeds, budgets, UCB constants, proxy weights, Groq model/budgets all centralized with inline rationale comments), which does most of the *substance* of this ask. What's missing is a standalone reproducibility appendix/table formatted for the paper itself, and pinned library/API versions (NAS-Bench-201 API version, `groq`/`ollama` package versions) aren't recorded anywhere.]**
- [x] Release cache builder + seeds if open-sourcing. **[DONE (Sep 24) — `release/archtree_repro_bundle.zip` written and confirmed (§0.9(e)); contains config, seeds, lock file, appendix, checksums, cache builders. Prior status kept for context: CODE-READY (Sep 20) — `build_repro_appendix.py --bundle` writes `release/archtree_repro_bundle.zip` (config, seeds, lock file, appendix, checksums, cache builders; benchmark files excluded for licence). Earlier status: DONE in spirit — `tools/build_nb201_cache.py` is already a clean, documented, rerunnable script, and seeds are centralized in `config.yaml`. No public release step taken yet, but the artifact itself is release-ready.]**

---

## 4. Suggested Order of Execution — where you actually are

1. **Seeds + re-run stats (§1.1)** — **DONE, executed end-to-end.** Phase 1→2→3→4→ablations→Phase 6 all reran successfully against corrected cache/backend/proxy code (see §0.2). `statistical_tests.csv` is fresh and real (96 tests). One bug found+fixed mid-pipeline (§0.2(d), dataset-blend in `eval_efficiency_summary`) — worth a final rerun of Phase 6 alone (cheap, seconds, no LLM calls) to regenerate `eval_efficiency_summary.csv`/figures with that fix applied, if not already done. **[Sep 20 caveat: "DONE" here predates the ZC-cache finding (§0.4(b)) — proxy-gated numbers and zico/rz_nas must be rerun; see §4a.]**
2. **NAS-Bench-101 (§1.2)** — still not started; still the single biggest remaining lift. Nothing in this pass touched it.
3. **Fill baseline gaps (§1.3)** — **DONE**, fully re-verified: Phase 0 (863 rows), Phase 1–4 (144/144/144/864 rows), and ablations (now dataset-complete per §0.2(b)) all confirmed gap-free.
4. **Trie ablation + TCR-H usage + dup-eval count (§2.1)** — **CODE-READY as of Sep 20, not yet run** (§0.4(a)). Run it after the clean rerun (§4a) — the "baseline is trustworthy" assumption in #1 is qualified by §0.4(b).
5. **Reframe narrative (§2.2)** — **unblocked, analysis done** per §0.2(e): equal-real-compute reframe flips the story from "loses to most baselines" to "wins/ties 5 of 6." Remaining work is writing it into a real output artifact (§2.2 above) and into the paper prose itself.
6. **LLM prior justification (§2.3)** — unchanged from §0.1: accuracy angle answered (negative/neutral), convergence/wall-clock angle still open, now genuinely unblocked since #1 is done (fresh wall-clock data exists in `phase3_main_results.csv`'s `wall_time_sec` column — but that is a per-run total only, so per-eval timing was added to the run logs; the wall-clock/convergence analysis is now CODE-READY as `experiments/run_convergence_study.py`, Sep 20, not yet run).
7. **§3 items** — CODE-READY as of Sep 20 (§3.1–3.4, none run); still lowest priority per the roadmap's own ordering, run after #9.
8. **[NEW, UPDATED] Phase 6 dupe-bug fix + alpha sweep (§0.3)** — dupe fix done, **still not yet re-verified against a clean Phase 6 rerun** (do this before trusting any `outputs/final/` numbers currently on disk). Alpha sweep: **executed, closed, alpha_init=0.0 now locked in** — this was the last open check on §2.3's LLM-prior accuracy framing; it's now final, not tentative.
9. **[NEW, follow-on from #8]** Full clean rerun of `outputs/phase3/ → phase4/ → ablations/ → final/` — required now that `alpha_init` changed from `0.1` to `0.0`; every downstream number currently on disk was generated under the old (unvalidated) value and is stale. **[NOT YET DONE]** — do this before reporting/finalizing ANY current numbers, including the §0.2(e) equal-real-compute table (that one used alpha=0.1 runs, will need at minimum a re-check once rerun completes under alpha=0.0, though the qualitative "trie+proxy wins the efficiency story" conclusion is expected to hold since it was never about the LLM prior).
9a. **[NEW Sep 20]** The rerun in #9 is wider than stated: also **phase 1, phase 2 (logs/results; keep `phase2/weights`), and phase 0 `zico` + `rz_nas`** because of the ZC score-cache contamination (§0.4(b)). Exact order, what to move (nothing hard-deleted) and LLM-call budgets: §4a. **[NOT YET DONE]**
10. ~~Optional: the alpha=0.2–1.0 anomaly~~ — **root-caused Sep 20, §0.4(c)**; optional llm_calls confirmation only.
11. **[NEW Oct 2] Reviewer-driven work (§0.11)** - the 20-item master checklist. Order: A1 (random + same proxy gate) → A6 (baseline audit) → A5 (UCB scale) → A7 (proxy cost) → A3/D17 (quick doc fixes) → B9 (baselines in Phase 4) → B8 (NB101, biggest lift) → B11-B13 (stats, proxy-vs-single, theory) → writing. A1-A7 alone take this from a decent project to a solid seminar paper. **[NOT STARTED]**

### 4a. [NEW — Sep 20] Execution order for the rerun + new experiments

**Before anything — back up, then MOVE (nothing is hard-deleted):** full copy of `outputs/` to `outputs_backup_<timestamp>/`, then move into it: `outputs/phase1`, `outputs/phase2/{logs,results,plots}`, `outputs/ablations`, and `outputs/phase0/logs/{zico,rz_nas}__*`. **Keep:** `outputs/phase2/weights`, `outputs/_zc_raw_cache.json`, `outputs/alpha_sweep`, phase-0 random/enas/mcts/llm_nas logs. Rename `_zc_score_cache.json` → `.OLD`. Then remove the stale zico/rz_nas rows from `baseline_summary.csv` with `tools/filter_phase0_summary.py` (dry run first, `--apply` makes a `.bak`; needs the logs already moved) — a partial phase-0 rerun otherwise appends next to the old rows.

| # | Command | Needs |
|---|---|---|
| 1 | `python experiments/run_phase0.py` (only zico + rz_nas actually rerun; `run_phase0_llm_nas_only.py` not needed) | GPU |
| 2 | `python experiments/run_phase1.py` | — |
| 3 | `python experiments/run_phase2.py` (slowest; refills the raw proxy cache) | GPU |
| 4 | `python experiments/run_phase3.py` (its built-in A4 sweep is the heaviest LLM user and its answer is closed — Ctrl-C after the main runs if you don't need it) | LLM |
| 5 | `python tools/build_igd_reference_front.py` then `python experiments/run_phase4.py --yes` | — |
| 6 | `python experiments/run_ablations.py` (A1–A6) | LLM |
| 7 | `python experiments/run_trie_ablation.py` (A7) | — |
| 8 | `python experiments/run_proxy_ablation.py` | GPU |
| 9 | `python experiments/run_convergence_study.py` | LLM |
| 10 | `python experiments/run_llm_size_sensitivity.py --models groq:openai/gpt-oss-20b groq:llama-3.1-8b-instant` (probe first with `--probe-only`) | LLM |
| 11 | `python experiments/run_phase6_aggregate.py` — then verify no duplicate rows (`groupby(...).size()` all 1s) | — |
| 12 | `tools/analyze_duplicate_evals.py`, `analyze_tcr_h.py`, `analyze_regret.py`, `analyze_cross_device_pareto.py` | — |
| 13 | `python tools/build_repro_appendix.py --bundle` | — |

`run_full_pipeline.ps1` chains all of the above with the two confirmations (move, CSV filter); flags `-From N -To N -Backend groq`.

**LLM backend decision:** Groq only (13-key pool, ≈950 requests and 190k tokens per key per day ≈ 12k requests/day); Ollama is optional and only needed for a 3B model. Upper-bound LLM calls (`max_calls=60` per run): A4 sweep ≈ 3,600, convergence ≈ 1,440, size study ≈ 760 per model, A2/A3/A5/A6 not estimated — split across days if the pool runs dry; don't run two LLM steps concurrently (shared budgets).

**Ordering constraints:** Phase 2 before 3; Phase 3 and 4 before ablations; each script refuses to start otherwise (`storage.verify_phase_inputs`). The new experiments (A7, proxy ablation, convergence, size study) and analysis scripts are **not** wired into `run_phase6_aggregate.py`.


---

## 5. What "Done" Looks Like — current honest answers

1. **Is the accuracy gain real or noise?** → **CONFIRMED, Sep 24, §0.8 — not just answerable, now measured on the final rerun data with the comparison bias-checked.** At nominal equal budget, archtree_llm loses/ties most baselines (22% sig rate, mean |d|=0.455) — expected, unfair comparison (proxy gating means far fewer real evals than nominal budget). At matched real-compute, STRICT (ceiling-matched, verified never to favor archtree_llm): **wins-or-ties 71 of 72 (98.6%) vs the 6 external baselines across 3 datasets × 4 budgets, one significant named exception (llm_nas, cifar10, b=25)** [the earlier "95 of 96 vs 9 baselines" counted 2 of our own variants as baselines, §0.10(d)]. Most of these are TIES, not wins - report them separately. Nuance that matters for the writeup: wins are large/significant against weak baselines (rz_nas, random_search, mcts_nas, enas), but TIES (not wins) against the two strongest baselines (zico, llm_nas). **The honest headline is an efficiency result — comparable-to-superior accuracy at a fraction of the real-eval cost — not a raw-accuracy-superiority result.** This framing is now final, not provisional.
2. **Does this hold beyond one benchmark?** → **No.** NB101 wrapper exists but is explicitly not wired into any search phase. Unchanged this pass.
3. **Is the trie doing something a flat search couldn't?** → **ANSWERED (Sep 22, A7, §0.6(a)): mostly no.** trie vs `flat_set` (bare Python `set()` dedup, no trie structure): no significant difference in any of 12 (dataset, budget) cells (p 0.06-0.95). trie vs `flat_none` (no dedup): trie wins, often significantly — but `flat_none` wastes 78-98% of its evals on duplicates, so this is a dedup-vs-no-dedup comparison, not a structure comparison. **The trie's measured value is duplicate-avoidance, not its hierarchical/prefix-conditioned structure.**
4. **Is every title component earning its place?** → **ZC proxy: yes, and now quantified** — it's the entire reason the equal-real-compute story in §0.2(e) works, and the composition ablation (§0.6(b)) validates the design (tau predicts search quality on 2/3 datasets). **Trie: PARTIALLY — dedup is real and provable (T2), but the hierarchical STRUCTURE beyond dedup is NOT shown to earn its place (A7, §0.6(a): a plain `set()` matches it).** **LLM prior (C7): implementation verified correct, but now CLOSED as harmful-to-neutral across FOUR independent tests — never helps accuracy, and significantly HURTS early convergence (§0.7(a)).** **Honest summary as of Sep 23: 2 of 5 unifying ideas (ZC proxy, TCR-H/hardware-awareness) are strong and title-worthy; 1 (trie) is real but narrower in scope than claimed (dedup, not structure); 2 (LLM guidance, formal regret bound on structure) are the title's weakest claims and should be reframed as tested-and-rejected hypotheses, not headline contributions. See §0.7(b) for the full re-scoping discussion — project target changed to a master's seminar paper, where this is a legitimate and well-supported set of findings.**
5. **One-sentence headline?** → **FINALIZED, Sep 24 — see §0.8(e) for the full paragraph.** Short form: *"ArchTree-LLM matches or beats the 6 external NAS baselines in 71 of 72 comparisons (98.6%) at matched real-evaluation cost — verified bias-free — via trie deduplication + zero-cost proxy gating alone; the LLM prior's accuracy contribution remains unproven and significantly hurts early convergence (§0.7)."* This number is measured on the full, clean, post-rerun data (§0.5's ZC-cache fix, §0.4(f)'s search.py patches) and the matching methodology has been adversarially checked for bias in archtree_llm's favor (§0.8(c)/(d)). Stale 92.51% CIFAR-10 number: fully superseded, do not use. Still open: this framing has not been cross-checked against NAS-Bench-101 (§1.2, still blocked) — state NB201-only scope explicitly in the paper.

6. **Is the efficiency gain attributable to ArchTree, or to the proxy gate alone?** → **NOT YET ANSWERED (Oct 2, §0.11).** Phase 1 (trie + UCB1, no proxy) is 92.14 vs random search 92.19 at budget 200 on cifar10, i.e. the trie/UCB machinery behaves like random on NB201. The missing control is random / MCTS-NAS / ENAS **with the same proxy gate at matched real evals**. Until it exists, do not claim the gain is from ArchTree's search.
7. **Is it novel?** → **NOT VERIFIED (Oct 2).** The combination is ours; each component exists. Literature search (trie/prefix-tree NAS; LLM-guided MCTS/UCB NAS; evaluation caching in NAS) is still open. No "first" claims until done.

---

## 6. Stretch Goals
No change — none started, all still post-submission scope per the original doc.


---

## 0.10 [NEW — Sep 29, 2026] Which models are ours, what each one is, and what we discussed

### (a) Model table

Everything named `archtree*` is one system built in stages, not four separate methods. Each stage adds one component to the previous one.

**Ours (the ArchTree family)**

| Model (code name) | Search fn | What it is | Components on | Status in results | Role in the paper |
|---|---|---|---|---|---|
| `archtree_v1` | `search_phase1` | Trie + UCB1 + weight bank. No proxy, no LLM. | trie, UCB1 (mean backup, accuracy/100 scale), dedup | Phase 1, 144 runs | Intermediate stage / ablation (A-series baseline for "what does the proxy add") |
| `archtree_zc` | `search_phase2` | `archtree_v1` + zero-cost proxy gate (E/S/T/C ensemble, top-35% pass). | + ZC gate | Phase 2, 144 runs. About 79 real evals of 200 at budget 200. | **Headline model in `backend.py` (`OUR_METHOD`) and therefore in the frontend.** Cleanest match to the efficiency story. |
| `archtree_llm` | `search_phase3` | `archtree_zc` + LLM warm-start prior on never-visited siblings. | + LLM prior (alpha), UCB on 0-100 scale, `backup_mode="max"` | Phase 3, 144 runs. **`llm_calls` = 0 in all 144 main-sweep rows** (`alpha_init: 0.0`), so as measured this is trie + proxy with the LLM off. About 63 real evals of 200 at budget 200. | Name used in the Phase 6 headline (§0.8). Its LLM component is closed as neutral-to-harmful (§0.1, §0.3, §0.7). |
| `archtree_llm_hw` | `search_phase4` | Hardware-aware variant: (accuracy, latency) objective, Pareto archive, TCR-H / HV / IGD. | trie, UCB, latency-penalised reward; **no proxy gate, no LLM** | Phase 4, 288 runs (6 devices). Every run uses its full budget as real evals. | The C9 / hardware contribution; cross-device analysis (§0.9(d)) is its strongest result. |

**Ours, but only inside the live demo (`backend.py` ARMS)**

| Arm | What it is |
|---|---|
| `archtree` | Trie + UCB1 with dedup, no gate. Live analogue of `archtree_v1`. |
| `archtree_gate` | Same plus the proxy gate. Live analogue of `archtree_zc`. |
| `archtree_nodedup` | Trie UCB1 with dedup OFF. Really an ablation of our own method (A7 `flat_none` analogue). Tagged "(ours)" for now, see (c). |

**Not ours: baselines (our own simplified re-implementations of existing ideas)**

| Baseline | What it really is | Say this in the paper |
|---|---|---|
| `random_search` | Uniform sampling without replacement | Random search |
| `enas` | Running-mean-per-op sampler, no supernet | "ENAS-style", not ENAS |
| `mcts_nas` | Flat MCTS that pads short paths with random ops; no trie | "MCTS-style" |
| `zico` | Flat sample-and-rank using **our own** ZC ensemble with equal weights | "ZiCo-style", not the original ZiCo scoring |
| `rz_nas` | Mutation walk guided by our own ZC ensemble | "RZ-NAS-style" |
| `llm_nas` | LLM proposes batches of architectures directly, top-K context; falls back to mutation | LLM-proposal baseline. The general idea exists in prior LLM+NAS work, cite it, don't claim it |

**Controls that appear in ablations (not methods, not ours)**: `flat_set` and `flat_none` (flat position-wise UCB1 with set dedup / no dedup, A7); proxy-composition arms `E S T C ... ESTC`, `saved`, `none` (`ESTC` = the full ensemble we deploy).

### (b) What we discussed, in order

1. **"Are `archtree_v1`, `archtree_llm`, `llm_nas`, `archtree_zc` my own versions? Do they exist already?"** Answer: the four `archtree*` names are labels for stages of our own system, so nobody can have published them under those names. The *ingredients* (UCB, tries, zero-cost proxies, LLM priors) all exist separately; what is ours is the combination (prefix trie as search state + dedup + proxy gate + LLM prior on unvisited siblings). `llm_nas` and the other baselines are our implementations of existing ideas and should be named "-style" and cited. **Not yet done:** a real literature search for "trie / prefix tree + NAS" and "LLM-guided UCB/MCTS NAS". Novelty is not verified.
2. **"Which one is my actual final model? There should be one."** Answer: one system, built in stages. By design the final one is `archtree_llm` (plus `archtree_llm_hw` as its hardware extension). By the evidence it is closer to `archtree_zc`, because with `alpha_init = 0.0` the LLM never runs. Two honest framings were offered: **(1)** keep `archtree_llm` as the headline and report the LLM prior as tested-and-neutral; **(2)** make trie + UCB + proxy gate the headline and report the LLM prior as a negative-result ablation. I leaned toward (2). **The choice between (1) and (2) has not been explicitly made.** Until it is, the frontend headline is `archtree_zc` (already the `OUR_METHOD` value in `backend.py`), while the roadmap/Phase 6 headline still says `archtree_llm`. Pick one name and use it everywhere.
3. **"Mark all my models as ours in the frontend and highlight accordingly."** Done, see (c).
4. **Side-finding while reading the code:** `archtree_llm` and `archtree_zc` differ by more than the LLM. `search_phase3` scores UCB on the raw 0-100 accuracy scale (so the exploration bonus is small relative to accuracy gaps) and uses `backup_mode="max"`; `search_phase2` divides accuracy by 100 and uses the `config.yaml` backup mode (`mean`). So `archtree_llm - archtree_zc` is not a clean "LLM effect". The clean LLM-on vs LLM-off comparison is A1 / A4 / alpha sweep / convergence study, which hold every other setting fixed.

### (c) Frontend / backend changes made (`index.html`, `backend.py`)

- **Every ArchTree variant** (`archtree*`, plus the explicit set `archtree_v1/_zc/_llm/_llm_hw/archtree/_gate/_nodedup`) gets the "(ours)" tag and a purple-edged highlighted row in all three accuracy tables (standard, stacked, hardware-grouped).
- **Headline model** (`archtree_zc`) gets a stronger marker ("★ ours · headline") and a cobalt edge. Legends now read "ArchTree variants (ours)" + "<headline> — headline model" + "best value in that column".
- **Break-even tab:** headline = bold solid line; our other variants = lighter purple dotted lines, not counted as rivals; "Versus …" column is relative to the headline model.
- **Win/loss sentence** ("wins X of Y cells against N baselines") now counts **external baselines only**. Otherwise our own variants would show up as "beating" or "losing to" the headline.
- **Hardware table** highlights `archtree_llm_hw` (backend sends `our_method_hw`), instead of falling back to the first row.
- **Headline fallback:** if `archtree_zc` is absent from a results file, the backend falls back to our best variant, not the best method overall (which could be a baseline).
- Live-demo arm labels now read "(ours)". JS syntax-checked and backend logic tested on synthetic data; **not yet viewed in a browser against real outputs.**
- **Judgement call to revisit:** `archtree_nodedup` is tagged "ours" because it is our trie with dedup off. If you would rather present it as a control, remove it from `OUR_METHODS` and tighten the `archtree*` prefix rule in `is_ours()`.

### (d) Correction to the Sep 24 headline (§0.8, §5 item 5)

The Sep 24 text says "95 of 96 comparisons vs 9 baselines". Checking `equal_compute_statistical_tests_strict.csv`: the 96 rows are **12 (dataset × budget) cells × 8 comparators**, and the 8 comparators are `random_search, enas, mcts_nas, zico, rz_nas, llm_nas` **plus our own `archtree_v1` and `archtree_zc`** (Phase 6 treats every method other than `archtree_llm` as a "baseline"). So:

| Comparator set | Comparisons | Wins-or-ties | Rate |
|---|---|---|---|
| As reported (8 comparators, incl. 2 of our own variants) | 96 | 95 | 99% |
| **External baselines only (6)** | **72** | **71** | **98.6%** |
| Our own variants only (`archtree_v1`, `archtree_zc`) | 24 | 24 | 100% |

The single exception is unchanged (`llm_nas`, cifar10, nominal budget 25, p = 0.021). Use the **72-comparison, external-only** figure in the paper and drop "9 baselines" (there are 6). The 24 own-variant comparisons are a fair internal ablation ("adding the proxy / LLM machinery does not lose accuracy") but are not evidence of beating other methods. The frontend win-count already follows this rule; `run_phase6_aggregate.py` itself is **unchanged**, so its CSVs still contain the 8-comparator numbers.

### (e) Open items from this pass

- [ ] Decide the single headline name, (1) `archtree_llm` or (2) `archtree_zc` / "trie + proxy gate", and make roadmap, paper, and `OUR_METHOD` agree.
- [ ] Literature search to actually verify novelty (trie/prefix-tree NAS; LLM-guided MCTS/UCB NAS). Not done.
- [ ] Rename baselines to "-style" in the paper text and cite the originals.
- [ ] Optionally filter `archtree_*` out of the "baselines" list in `run_phase6_aggregate.py` so the CSV headline matches the corrected 71/72 figure.
- [ ] Open the updated frontend against real outputs and eyeball the Overview and Break-even tabs.
- [ ] Decide whether `archtree_nodedup` is presented as ours or as a control.
- *(Oct 2: these open items are folded into the master checklist, §0.11(c) - items A4 [headline name], A2 [literature search], D17 ["-style" names], D20 [nodedup, frontend check].)*


---

## 0.11 [NEW - Oct 2, 2026] Technical-reviewer assessment, title decision, and master checklist

**Scope of this pass:** a self-review written the way a skeptical NAS reviewer would read the project (goal README + this roadmap + the progress report). **Nothing was run.** Every checklist item below is `[ ]` unless stated. Numbers quoted from earlier sections keep their original provenance (see the caveat in (b)).

### (a) Title

- **Earlier working title (superseded by the final title below):** *Neural Architecture Search via Trie-Based Deduplication and Zero-Cost Proxies: An Empirical Study of LLM-Guided Warm-Start*. Much more honest than the original "LLM-Guided UCB Traversal" title and consistent with §0.7(b).
- **Concerns raised:** (1) "Trie-Based Deduplication" slightly oversells the trie, because A7 shows a plain `set()` matches it (§0.6(a)) - say so in the abstract, or title it "Evaluation Deduplication". (2) Hardware-aware search (C9, the strongest result) is invisible in the title. (3) "Dedup + proxies" is presented as what works, so the random + same-proxy control (A1) is what backs the title's central claim.
- **Reviewer-suggested alternative:** *Duplicate-Free, Proxy-Gated Neural Architecture Search: Hardware-Aware Evaluation and an Empirical Study of LLM-Guided Warm-Start*.
- **Other options:** (i) closest to the working title, adding "with Hardware-Aware Coverage"; (ii) *What Actually Helps in Trie-Based NAS? Deduplication, Zero-Cost Proxies, and Hardware-Aware Search, with a Negative Result on LLM Warm-Start*; (iii) short form *Deduplication and Proxy Gating for Sample-Efficient Hardware-Aware NAS*.
- [x] **Final title DECIDED (Oct 2, 2026):** *Neural Architecture Search via Trie-Based Deduplication and Zero-Cost Proxies, with Hardware-Aware Coverage: An Empirical Study of LLM-Guided Warm-Start* - option (i) above. Applied in README, progress report and this roadmap; still to be used in the paper itself.

### (b) Reviewer scores

| Dimension | Score | Why |
|---|---|---|
| Rigor and honesty | 8/10 | 12 seeds, paired tests, effect sizes; self-caught biased matching and the 95/96 miscount; negative results reported plainly |
| Experimental depth (within NB201) | 7/10 | 5 phases, 7 ablations, convergence study, analysis scripts, repro bundle |
| Novelty | 3/10 | Combination is new, components are not; dedup/caching is standard; literature check not done |
| Real-world usability | 3/10 | Lookup-table benchmark, no real training, 15,625-architecture space |
| Breadth (benchmarks, baselines, real training) | 3/10 | NB201 only (NB101 blocked), several baselines are own re-implementations |
| **Overall - master's seminar paper** | **about 7/10** | |
| **Overall - top venue (AAAI/IJCAI/ECAI)** | **about 3/10** | |
| **If A1-A7 and B8 done** | **about 5-6/10** | workshop / negative-results level |

**Provenance caveat:** the matched-compute per-baseline figures quoted in the review discussion (e.g. -0.18pp vs random search, p=0.97; +2.32pp vs rz_nas) come from §0.2(e), which was computed **before** the clean rerun and is marked provisional there. The final per-baseline numbers must be re-read from `equal_compute_statistical_tests_strict.csv` before they go into the paper.

### (c) Master checklist (prioritized)

**A. Must fix - these decide whether the paper holds up**

- [ ] **A1. Random search + the same proxy gate.** Run random, MCTS-NAS and ENAS with the identical proxy filter at matched real evals. Phase 1 (trie + UCB1, no proxy) = 92.14 vs random 92.19 (cifar10, b=200, §0.5(b)), so without this control the efficiency gain cannot be attributed to ArchTree rather than to the proxy filter (which ZiCo / AZ-NAS already use). *Needs: GPU for proxy scoring, no LLM.*
- [~] **A2. Novelty / literature check.** FIRST PASS DONE Oct 2 (web search, abstracts/snippets only; results and consequences in §0.12). Still to do: read the key papers in full, search Google Scholar / Semantic Scholar / dblp with the query list in §0.12(d), and build the related-work section. "First" stays removed from every claim.
- [ ] **A3. Headline number.** Use **71 of 72 (98.6%) vs 6 external baselines**; report **wins and ties separately, per baseline** (most are ties; large wins are mostly vs the weak rz_nas). Optionally filter `archtree_*` out of the baselines list in `run_phase6_aggregate.py` so the CSV matches.
- [x] **A4. One headline method name - DECIDED Oct 2 (see §0.12(e)):** headline = the Phase 3 system run with the LLM prior OFF (alpha = 0; code name `archtree_llm`), because every headline number (71/72) was measured on it. Present it in the paper as "ArchTree (LLM prior off)", not as an LLM method. `archtree_zc` (Phase 2) stays an ablation arm. **Still to do in code:** change `OUR_METHOD` in `backend.py` from `archtree_zc` to `archtree_llm`, and update the frontend headline label.
- [ ] **A5. UCB scale confound.** Phase 1/2 divide accuracy by 100 and Phase 3/4 do not (§0.4(d)); this also drives the roughly 60% wasted iterations at high budget (§0.5(f)). Unify the scale and add a sensitivity check on `C`. Note `ucb.C = 2.0` in `config.yaml` is never read (code default 1.414). Phase 1/2 vs 3/4 comparisons are confounded until this is done.
- [ ] **A6. Audit the weak baselines.** rz_nas = 90.58 vs random 92.19 (cifar10, b=200) looks like a bug or poor tuning; a reviewer will call it a strawman. Check its implementation (re-evaluates the current architecture on every rejected mutation: 82% duplicates), and check llm_nas for fairness. Rename all baselines "-style" and cite the originals.
- [ ] **A7. Count the proxy's own cost.** "Zero-cost" still needs a forward (+ backward) pass per proposal; the held-out Kendall tau is only about 0.4-0.5. Report proxy wall-clock / FLOPs and give a cost-adjusted efficiency number.

**B. Should fix - makes it stronger**

- [ ] **B8. Second benchmark (NB101).** NB201 is saturated and the A7 conclusion ("structure = a plain set") may be an artifact of depth 6. NB101 has token length 26, which is where prefix structure could actually matter. Unblock via Linux / WSL / Docker / Colab or a TF-free port (§1.2). If it stays blocked, state it as a limitation with the reason.
- [ ] **B9. Baselines in the hardware phase.** Phase 4 is ArchTree-only (§3.2). Add random search and an NSGA-II-style baseline, otherwise C9 says nothing about the method.
- [ ] **B10. Reframe TCR-H.** Present it as a metric definition. Do not cite the TCR-H-vs-accuracy correlation or the within-run rho=0.83 (mechanical artifact, §0.9(b)). Lead with the cross-device Pareto analysis, and say plainly that "1 of 15,625 Pareto-optimal on all 6 devices" is a property of the HW-NAS-Bench data. Phase 4 found-fronts have low precision/recall (0.00-0.08 / 0.00-0.03) but high HV ratio (0.89-0.98).
- [ ] **B11. Statistics.** Multiple-testing correction (Holm) over the 72-96 tests (§0.9(b) notes none was applied) and confidence intervals alongside p-values.
- [ ] **B12. Proxy ensemble vs best single proxy.** Test significance. On cifar10 SynFlow has weight 0 and no dataset's best subset is the full 4-component ensemble (§0.6(b)).
- [ ] **B13. Theory.** Verify T4: percentile pruning cannot guarantee it never removes the optimum unless the prune uses confidence bounds - re-check the proof or restate the claim. State T3/T6 as informal arguments, not theorems (T6 is near-trivial since the prior only reorders never-visited siblings, §0.4(c)). Keep T1, T2, T5; call T7 definitional.

**C. Scope the LLM claim carefully**

- [ ] **C14.** The prior only reorders never-visited siblings (§0.4(c)), so a narrow integration was tested. Write "this form of LLM prior does not help", not "LLMs do not help NAS". Alpha experiments measured the fraction of the run the LLM is active, not a mixing weight.
- [ ] **C15.** Contrast with `llm_nas`, where the LLM is a generator and gives the best accuracy (92.98 on cifar10, b=200). That contrast deserves a section.
- [ ] **C16.** (Optional) LLM-size probe when the Groq budget resets (§3.3). Do not cite the convergence-study break-even figures (§0.7(a)).

**D. Writing and housekeeping**

- [ ] **D17.** Update docx report and README: 6 baselines, 71/72, 12 seeds (README says 5), seminar target, chosen title; "-style" baseline names.
- [ ] **D18.** Give C7 and A7 their own sections. State limitations plainly: NB201 only, lookup table with no real training, small search space.
- [ ] **D19.** Abstract states up front: the benefit is dedup + proxy gating, not tree structure; LLM guidance did not help.
- [ ] **D20.** Small cleanups: decide `archtree_nodedup` ours-vs-control; fix the `Mean of empty slice` warning in convergence figures; open the frontend against real outputs; report effective constants not config comments; fix the seed list in §0.7(b) item 6 (27182 is listed twice - copy the real seed list from `config.yaml`).

### (d) What did NOT change in this pass

No experiment, script, config or output file was modified. The only edits are to this document: the headline correction (§0.8(b)/(d)/(e), §2.2, §5 items 1 and 5), the title pointer in §0.7(b), the new items in §4 and §5, and this section.


---

## 0.12 [NEW - Oct 2, 2026] Literature check (first pass) and headline-method decision

**Method and limits.** Web search only (about 6 queries: trie/prefix-tree NAS; LLM + MCTS/UCB NAS; LLM-NAS with zero-cost proxies; tree-based NAS with training-free metrics; RZ-NAS). I read **abstracts and snippets, not full papers**, and I did not search Google Scholar, Semantic Scholar or dblp. "Not found" below means "not found in this pass", not "does not exist". Cite only what you have read yourself.

### (a) Closest prior work found, and what it does to each claim

| Work | What it does (from the abstract/snippet) | Overlaps with | Consequence |
|---|---|---|---|
| Vo, Le, Luong, *Efficient Multi-Objective NAS via Tree Search with Training-Free Metrics*, SN Computer Science (2026), doi 10.1007/s42979-026-04749-4 | Title and listing only were seen; a related snippet describes a "Training-Free Multi-Objective Tree-based NAS" using SynFlow and Jacobian covariance to pick promising nodes in an architecture search tree, evaluated on NB201 (attribution of that snippet was garbled - verify it is the same work) | C8 (proxy inside tree search) + C9 (multi-objective) | **Closest competitor found. Read it first.** The claim "proxy embedded in a tree search, multi-objective" is NOT unique to ArchTree. |
| Vo and Luong, *MTF-PDNS* (arXiv 2407.20656) | Multi-objective NAS with several training-free metrics, novelty search | C8 (multi-metric ensemble) | Prior art for multi-proxy use; cite. |
| Qian et al., *When NAS Meets Trees (TNAS)*, CVPR Workshops 2022 | Architecture tree + binary operation tree, bi-level BFS; reports finding the NB201 CIFAR-10 optimum in 4 GPU hours | C1 (tree-factorized search space) | Prior art for tree-structured NAS on NB201 - must be a baseline or at least discussed. |
| Roshtkhari, *NAS by Learning a Hierarchical Search Space* (arXiv 2503.21061) | MCTS over a learned hierarchical search space; visiting order of branches | C1/C2 | Prior art for hierarchical-tree + MCTS NAS. |
| AlphaX; LaNAS | MCTS-based NAS; LaNAS reports large sample-efficiency gains over random, regularized evolution and MCTS on NB101 | C2 | Standard MCTS-NAS references. |
| RZ-NAS (ICML 2025); CoLLM-NAS (arXiv 2509.26037); EvoPrompting, LLMatic, SEKI, LM-Searcher (seen in reference lists only) | LLM-as-generator / mutator NAS, several with zero-cost proxies | C7 | LLM-to-NAS is a crowded area; your angle (LLM as a prior over unvisited tree siblings) was not found in this pass. |
| **LLM-NAS: LLM-driven Hardware-Aware NAS** (arXiv 2510.01472) | HW-NAS-Bench, LLM + zero-cost predictor; **reports an "exploration bias" - the LLM repeatedly proposes the same architectures**; claims near-complete coverage across latency ranges | C9, C7, and your `llm_nas` baseline | (1) **Corroborates your 41% duplicate-eval finding for llm_nas** - cite it. (2) Hardware-aware + coverage-across-latency already exists, so "TCR-H, first of its kind" cannot be claimed. (3) Confirm which paper your `llm_nas` baseline actually implements. |
| ZAPS (arXiv 2609.14184) | Zero-cost proxy selection + active learning under a fixed budget, NB201 and NB101, 200 seeds | C8 efficiency claim | Recent strong competitor for "proxy-gated sample-efficient search"; consider as a discussion point or extra baseline. |
| CoRA-NAS (arXiv 2609.11884) | States that under one protocol no single zero-cost proxy robustly beats the trivial #Params/FLOPs baseline | C8 | **Add a #Params/FLOPs gate as a control** (extends A1). A reviewer who knows this paper will ask for it. |

### (b) What was NOT found in this pass
- A prefix trie used for **exact deduplication** of evaluated architectures in NAS.
- An LLM used as a **prior over never-visited sibling branches inside a UCB tree** for NAS.

This is encouraging but not proof. These two are the only parts that still look possibly new, and the A7 ablation already shows the first one is replaceable by a plain `set()`.

### (c) Consequences for the paper
1. Keep every "first" claim removed (A2).
2. Reword the contribution as a **study**, not a new-method claim: dedup + proxy gating + hardware-aware coverage evaluated and ablated on NB201, with the LLM prior and trie structure reported as negative results.
3. Related-work section must cover: TNAS, Vo et al. (tree search + training-free metrics), MTF-PDNS, Roshtkhari, AlphaX/LaNAS, RZ-NAS, LLM-NAS (HW), CoLLM-NAS, ZAPS, CoRA-NAS.
4. TCR-H: present as a metric definition and use the cross-device analysis; do not claim novelty of hardware-aware coverage in general.
5. Add a **#Params/FLOPs gate** control, and consider TNAS as an additional baseline.

### (d) Queries to run yourself (Google Scholar / Semantic Scholar / dblp)
`"prefix tree" "neural architecture search"` · `trie NAS duplicate evaluation` · `"evaluation cache" OR "memoization" neural architecture search` · `LLM prior MCTS neural architecture search` · `LLM "UCB" neural architecture search NAS-Bench-201` · `tree search "training-free" multi-objective NAS` · `hardware-aware NAS Pareto coverage metric HW-NAS-Bench` · `zero-cost proxy gating budget NAS-Bench-201`. Also read the related-work sections of Vo et al. and LLM-NAS (2510.01472), which will point to anything missed.

### (e) Headline-method decision (A4)
- **Facts:** the 71/72 equal-compute result and the 92.50 / 63.4-real-evals row are for the Phase 3 system run with the LLM prior off (alpha = 0; code name `archtree_llm`). `archtree_zc` is Phase 2 (92.30 / 78.7 evals) and differs in UCB scale and backup mode (§0.10(b) point 4), so it is not the system the headline was measured on.
- **Decision:** headline = Phase 3 system with alpha = 0, presented as "ArchTree (LLM prior off)". `archtree_zc` remains an ablation arm. This matches the title (dedup + proxies + hardware-aware, LLM as a studied add-on).
- **Caution:** if you prefer `archtree_zc` as the headline instead, `run_phase6_aggregate.py` must be rerun with it as the focal method, because the 71/72 figure would no longer apply.
- **Still to do (code):** `OUR_METHOD` in `backend.py` and the frontend headline label (currently `archtree_zc`).
