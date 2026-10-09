# ArchTree-LLM — detailed results log (Oct 2-4, 2026)

*This is the full working log that used to be the project README (moved here Oct 4, 2026). The project's front page is now the master `README.md` in the repository root; section numbers (§0-§8) quoted in the other documents refer to THIS file.*

Neural architecture search on tabular benchmarks: a prefix **trie** over architecture tokens, **UCB1**
traversal, **deduplication** (never pay twice for the same architecture), a **zero-cost-proxy gate**
(only the top 35% of proposals by proxy score get a real evaluation), an optional **LLM prior** for
ordering unvisited branches, and a **hardware-aware** variant (accuracy vs latency).
Benchmarks: NAS-Bench-201 (cifar10, cifar100, ImageNet16-120), HW-NAS-Bench (6 devices) and,
new, NAS-Bench-101 (cifar10).

Detailed notes: `ReadMe/FIXES_OCT2026.md` (every change), `ReadMe/COMPLETE_GUIDE_OCT2026.md` (how to run),
`ReadMe/RUN_PLAN_OCT2026.md` (per-step table). Every number produced before Oct 2, 2026 is invalid
(see §2); the old outputs are kept in `outputs_pre_oct2026_20261002_1215\` for reference only.

---

**Oct 9, 2026 corrections (re-checked against the CSVs):** NB101 vs random is 14 W / 2 T (was 15 / 1); GPU-hour
saving vs RE under the same gate is 19-23% (IN16 19%); "1st on NB201 at equal GPU-hours" is tied-best (4.75 vs
4.84, CD 1.91); the LLM-size +0.28 (p = 0.027) is unadjusted; duplicate rates 30% / 67% are means over budgets
(55% / 74% at b=200); the gate-cost model is claimed for its sign only (it under-predicts the size).

## 0. Oct 3 audit and ArchTree v2 (read first)

**Audit of every result file:** no duplicates, NaNs or out-of-range accuracies; 0 duplicate real
evaluations; Phase 3 made 0 Groq calls; an offline replica reproduced the device runs exactly.

**Finding (statistics):** Holm was applied over *all* comparisons at once. Applied per family
(gate × baseline, as it should be), **regularized evolution beats v1 ArchTree** in 6/12 cells
(label-free gate: budgets 100 and 200 on every dataset). v1 numbers are in §5.3 for reference.

**Finding (cost):** counting real evaluations understates the gate's cost — the gate prefers bigger
networks (+18% training time per evaluation). In GPU-hours (benchmark training time of the evaluated
architectures) the label-free gate still saves ~52% at budget 200
(cifar10 50 vs 104 h, cifar100 97 vs 207 h, ImageNet16-120 302 vs 625 h). New: `tools/analyze_train_cost.py`.

**ArchTree v2 (config `ucb.reward: rank`, `ucb.C: 0.3`, `search.mutation_prob: 0.5`):**
1. UCB reward = percentile rank of the accuracy among the run's evaluations (accuracy/100 squeezed
   all values into ~0.85-0.95, so exploration swamped exploitation);
2. C = 0.3;
3. after 10 evaluations, half the proposals mutate the best of 5 random evaluated architectures.

Chosen on the main seeds, then **validated on 12 held-out seeds** (`search.heldout_seeds`,
`experiments/run_design_ablation.py`, label-free gate, budgets 25-200, Holm per arm):

| Arm vs v2 | mean Δ test acc | v2 significantly better | real evals b=200 (cifar10) |
|---|---|---|---|
| v1 (raw reward, C=1.414) | −0.63 | 3 of 12 | 84 |
| rank reward only | −0.18 | 1 of 12 | 71 |
| mutation only (p=1) | +0.03 | 0 (ties) | 94 |
| regularized evolution, same gate | +0.14 | 0 (ties) | 99 |
| **v2** | — | — | **84** |

v2 matches RE in accuracy with **12-19% fewer GPU-hours** at budget 200 (full numbers after the core
rerun: §5.4). Honest reading: the accuracy comes from the evolutionary
step; the trie/UCB part mainly saves cost (dedup + fewer, cheaper evaluations).

Also new: RE is now a Phase 0 method (7 external baselines in Phase 6), Holm per family in
`run_gated_baselines.py` / `run_nb101.py`, `train_hours` in every result row, C grid includes 0.3.
v1 behaviour is exactly reproducible (`ucb.reward: raw`, `ucb.C: 1.414`, `search.mutation_prob: 0`).

**Done:** core rerun with v2 (Oct 3, 08:51). The device runs reproduce the offline v2 replica exactly
(Phase 3 108/108, design ablation 540/540). Results in §5. NB101 done (§5.6b). Next: groq, final (§6).

---

## 0b. Oct 3 (afternoon): additions that close the reviewer attacks (code done, runs pending)

| # | Addition | Closes the attack | Code | Pipeline |
|---|---|---|---|---|
| 1 | **Knowledge-free size prior** (op score ∝ #params, no LLM) as a convergence-study arm | "the LLM only knows bigger is better" | `archtree/llm_prior.py` (backend `heuristic`), `run_convergence_study.py` arm `size_prior` | step 15 |
| 2 | **Contamination test**: same LLM, prompt without the benchmark name / NB201 op names (`described`), labels only (`anonymous`, negative control) | "the LLM just remembers NAS-Bench-201" | `llm_prior.py` prompt modes, `experiments/run_llm_contamination.py`, convergence arm `llm_described` | steps 24, 15 |
| 3 | **Local search** (White et al. 2021) and **RE + zero-cost warm-up** (Abdelfattah et al. 2021, closest prior work) | "strong / closest baselines missing" | `baselines/local_search.py`, `baselines/regularized_evolution.py` (`init_pool`, `ZCWarmupRESearch`) | steps 4, 12, 22 |
| 4 | **Third benchmark: NATS-Bench size space** (32,768 networks, only widths change, so cost varies a lot) | "only NB201-like spaces"; tests the cost finding where cost is not tied to cell design | `benchmarks/natssss.py`, `tools/build_natsbench_sss_cache.py`, `SSSProxyNet` in `zc_proxy.py` | steps 23, 22 |
| 5 | **Gate-cost model**: closed-form criterion + simulation that predicts, before the runs, whether correcting the gate for cost helps; checked against measured arms | turns "we observed" into "we predicted and confirmed" | `tools/gate_cost_model.py`, `run_equal_gpu_hours.py` (`gate_cost_model.csv`, `gate_cost_validation.csv`) | step 22 |

**Proposition (gate-cost model).** In a Gaussian-copula model of (accuracy A, proxy P, predicted cost D),
the gate score S = z(P) − λ z(D) has corr(S, A) = (r_PA − λ r_DA) / √(1 + λ² − 2λ r_PD), so
d corr(S, A)/dλ at λ = 0 = **r_PA·r_PD − r_DA**: a small cost penalty helps only if accuracy is less
cost-coupled than the proxy's cost bias implies. NB201: r_PA ≈ 0.83–0.86, r_PD ≈ 0.79, r_DA ≈ 0.66–0.72
→ slope −0.02 to −0.04 → the penalty should hurt, as measured (§5.6c). The simulation of gated random
search reproduces the measured plain-gate accuracy (k = 100: predicted 93.89 / 71.96 / 45.67, measured
93.92 / 71.92 / 45.45) and the sign of the penalty's effect at λ = 1; it underestimates its size
(ArchTree reacts more strongly than random search). NATS-SSS is where the criterion can flip sign.

**`-Plan extend` done on the PC (Oct 3, 13:32):** Phase 0 now 9 methods x 144 runs (0 errors); local search
in the same-gate study: 48 of 48 comparisons with ArchTree v2 are ties (Holm); NATS-SSS cache built
(32,768 archs, max test 93.65 / 71.34 / 47.40, no missing values). Note: on NATS-SSS #params varies 60x
but the recorded training time only ~2x (cifar10 255-529 s), so GPU-hour differences will be small there.

| Phase 0, test acc (12 seeds) | b=25 c10 / c100 / IN16 | b=200 c10 / c100 / IN16 |
|---|---|---|
| local search | 93.75 / 71.44 / 45.23 | 94.37 / 73.51 / 46.52 |
| RE + zero-cost warm-up | 93.91 / 72.00 / 46.11 | 94.28 / 73.36 / 46.34 |
| RE | 93.42 / 70.43 / 44.53 | 94.31 / 73.37 / 46.48 |
| ArchTree v2 (gate, ~40% of evals) | 93.60 / 71.51 / 45.28 | 94.28 / 73.40 / 46.41 |

**Already measured (real NB201 data, offline, same code):** local search with the full budget vs the
v2 headline (~40% of the evaluations): 11 of 12 cells tie, LS ahead at cifar10 b=200 (+0.09). In the
first equal-GPU-hour test (2 seeds) ArchTree v2 + gate had no losses against any of the 13 other arms,
including LS and RE warm-up. Full 12-seed numbers: step 22 on your PC.

---

## 0c. Oct 4: faster pipeline + reviewer-grade statistics

### Speed-ups (results are identical - checked byte for byte / run for run)

| Change | Where | Effect |
|---|---|---|
| Per-iteration TCR rows buffered and written in blocks (was: one file open per search iteration, scanned by the antivirus on Windows) | `archtree/storage.py` (`append_result_row_buffered`, `append_result_rows`), `archtree/search.py` | output CSVs byte-identical; far fewer file operations |
| Proxy cache written at most every 2 min (was: the whole 6-15 MB JSON rewritten every 25 new architectures) | `archtree/zc_proxy.py` | big saving on NB101 / NATS-SSS |
| Proxy cache merge-on-write under a file lock | `archtree/zc_proxy.py` | several processes can share it without losing entries (tested: 4 workers, 0 lost) |
| **Parallel proxy pre-computation**: 3 GPU processes fill the cache for all 32,768 NATS-SSS networks and the NB101 samples | `tools/precompute_proxies.py`, new **step 25** | step 22 then only reads proxies |
| **Parallel search workers in step 22** (`runtime.workers: 3`) - each run seeds its own RNGs | `experiments/run_equal_gpu_hours.py` | identical results for 1 or 3 workers (252/252 runs); side files of the searches switched off there |

Tune in `config.yaml` -> `runtime:` (`workers`, `proxy_workers`; lower to 2 if the GPU runs out of memory).

### New statistics (`tools/analyze_rank_equivalence.py`, in step 18)

* **Friedman test + Nemenyi critical difference + CD diagram** (Demšar 2006) over all methods; blocks =
  (dataset, budget, seed).
* **TOST equivalence** (±0.5 and ±0.25 test points, Holm): "no significant difference" is not evidence of
  equality - TOST is.
* **Probability of improvement** P(ArchTree v2 > method) with a stratified bootstrap 95% CI.

First numbers (real device results, NB201, Phase 0 at FULL budget vs ArchTree v2 with ~40% of the evaluations):

| Method | avg rank (1 = best, CD = 1.13) | P(v2 better) [95% CI] |
|---|---|---|
| RE + zero-cost warm-up | 3.67 | 0.41 [0.34, 0.48] |
| local search | 3.90 | 0.43 [0.36, 0.49] |
| **ArchTree v2** | **4.41** | – |
| RZ-style | 4.41 | 0.50 [0.42, 0.57] |
| ZiCo-style | 4.47 | 0.51 [0.44, 0.58] |
| RE | 5.07 | 0.57 [0.50, 0.64] |
| ENAS-style | 6.57 | 0.77 [0.70, 0.83] |
| LLM-NAS | 6.70 | 0.75 [0.69, 0.81] |
| random | 7.85 | 0.81 [0.75, 0.88] |
| MCTS-style | 7.97 | 0.85 [0.79, 0.90] |

Friedman p = 1e-76. ArchTree v2 is within the critical difference of the best method (not
significantly worse) while using ~40% of its evaluations; local search and RE + warm-up are slightly
more often better at full budget. **TOST:** equivalence within ±0.5 points is shown in only 1-5 of 12
cells - with 12 seeds the paper must say "no significant difference", not "equivalent". The fair
comparison is equal GPU-hours (step 22, device run Oct 4, §5.6c): ArchTree v2 + gate has 0 losses vs 13 arms
and is within the critical difference of the best method on NB201 (**tied-best**, 4.75 vs 4.84 for RE + ZC warm-up), NB101 (2nd) and NATS-SSS (4th).
NB101 (label-free gate): P(v2 > RE) = 0.44 [0.30, 0.57], P(v2 > random) = 0.83 [0.73, 0.94].

### All figures in one folder, each with its data points (`plots/figures/`)

`plots/plot_results.py all` now runs in **step 18** (before: only Phases 3 / ablations / 6 drew figures, the
rest needed a manual call). New / rewritten figures: 4-arm convergence study (`convergence_early_speed`,
`convergence_effect_sizes`), equal GPU-hours (`egh_acc_vs_gpu_hours`), gate-cost model predicted vs
measured (`egh_gate_cost_model_check`), proxies vs training time (`egh_proxy_cost_correlation`), LLM
contamination probe (`llm_contamination_probe`).

Every figure saved by any script (plot_results, CD diagrams, regret, cross-device, trie / proxy ablation,
convergence) is ALSO copied to `plots/figures/<experiment>__<name>.png` next to
`plots/figures/<experiment>__<name>.csv` = the exact points drawn (columns: panel, series, kind, x, y,
value, x_label, x_axis, y_axis), read back from the figure itself (`plots/figure_export.py`). The original
files stay where they were. Rebuild the folder any time: `python plots/plot_results.py all` and
`python tools/analyze_rank_equivalence.py`.

---

## 1. Status board

| Block | Command | Status | Result |
|---|---|---|---|
| Data: NB201 v2 cache, HW fronts | steps 1-3 | done | cache verified against the NB201 paper (§4) |
| Unit tests + NB101 cache | `-Only 0,20` | done | 423,624 graphs, 0 hash mismatches |
| LLM-free NB201 experiments | `-Plan core` | done (SynFlow fixed) | §5 |
| Decision 1: calibration mode | — | **decided: label-free** (`zc_proxy.calibration_mode: unsupervised`) | §5.3 |
| Steps 7, 8, 10-13 with the label-free gate (v1) | `-Only 7,8,10,11,12,13 -NoGroq` | done (Oct 2), superseded by v2 | archived |
| Oct 3 audit + ArchTree v2 (rank reward, C=0.3, 50% mutation) | `-Plan core -ArchivePhases ...` | **done (Oct 3, 08:51)**, verified | §0, §5 |
| NB101 experiment (v2) | `-Plan nb101` | **done (Oct 3, 09:39)**, 576/576 runs | §5.6b |
| Groq studies (LLM prior C7, llm_nas baseline) | `-Plan groq` | **done (Oct 3, 11:09)**; 2 alpha-sweep rows (IN16) discarded for fallbacks | §5.6d |
| Equal-GPU-hour study + proxy cost analysis + gate-cost model, NB201 + NB101 + NATS-SSS (step 22) | `-Plan final` | **done Oct 4 (device)** | §5.6c, §0b |
| Oct 3 additions: local search, RE warm-up, NATS-SSS cache, size prior, contamination probe | `-Plan extend`, `-Plan groq` | **all done** (extend Oct 3 13:32, groq Oct 4 12:53) | §0b, §5.6d |
| Final aggregate, analyses, repro bundle | `-Plan final` | **done Oct 4, 13:04** | §5.9 |

---

## 2. Problems found and fixed

| # | Problem | Effect on the old (Sep 2026) results | Fix | Verified by |
|---|---|---|---|---|
| 1 | HW-NAS-Bench indexed in a different architecture order than NB201 | Phase 4, cross-device and TCR-H numbers belonged to the wrong architectures | canonical→HW-NAS-Bench index map from the pickle's arch strings (`benchmarks/hwnb.py`) | accuracy–latency alignment ρ = 0.624 (was 0.001) |
| 2 | NB201 cache = 12-epoch results and **cifar10 TRAIN accuracy** | every NB201 accuracy | v2 cache: 200-epoch, search on validation, report test (`tools/build_nb201_cache.py`) | matches the NB201 paper (§4) |
| 3 | LLM-prior bias: children visited only through proxy-skips got prior 0.5, truly unvisited ones ~1/k | "LLM slows convergence" was an artifact | one notion of "cold" child, one accuracy scale (`archtree/ucb.py`) | zero-information LLM ≡ LLM off on 24/24 seeds |
| 4 | **LLM prompt contained example scores `{"none":0.7,...}`; the model copied them for every prefix** | the prior always favoured `none`, the worst op | no example values, integer 0-9 scores, strict JSON schema (`archtree/llm_prior.py`) | Sep caches show identical scores for every prefix |
| 5 | **SynFlow computed with BatchNorm in train mode on one all-ones image** | SynFlow carried no signal (τ ≈ 0.05); ensemble and label-free gate weaker | BatchNorm in eval mode (`archtree/zc_proxy.py`); cached S values recomputed automatically | τ = 0.55-0.59 (§5.1) |
| 6 | UCB scale changed after the runs (0-100 vs 0-1) | old runs not reproducible | one scale (accuracy/100) everywhere | phase2(max) ≡ phase3(α=0) on 12/12 |
| 7 | Pruning removed the LEAST-visited nodes | could cut unexplored optima; theorem T4 false as stated | value-based pruning, never the last sibling; T4 restated as heuristic | `REGRET_ARGUMENT.md` §5b |
| 8 | Silent LLM fallbacks (failed calls returned uniform scores) | "LLM-on" runs partly ran without an LLM | fallbacks counted; such runs discarded and redone | `llm_fallbacks` column |
| 9 | Config keys ignored (C, pruning, latency penalty) | config did not match the code | `archtree/settings.py` reads them all | — |
| 10 | 1,500 calibration labels not counted as cost | efficiency overstated | `labels_used` tracked; label-free mode added and chosen | `tools/analyze_proxy_cost.py` |
| 11 | Weak / mislabelled baselines (rz_nas paid for re-evals, zico sampled with replacement, enas had no policy) | baselines handicapped | fixed and renamed "-style" | — |
| 12 | NAS-Bench-101 blocked (needs TensorFlow 1) | NB201-only paper | TF-free NB101 layer (§3) | all 423,624 hashes reproduced |
| 13 | Groq budget wasted (dead keys retried, no TPM pacing, unparseable answers) | discarded LLM runs | `utils/groq_client.py` (§3) | offline tests with a fake Groq SDK |
| 14 | NASWOT stored every layer's codes (~1.5 GB GPU peak) | OOM risk on 4 GB GPUs | kernel accumulated per layer | bit-identical scores, peak ~tens of MB |

---

## 3. What was added

| Addition | Files | Purpose |
|---|---|---|
| NAS-Bench-101 without TensorFlow | `tools/build_nb101_cache.py`, `benchmarks/nb101.py`, `tools/test_nb101.py` | pure-Python TFRecord/protobuf reader, ports of the official prune / hash / validity code |
| Bench-generic search | `archtree/ucb.py` (`vocab_at`), `archtree/trie.py` (`mark_dead`), `archtree/search.py` (`_fix_proposal`) | per-position vocabularies; invalid / isomorphic NB101 proposals redirected. NB201 path bit-identical (6/6 regression runs) |
| NB101 proxy network | `archtree/zc_proxy.py` (`NB101ProxyNet`) | proxy-size port of the NB101 network |
| Regularized evolution baseline | `baselines/regularized_evolution.py` | standard strong NB101/NB201 baseline; also in step 12 |
| Same-gate baselines | `experiments/run_gated_baselines.py` | "is it the trie or just the gate?" |
| NB101 experiment | `experiments/run_nb101.py` | archtree / random / RE × 4 gates, paired + equal-compute tests |
| HW baselines, UCB sweep, proxy cost | `run_phase4_baselines.py`, `run_ucb_sensitivity.py`, `tools/analyze_proxy_cost.py` | reviewer items |
| Groq efficiency | `utils/groq_client.py`, `utils/rate_limiter.py`, `utils/groq_key_pool.py` | strict JSON schema, per-minute token pacing, per-day 429 → next key, shared answer cache, token accounting |
| Pipeline | `run_full_pipeline.ps1`, `tools/preflight.py` | plans (`core`/`nb101`/`groq`/`final`), preflight, logs in `pipeline_logs\`, `-ArchivePhases`, `-DryRun` |

---

## 4. Data checks

| Check | Expected | Got |
|---|---|---|
| NB201 v2 cache size | 15,625 | 15,625, no gaps |
| all-`none` cell, cifar10 test | 10.00 | 10.00 |
| NB201 best valid → its test, cifar10 | 91.61 → 94.37 | 91.61 → 94.37 |
| NB201 best valid → its test, cifar100 | 73.49 → 73.51 | 73.49 → 73.51 |
| NB201 best valid → its test, ImageNet16-120 | 46.73 → 46.20 | 46.73 → 46.20 |
| NB101 graphs / repeats | 423,624 / 3 each | 423,624 / 3 each |
| NB101 hashes reproduced by our port | all | 423,624 of 423,624 |
| NB101 best mean test | ≈ 94.3 | 94.32 (best valid 95.06 → test 94.23) |
| HW accuracy–latency alignment | ≈ 0.6 | 0.624 |
| unit tests | all pass | storage 28/28, NB101 all |

---

## 5. Results (ArchTree v2, NB201, 12 seeds, TEST accuracy of the best-validation architecture)

Core rerun with v2, Oct 3, 08:51 (`ucb.reward: rank`, `ucb.C: 0.3`, `search.mutation_prob: 0.5`,
label-free gate). Integrity: Phase 0 864 rows (6 methods), Phase 1/2/3 144 each, gated baselines 2,880,
UCB sweep 1,008, design ablation 720; no duplicates, no missing values; A1 = Phase 3 on 144/144 runs.
The v1 results are in `outputs_archive\20261003_0828\` (label-free v1, source of the v1 row in §5.3; keep until
the paper is submitted). `20261002_1642` (supervised v1) is superseded and can be deleted.

### 5.1 Zero-cost proxies (Kendall τ with validation accuracy, 1,500 architectures)

| Dataset | E (SVD entropy) | S (SynFlow) | T (NASWOT) | C (complexity) | Ensemble, supervised (1,500 labels) | Ensemble, **label-free** (0 labels, used) |
|---|---|---|---|---|---|---|
| cifar10 | 0.55 | 0.55 | 0.59 | −0.50 | 0.67 | **0.64** |
| cifar100 | 0.54 | 0.59 | 0.64 | −0.53 | 0.72 | **0.68** |
| ImageNet16-120 | 0.49 | 0.56 | 0.59 | −0.48 | 0.63 | **0.63** |

C is negative by design (it penalises size). The label-free ensemble = z-scored E, S, T with equal weights;
labels were used only to measure its τ, never to set its weights.

### 5.2 Cost saved by the label-free gate (v2 gated vs v2 ungated, same proposals budget)

| Budget | real evals saved (c10 / c100 / IN16) | GPU-hours saved (c10 / c100 / IN16) | test acc. change (c10 / c100 / IN16) |
|---|---|---|---|
| 25 | 34 / 34 / 34% | 32 / 32 / 30% | −0.08 / +0.06 / +0.04 |
| 50 | 46 / 46 / 50% | 41 / 41 / 46% | +0.05 / −0.16 / +0.36 |
| 100 | 52 / 53 / 55% | 46 / 48 / 49% | +0.11 / +0.11 / +0.27 |
| 200 | **60 / 59 / 59%** | **53 / 53 / 53%** | +0.16 / +0.43 / +0.02 |

GPU-hours = NB201 training time of the really evaluated architectures. GPU-hour savings are ~6 points
smaller than evaluation savings because the gate prefers larger (slower-to-train) networks.

### 5.3 Headline system vs baselines at budget 200 (test accuracy, mean of 12 seeds)

| Method | real evals | GPU-h (c10 / c100 / IN16) | cifar10 | cifar100 | ImageNet16-120 |
|---|---|---|---|---|---|
| **ArchTree v2 (label-free gate, LLM off)** | **81-83** | **55 / 115 / 351** | 94.28 | **73.40** | 46.41 |
| ArchTree v2, supervised gate (1,500 labels) | 76-81 | 54 / 112 / 348 | 94.26 | 73.27 | 46.43 |
| ArchTree v2, no gate | 200 | 117 / 243 / 742 | 94.12 | 72.97 | 46.39 |
| ArchTree v1 (Oct 2, label-free gate) | 79-83 | 50 / 97 / 302 | 93.85 | 72.31 | 45.54 |
| Random search | 200 | | 93.92 | 71.92 | 45.47 |
| ENAS-style | 200 | | 94.09 | 72.59 | 46.16 |
| MCTS-style | 200 | | 93.89 | 71.94 | 45.44 |
| ZiCo-style (label-free proxy) | 200 + proxy | | 94.29 | 72.97 | 46.28 |
| RZ-style ZC mutation walk | 200 + proxy | | 94.33 | 73.39 | 46.32 |
| Regularized evolution (no gate) | 200 | 121 / 258 / 782 | 94.31 | 73.37 | **46.48** |
| LLM-NAS (gpt-oss-20b, batch 8) | 200 | 0 | 94.28 | 72.14 | 45.37 |
| Oracle (best validation arch) | — | — | 94.37 | 73.51 | 46.20 |

**Paired tests, v2 headline vs the 6 Phase-0 baselines** (Wilcoxon, Holm within each baseline, 12 cells
each = 3 datasets × 4 budgets; baselines get the full budget, ArchTree ~40% of it):

| Baseline | wins | ties | losses |
|---|---|---|---|
| Random search | 7 | 5 | 0 |
| MCTS-style | 6 | 6 | 0 |
| ENAS-style | 3 | 9 | 0 |
| ZiCo-style | 1 | 11 | 0 |
| RZ-style | 0 | 12 | 0 |
| Regularized evolution | 0 | 12 | 0 |
| **Total** | **17** | **55** | **0** |

v1 lost to RE in 6 of 12 cells after per-family Holm; v2 has no losses. The formal equal-compute
comparison (each baseline at ArchTree's real-evaluation count) is step 17 (`-Plan final`).

### 5.4 Same gate for every search strategy (step 12)

Budget 200, **label-free gate** (`zc_unsup`):

| Proposer | real evals (c10 / c100 / IN16) | GPU-h (c10 / c100 / IN16) | cifar10 | cifar100 | ImageNet16-120 |
|---|---|---|---|---|---|
| **ArchTree v2** | 81 / 82 / 83 | **55 / 115 / 351** | 94.28 | 73.40 | 46.41 |
| Regularized evolution | 105 / 106 / 103 | 72 / 146 / 434 | 94.37 | 73.44 | 46.39 |
| ENAS-style | 77 / 79 / 75 | 50 / 103 / 303 | 93.96 | 72.41 | 45.86 |
| Random | 79 / 79 / 79 | 48 / 96 / 290 | 93.92 | 71.92 | 45.47 |
| MCTS-style | 56 / 61 / 54 | 31 / 67 / 178 | 93.62 | 71.29 | 44.87 |

Paired tests (Holm per gate × proposer family, 12 cells each), ArchTree v2 vs:

| Gate | Random | ENAS | MCTS | RE |
|---|---|---|---|---|
| none | 3 W / 9 T | 12 T | 2 W / 10 T | 12 T |
| #params | 6 W / 6 T | 1 W / 11 T | 6 W / 6 T | 12 T |
| ZC supervised | 2 W / 10 T | 3 W / 9 T | 4 W / 8 T | 1 W / 11 T |
| **ZC label-free** | 7 W / 5 T | 3 W / 9 T | 9 W / 3 T | **12 T** |

No losses anywhere (0 of 192). Against RE under the same gate: equal accuracy with **19-23% fewer
GPU-hours** (c10 23%, c100 21%, IN16 19%) and ~22% fewer real evaluations.

**Held-out seeds (design ablation, `design_paired_tests.csv`, 12 unseen seeds × 3 datasets × 4 budgets):**

| Arm vs v2 | mean Δ test | v2 significantly better | real evals b=200 (c10 / c100 / IN16) | GPU-h b=200 (c10 / c100 / IN16) |
|---|---|---|---|---|
| v1 (raw reward, C=1.414, no mutation) | −0.63 | 3 of 12 | 84 / 83 / 83 | 51 / 101 / 304 |
| rank reward only | −0.18 | 1 of 12 | 71 / 66 / 71 | 45 / 85 / 275 |
| mutation only | +0.03 | 0 (ties) | 94 / 93 / 99 | 65 / 132 / 417 |
| RE, same gate | +0.14 | 0 (ties) | 99 / 105 / 102 | 68 / 147 / 425 |
| **v2** | — | — | 84 / 84 / 90 | **57 / 118 / 372** |

On unseen seeds v2 ties RE with 12-19% fewer GPU-hours, and beats v1.

### 5.5 Does the tree structure matter? (A7, trie vs flat per-position UCB, v2, label-free gate)

| Comparison | significant (unadjusted p < 0.05) | mean difference |
|---|---|---|
| trie vs flat UCB **with** dedup | 0 of 12 | +0.06 points (tie) |
| trie vs flat UCB **without** dedup | 3 of 12, trie better | +0.53 points |

→ the gain over plain UCB comes from **deduplication**; with dedup, trie and flat UCB are equivalent.

### 5.6 Other checks

| Study | Result |
|---|---|
| A1 cross-check (ablation "no LLM" = Phase 3 headline) | identical on 144/144 runs |
| Device rerun vs offline v2 replica | Phase 3 108/108, design ablation 540/540 identical |
| Hardware-aware search vs random / NSGA-II-style (HV and IGD, 6 devices × 4 budgets) | **96 of 96 ties** after Holm |
| UCB constant C ∈ {0.1 … 2} × backup {mean, max} (v2) | no significant effect (0 of 78 tests, Holm) — v2 is robust to C |
| Proxy composition ablation | 2 of 48 tests significant (Holm); best single proxy vs ensemble: no difference (Holm p = 1.0) |

### 5.6b NAS-Bench-101 (cifar10, 12 seeds, v2, Oct 3)

576 runs (3 proposers × 4 gates × 4 budgets × 12 seeds), no duplicates or missing values. Proxy ranking is
weaker than on NB201 (supervised ensemble τ = 0.50; E 0.39, S 0.26, T 0.27).

| Gate, budget 400 | ArchTree v2 | Random | RE | real evals (ArchTree) | GPU-h (ArchTree / RE) |
|---|---|---|---|---|---|
| none | 93.94 | 93.68 | 94.13 | 400 | 200 / 216 |
| #params | 93.91 | 93.68 | 94.07 | 163 | 111 / 121 |
| ZC supervised | 93.96 | 93.68 | 94.05 | 158 | 90 / 98 |
| **ZC label-free** | **93.96** | 93.68 | 94.07 | **163** | **105 / 110** |

Paired tests (Holm per gate × proposer, 4 budgets each): ArchTree vs random **14 wins, 2 ties** (ties: gates none and #params at b=50); vs RE
**15 ties, 1 loss** (#params gate, b=100, −0.31). Equal compute: the label-free-gated ArchTree ties ungated
ArchTree at 4/4 budgets with ~59% fewer real evaluations and ~47% fewer GPU-hours, and beats ungated random 4/4.

→ The NB201 findings transfer: the gate halves cost at equal accuracy; ArchTree ≈ RE in accuracy (RE
numerically 0.1-0.2 ahead, not significant); the GPU-hour edge over RE is small on NB101 (~5%).

### 5.6c Equal GPU-hours (step 22) — DEVICE RUN Oct 4, 2026 (1128 runs, 12 seeds, 3 benchmarks)

**Device results** (`outputs/equal_gpu_hours/results/`; ArchTree v2 + label-free gate vs every arm, Holm within
arm, 21 cells = NB201 3 datasets × 3 budgets + NB101 3 budgets + NATS-SSS 3 datasets × 3 budgets):

| vs | mean diff (pts) | wins / ties / losses |
|---|---|---|
| ArchTree + cost-penalised gate (λ = 1) | +1.01 | 16 / 5 / 0 |
| random + cost-penalised gate | +1.36 | 15 / 3 / 0 |
| random, no gate | +0.67 | 10 / 11 / 0 |
| random + same gate | +0.50 | 6 / 12 / 0 |
| ArchTree, no gate | +0.38 | 2 / 19 / 0 |
| RE, no gate | +0.36 | 1 / 20 / 0 |
| local search, no gate | +0.16 | 0 / 21 / 0 |
| RE + same gate / LS + same gate / RE + ZC warm-up | +0.10 / −0.03 / −0.07 | 0 / 21 / 0 each |

**0 losses in 13 comparisons × 21 cells.** Average rank at equal GPU-hours (Friedman; blocks = dataset ×
budget k × seed; 1 = best; CD = Nemenyi critical difference):

| Benchmark | blocks | ArchTree v2 + gate | best method | CD | within CD of best? |
|---|---|---|---|---|---|
| NB201 (14 arms) | 108 | **tied-best**, 4.75 | ArchTree v2 4.75 ≈ RE + ZC warm-up 4.84 (Δ 0.09 ≪ CD) | 1.91 | best |
| NB101 (10 arms) | 36 | **2nd**, 4.38 | LS + same gate, 3.11 | 2.26 | yes |
| NATS-SSS (14 arms) | 108 | **4th**, 5.99 | LS + same gate, 4.19 | 1.91 | yes |

Full ranks: NB201 **v2 4.75**, RE+warm-up 4.84, v2+costpen(λ=0.25) 5.36, v2+#params 5.47, LS+gate 5.50, RE+gate 6.03,
LS 6.91, RE 8.04, v2 no gate 8.16, random+gate 9.07, random+costpen025 9.15, v2+costpen 9.58, random 10.20,
random+costpen 11.94. NB101 LS+gate 3.11, **v2 4.38**, RE+gate 4.56, LS 5.03, v2+#params 5.08, RE+warm-up 5.11,
RE 5.81, v2 no gate 5.88, v2+costpen 7.57, random 8.49. NATS-SSS LS+gate 4.19, RE+warm-up 5.45, LS 5.55,
**v2 5.99**, RE+gate 6.01, v2+#params 6.15, v2+costpen025 7.42, RE 7.46, random+gate 7.80, random+costpen025 7.94,
random 8.66, v2 no gate 8.74, random+costpen 11.74, v2+costpen 11.93. (Final, `-Plan final` Oct 4: all 1128 runs, NB201 108 blocks;
`outputs/final/results/stats_avg_ranks.csv`.)

**How to frame it (paper):** the **label-free gate is the main, portable contribution** - it improves every
proposer at equal GPU-hours (mean gain from the gate: random +0.17, RE +0.26, LS +0.19, ArchTree +0.38 pts) and
LS + gate is the strongest single arm on NB101 / NATS-SSS. ArchTree v2 is one strong way to use the gate
(never worse than any arm; tied-best on NB201, 2nd on NB101, 4th on NATS-SSS, always within the critical difference),
not "a search strategy that beats all others". Probability of improvement at equal GPU-hours: P(v2 > LS + gate) =
0.50 NB201, 0.22 [0.08, 0.36] NB101, 0.38 NATS-SSS - LS + gate is the method to beat outside NB201; say so.

**Gate-cost model:** predicted "cost penalty hurts" (slope < 0) on all 7 benchmark/dataset pairs; measured:
λ = 1 loses on 16 / 21 cells (0 wins). Predicted vs measured change: sign agreement 72 % (ArchTree) / 69 %
(random), Spearman +0.78 / +0.86 over 39 / 36 cells; the predicted changes are several times smaller than the
measured ones, so the model is used for the sign only. **Proxies are cost detectors in the cell spaces**
(label-free ensemble ρ with training time: NB201 0.78, NB101 0.63) **but not in the width space**
(NATS-SSS 0.20; #params 0.74 there) - yet the penalty still hurts on NATS-SSS because accuracy itself is
tied to cost there (r_DA = 0.75-0.89, slope −0.19 to −0.39). The criterion explains both regimes.
Known: 1 of 1128 runs failed on Oct 4 (WinError 32 on an unused checkpoint file, fixed) and was redone by
`-Plan final` - all 1128 runs complete; 5 runs ran out of proposals just before their budget (flagged `under_budget`).

Earlier offline NB201 numbers (same code, kept for reference):

Every method is compared at the same **training cost**, not the same number of evaluations:
budget H = k × (mean training time of one architecture), k = 25 / 50 / 100; reported = test accuracy of
the best-validation architecture found within H (`experiments/run_equal_gpu_hours.py`).

| Arm (NB201, k = 100) | cifar10 | cifar100 | ImageNet16-120 | vs ArchTree v2 + label-free gate (9 cells, Holm) |
|---|---|---|---|---|
| **ArchTree v2 + label-free ZC gate** | **94.28** | **73.40** | **46.41** | — |
| RE + same gate | 94.35 | 73.28 | 46.39 | 9 ties |
| ArchTree v2 + #params gate | 94.25 | 73.21 | 46.32 | 1 win, 8 ties |
| RE, no gate | 94.08 | 72.76 | 46.24 | **6 wins**, 3 ties |
| ArchTree v2, no gate | 94.09 | 72.84 | 45.91 | 2 wins, 7 ties |
| Random + same gate | 93.92 | 71.92 | 45.45 | 6 wins, 3 ties |
| Random, no gate | 93.82 | 71.54 | 45.53 | 7 wins, 2 ties |
| ArchTree v2 + **cost-penalised** gate | 93.67 | 71.85 | 45.05 | 6 wins, 3 ties |

No losses. Small budgets show the largest gap (k = 25, cifar100: 72.09 vs 70.77 for ungated RE).

**Proxy cost analysis** (`cost_correlation.csv`, NB201 cifar10, all 15,625 architectures):

| Score | ρ with training time | τ with valid acc | top-35% cost / mean | excess vs truly-best 35% |
|---|---|---|---|---|
| NASWOT (T) | 0.82 | 0.59 | 1.23 | +6% |
| SynFlow (S) | 0.75 | 0.55 | 1.21 | +4% |
| SVD entropy (E) | 0.44 | 0.55 | 1.13 | −3% |
| label-free ensemble | 0.78 | 0.64 | 1.22 | +5% |
| #params | 0.83 | 0.53 | 1.21 | +5% |
| validation accuracy (oracle) | 0.59 | 1.00 | 1.16 | 0 |

**Finding:** zero-cost proxies are largely cost detectors (NASWOT correlates with training time as much
as #params does), so a proxy gate raises the cost of each evaluation. But accuracy itself is tied to
cost (ρ ≈ 0.6), so only ~5% of that is waste; measured fairly in GPU-hours the gate still wins.
**Correcting for cost hurts:** a gate that penalises predicted cost (z(ensemble) − z(log #params)) loses
~1 point (6 of 9 cells significantly worse) - cheap architectures are cheap because they are weaker.
The #params gate comes close to the proxy ensemble, consistent with proxies ≈ cost.

### 5.6d LLM studies (Groq; complete Oct 3, 11:09; knowledge controls complete Oct 4, 12:53)

**llm_nas baseline (step 5):** 144/144 runs, ~12 calls and ~10.6k tokens per run, 0.37 fallback
architectures per run on average (0 Ollama fallbacks, 0 truncated replies).

| Budget 200, test acc | cifar10 | cifar100 | ImageNet16-120 |
|---|---|---|---|
| LLM-NAS | 94.28 | 72.14 | 45.37 |
| ArchTree v2 (label-free gate, ~40% of the evaluations) | 94.28 | 73.40 | 46.41 |

Strong on cifar10, weak on ImageNet16-120 (≈ random). Formal tests: Phase 6 (`-Plan final`).

**Convergence study (step 15, alpha = 0.3, 12 seeds, all 3 datasets) - the positive LLM result:**

| Dataset | evals to get within 1 pt of final (LLM on / off, p) | best after 10 evals (on / off, p) | final best |
|---|---|---|---|
| cifar10 | **5.0 / 15.5**, p = 0.008 | 91.04 / 90.38, p = 0.012 | tie |
| cifar100 | **15.6 / 30.8**, p = 0.009 | 70.75 / 70.21, p = 0.20 | tie |
| ImageNet16-120 | 23.2 / 17.8, p = 0.23 (n.s.) | 43.90 / 43.75, p = 0.91 | tie |

(validation accuracy = the search metric; budgets 100 and 200 agree.) → the LLM prior makes the search
reach near-final quality **2-3x faster on cifar10 and cifar100**, not on ImageNet16-120, and never changes
the final result.

**Alpha sweep (step 14, all 3 datasets, budgets 50/100, 12 seeds, 646 runs, 0 fallbacks):** 22 of 30
distinct comparisons favour the LLM prior (mean +0.16 test points; cifar100 b=50 up to +0.85), **none
significant after Holm**; ImageNet16-120 slightly negative. alpha ≥ 0.2 gives identical runs for every
seed (the prior saturates). `alpha_init` stays 0 for the headline (choosing it here = selecting on test).

**A4 alpha grid (step 8):** complete, 72/72, 0 fallbacks; alpha > 0 is +0.11-0.16 test points vs alpha = 0
on cifar10 (same picture as the sweep).

**LLM size (step 16, cifar10, budget 100, 12 seeds):**

| Model | prefix ranking ρ (chance 0) | top-1 hit (chance 0.20) | invalid op names / call | best after 25 evals vs off | final vs off |
|---|---|---|---|---|---|
| allam-2-7b (7B, other family) | 0.41 | 0.50 | 0.50 | 1 run only (dropped: unreliable answers) | – |
| gpt-oss-20b | 0.60 | 0.46 | 0 | **+0.28 pts, p = 0.027 (unadjusted; not Holm-corrected)** | +0.08 (tie) |
| gpt-oss-120b | 0.61 | 0.65 | 0 | +0.04 (tie) | −0.01 (tie) |

→ the bigger model ranks operations better, but that does not turn into a better search: prior
quality is not the bottleneck. The 7B model often returns wrong op names (half its calls).

**Knowledge controls (Oct 4, steps 15 + 24, all complete: 4 arms × 3 datasets × 2 budgets × 12 seeds, 0 fallbacks).**

*Contamination probe (step 24, gpt-oss-20b, 56 trie prefixes per dataset; ρ = Spearman of the predicted op
ranking with the true NB201 ranking, chance 0; top-1 chance 0.20; Holm over tests, all p_holm < 0.03):*

| Prompt | cifar10 ρ / top-1 | cifar100 ρ / top-1 | ImageNet16-120 ρ / top-1 |
|---|---|---|---|
| anonymous (op_A..op_E, scrambled, no meaning) | 0.14 / 0.11 | 0.19 / 0.11 | 0.18 / 0.18 |
| standard (NB201 op names) | 0.57 / 0.52 | 0.50 / 0.55 | 0.52 / 0.45 |
| described (what each op does, no NB201 names) | **0.70 / 0.57** | **0.64 / 0.52** | **0.66 / 0.54** |
| size prior (score ∝ op size, no LLM) | 0.76 / 0.80 | 0.75 / 0.86 | 0.73 / 0.73 |

→ Without meaningful op names the LLM is near chance; **describing the ops beats the exact NB201 names**
(+0.13-0.15 ρ, significant), which argues against memorised benchmark tables and for general
architecture knowledge. But the knowledge-free size prior ranks single ops best of all.

*Convergence with the controls (step 15, budget 100; budget 200 agrees; Holm within pair × metric):*

| | cifar10 | cifar100 | ImageNet16-120 |
|---|---|---|---|
| evals to within 1 pt: LLM / size prior / described / off | **5.0** / 16.4 / 7.8 / 15.5 | **15.6** / 19.7 / 25.4 / 30.8 | 23.2 / 31.3 / 19.4 / **17.8** |
| LLM vs off | −10.5, p_holm 0.047 | −15.2, p_holm 0.047 | +5.4, n.s. |
| LLM vs size prior | −11.4, p_holm 0.003 | −4.1, n.s. (p 0.057) | −8.1 (b=200: −14.0, p_holm 0.039) |
| size prior vs off | +0.9, n.s. | −11.1, p 0.034 (n.s. after Holm); auc30 +1.18, p_holm 0.015 | **+13.5 slower, p_holm 0.018** |
| described vs off | −7.8, n.s. (p 0.074) | −5.3, n.s. | auc30 **+1.14, p_holm 0.029** |
| final best | all tie | all tie | all tie |

→ **The LLM is not just "bigger is better":** the size prior ranks ops better than the LLM but searches
worse - no gain on cifar10, and significantly SLOWER than no prior on ImageNet16-120 (it over-commits to
large ops). The LLM prior beats the size prior on cifar10 (and ImageNet16-120 at b=200) and is the only
arm that is faster than "off" on cifar10 and cifar100. The described prompt is the only arm that helps on
ImageNet16-120 (area under the curve). Final accuracy is unchanged by any prior.

**Statistical caveat (found Oct 4, fixed in the analysis):** a deterministic prior (LLM / size prior) makes
the first proposals identical for every seed - e.g. LLM evals-to-1pt on cifar10 is 5 for all 12 seeds,
best-after-10 takes one value per arm. Comparisons where the paired difference is the same for every
seed (4 cells, e.g. LLM vs size prior best-after-10) are effectively n = 1 and are now reported without a
test (`test_note = constant_difference`); where one arm is constant the test is a one-sample test of the
other arm against that value (`one_deterministic`). Paper wording: "the LLM-guided early trajectory is
seed-independent; the p-values measure how often the seed-dependent baseline does as well".
`experiments/run_convergence_study.py --analyze-only` (in step 17) rewrites `convergence_paired_tests.csv`.

**Groq client fixes made during the run** (`utils/groq_client.py`):
* `json_validate_failed` (one off-enum value, e.g. "conv_3x1"): the answer in `failed_generation` is
  parsed and its valid items kept (INFO line "answer broke the schema in one value") - before, the whole
  batch became random mutations.
* `reasoning_effort` is sent only to gpt-oss models (others reject it); auto-dropped if a model rejects it.
* models without strict `json_schema` support fall back once to `json_object` (WARNING line, expected).
* step 16 skips a failing model instead of stopping all models.

### 5.7 What this means for the paper

* Supported: **deduplication + zero-cost gating cut real evaluations by ~57% (GPU-hours ~52%) at
  (near) equal accuracy, with zero labels** (label-free gate); hardware-aware search matches random / NSGA-II-style.
* Not supported (v1): ArchTree beating other search strategies, or the trie structure itself adding value.
  Regularized evolution is the strongest baseline and must be reported (it beats v1 after per-family Holm).
* Equal GPU-hours (§5.6c, device run, 3 benchmarks): ArchTree v2 + gate 0 losses vs 13 arms, within the
  critical difference of the best everywhere (tied-best NB201, 2nd NB101, 4th NATS-SSS); the gate helps every
  proposer. Proxies are cost detectors in cell spaces (ρ 0.78 / 0.63) but not in the width space (0.20); a
  cost penalty hurts everywhere (16 / 21 losses, 0 wins), in the direction the gate-cost model predicts
  (Spearman +0.78 / +0.86 predicted vs measured; magnitude under-predicted several-fold).
* v2 (§0, §5.3-5.4): **0 losses** vs every baseline; ties RE in accuracy at 19-23% fewer GPU-hours on the
  main seeds and 12-19% on held-out seeds; gate saves ~53% GPU-hours vs ungated v2.
* LLM prior (C7, §5.6d): 2-3x faster early convergence on cifar10/cifar100 (significant), none on
  ImageNet16-120, same final accuracy everywhere; bigger LLM ranks better but searches no better.
  Knowledge controls (Oct 4): near chance with anonymous op names, better with op descriptions than with
  NB201 names (not memorisation); a size prior ranks ops better but searches worse (slower than no prior
  on ImageNet16-120) - the LLM adds more than "bigger is better" for search.

### 5.8 Novelty — honest reviewer view

**As a new method: modest.** Every component exists: MCTS/UCB for NAS (AlphaX, LaNAS), zero-cost
proxies (Abdelfattah et al. 2021, ZiCo, NAS-Bench-Suite-Zero), proxy-filtered evolution (Abdelfattah et al.
2021 — closest prior work, must be cited), regularized evolution (Real et al. 2019), LLM-guided NAS
(GENIUS, EvoPrompting). v2 ties RE on accuracy; the ablation shows accuracy comes from the evolutionary
step and the trie/UCB mainly saves cost. Tabular benchmarks only; NB201 is saturated (0.2-pt gaps).

**Expected reviewer comments:** "combination of known parts", "does not beat RE", "tabular only",
"LLM part weak" (answered: knowledge controls, §5.6d). Strengths they will credit: 12 + 12 held-out seeds, paired
Wilcoxon + per-family Holm, equal-compute comparisons, GPU-hours accounting, honest negative results.

**Where the contribution is (frame as an empirical study, not "a new SOTA method"):**
1. Proxy gating biases search toward **more expensive architectures** (+18% training time per eval):
   in cell spaces (NB201, NB101) zero-cost proxies are largely cost detectors; in the width space (NATS-SSS)
   they are not. Yet at **equal GPU-hours** the gate wins on all three (§5.6c) and a cost-penalised gate is
   significantly worse everywhere. A **closed-form gate-cost criterion** (slope r_PA·r_PD − r_DA) predicts
   the sign in advance for all 7 benchmark/dataset pairs and ranks the measured effects well
   (Spearman +0.78 / +0.86), though it under-predicts their size several-fold - the most novel result of the paper.
2. A **label-free** gate saves ~52% GPU-hours at near-equal accuracy.
3. **What actually helps:** dedup + gating yes; tree structure little; LLM prior = faster early search on
   cifar10/cifar100 from general op knowledge (not memorisation, not just a size prior), no final gain.
4. **Reward-scale flaw of UCB in NAS:** raw accuracy rewards (~0.85-0.95) make exploration swamp
   exploitation; rank reward fixes it, validated on held-out seeds.

**Verdict:** seminar / student workshop — clearly fine; AutoML workshop or mid-tier venue — possible
with the empirical-study framing; top venue — would be rejected on novelty.

**To strengthen before writing:** accuracy-vs-GPU-hours plots for all methods as the main figure;
the expensive-architecture bias as a headline finding; NB101 to show the findings transfer;
explicit comparison to Abdelfattah et al.'s proxy-filtered evolution.

---

### 5.9 Final run (`-Plan final`, Oct 4, 13:04) - what the paper reports

* **Phase 6, external baselines (9: ENAS, LLM-NAS, local search, MCTS, random, RE, RE + ZC warm-up, RZ, ZiCo):**
  nominal budget 0 wins / 108 ties / 0 losses (Holm); equal real-evaluation compute (nearest and strict)
  0 / 108 / 0 (Holm; unadjusted 43 / 62 / 3 and 33 / 70 / 5). Internal comparators: 2 / 22 / 0 and 1 / 23 / 0.
* **Friedman + Nemenyi CD (`stats_friedman.csv`, `cd_*.png`):**

| Source | methods | blocks | Friedman p | CD | ArchTree v2 avg rank | best |
|---|---|---|---|---|---|---|
| NB201 full budget (v2 at ~40% evals) | 10 | 144 | 1e-76 | 1.13 | 4.41 | RE + ZC warm-up 3.67 |
| NB201 equal GPU-hours | 14 | 108 | 3e-83 | 1.91 | **4.75 (tied-best)** | ArchTree v2 (RE + ZC warm-up 4.84) |
| NB101 equal GPU-hours | 10 | 36 | 8e-15 | 2.26 | 4.38 (2nd) | LS + gate 3.11 |
| NATS-SSS equal GPU-hours | 14 | 108 | 6e-81 | 1.91 | 5.99 (4th) | LS + gate 4.19 |
| NB101 same budget, label-free gate | 3 | 48 | 2e-8 | 0.48 | 1.73 | RE 1.58 |

* **Probability of improvement P(v2 > X) [95% CI], equal GPU-hours:** NB201 vs random 0.90, RE 0.79, ungated v2
  0.76, cost-penalised v2 0.85, RE + warm-up 0.51, LS + gate 0.50; NB101 vs random 0.92, LS + gate 0.22 [0.08, 0.36];
  NATS-SSS vs random 0.72, RE + warm-up 0.40, LS + gate 0.38. Full budget NB201: vs MCTS 0.85, random 0.81,
  ENAS 0.77, LLM-NAS 0.75, RE 0.57, LS 0.43, RE + warm-up 0.41.
* **TOST:** equivalence within ±0.5 pts is shown in a minority of cells (e.g. NB201 equal GPU-hours vs RE + gate
  4 / 9, vs LS + gate 4 / 9) - wording stays "no significant difference", not "equivalent".
* **Convergence re-test (step 17):** 8 comparisons not tested (paired difference identical for every seed);
  the LLM-vs-off results on cifar10 / cifar100 are unchanged (§5.6d).
* **Figures:** `plots\figures\` - 50 figures (Phases 3-4, ablations A1-A7, proxy ablation, convergence, LLM size,
  contamination, equal GPU-hours, gate-cost model check, CD diagrams, regret, cross-device, F1-F7), each with a
  CSV of the exact plotted points.
* **Reproducibility:** `release\archtree_repro_bundle.zip` (87 files: code, config, seeds, checksums, environment,
  appendix .tex); checked: no `.env`, no API keys inside.

## 6. How to run (execution sequence from Oct 3)

Environment (Oct 4, 2026; paper runs: Python 3.10.9, torch 2.5.1 + CUDA 12.4, RTX 2050 4 GB, Windows 11).
Exact package versions of the runs: `outputs\final\repro\requirements.lock.txt`.

| File | What it is |
|---|---|
| `setup_env.ps1` (new, Windows) | one-time setup: uses the active conda env / venv (or creates `.\venv`), installs GPU torch (`-Cuda cu124`), `requirements.txt`, verifies packages + CUDA, checks data / `third_party` / `.env` (presence only, keys never printed), runs `tools\preflight.py`. `-CheckOnly` = verify only. |
| `setup_env.sh` | the same for Linux / macOS |
| `requirements.txt` | only what the code imports (torch, numpy, pandas, scipy ≥ 1.11, matplotlib, pyyaml, python-dotenv, loguru, psutil, nas-bench-201, nats_bench, groq ≥ 1.0; optional ollama, flask). No TensorFlow / nasbench, streamlit, seaborn. |
| `config.yaml` | all hyperparameters; values unchanged since the paper runs, comments brought up to date, unused keys marked "informational" |
| `.env` | `GROQ_API_KEYS=key1,key2,...` (or `GROQ_API_KEY=`) - never committed, never in the bundle |
| `tools\preflight.py` | now checks steps 0-25 (NATS-SSS cache / source, nats_bench, groq, matplotlib, proxy cache) |
| `command.txt` | the command sequence (all done) |
| `data/README.md` | every dataset / benchmark: what it is, download links, how each cache is built, current folder contents, SHA-256 checksums, licences and citations |
| `third_party/README.md` | HW-NAS-Bench (the only vendored code): install, contents, how `benchmarks/hwnb.py` uses it (index map, unmeasured pairs), checksum, licence |

Setup (every new PowerShell window):
```powershell
cd "C:\Users\Iyaad Hajwane\Desktop\archtree-llm V4"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
conda activate cuda            # the env used for the paper runs (or .\venv\Scripts\activate)
.\setup_env.ps1 -CheckOnly    # optional: verify packages, GPU, data, keys
```

| # | Command | What it does | Time / cost |
|---|---|---|---|
| 1 | `.\run_full_pipeline.ps1 -Plan core -ArchivePhases phase0,phase1,phase2,phase3,phase4,ablations,ucb_sensitivity,gated_baselines,proxy_ablation` | archives the v1 results, reruns all LLM-free NB201 experiments with v2 | **done Oct 3** |
| 2 | `.\run_full_pipeline.ps1 -Plan nb101` | NAS-Bench-101: v2 vs baselines, same gates | **done Oct 3** |
| 3 | `.\run_full_pipeline.ps1 -Plan groq` | LLM prior (C7) and llm_nas baseline | **done Oct 3** |
| 4 (**done**) | download `NATS-sss-v1_0-50262-simple.tar` (1.01 GB, NATS-Bench Google Drive, link in https://github.com/D-X-Y/NATS-Bench), then `tar -xf <path>\NATS-sss-v1_0-50262-simple.tar -C data` -> `data\NATS-sss-v1_0-50262-simple\` | NATS-Bench size-space data (low RAM; the 874 MB `.pickle.pbz2` also works but is unpacked into RAM, ~10 GB+) | once |
| 5 (**done Oct 3**) | `.\run_full_pipeline.ps1 -Plan extend` | step 4: local search + RE warm-up (only the new methods run), step 12: local search under the same gates, step 23: build `data\natssss_cache.json` | ~15-30 min, no Groq |
| 6 (**done Oct 4**: step 25 ~5 h, step 22 25 min, 1127 / 1128 runs) | `.\run_full_pipeline.ps1 -Only 25,22` | step 25: proxies for all NATS-SSS / NB101-sample archs in 3 parallel GPU workers; step 22: equal GPU-hours on NB201 + NB101 + NATS-SSS with 3 parallel workers, cost analysis, gate-cost model | ~40-90 min, GPU, no Groq |
| 7 (**done Oct 4, 12:53**, 12 min) | `.\run_full_pipeline.ps1 -Plan groq` | step 15: size-prior + described-prompt arms, step 24: contamination probe, fills any gaps (A4, sweep) | ~30-60 min, ~0.3-0.5M tokens |
| 8 (**done Oct 4, 13:04**, ~7 min) | `.\run_full_pipeline.ps1 -Plan final` | steps 25 + 22 (finished work skipped; redoes the 1 failed step-22 run in seconds), 17 convergence re-test + Phase 6 (9 external baselines), 18 analyses + Friedman / CD / TOST / P(improvement), 19 repro bundle | ~15-30 min |
| 9 | `Remove-Item -Recurse -Force "outputs_archive\20261002_1642"` | delete superseded v1 supervised archive (any time) | – |
| 10 | `Remove-Item -Recurse -Force "outputs_archive\20261003_0828"` | delete v1 label-free archive (after the paper is submitted) | – |
| 11 (**next**) | paper writing | last step; figures in `plots\figures\` (50 figures, each with a data-point CSV), tables in `outputs\final\results\` | – |
| 12 (while writing) | `.\run_full_pipeline.ps1 -Plan report` | steps 17 + 18 + 19 only: rebuild tables, statistics, all figures (`plots\figures`) and the repro bundle from the existing results | ~7 min, no Groq |

Check results after every step before starting the next. `-DryRun` previews any command.
Never rerun step 1 of the pipeline (data download). Finished runs are skipped, so after a config
change move the affected phases aside with `-ArchivePhases`.
Dashboard: `python frontend\backend.py` -> http://localhost:5000

| Results | Where |
|---|---|
| per experiment | `outputs\<phase>\results\*.csv`, logs in `outputs\<phase>\logs\` |
| final tables | `outputs\final\results\` |
| pipeline output | `pipeline_logs\` |
| archived reruns | `outputs_archive\` |
| invalid Sep-2026 results | `outputs_pre_oct2026_20261002_1215\` |

---

## 7. Limitations (state in the paper)

* Tabular benchmarks only: nothing is trained; weight inheritance is simulated.
* NB101: cifar10 only, weaker proxies (τ 0.50), no LLM prior (prompt is NB201-specific), no hardware part.
* The proxy network always uses CIFAR-10 images, also for cifar100 / ImageNet16-120.
* Deterministic accuracies (mean over the benchmark's repeats), so no evaluation noise.
* edgetpu: ~30% of architectures have no latency in HW-NAS-Bench.
* LLM answers are memoised per (model, prompt) across seeds (`llm.shared_cache`).
* The LLM may know NAS-Bench-201 from its training data (possible contamination of the LLM prior).
* LLM-size study limited to models available on Groq (Oct 2026); allam-2-7b is a different family.
* The gate-cost model is a model of gated RANDOM search; it predicts the direction of the cost-penalty
  effect for ArchTree but underestimates its size several-fold (claim the sign only).
* NATS-SSS proxies use a proxy-size port of the SSS network (one cell per stage, CIFAR-10 images).
* 12 seeds are enough to detect differences of ~0.3 points but not to PROVE equivalence within ±0.5
  points in most cells (TOST): claims are "no significant difference".
* v2 design was chosen on the main seeds; only the held-out design ablation is an unbiased test of it.
* GPU-hours are the benchmark's recorded training times (NB201 200-epoch, NB101 108-epoch), not measured.

## 8. Next

1. Execution sequence in §6 (download NATS-SSS -> extend -> step 22 -> groq -> final), checking results after each.
2. Delete `outputs_archive\20261002_1642\` now; `outputs_archive\20261003_0828\` after submission.
3. `ReadMe/goalREADME.md` updated Oct 4 (goals C1-C9 + new goals, each with its final outcome and status).
4. Paper writing (last step), using the framing in §5.8.
