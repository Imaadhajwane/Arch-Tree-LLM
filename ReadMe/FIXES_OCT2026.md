# ArchTree-LLM — Oct 2, 2026 audit: what was wrong, what changed

Every Sep-2026 number is superseded. The fixes below were tested end-to-end on a CPU copy of the
project with stub data (all runners, Phase 6, analysis tools, backend import, storage unit test).
Key checks that passed:

* `hwnb.py` index map: Spearman(latency, #conv of the same arch) = **0.624** (was 0.001).
* `search_phase2(backup=max)` == `search_phase3(alpha=0)`: **12/12** seeds identical (one UCB scale).
* zero-information LLM (alpha 0.3, no calls) == LLM off: **12/12** seeds identical (prior bias fixed;
  before the fix it reproduced the "LLM slows convergence" effect).
* A1 ablation == Phase 3 headline: **12/12** rows identical (ablations now use the headline config).
* `prune(0)` prunes nothing; `tools/test_storage.py` 28/28 pass.

## 1. Critical data / code fixes

| # | Problem | Fix (files) |
|---|---|---|
| 1 | HW-NAS-Bench indexed by NB201's original order; code passed the base-5 index -> every (accuracy, latency) pair mixed two architectures (15,624/15,625 indices differ). Phase 4, true fronts, cross-device, TCR-H, HV, IGD invalid. | `benchmarks/hwnb.py` builds a canonical->HW-NAS-Bench map from each entry's `arch_str` and checks it is a bijection; `check_alignment()` sanity test. `tools/build_igd_reference_front.py` now overwrites old fronts and prints the alignment check. `tools/analyze_cross_device_pareto.py` ignores old `hw_table__*.npz` (version tag). |
| 2 | Accuracy cache = 12-epoch schedule (`hp='12'` default) and cifar10 = TRAINING accuracy (`x-test`/`test` don't exist for `cifar10`, fallback hit `train`). | `tools/build_nb201_cache.py` v2: 200 epochs; per dataset `valid` (search: cifar10-valid/x-valid, cifar100 & IN16 x-valid), `test` (report: cifar10 ori-test, x-test), `train_sec`; sanity checks (all-`none` must be 10.00 on cifar10 test). `benchmarks/nb201.py`: `query()` = search metric, `query_test()` = report metric, `train_sec()`, `best_from_history()`; sampling without replacement. Config `paths.nb201_cache` -> `data/nb201_cache_v2.json`. |
| 3 | LLM prior bias: proxy-skipped-only children (visits>0, no accuracy) were "cold" and got a default prior 0.5 vs ~1/k for unvisited siblings -> every alpha>0 re-walked proxy-rejected branches. Produced the C7 "slower convergence" result. | `archtree/ucb.py`: a child is cold only if never visited; visited-without-accuracy children are scored like plain UCB1 (value 0 + bonus); neutral prior = 1/k. |
| 4 | Results came from raw 0-100 UCB in phase 3/4; source changed to /100 on Sep 30 after the runs; `search_flat_ucb` still 0-100. | One scale (accuracy/100) in `ucb.py` and `search_flat_ucb`. Rerun everything. |
| 5 | `run_ablations.py` passed no gate pct (silent 50, headline 35) and no alpha (default 0.3 -> LLM ON in A2/A3/A5/A6). | Ablations use `alpha_init` and the config gate; rows carry `best_test_acc`, `llm_fallbacks`. Every runner now defaults the gate to 35. |
| 6 | Silent LLM fallback: exhausted keys / errors returned uniform scores that were cached and counted as calls (64/72 convergence LLM-on runs; LLM-size probe 100%). | `archtree/llm_prior.py`: `fallback_count`, failures never cached, no more calls once the pool is exhausted, counters reset per run (old checkpoints shrank the call budget). A4, alpha sweep, convergence and LLM-size studies discard / stop on any fallback. |

## 2. Other code changes

| Item | Change (files) |
|---|---|
| Constants ignored from config (`ucb.C`, `trie.*`, `hw.*`) | New `archtree/settings.py`; `search.py`, `ucb.py`, `trie.py` read it. `config.yaml` cleaned and documented. |
| Pruning removed least-visited (= unexplored) nodes; pct=0 still pruned | `archtree/trie.py`: value rule (lowest backed-up accuracy among nodes with >= 2 visits, never the last sibling), pct=0 = none; legacy rule kept as `prune_rule: visits`. |
| Proxy calibration used 1,500 true accuracies (uncounted) | `archtree/zc_proxy.py`: `calibration_mode` supervised / unsupervised (label-free z-score, E=S=T=1/3, C=0); `labels_used` saved with the calibration; Phase 2 builds both; storage helper `_zc_cal_path` (legacy cifar10-for-all fallback removed). Raw cache path is now absolute. |
| Report metric | Every search result row/log has `best_test_acc`; `storage.to_report_metric()` / `load_report_csv()` make tables, tests, figures and the dashboard use test accuracy. |
| Weak / mislabelled baselines | `rz_nas.py`: no paid re-evaluation of rejected mutations (was 82% duplicates), label-free proxy, renamed "ZC mutation walk (rz-style)". `enas_proxy.py`: real per-position policy (old sampler was random search). `zico.py`: pool without replacement, label-free proxy, "ZiCo-style". `mcts_nas.py`: "MCTS-style". Phase 0 rows add `best_test_acc`, `proxy_evals`. |
| Phase 6 statistics | External baselines only for the headline (internal variants separate); win / tie / loss columns; Holm within family; 95% bootstrap CIs; strict equal-compute matched on the MAX real evals over seeds (not the mean); gated-baseline tests folded into the extended summary. |
| Phase 4 | eval history logged; prune pct from config; method tag. |
| Repro bundle | effective constants read from `archtree/settings.py`; bundle now contains the full source (never `.env`). |
| Housekeeping | `.gitignore` (was named `gitignore`); requirements add flask, flask-cors, loguru, psutil; `tools/test_storage.py` path + stale tests fixed; `run_phase0.py` needs only the cache, not the 5 GB .pth; walkthrough reads the v2 cache; `backend.py` OUR_METHOD = archtree_llm, claims marked "pending rerun", live demo C now 1.414 via config; `index.html` stale "95 of 96" / "1 of 15,625" text replaced. |

## 3. New experiment / tool files

| File | Reviewer item |
|---|---|
| `experiments/run_gated_baselines.py` | A1: random / ENAS / MCTS with the same gate (calibrated zc, label-free zc, #params), Holm + CIs |
| `experiments/run_phase4_baselines.py` | B9: random and NSGA-II-style hardware search, paired HV / IGD tests vs ArchTree |
| `experiments/run_ucb_sensitivity.py` | A5: C x backup grid vs the default |
| `tools/analyze_proxy_cost.py` | A7: proxy time (measured on GPU) + calibration labels -> cost in real-eval equivalents |
| `tools/archive_legacy.ps1` | moves (never deletes) stale files out of the way |
| `ReadMe/RUN_PLAN_OCT2026.md` | execution order, timings, decisions |

## 4. Reviewer checklist — where each item stands

| Item | Status |
|---|---|
| A1 same-gate baselines | code ready (`run_gated_baselines.py`) — run step 12 |
| A2 literature check | **you**: not code. Read Vo et al., TNAS, LLM-NAS (2510.01472), ZAPS, CoRA-NAS, RZ-NAS (note: our `rz_nas` is not RZ-NAS) |
| A3 headline wins/ties | code ready (Phase 6 prints wins / ties / losses per baseline, Holm) — the 71/72 number must be re-derived |
| A4 headline name | done in code: `archtree_llm`, present as "ArchTree (LLM prior off)" |
| A5 UCB scale + C | done (one scale) + `run_ucb_sensitivity.py` |
| A6 baseline audit | done (rz_nas, enas, zico fixed and renamed "-style"); cite originals in the paper |
| A7 proxy cost | code ready (`analyze_proxy_cost.py`), includes the 1,500 calibration labels |
| B8 NAS-Bench-101 | **done in code (Oct 2026, §6)**: TensorFlow-free NB101, bench-generic search, NB101 proxy net, regularized-evolution baseline, `run_nb101.py` — run steps 20-21 |
| B9 HW baselines | code ready (`run_phase4_baselines.py`) |
| B10 TCR-H reframe | writing; cross-device numbers must be recomputed first (fix #1) |
| B11 Holm + CIs | done in Phase 6 and every new script |
| B12 ensemble vs best single proxy | done (`run_proxy_ablation.py` prints best single vs ensemble with Holm) |
| B13 theory | done in `ReadMe/REGRET_ARGUMENT.md` §5b (T2 by construction, T4 false as stated, T3/T6 informal, T7 definitional) |
| C14 narrow LLM claim | re-test C7 first (fix #3); write "this form of prior", not "LLMs" |
| C15 llm_nas contrast | writing (llm_nas is the strongest baseline; its fallback rate was ~1%) |
| C16 LLM-size probe | code fixed; needs a fresh Groq budget |
| D17-D20 writing | after the rerun. Seed list in config is correct (12 unique); the duplicate 27182 is only in the progress roadmap text |

## 5. Limitations that stay (state them in the paper)
* Tabular benchmark only (NB201 + HW-NAS-Bench): nothing is trained; weight inheritance is simulated.
* NB101: cifar10 only, no hardware latencies, LLM prior not run (its prompt is NB201-specific).
* The proxy network always uses CIFAR-10 images and a 10-class head, also for cifar100 / ImageNet16-120.
* Deterministic accuracy (mean over NB201 seeds), so no evaluation noise.
* Supervised calibration consumes 1,500 labels per dataset — report the cost or use `calibration_mode: unsupervised`.
* edgetpu: ~30% of architectures have no latency in HW-NAS-Bench (with the correct index map).

## 6. NAS-Bench-101 (added Oct 2026, second session)

Why it was blocked: the official `nasbench` API needs TensorFlow 1.x. Now nothing from TF is used.

| File | What |
|---|---|
| `tools/build_nb101_cache.py` (new) | reads `data/nasbench_only108.tfrecord` with a pure-Python TFRecord + protobuf reader, writes `data/nb101_cache.npz` (hash, pruned adjacency, ops, params, 3 repeats of valid / test / train time). Refuses to write unless all 423,624 hashes are reproduced by our hash port and every graph has 3 repeats |
| `benchmarks/nb101.py` (rewritten) | line-by-line ports of `ModelSpec._prune`, `graph_util.hash_module`, the validity check, and the NB101 evolution mutation. Checked against the official code on 20,000 random specs (identical). Tokens = 5 interior ops then 21 adjacency bits; `tokens_to_index` returns None for invalid cells; search on mean valid acc, report mean test acc |
| `archtree/ucb.py`, `trie.py`, `search.py` | per-position vocabulary (`vocab_at`), `trie.mark_dead` for invalid leaves, `_fix_proposal`: invalid cells and isomorphic duplicates are redirected to a random unevaluated graph (no real eval spent). **NB201 code path unchanged: 6/6 regression runs bit-identical** |
| `archtree/zc_proxy.py` | `NB101ProxyNet` (proxy-size port of the NB101 network, official cell builder incl. channel split); cache key = pruned-graph tokens |
| `baselines/regularized_evolution.py` (new) | aging evolution (Real et al. 2019), standard NB101/NB201 baseline; also added as proposer `re` to `run_gated_baselines.py` (NB201 step 12) |
| `experiments/run_nb101.py` (new) | archtree / random / RE × gates none / params / zc_unsup / zc, budgets 50-400, 12 seeds; paired tests + equal-real-eval tests (Holm, bootstrap CI) |
| `tools/test_nb101.py` (new) | unit tests (no data) + cache tests (after step 20) |
| `run_full_pipeline.ps1` | step 20 (build cache + tests), step 21 (experiment) |

## 7. Groq token efficiency (Oct 2, afternoon)

| Change | Why it saves tokens / runs |
|---|---|
| `utils/groq_client.py` (new) `GroqChat`: every Groq call (LLM prior + llm_nas) goes through it | one tested code path |
| strict `json_schema` output (gpt-oss-20b supports constrained decoding on Groq), auto-downgrade to `json_object` | no unparseable answers = no wasted calls, no fallbacks that invalidate a whole run |
| **prompt bug fixed**: the prior prompt showed example scores `{"none":0.7,...}`; the Sep-2026 caches show the model copied them for every prefix (prior always favoured `none`, the worst op) | the C7 re-test now measures a real LLM opinion; integer 0-9 scores, schema-enforced |
| per-key per-minute pacing on requests AND tokens (`MinuteLimiter`, config `llm.groq.tpm_per_key: 7000`) | the 8K TPM free-tier cap binds before 30 RPM; pacing avoids 429s |
| 429 by kind: per-minute -> wait `retry-after`; per-DAY -> key skipped until reset, next key used at once | before, a dead key was retried 3x and the call failed although other keys had budget |
| SDK retries off (`max_retries=0`) | no hidden retries on a dead key |
| shared answer cache across seeds / runs / alphas (`llm.shared_cache: true`, `outputs/_llm_shared_cache.json`) | the same (model, prompt) is asked once; mention in the paper ("LLM answers memoised per prompt") |
| no call when only one child is unvisited | the prior cannot change that choice |
| single key = 1-key pool (no 24 h blocking sleep), fixed instructions first (Groq prompt caching), `max_tokens` sized to the answer | |
| token / truncation / 429 counts logged (`llm_tokens`, `truncated_replies` in cache-stats and llm_nas diagnostics) | cost can be reported |
Tested offline with a fake Groq SDK: schema path, downgrade path, minute 429 + retry-after, day 429 -> rotation -> all-exhausted fallback (logged once).


---

## Oct 3, 2026 — audit, statistics, cost, ArchTree v2

| # | Change | Files |
|---|---|---|
| 1 | Holm correction per family (gate × baseline) instead of over all tests at once; with it RE beats v1 in 6/12 (label-free gate) | `run_gated_baselines.py`, `run_nb101.py` |
| 2 | GPU-hours: `train_hours` (benchmark training time of the real evaluations) in every result row; new aggregator | `archtree/search.py`, `tools/analyze_train_cost.py` |
| 3 | Rank-normalised UCB reward (`ucb.reward: rank`), C = 0.3 | `archtree/ucb.py`, `archtree/settings.py`, `config.yaml` |
| 4 | Evolutionary expansion: after 10 evals, p = 0.5 of proposals mutate the best of 5 sampled evaluated archs | `archtree/search.py` (phase3, flat UCB) |
| 5 | Regularized evolution added to Phase 0 (7th external baseline) | `baselines/regularized_evolution.py`, `run_phase0.py`, `run_phase6_aggregate.py` |
| 6 | Design ablation v1 / rank / v2 / mutation-only / RE on 12 held-out seeds | `experiments/run_design_ablation.py` (step 11), `search.heldout_seeds` |
| 7 | UCB C grid includes 0.3, default = config C | `run_ucb_sensitivity.py` |

Regression: v1 settings reproduce the device Phase 1/2/3 and flat-UCB runs exactly; RE reimplementation
matches the gated-baseline RE runs 54/54. Results: RESULTS_LOG_OCT2026.md §0.

### Oct 3, 2026 (later) — equal-GPU-hour protocol, proxy cost analysis, Groq schema salvage

| # | Change | Files |
|---|---|---|
| 8 | Groq 400 `json_validate_failed` (one off-enum value, e.g. "conv_3x1"): the answer in `failed_generation` is parsed and its valid items kept, instead of replacing the whole batch with mutations | `utils/groq_client.py` |
| 9 | Equal-GPU-hour comparison: each run's history cut at k x mean architecture training time (k = 25/50/100), NB201 + NB101, 8 arms incl. RE / random with and without the gate, #params gate, cost-penalised gate | `experiments/run_equal_gpu_hours.py` (pipeline step 22, in `-Plan final`), Phase 6 extended source |
| 10 | Proxy cost analysis: rho(proxy, training time), tau(proxy, valid acc), cost of the top-35% vs the truly best 35% | same script, `cost_correlation.csv` |
| 11 | Cost-penalised gate tested and rejected (negative result, kept as an arm) | same script |

Result (NB201, offline run of the same code): ArchTree v2 + label-free gate 0 losses at equal GPU-hours;
beats ungated RE 6/9, random 7/9, the cost-penalised gate 6/9; ties gated RE 9/9. RESULTS_LOG_OCT2026.md §5.6c.

### Oct 3, 2026 (Groq run) — LLM-study fixes

| # | Change | Files |
|---|---|---|
| 12 | `reasoning_effort` only for gpt-oss models; auto-dropped if a model rejects it | `utils/groq_client.py` |
| 13 | `llama-3.1-8b-instant` retired by Groq (404): step 16 models = gpt-oss-20b, gpt-oss-120b, allam-2-7b; new `tools/list_groq_models.py` | `run_full_pipeline.ps1`, `tools/list_groq_models.py` |
| 14 | Step 16 skips a failing model instead of stopping every model | `experiments/run_llm_size_sensitivity.py` |
| 15 | Alpha sweep on all 3 datasets (was cifar10 only); analysis with Holm, identical-run grouping (alpha >= 0.2 saturates), no "set alpha_init" advice | `experiments/run_alpha_sweep.py`, `experiments/analyze_alpha_sweep.py` |

Results so far: RESULTS_LOG_OCT2026.md §5.6d.

### Oct 3, 2026 (afternoon) — additions for the paper (novelty / reviewer-proofing)

| # | Change | Files |
|---|---|---|
| 16 | Knowledge-free size prior (backend `heuristic`, op score ~ #params) - control for the LLM prior | `archtree/llm_prior.py`, `archtree/search.py` |
| 17 | Prompt modes `described` (no benchmark / op names; labels + descriptions) and `anonymous` (labels only, negative control) - contamination test; the `standard` prompt is byte-identical, so the shared answer cache stays valid | `archtree/llm_prior.py` |
| 18 | Convergence study: 4 arms (llm_off, llm_on, size_prior, llm_described), 5 paired comparisons, Holm per pair x metric | `experiments/run_convergence_study.py` |
| 19 | Contamination probe (op-ranking quality per prompt mode, vs chance, Holm) - new step 24 | `experiments/run_llm_contamination.py` |
| 20 | Local search baseline (White et al. 2021) in Phase 0, the same-gate study and equal GPU-hours | `baselines/local_search.py`, `run_phase0.py`, `run_gated_baselines.py` |
| 21 | RE + zero-cost warm-up (Abdelfattah et al. 2021) in Phase 0 and equal GPU-hours; plain RE unchanged (reproduces the device runs) | `baselines/regularized_evolution.py`, `run_phase0.py` |
| 22 | NATS-Bench size space as 3rd benchmark: cache builder (nats_bench API, hp=90, valid/test split as for NB201), benchmark class, SSS proxy network | `tools/build_natsbench_sss_cache.py`, `benchmarks/natssss.py`, `archtree/zc_proxy.py`, `config.yaml` (paths) - new step 23 |
| 23 | Gate-cost model: Gaussian-copula fit, closed-form criterion r_PA r_PD - r_DA, simulated delta of a cost-penalised gate, predicted vs measured | `tools/gate_cost_model.py`, `experiments/run_equal_gpu_hours.py` (v2: 14 arms, 3 benchmarks) |
| 24 | Pipeline: plan `extend` (4, 12, 23), step 24 in `groq`; step 16 search without allam-2-7b | `run_full_pipeline.ps1` |

Offline checks: standard prompt == cached prompts; RE reproduces device runs; local search never
re-evaluates; NATS-SSS builder tested with a synthetic API (same method names / keys as nats_bench);
gate rule in the model identical to np.percentile; NB201 equal-GPU-hours (14 arms) end to end.

### Oct 4, 2026 — speed-ups (identical results) and new statistics

| # | Change | Files |
|---|---|---|
| 25 | Per-iteration TCR rows buffered, written in blocks (byte-identical CSVs) | `archtree/storage.py`, `archtree/search.py` |
| 26 | Proxy cache: flush at most every 2 min; merge-on-write under a file lock (multi-process safe) | `archtree/zc_proxy.py` |
| 27 | Parallel proxy pre-computation (GPU workers) - new step 25 | `tools/precompute_proxies.py` |
| 28 | Step 22 with parallel worker processes (`runtime.workers`), identical to sequential | `experiments/run_equal_gpu_hours.py`, `config.yaml` (`runtime`) |
| 29 | Friedman + Nemenyi CD diagram, TOST equivalence, probability of improvement with bootstrap CI - in step 18 | `tools/analyze_rank_equivalence.py`, `run_full_pipeline.ps1` |
| 30 | `plot_results.py all` in step 18; convergence plot rewritten for 4 arms; new equal-GPU-hours, gate-model, proxy-cost and contamination figures; every figure also copied to `plots/figures/` with a CSV of its data points | `plots/plot_results.py`, `plots/figure_export.py`, 6 scripts with figures, `run_full_pipeline.ps1` |
| 31 | Step 22: unused LLM-prior checkpoint writes switched off (two arms of one seed in parallel workers collided on Windows, WinError 32, 1 of 1128 runs) | `experiments/run_equal_gpu_hours.py` |
| 32 | Convergence tests: comparisons whose paired difference is identical for every seed (deterministic prior -> same first proposals; n = 1 in effect) are no longer tested; `test_note`, `n_unique_a/b` columns; `--analyze-only` re-run added to step 17 | `experiments/run_convergence_study.py`, `run_full_pipeline.ps1` |
| 33 | Figure data export: masked-point warnings silenced (NaN cells written as empty) | `plots/figure_export.py` |
| 34 | Environment brought up to date: `requirements.txt` = actual imports (+ nats_bench, groq >= 1.0; - TF/nasbench, streamlit, seaborn), new `setup_env.ps1` (Windows), `setup_env.sh` rewritten, `config.yaml` comments updated (values unchanged, unused keys marked), preflight covers steps 0-25, `-Plan report` (17-19), repro bundle also ships setup scripts + README | `requirements.txt`, `setup_env.ps1`, `setup_env.sh`, `config.yaml`, `tools/preflight.py`, `tools/build_repro_appendix.py`, `run_full_pipeline.ps1`, `command.txt` |
| 35 | `data/README.md` rewritten (old one listed wrong files: nasbench_full.tfrecord, TF nasbench, data/ HW pickle) and new `third_party/README.md`; `.gitignore` also excludes `data/natssss_cache.json` and the NATS raw download | `data/README.md`, `third_party/README.md`, `.gitignore` |
| 36 | New master `README.md` (GitHub front page: summary, objectives, corrected pipeline diagram, benchmarks, baselines, protocol, statistics, results, install, steps 0-25, outputs, structure, reproducibility, limitations, citation); the old working README moved to `ReadMe/RESULTS_LOG_OCT2026.md`; pipeline diagram also as `ReadMe/assets/archtree_pipeline.png/.svg` | `README.md`, `ReadMe/RESULTS_LOG_OCT2026.md`, `ReadMe/assets/` |

Checks: TCR / results CSVs before vs after identical; 1 vs 3 workers 252/252 identical runs and identical
to the previous version; 4 proxy workers lose 0 cache entries (before the lock: 10 of 30 lost).
