# Run plan after the Oct-2026 fixes

Run it in 5 blocks with `run_full_pipeline.ps1 -Only 0,20`, `-Plan core`, `-Plan nb101`, `-Plan groq`, `-Plan final`
(full walk-through: `COMPLETE_GUIDE_OCT2026.md`). `-DryRun` previews, `-Only 7,8` picks steps, `-ArchivePhases` moves
finished phases aside before a rerun, logs go to `pipeline_logs\`.
Below is the same sequence with what each step needs and what to check before moving on.
What changed and why: `ReadMe/FIXES_OCT2026.md`.

## Before step 0 — once
1. `venv\Scripts\activate`, then `pip install -r requirements.txt` (adds flask, flask-cors, loguru, psutil).
2. Optional clean-up: `.\tools\archive_legacy.ps1` (moves stale files to `_archive_pre_oct2026\`, deletes nothing).
3. Keep `.env` (Groq keys) out of anything you zip or share.

## The sequence

| Step | Command | Needs | Check before continuing |
|---|---|---|---|
| 0 | `python tools/test_storage.py` | – | "ALL TESTS PASSED" |
| 1 | (pipeline step) move `outputs\` and old `data\true_pareto_front__*`, `data\hw_table__*` aside | YES prompt | new empty `outputs\` with only `_zc_raw_cache.json` |
| 2 | `python tools/build_nb201_cache.py` | `data\NAS-Bench-201-v1_1-096897.pth`, 10-16 GB RAM, ~10-30 min | prints all-'none' cifar10 test = 10.00 and best ≈ 94.4; `data\nb201_cache_v2.json` exists |
| 3 | `python tools/build_igd_reference_front.py` | HW-NAS-Bench repo | alignment check ≈ 0.6 (not ≈ 0) |
| 4 | `python experiments/run_phase0.py` | GPU (zico / rz_nas proxy) | 8 methods (incl. regularized evolution, local search, RE + zero-cost warm-up) x 3 datasets x 12 seeds x 4 budgets, 0 errors |
| 5 | `python experiments/run_phase0_llm_nas_only.py` | Groq pool (~1,700 calls) | 144 rows; `llm_nas_call_diagnostics.csv` fallback_archs small |
| 6 | `python experiments/run_phase1.py` | – | 144 runs |
| 7 | `python experiments/run_phase2.py` | GPU (slowest step) | both calibrations per dataset exist (`..._unsupervised.pt` too), tau ≈ 0.4 |
| 8 | `python experiments/run_phase3.py` (answer 2 = Groq) | alpha=0: no LLM calls in the main sweep; A4 grid uses Groq | 144 main rows; A4 rows only where llm_fallbacks = 0 (rerun later to fill) |
| 9 | `python experiments/run_phase4.py --yes` then `python experiments/run_phase4_baselines.py` | – | `phase4_hw_paired_tests.csv` |
| 10 | `python experiments/run_ablations.py` | – (alpha_init = 0) | A1 rows identical to Phase 3 (built-in cross-check) |
| 11 | `python experiments/run_trie_ablation.py`, `python experiments/run_ucb_sensitivity.py`, `python experiments/run_design_ablation.py` | – | A7 / ucb summaries, `design_paired_tests.csv` (v1→v2 on held-out seeds) |
| 12 | `python experiments/run_gated_baselines.py` | GPU | `gated_paired_tests.csv` — this answers "ArchTree or just the gate?" |
| 13 | `python experiments/run_proxy_ablation.py` | GPU for uncached archs | best single proxy vs ensemble line |
| 14 | `python experiments/run_alpha_sweep.py`, `python experiments/analyze_alpha_sweep.py` | Groq (needs a full daily budget) | rerun on following days until 216/216 rows (rows with fallbacks are skipped) |
| 15 | `python experiments/run_convergence_study.py` | Groq | stops at the first fallback; rerun the next day to resume |
| 16 | `python experiments/run_llm_size_sensitivity.py --models groq:openai/gpt-oss-20b groq:llama-3.1-8b-instant` | Groq | probe rows must show uniform_fallback_rate ≈ 0 |
| 22 | `python experiments/run_equal_gpu_hours.py` | GPU (new NB101 proxies) | `egh_paired_tests.csv` (equal GPU-hours, NB201 + NB101 + NATS-SSS), `gate_cost_model.csv`, `cost_correlation.csv`; runs first in `-Plan final` |
| 17 | `python experiments/run_phase6_aggregate.py` | – | prints wins / ties / losses vs the 9 external baselines (Holm) and per baseline |
| 18 | analysis tools + `python tools/analyze_proxy_cost.py --measure 200` + `python tools/analyze_train_cost.py` | GPU for --measure | `proxy_cost_summary.csv`, `train_cost_summary.csv` (GPU-hours) |
| 19 | `python tools/build_repro_appendix.py --bundle` | – | `release\archtree_repro_bundle.zip` (rerun after 21) |
| 20 | `python tools/build_nb101_cache.py`, `python tools/test_nb101.py` | `data\nasbench_only108.tfrecord`, ~2 GB RAM, ~5 min | "ALL CHECKS PASSED" (0 bad hashes, 423,624 graphs, best mean test ≈ 94.3) and "ALL NB101 TESTS PASSED" |
| 21 | `python experiments/run_nb101.py` | GPU for the zc gates | `outputs\nb101\results\nb101_paired_tests.csv`, `nb101_equal_compute_tests.csv` |

LLM budget: steps 5, 8 (A4 only), 14, 15, 16 use Groq. Never run two of them at the same time
(shared key budgets). Everything else is LLM-free and can run in one sitting.

Live dashboard after step 17: `python frontend_practise/backend.py`, open http://localhost:5000
(backend.py finds the project root itself, so it works from any folder).

## Decisions you still have to make
1. **Calibration mode for the headline.** Default `supervised` (1,500 labels, must be reported as
   cost). `unsupervised` uses zero labels and makes the efficiency claim clean. Step 12 runs both
   gates side by side; choose after looking at it, then set `zc_proxy.calibration_mode` and rerun
   from step 7.
2. **alpha_init.** Leave 0.0 unless the re-tested C7 (steps 14-15) shows a significant benefit.
3. **NAS-Bench-101.** Implemented (FIXES §6), no WSL/Colab/TensorFlow needed: steps 20-21.

## Files: keep / changed / new / remove

**Changed (already written into your folder):** `config.yaml`, `archtree/{ucb,trie,search,llm_prior,zc_proxy,storage}.py`,
`benchmarks/{nb201,hwnb}.py`, `baselines/{rz_nas,zico,enas_proxy,mcts_nas}.py`,
`experiments/{analyze_alpha_sweep,run_phase0,run_phase0_llm_nas_only,run_phase2,run_phase3,run_ablations,run_trie_ablation,run_proxy_ablation,run_alpha_sweep,run_convergence_study,run_llm_size_sensitivity,run_phase6_aggregate}.py`,
`tools/{build_nb201_cache,build_igd_reference_front,analyze_cross_device_pareto,build_repro_appendix,test_storage,archtree_walkthrough}.py`,
`plots/plot_results.py`, `frontend_practise/backend.py`, `frontend_practise/index.html`, `requirements.txt`, `run_full_pipeline.ps1`,
`ReadMe/{goalREADME,REGRET_ARGUMENT}.md`.

**New:** `archtree/settings.py`, `experiments/{run_gated_baselines,run_phase4_baselines,run_ucb_sensitivity}.py`,
`tools/{analyze_proxy_cost.py,archive_legacy.ps1}`, `.gitignore`, `ReadMe/{FIXES_OCT2026,RUN_PLAN_OCT2026}.md`.

**Move out (via `tools/archive_legacy.ps1` / pipeline step 1, nothing deleted):** old `outputs\`,
`data\nb201_cifar10_cache.json` (v1), old `data\true_pareto_front__*.json` and `data\hw_table__*.npz`,
every `__pycache__\`, `demo_output\`, `Exttra\`, `gitignore`, `Commands.txt`, one-off fix/debug tools,
roadmap versions 1-13. Safe to delete later: `outputs\_zc_score_cache.OLD`.

**Unchanged and still valid:** `archtree/{encoder,tcr,weight_bank}.py`, `baselines/{base,random_search,llm_nas}.py`,
`benchmarks/nb101.py`, `utils/*`, `experiments/{run_phase1,run_phase4}.py`,
`tools/{analyze_duplicate_evals,analyze_tcr_h,analyze_regret,check_phase1_determinism}.py`,
`third_party/`, `data/cifar10/`.

**Docs to rewrite after the rerun:** `ReadMe/ARCHTREE_ROADMAP_progress.md` (§0.5-0.12 numbers),
`ReadMe/goalREADME.md` Implementation Status, the paper.


## Oct 3 additions (steps 23, 24, plan `extend`)

| Step | Command | Needs | Output |
|---|---|---|---|
| 23 | `python tools/build_natsbench_sss_cache.py` (after `pip install nats_bench`) | `data\NATS-sss-v1_0-50262-simple\` (1.01 GB tar, extract into data\; or the 874 MB .pickle.pbz2, needs ~10 GB+ RAM) | `data\natssss_cache.json` |
| 24 | `python experiments/run_llm_contamination.py` | Groq (~170 calls) | `outputs\llm_contamination\results\contamination_*.csv` |
| 15 | `python experiments/run_convergence_study.py` (now 4 arms) | Groq for llm_described | `convergence_paired_tests.csv` with arm_a / arm_b |

| 25 | `python tools/precompute_proxies.py` | GPU (3 parallel workers, `runtime.proxy_workers`) | fills `outputs\_zc_raw_cache.json` (all NATS-SSS, NB101 sample); resumable |
| 18 | + `python tools/analyze_rank_equivalence.py` | – | `outputs\final\results\stats_friedman.csv`, `stats_tost.csv`, `stats_prob_improvement.csv`, `outputs\final\plots\figures\cd_*.png` |

Speed (Oct 4): step 22 runs `runtime.workers` (3) parallel processes; identical results to 1 worker.

| 18 | + `python plots/plot_results.py all` | – | every figure; each also copied to `plots\figures\<experiment>__<name>.png` + `.csv` of its data points |
| 17 | + `python experiments/run_convergence_study.py --analyze-only` | – | re-tests `convergence_paired_tests.csv` (`test_note`: no test when the paired difference is identical for every seed) |

Status (Oct 4, 12:57):

| Block | Status | Time it took |
|---|---|---|
| `-Plan extend` | done Oct 3 | ~15 min |
| `-Only 25,22` | done Oct 4 | step 25 ~5 h (32,768 NATS-SSS + NB101 sample), step 22 25 min (1127 / 1128 runs; 1 failed on a file lock, fixed, redone by `-Plan final`) |
| `-Plan groq` | done Oct 4, 12:53 | 12 min (steps 15 + 24; 5, 8, 14, 16 were already complete) |
| `-Plan final` | done Oct 4, 13:04 | ~7 min; all 1128 step-22 runs, Phase 6 0 losses vs 9 external baselines, 50 figures in `plots\figures`, `release\archtree_repro_bundle.zip` |
| paper writing | **next** | results summary: README.md §9, ReadMe/RESULTS_LOG_OCT2026.md §5.9 |

Key results: ReadMe/RESULTS_LOG_OCT2026.md §5.6c (equal GPU-hours: 0 losses vs 13 arms, within CD of the best on NB201 / NB101 /
NATS-SSS; gate-cost model confirmed) and §5.6d (LLM knowledge controls: not memorisation, not just a size
prior; faster early search on cifar10 / cifar100, no final gain).

Order: `-Plan extend` (done) -> `-Only 25,22` (done) -> `-Plan groq` (done) -> `-Plan final` (done) -> **paper**.

While writing: `.\run_full_pipeline.ps1 -Plan report` (steps 17-19, ~7 min) rebuilds tables, statistics, figures and the
repro bundle from the existing results. New machine: `.\setup_env.ps1` (packages, GPU torch, data / key check, preflight);
`requirements.txt`, `setup_env.sh`, `config.yaml` comments and `tools\preflight.py` (steps 0-25) updated Oct 4 (FIXES #34).
