# ArchTree-LLM — complete run guide (Oct 2026, updated Oct 2 afternoon)

Background: `FIXES_OCT2026.md` (what changed, §6 = NAS-Bench-101), `RUN_PLAN_OCT2026.md` (per-step table).
Everything runs through `run_full_pipeline.ps1`. It checks prerequisites first (`tools\preflight.py`),
logs every step to `pipeline_logs\`, prints how long each step took, and never deletes anything.
Finished runs are skipped on a rerun, so after any crash: fix it, run the same command again.

**Already done (Oct 2):** cleanup, step 1 (old outputs moved to `outputs_pre_oct2026_20261002_1215\`),
step 2 (NB201 v2 cache, verified), step 3 (HW fronts). Never run step 1 again (it needs `-ResetOutputs` now).

---

## The sequence

Open PowerShell in the project folder once per session:
```powershell
cd "C:\Users\Iyaad Hajwane\Desktop\archtree-llm V4"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\activate
```
Any command can be previewed with `-DryRun` (prints the steps, runs nothing).

### Run 1 — quick checks (~10 min)
```powershell
.\run_full_pipeline.ps1 -Only 0,20
```
| Step | Does | Must see |
|---|---|---|
| preflight | packages, GPU, data files, keys (count only) | `PREFLIGHT OK` |
| 0 | storage + NB101 unit tests | `ALL TESTS PASSED`, `ALL NB101 TESTS PASSED` |
| 20 | builds `data\nb101_cache.npz` from the tfrecord (no TensorFlow), then the NB101 tests on the real data | `check 2 ... 0 of 423,624`, `ALL CHECKS PASSED`, best mean test ≈ 94.3 |

**Send me `pipeline_logs\step20_*.log`** (or a screenshot of the end). If anything says FAILED, stop here.

### Run 2 — all LLM-free NB201 experiments (GPU, 1-2 days, can run overnight)
```powershell
.\run_full_pipeline.ps1 -Plan core
```
Steps 0, 4, 6, 7, 8 (main sweep only, A4 grid skipped), 9, 10, 11, 12, 13. No Groq call is made.

| Step | Does | Check in the log |
|---|---|---|
| 4 | Phase 0 baselines: random, enas, mcts, zico, rz_nas, regularized evolution, local search, RE + zero-cost warm-up | `Errors: 0` |
| 6 | Phase 1 trie + UCB | 144 runs |
| 7 | Phase 2: proxy calibration (supervised + label-free) + gated search — slowest | tau ≈ 0.4 per dataset |
| 8 | Phase 3 headline (alpha = 0) | 144 main rows; "A4 alpha grid skipped" warning is expected |
| 9 | Phase 4 hardware + random / NSGA-II baselines | `phase4_hw_paired_tests.csv` |
| 10 | Ablations A1-A6 | A1 numbers = Phase 3 numbers |
| 11 | A7 trie vs flat, UCB C sweep, v1→v2 design ablation (held-out seeds) | `A7_paired_tests.csv`, `ucb_paired_tests.csv`, `design_paired_tests.csv` |
| 12 | Same-gate baselines (incl. regularized evolution and local search) | `gated_paired_tests.csv` |
| 13 | Proxy composition ablation | "best single proxy ... vs full ensemble" |

**Then send me** `outputs\gated_baselines\results\gated_summary.csv` → **Decision 1** (calibration mode):
* `zc` (1,500 labels) and `zc_unsup` (0 labels) similar → set `zc_proxy.calibration_mode: unsupervised` in
  `config.yaml` and redo the affected steps (the archive switch moves the old results aside so they are
  not skipped):
  ```powershell
  .\run_full_pipeline.ps1 -ArchivePhases phase2,phase3,ablations,ucb_sensitivity,gated_baselines -Only 7,8,10,11,12 -NoGroq
  ```
* `zc` clearly better → keep `supervised`; the label cost is reported by step 18.

Make Decision 1 BEFORE Run 4: the archive above would also move Groq results stored in `ablations\`.

### Run 3 — NAS-Bench-101 experiment (GPU, ~3-5 h)
```powershell
.\run_full_pipeline.ps1 -Plan nb101
```
Step 20 only re-runs its tests (cache exists); step 21 runs ArchTree vs random vs regularized evolution
with gates none / params / zc_unsup / zc, budgets 50-400, 12 seeds.
Check the last lines: "same gate ... wins / ties / losses" and "equal real-eval compute ...".
CPU-only part if the GPU is busy: `python experiments\run_nb101.py --gates none params`.

### Run 4 — Groq studies (days 2-4; one command per day until complete)
```powershell
.\run_full_pipeline.ps1 -Plan groq
```
Steps 5 (llm_nas baseline), 8 (fills the A4 alpha grid; main sweep is skipped as done), 14 (alpha sweep),
15 (convergence), 16 (LLM-size). When the keys run out, rows with LLM failures are NOT written; run the
same command the next day and it continues. Never run two Groq commands at once.
Done when: `alpha_sweep_results.csv` has 216 rows, convergence has 12 seeds per cell, the LLM-size probe
shows `uniform_fallback_rate` ≈ 0.

**Decision 2** (alpha_init): keep `llm.alpha_init: 0.0` unless steps 14-15 show a Holm-significant gain.
If they do, tell me before changing anything (several phases depend on it).

### Run 5 — final tables, analyses, bundle (~30-60 min)
```powershell
.\run_full_pipeline.ps1 -Plan final
python frontend_practise\backend.py         # dashboard -> http://localhost:5000
```
| Step | Output |
|---|---|
| 22 | `outputs\equal_gpu_hours\results\egh_paired_tests.csv` (every method at equal GPU-hours, NB201 + NB101 + NATS-SSS) + `gate_cost_validation.csv` (predicted vs measured) + `cost_correlation.csv` (proxies vs training cost); runs first in `-Plan final` |
| 17 | `outputs\final\results\equal_compute_statistical_tests_strict.csv` (headline wins / ties / losses vs the 9 external baselines, Holm) + `extended_findings_summary.csv` (gated baselines, A7, proxy ablation, convergence, LLM size, NB101, equal GPU-hours) |
| 18 | duplicate evals, TCR-H, regret, cross-device Pareto, `proxy_cost_summary.csv`, `train_cost_summary.csv` (GPU-hours) |
| 19 | `outputs\final\repro\` + `release\archtree_repro_bundle.zip` (never contains `.env`) |

Send me the step 17 log → we write the paper numbers.

---

## Where results go
| What | Where |
|---|---|
| per-experiment CSVs | `outputs\<phase>\results\` (phase0-4, ablations, gated_baselines, proxy_ablation, ucb_sensitivity, alpha_sweep, convergence, llm_size, nb101) |
| run logs (JSON per run) | `outputs\<phase>\logs\` |
| figures | `outputs\<phase>\plots\figures\` |
| final tables | `outputs\final\results\` |
| pipeline logs (what to send me) | `pipeline_logs\stepNN_<time>.log` |
| archived results (`-ArchivePhases`) | `outputs_archive\<time>\` |
| old Sep-2026 results (invalid) | `outputs_pre_oct2026_20261002_1215\` |

## Quick reference
| Situation | Do |
|---|---|
| a step crashed | fix it, rerun the same command (done runs are skipped); send me its `pipeline_logs` file |
| preflight says MISS | fix that item (it says what), rerun |
| Groq keys exhausted | wait for the daily reset, rerun `-Plan groq` |
| changed `config.yaml` | archive the affected phases and rerun their steps (example under Decision 1) |
| see what a command would do | add `-DryRun` |
| dashboard shows old numbers | rerun step 17, refresh the page |
| free disk later | `data\nasbench_only108.tfrecord` (after step 20) and `data\NAS-Bench-201-v1_1-096897.pth` (after step 2) can be moved off the PC; keep a copy somewhere |

## Writing (after Run 5)
1. Update `ReadMe\ARCHTREE_ROADMAP_progress.md` (§0.5-0.12) and `goalREADME.md` Implementation Status; remove the "superseded" banner.
2. Literature check: Vo et al. (tree search + training-free metrics), TNAS, MTF-PDNS, LaNAS/AlphaX, RZ-NAS, LLM-NAS 2510.01472, CoLLM-NAS, ZAPS, CoRA-NAS. No "first" claims.
3. Name baselines "-style" and cite the originals; our `rz_nas` is a ZC mutation walk, not RZ-NAS. Regularized evolution = Real et al. 2019.
4. Paper sections:
   * Headline: wins / ties / losses vs external baselines at matched real cost (test accuracy).
   * Step 12: does the trie add anything over the same gate with random / RE / ENAS / MCTS?
   * Generalisation: NB101 (same-gate and equal-real-eval tests).
   * Hardware: Phase 4 vs random / NSGA-II, cross-device overlap.
   * C7 (LLM prior) and A7 (trie structure) as their own sections, scoped narrowly.
   * Proxy cost incl. calibration labels.
   * Theory: T1, T2 (by construction), T5 hold; T3/T6 informal; T4 restated as heuristic; T7 definitional (`REGRET_ARGUMENT.md` §5b).
   * Limitations: tabular only; NB101 cifar10 only and without the LLM prior / hardware part; proxy uses CIFAR-10 images for every dataset; deterministic accuracies (mean over repeats); simulated weight bank; edgetpu ~30% archs without latency.
5. Abstract sentence: the benefit comes from deduplication + proxy gating (and hardware-aware search), not from tree structure; the LLM prior in this form did not help (if the re-test confirms it).


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
