# run_full_pipeline.ps1 -- ArchTree-LLM full pipeline (PowerShell, Windows), steps 0-25
#
# STATUS Oct 4, 2026 13:04: EVERY step has run - all results of the paper are in outputs\ (README §5.9).
# Nothing needs to be rerun. While writing the paper, `-Plan report` rebuilds tables, statistics and
# figures from the existing results (~7 min, no GPU search, no Groq).
#
# First time on a new machine:  .\setup_env.ps1   (packages, GPU, data check, preflight)
# Run from the project root (next to config.yaml). Full order (details: ReadMe\COMPLETE_GUIDE_OCT2026.md):
#     .\run_full_pipeline.ps1 -Only 0,20        # unit tests + NB101 cache (5-10 min) - quick check first
#     .\run_full_pipeline.ps1 -Plan core        # all LLM-free NB201 experiments (GPU, many hours)
#     .\run_full_pipeline.ps1 -Plan nb101       # NB101 experiment (GPU, ~3-5 h)
#     .\run_full_pipeline.ps1 -Plan extend      # Oct 3 additions, no Groq: local search + RE warm-up (4),
#                                               # local search in the same-gate study (12), NATS-SSS cache (23)
#     .\run_full_pipeline.ps1 -Plan groq        # LLM studies (Groq, repeat on later days until complete)
#     .\run_full_pipeline.ps1 -Plan final       # proxy pre-computation (25), equal-GPU-hour study (22),
#                                               # aggregate, analyses + rank / equivalence statistics, repro bundle
#     .\run_full_pipeline.ps1 -Plan report      # 17 + 18 + 19 only: tables, statistics, all figures
#                                               # (plots\figures\ with data CSVs), repro bundle
#     .\run_full_pipeline.ps1 -Plan all         # everything in dependency order (fresh machine; ~1-2 days)
#
# Measured times (Oct 3-4, RTX 2050): extend ~15 min, step 25 ~5 h (first time), step 22 25 min,
# groq 12 min (complete caches), final ~7 min once 25/22 are done.
#
# Other ways to pick steps:
#     -Only 7,8,12              exactly these steps, in this order
#     -From 6 -To 13            a range (step 1 is never part of a range, see -ResetOutputs)
#     -NoGroq                   skip Groq steps (5, 14-16, 24) and the Groq part of step 8 (A4 grid)
#     -ArchivePhases phase2,phase3   MOVE outputs\phase2, outputs\phase3 to outputs_archive\<time>\ first
#                               (needed after a config change: finished runs are otherwise SKIPPED on resume)
#     -ResetOutputs             allow step 1 (moves ALL of outputs\ aside). Already done on Oct 2 - don't repeat.
#     -RebuildFronts            step 3 rebuilds HW fronts even if they exist
#     -SkipPreflight            don't run tools\preflight.py first
#     -DryRun                   only print which steps would run, then stop
#
# Every python call is logged to pipeline_logs\stepNN_<time>.log (send these to Claude when something fails).
# If a step fails the script stops; fix it and rerun the same command - finished runs are skipped.
# Nothing is ever deleted.

param(
    [ValidateSet("", "core", "nb101", "extend", "groq", "final", "report", "all")] [string]$Plan = "",
    [int[]]$Only = @(),
    [int]$From = 0,
    [int]$To = 25,
    [ValidateSet("groq", "ollama")] [string]$Backend = "groq",
    [switch]$NoGroq,
    [string[]]$ArchivePhases = @(),
    [switch]$ResetOutputs,
    [switch]$RebuildFronts,
    [switch]$SkipPreflight,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# ── which steps ───────────────────────────────────────────────────────────────
$Plans = @{
    "core"  = @(0, 4, 6, 7, 8, 9, 10, 11, 12, 13)
    "nb101" = @(20, 21)
    "extend" = @(4, 12, 23)
    "groq"  = @(5, 8, 14, 15, 16, 24)
    "final" = @(25, 22, 17, 18, 19)
    "report" = @(17, 18, 19)
    "all"   = @(0, 2, 3, 20, 23, 4, 6, 7, 8, 9, 10, 11, 12, 13, 21, 5, 14, 15, 16, 24, 25, 22, 17, 18, 19)
}
if ($Plan -ne "")      { $Steps = $Plans[$Plan] }
elseif ($Only.Count)   { $Steps = $Only }
else                   { $Steps = @($From..$To | Where-Object { $_ -ne 1 }) }
if ($ResetOutputs -and -not ($Steps -contains 1)) { $Steps = @(1) + $Steps }
if ($Plan -eq "core")  { $NoGroq = $true }
if ($NoGroq)           { $Steps = @($Steps | Where-Object { @(5, 14, 15, 16, 24) -notcontains $_ }) }

# ── environment for every python call ────────────────────────────────────────
$env:ARCHTREE_LLM_BACKEND = $Backend         # no interactive backend prompts
$env:PYTHONUNBUFFERED = "1"                  # live output while logging
$env:PYTHONIOENCODING = "utf-8"              # no UnicodeEncodeError when output is piped
$env:PYTHONUTF8 = "1"
if ($NoGroq) { $env:ARCHTREE_SKIP_A4 = "1" } else { Remove-Item Env:ARCHTREE_SKIP_A4 -ErrorAction SilentlyContinue }
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

$LogDir = Join-Path $PSScriptRoot "pipeline_logs"
New-Item -ItemType Directory $LogDir -Force | Out-Null
$script:CurStep = 0
$script:Summary = @()

function Py([string]$cmd) {
    Write-Host "> python $cmd" -ForegroundColor Yellow
    $parts = $cmd -split ' '
    $log = Join-Path $LogDir ("step{0:D2}_{1}.log" -f $script:CurStep, (Get-Date -Format "yyyyMMdd_HHmmss"))
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"      # python writes progress to stderr (loguru); not an error
    try {
        & python @parts 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $prev }
    if ($code -ne 0) { throw "FAILED (exit $code): python $cmd   -> log: $log" }
}

function Step([int]$n, [string]$title, [scriptblock]$body) {
    $script:CurStep = $n
    Write-Host "`n=== STEP $n : $title ===" -ForegroundColor Cyan
    $t0 = Get-Date
    & $body
    $min = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    $script:Summary += ("  step {0,2}  {1,7} min  {2}" -f $n, $min, $title)
    Write-Host "--- step $n done in $min min" -ForegroundColor Green
}

$note = if ($NoGroq) { "   [no Groq: steps 5/14-16/24 and the A4 grid of step 8 skipped]" } else { "" }
Write-Host "Steps to run (in this order): $($Steps -join ', ')$note" -ForegroundColor Cyan
if ($DryRun) { return }

# ── optional: archive phase folders before re-running them ───────────────────
if ($ArchivePhases.Count) {
    $dst = "outputs_archive\" + (Get-Date -Format "yyyyMMdd_HHmm")
    New-Item -ItemType Directory $dst -Force | Out-Null
    foreach ($p in $ArchivePhases) {
        if (Test-Path "outputs\$p") { Move-Item "outputs\$p" "$dst\$p"; Write-Host "archived outputs\$p -> $dst\$p" }
        else { Write-Host "outputs\$p not found - nothing to archive" -ForegroundColor DarkGray }
    }
}

# ── preflight ─────────────────────────────────────────────────────────────────
if (-not $SkipPreflight) {
    $script:CurStep = 0
    $pf = "tools/preflight.py --steps " + ($Steps -join ' ')
    if ($NoGroq) { $pf += " --skip-a4" }
    Py $pf
}

try {
foreach ($s in $Steps) {
switch ($s) {

0 { Step 0 "sanity: unit tests (storage + NB101 pure functions)" {
    Py "tools/test_storage.py"
    Py "tools/test_nb101.py"
} }

1 { Step 1 "move ALL old outputs + old HW fronts aside (needs -ResetOutputs and YES)" {
    $ts = Get-Date -Format "yyyyMMdd_HHmm"
    $bk = "outputs_pre_oct2026_$ts"
    Write-Host "Will MOVE outputs\ -> $bk\ (kept, never deleted), plus data\true_pareto_front__*.json and"
    Write-Host "data\hw_table__*.npz. Only the proxy raw-score cache is copied back (still valid)."
    if ((Read-Host "Type YES to proceed") -ne "YES") { throw "aborted by user" }
    if (Test-Path outputs) { Move-Item outputs $bk } else { New-Item -ItemType Directory $bk | Out-Null }
    New-Item -ItemType Directory outputs -Force | Out-Null
    if (Test-Path "$bk\_zc_raw_cache.json") { Copy-Item "$bk\_zc_raw_cache.json" outputs\_zc_raw_cache.json }
    New-Item -ItemType Directory "$bk\_old_data" -Force | Out-Null
    Get-ChildItem data -Filter "true_pareto_front__*.json" -ErrorAction SilentlyContinue | Move-Item -Destination "$bk\_old_data"
    Get-ChildItem data -Filter "hw_table__*.npz" -ErrorAction SilentlyContinue | Move-Item -Destination "$bk\_old_data"
    Write-Host "Done. Old outputs in $bk" -ForegroundColor Green
} }

2 { Step 2 "build the v2 NB201 cache (200-epoch, valid + test)  [5 GB .pth, ~10-16 GB RAM]" {
    if (-not (Test-Path data\nb201_cache_v2.json)) { Py "tools/build_nb201_cache.py" }
    else { Write-Host "data\nb201_cache_v2.json exists - skipping" }
} }

3 { Step 3 "HW reference fronts (index-mapped hwnb.py, alignment check ~0.6)" {
    if ($RebuildFronts) { Py "tools/build_igd_reference_front.py" }
    else { Py "tools/build_igd_reference_front.py --skip-existing" }
} }

4 { Step 4 "PHASE 0 baselines: random, enas, mcts, zico, rz_nas, RE, local search, RE + zero-cost warm-up  [GPU]" {
    Py "experiments/run_phase0.py"
} }

5 { Step 5 "PHASE 0 llm_nas baseline  [Groq, ~1,700 calls]" {
    Py "experiments/run_phase0_llm_nas_only.py"
} }

6 { Step 6 "PHASE 1 - trie + UCB1" { Py "experiments/run_phase1.py" } }

7 { Step 7 "PHASE 2 - proxy calibration (supervised + label-free) + gated search  [GPU]" {
    Py "experiments/run_phase2.py"
} }

8 { Step 8 "PHASE 3 - headline system (alpha=0) + A4 alpha grid (A4 needs Groq; skipped with -NoGroq)" {
    Py "experiments/run_phase3.py"
} }

9 { Step 9 "PHASE 4 - hardware-aware ArchTree, then random / NSGA-II-style baselines" {
    Py "experiments/run_phase4.py --yes"
    Py "experiments/run_phase4_baselines.py"
} }

10 { Step 10 "ABLATIONS A1-A6 (no LLM calls while alpha_init=0)" {
    Py "experiments/run_ablations.py"
} }

11 { Step 11 "A7 trie vs flat UCB1, UCB C / backup sensitivity, v1->v2 design ablation (held-out seeds)" {
    Py "experiments/run_trie_ablation.py"
    Py "experiments/run_ucb_sensitivity.py"
    Py "experiments/run_design_ablation.py"
} }

12 { Step 12 "same-gate baselines: archtree vs random / RE / local search / ENAS / MCTS (zc, zc_unsup, params, none)  [GPU]" {
    Py "experiments/run_gated_baselines.py"
} }

13 { Step 13 "proxy composition ablation (+ best single proxy vs ensemble)  [GPU for uncached]" {
    Py "experiments/run_proxy_ablation.py"
} }

14 { Step 14 "C7 re-test 1: alpha sweep  [Groq; failed-LLM rows skipped - rerun next day to fill]" {
    Py "experiments/run_alpha_sweep.py"
    Py "experiments/analyze_alpha_sweep.py"
} }

15 { Step 15 "C7 re-test 2: convergence study - LLM vs size prior vs described prompt  [Groq; stops at the first LLM fallback, resumable]" {
    Py "experiments/run_convergence_study.py"
} }

16 { Step 16 "C7 optional: LLM-size probe + search  [Groq]" {
    Py "experiments/run_llm_size_sensitivity.py --models groq:openai/gpt-oss-20b groq:openai/gpt-oss-120b"   # llama-3.1-8b-instant retired (404); allam-2-7b probed Oct 3, search dropped (wrong op names in half its calls)
} }

22 { Step 22 "equal-GPU-hour comparison NB201 + NB101 + NATS-SSS, proxy cost analysis, gate-cost model  [GPU, parallel workers]" {
    Py "experiments/run_equal_gpu_hours.py"
} }

25 { Step 25 "pre-compute zero-cost proxies in parallel GPU workers (NATS-SSS all, NB101 sample)  [GPU, resumable]" {
    Py "tools/precompute_proxies.py"
} }

23 { Step 23 "NATS-Bench size space: build data\natssss_cache.json (pip nats_bench, no GPU)" {
    if (Test-Path data\natssss_cache.json) { Write-Host "data\natssss_cache.json exists - skipping build" }
    elseif ((Test-Path data\NATS-sss-v1_0-50262.pickle.pbz2) -or (Test-Path data\NATS-sss-v1_0-50262-simple)) {
        Py "-m pip install nats_bench"
        Py "tools/build_natsbench_sss_cache.py"
    }
    else { Write-Host "NATS-Bench SSS file missing in data\ - step 22 will skip NATS-SSS (README section 6 says where to get it)" -ForegroundColor Yellow }
} }

24 { Step 24 "LLM contamination probe: standard vs described vs anonymous prompt vs size prior  [Groq, ~170 calls]" {
    Py "experiments/run_llm_contamination.py"
} }

17 { Step 17 "PHASE 6 aggregate (test metric, external-only headline, Holm, CIs; folds in steps 11, 12, 15, 21, 22, 24)" {
    Py "experiments/run_convergence_study.py --analyze-only"   # re-test with the deterministic-prior check (no GPU / no Groq)
    Py "experiments/run_phase6_aggregate.py"
} }

18 { Step 18 "analysis scripts + proxy cost  [GPU for --measure]" {
    Py "tools/analyze_duplicate_evals.py"
    Py "tools/analyze_tcr_h.py"
    Py "tools/analyze_regret.py"
    Py "tools/analyze_cross_device_pareto.py"
    Py "tools/analyze_proxy_cost.py --measure 200"
    Py "tools/analyze_train_cost.py"
    Py "tools/analyze_rank_equivalence.py"
    Py "plots/plot_results.py all"          # every figure (convergence, equal GPU-h, contamination, ...)
} }

19 { Step 19 "reproducibility appendix + release bundle (run last)" {
    Py "tools/build_repro_appendix.py --bundle"
} }

20 { Step 20 "NAS-Bench-101: build data\nb101_cache.npz from the tfrecord (no TensorFlow) + tests" {
    if (-not (Test-Path data\nb101_cache.npz)) { Py "tools/build_nb101_cache.py" }
    else { Write-Host "data\nb101_cache.npz exists - skipping build" }
    Py "tools/test_nb101.py"
} }

21 { Step 21 "NAS-Bench-101 experiment: ArchTree vs random / regularized evolution, 4 gates  [GPU]" {
    Py "experiments/run_nb101.py"
} }

default { Write-Host "unknown step $s - ignored" -ForegroundColor DarkGray }
}
}
}
finally {
    if ($script:Summary.Count) {
        Write-Host "`nFinished steps:" -ForegroundColor Green
        $script:Summary | ForEach-Object { Write-Host $_ }
    }
    Write-Host "Logs: $LogDir"
}
Write-Host "`nPipeline finished (steps $($Steps -join ', '))." -ForegroundColor Green
