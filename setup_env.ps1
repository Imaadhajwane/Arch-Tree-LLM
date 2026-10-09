# setup_env.ps1 - one-time environment setup on Windows (Oct 4, 2026)
#
# Run from the project root in PowerShell:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
#   .\setup_env.ps1                 # uses the ACTIVE conda env / venv if one is active, else creates .\venv
#   .\setup_env.ps1 -NewVenv        # always create / use .\venv
#   .\setup_env.ps1 -Cuda cu121     # other CUDA wheel index (default cu124 = the paper runs)
#   .\setup_env.ps1 -CheckOnly      # no installs: only verify packages, GPU, data, .env
#
# What it does: (1) picks the Python env, (2) installs GPU torch from the CUDA index, (3) installs
# requirements.txt, (4) verifies imports + CUDA, (5) checks data files / third_party / .env (presence
# only - keys are never printed), (6) runs tools/preflight.py for the whole pipeline.
param(
    [switch]$NewVenv,
    [string]$Cuda = "cu124",
    [switch]$CheckOnly
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
function Say($m, $c = "Cyan") { Write-Host $m -ForegroundColor $c }

Say "============================================"
Say " ArchTree-LLM environment setup (Windows)"
Say "============================================"

# -- 1. Python environment -----------------------------------------------------
$py = "python"
if ($NewVenv -or (-not $env:CONDA_PREFIX -and -not $env:VIRTUAL_ENV)) {
    if (-not (Test-Path ".\venv\Scripts\python.exe")) {
        Say "[1/6] creating .\venv (Python 3.10 recommended) ..."
        python -m venv venv
    } else { Say "[1/6] .\venv exists" }
    $py = ".\venv\Scripts\python.exe"
} else {
    $envName = if ($env:CONDA_PREFIX) { "conda: $env:CONDA_PREFIX" } else { "venv: $env:VIRTUAL_ENV" }
    Say "[1/6] using the active environment ($envName)"
}
& $py -c "import sys; print('      Python', sys.version.split()[0]); sys.exit(0 if sys.version_info >= (3, 9) else 1)"
if ($LASTEXITCODE -ne 0) { throw "Python >= 3.9 needed (paper runs: 3.10.9)" }

if (-not $CheckOnly) {
    # -- 2. GPU torch ----------------------------------------------------------
    $hasCuda = "False"
    $ErrorActionPreference = "Continue"          # torch may not be installed yet (stderr is not an error here)
    try { $hasCuda = & $py -c "import torch,sys; sys.stdout.write(str(torch.cuda.is_available()))" 2>$null } catch {}
    $ErrorActionPreference = "Stop"
    if ($hasCuda -eq "True") {
        Say "[2/6] CUDA torch already installed - skipping"
    } else {
        Say "[2/6] installing GPU torch ($Cuda) ..."
        & $py -m pip install --upgrade pip -q
        & $py -m pip install torch==2.5.1 torchvision==0.20.1 --index-url "https://download.pytorch.org/whl/$Cuda"
        if ($LASTEXITCODE -ne 0) { throw "torch install failed (try -Cuda cu121 or cu118 to match your driver)" }
    }
    # -- 3. requirements ------------------------------------------------------
    Say "[3/6] installing requirements.txt ..."
    & $py -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt failed" }
} else { Say "[2-3/6] -CheckOnly: no installs" }

# -- 4. verify imports + GPU ---------------------------------------------------
Say "[4/6] verifying packages ..."
& $py -c @"
import importlib, sys
req = ['torch','torchvision','numpy','pandas','scipy','matplotlib','yaml','dotenv','loguru','psutil','groq']
opt = ['nas_201_api','nats_bench','ollama','flask','flask_cors']
bad = []
for m in req:
    try: importlib.import_module(m); print('  ok   ', m)
    except Exception: print('  MISSING', m); bad.append(m)
for m in opt:
    try: importlib.import_module(m); print('  ok   ', m, '(optional)')
    except Exception: print('  --   ', m, '(optional, not installed)')
import torch
print('  CUDA :', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '- zero-cost proxies need a GPU')
sys.exit(1 if bad else 0)
"@
if ($LASTEXITCODE -ne 0) { throw "required packages missing - rerun without -CheckOnly" }

# -- 5. data / third_party / .env (presence only) -----------------------------
Say "[5/6] checking data files ..."
$items = @(
    @("data\nb201_cache_v2.json",              "NB201 cache (all NB201 experiments)"),
    @("data\nb101_cache.npz",                  "NB101 cache (step 21, 22)"),
    @("data\natssss_cache.json",               "NATS-SSS cache (step 22; built by step 23)"),
    @("data\HW-NAS-Bench-v1_0.pickle",         "HW-NAS-Bench pickle (Phase 4)"),
    @("third_party\HW-NAS-Bench",              "HW-NAS-Bench repo (Phase 4)"),
    @("outputs\_zc_raw_cache.json",            "zero-cost proxy cache (step 25; optional, saves hours)"),
    @(".env",                                  "Groq keys GROQ_API_KEY / GROQ_API_KEYS (LLM steps 5, 8, 14-16, 24)")
)
foreach ($it in $items) {
    if (Test-Path $it[0]) { Say ("  ok    {0,-36} {1}" -f $it[0], $it[1]) "Green" }
    else                  { Say ("  --    {0,-36} {1}" -f $it[0], $it[1]) "Yellow" }
}

# -- 6. pipeline preflight (all steps) ----------------------------------------
Say "[6/6] tools/preflight.py ..."
& $py tools/preflight.py
Say ""
Say "Done. Next:  .\run_full_pipeline.ps1 -DryRun -Plan all   (preview)   |   README.md section 6" "Green"
