#!/bin/bash
# setup_env.sh - one-time environment setup on Linux / macOS (updated Oct 4, 2026)
# Windows: use setup_env.ps1 instead (the paper runs were done on Windows).
#
#   bash setup_env.sh              # creates ./venv (or uses the active conda env / venv)
#   CUDA=cu121 bash setup_env.sh   # other CUDA wheel index (default cu124 = the paper runs)
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
CUDA="${CUDA:-cu124}"

echo "============================================"
echo " ArchTree-LLM environment setup"
echo "============================================"

# 1. Python environment
if [ -n "$CONDA_PREFIX" ] || [ -n "$VIRTUAL_ENV" ]; then
    PY=python
    echo "[1/5] using the active environment (${CONDA_PREFIX:-$VIRTUAL_ENV})"
else
    [ -d venv ] || python3 -m venv venv
    PY="$PROJECT_ROOT/venv/bin/python"
    echo "[1/5] using ./venv"
fi
$PY -c "import sys; print('      Python', sys.version.split()[0]); sys.exit(0 if sys.version_info >= (3, 9) else 1)"

# 2. GPU torch (from the CUDA index; plain `pip install torch` may be CPU-only)
if $PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[2/5] CUDA torch already installed"
else
    echo "[2/5] installing GPU torch ($CUDA) ..."
    $PY -m pip install --upgrade pip -q
    $PY -m pip install torch==2.5.1 torchvision==0.20.1 --index-url "https://download.pytorch.org/whl/$CUDA"
fi

# 3. requirements (no TensorFlow / nasbench needed: NB101 is read without TF)
echo "[3/5] installing requirements.txt ..."
$PY -m pip install -r requirements.txt

# 4. HW-NAS-Bench (not on PyPI; Phase 4 only)
if [ ! -d third_party/HW-NAS-Bench ]; then
    echo "[4/5] cloning HW-NAS-Bench into third_party/ ..."
    git clone https://github.com/GATECH-EIC/HW-NAS-Bench third_party/HW-NAS-Bench || \
        echo "  WARNING: clone failed - Phase 4 needs it"
else
    echo "[4/5] third_party/HW-NAS-Bench present"
fi

# 5. verify + preflight (Groq keys are read from .env and never printed)
echo "[5/5] verifying ..."
[ -f .env ] && echo "  ok  .env present (GROQ_API_KEY / GROQ_API_KEYS)" || \
    echo "  --  .env missing: create it with GROQ_API_KEYS=key1,key2,... for the LLM steps"
$PY tools/preflight.py || true

echo ""
echo "Done. Data files and the full step list: README.md section 6."
echo "Optional local LLM fallback: install Ollama (https://ollama.com) and 'ollama pull llama3.1:8b'."
