#!/usr/bin/env bash
# Set up the LoCA Python environment. Works on the CPU dev box and the GPU box.
# Usage: ./setup_env.sh [gpu]   # pass "gpu" to install the CUDA torch wheel
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.11 .venv 2>/dev/null || true
  source .venv/bin/activate
  uv pip install -r requirements.txt
else
  $PYTHON -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
fi

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
echo "[setup_env] done. Activate with: source .venv/bin/activate"
