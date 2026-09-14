#!/usr/bin/env bash
# Runs ON the pod (started by runpod/launch.sh setup): venv with pinned deps, then the
# fallback model (small, so the backend can be tested within minutes) and the 27B.
# Log: /workspace/setup.log. Touches /workspace/SETUP_DONE when finished.
set -euo pipefail
export HF_HOME=/workspace/hf PATH="$HOME/.local/bin:$PATH"
cd /workspace/probe-bench
set -a; [ -f .env ] && . ./.env; set +a
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B}"
REVISION="${REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
FALLBACK_MODEL_ID="${FALLBACK_MODEL_ID:-Qwen/Qwen2.5-1.5B}"

if [ ! -x .venv/bin/python ]; then
  echo "=== venv ($(date -u +%H:%M:%SZ))"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  uv venv --python 3.12 .venv >/dev/null
  # The pod image ships a CUDA 12.8 driver; the default torch wheel is the CUDA 13 build and
  # reports no GPU. Install torch from the cu128 index first, then everything else.
  uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0+cu128" 2>&1 | tail -1
  uv pip install --python .venv/bin/python -r backend/requirements.txt 2>&1 | tail -1
fi
echo "=== download $FALLBACK_MODEL_ID ($(date -u +%H:%M:%SZ))"
.venv/bin/hf download "$FALLBACK_MODEL_ID" --max-workers 16 >/dev/null
touch /workspace/FALLBACK_READY
echo "=== download $MODEL_ID @ $REVISION ($(date -u +%H:%M:%SZ))"
.venv/bin/hf download "$MODEL_ID" --revision "$REVISION" --max-workers 16 >/dev/null
echo "=== downloaded ($(date -u +%H:%M:%SZ)); $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
touch /workspace/SETUP_DONE
