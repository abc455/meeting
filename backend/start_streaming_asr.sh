#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen-asr-env

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
export TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export TORCH_CUDA_ARCH_LIST="12.1a"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

exec uvicorn app.streaming_asr_server:app --host "${ASR_STREAM_HOST:-127.0.0.1}" --port "${ASR_STREAM_PORT:-8005}"
