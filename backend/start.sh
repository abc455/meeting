#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate meeting-ai

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8001}"
