#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

npm install
npm run dev -- --host 0.0.0.0 --port "${PORT:-5173}"
