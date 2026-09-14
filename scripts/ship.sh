#!/usr/bin/env bash
# One-shot release: sync the backend to the pod, ship the reviewed sets into its cache, restart the
# server, wait for the model, precompute the prebuilt examples, build the frontend, deploy to Vercel.
#   scripts/ship.sh                 # full sequence
#   scripts/ship.sh precompute      # skip the pod steps
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="https://cvwordeo0ccw8x-8000.proxy.runpod.net"
# The API requires the access password; precompute sends it from PROBE_KEY (read from .env, never echoed).
PROBE_KEY="$(grep '^DEMO_PASSWORD=' .env | cut -d= -f2- || true)"
export PROBE_KEY
CONCEPTS="sycophancy sadness french-language sarcasm legal-language"

if [ "${1:-}" != "precompute" ]; then
  runpod/launch.sh sync-head
  runpod/launch.sh cache
  runpod/launch.sh run 'cd /workspace/probe-bench && DATA_DIR=/workspace/data runpod/remote_start.sh'
  for _ in $(seq 1 40); do
    sleep 5
    if curl -s -m 10 -A probe-bench-ship "$BASE/health" | grep -q '"model_loaded":true'; then echo "model loaded"; break; fi
  done
fi
python3 scripts/precompute.py --base "$BASE" --out frontend/public/examples --decoys-variant sycophancy -- $CONCEPTS
(cd frontend && npx tsc --noEmit && npm run build 2>&1 | grep -E "built in|rror" && vercel --prod --yes 2>&1 | grep -E "Aliased|rror" | tail -2)
echo "shipped"
