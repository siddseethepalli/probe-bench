#!/usr/bin/env bash
# Move the live API to another pod: ship the committed tree, the generation cache, and the fitted probes
# (so prebuilt example ids resolve there), restart it, point the frontend at it, redeploy.
#   scripts/switch_pod.sh runpod/.pod-id-2      # state file of the destination pod
set -euo pipefail
cd "$(dirname "$0")/.."
DEST_STATE="${1:?destination pod state file}"
export RUNPOD_API_KEY="${RUNPOD_API_KEY:-$(grep '^RUNPOD_API_KEY=' .env | cut -d= -f2-)}"
DEST_ID=$(cat "$DEST_STATE")
DEST_BASE="https://${DEST_ID}-8000.proxy.runpod.net"

echo "== committed tree and cache to $DEST_ID"
POD_STATE="$DEST_STATE" runpod/launch.sh sync-head
POD_STATE="$DEST_STATE" runpod/launch.sh cache
echo "== fitted probes from the current pod to $DEST_ID"
runpod/launch.sh run 'cd /workspace/data && tar -czf - probes' > /tmp/probe-bench-probes.tgz
POD_STATE="$DEST_STATE" runpod/launch.sh run 'mkdir -p /workspace/data && tar -xzf - -C /workspace/data && ls /workspace/data/probes | wc -l' < /tmp/probe-bench-probes.tgz
echo "== restart $DEST_ID"
POD_STATE="$DEST_STATE" runpod/launch.sh run 'cd /workspace/probe-bench && DATA_DIR=/workspace/data runpod/remote_start.sh'
for _ in $(seq 1 40); do
  sleep 5
  if curl -s -m 10 -A probe-bench-switch "$DEST_BASE/health" | grep -q '"model_loaded":true'; then echo "model loaded on $DEST_ID"; break; fi
done
echo "== frontend config, docs, deploy"
python3 - "$DEST_BASE" <<'PY'
import json, pathlib, re, sys
base = sys.argv[1]
cfg = pathlib.Path("frontend/public/config.json"); cfg.write_text(json.dumps({"apiBase": base}, indent=2) + "\n")
for name in ["README.md", "scripts/ship.sh"]:
    p = pathlib.Path(name); p.write_text(re.sub(r"https://[a-z0-9]+-8000\.proxy\.runpod\.net", base, p.read_text()))
PY
(cd frontend && npx tsc --noEmit && npm run build 2>&1 | grep -E "built in|rror" && vercel --prod --yes 2>&1 | grep -E "Aliased|rror" | tail -1)
git add frontend/public/config.json README.md scripts/ship.sh && git commit -qm "Live API moved to pod $DEST_ID" && git push -q
echo "switched to $DEST_BASE; the old pod is still running, kill it with: runpod/pod.sh kill"
