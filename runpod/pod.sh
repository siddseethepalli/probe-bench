#!/usr/bin/env bash
# Mint, inspect, ssh into, or terminate the RunPod GPU pod for this tutorial.
#
#   export RUNPOD_API_KEY=...   # https://www.runpod.io/console/user/settings
#   runpod/pod.sh mint          # one A100 80GB + 120 GB disk, secure cloud (GPU="NVIDIA H100 80GB HBM3" to override)
#   runpod/pod.sh status
#   runpod/pod.sh ssh           # prints the ssh command
#   runpod/pod.sh kill          # terminate, then verify it is gone
#
# The pod id is recorded in runpod/.pod-id. `kill` only ever terminates that pod,
# never anything else on the account.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
STATE="${POD_STATE:-$HERE/.pod-id}"   # set POD_STATE (and POD_NAME, GPU) to manage a second pod
POD_NAME="${POD_NAME:-probe-bench}"
GPU="${GPU:-NVIDIA A100-SXM4-80GB}"
IMAGE="${IMAGE:-runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04}"
# The 27B checkpoint is 56 GB. It goes on the container disk, not a network volume:
# RunPod network volumes write at single-digit MB/s, the container disk at GB/s.
# The download is lost when the pod is terminated, which is fine for a one-off run.
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-120}"
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"

gql() {
  curl -sS --max-time 60 "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
    -H 'Content-Type: application/json' -d "$1"
}
pod_id() { cat "$STATE" 2>/dev/null || { echo "no pod recorded in $STATE" >&2; exit 1; }; }

case "${1:-}" in
mint)
  if [ -f "$STATE" ]; then
    echo "refusing: $STATE already exists (pod $(cat "$STATE")). Run '$0 kill' first."; exit 1
  fi
  RESP=$(gql "{\"query\":\"mutation { podFindAndDeployOnDemand(input: { cloudType: SECURE, gpuCount: 1, volumeInGb: 0, containerDiskInGb: $CONTAINER_DISK_GB, minVcpuCount: 8, minMemoryInGb: 60, gpuTypeId: \\\"$GPU\\\", name: \\\"$POD_NAME\\\", imageName: \\\"$IMAGE\\\", volumeMountPath: \\\"/workspace\\\", ports: \\\"22/tcp,8000/http\\\", startSsh: true, env: [{key: \\\"HF_HOME\\\", value: \\\"/workspace/hf\\\"}] }) { id name machine { gpuDisplayName } costPerHr } }\"}")
  echo "$RESP"
  ID=$(echo "$RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); p=(d.get('data') or {}).get('podFindAndDeployOnDemand') or {}; print(p.get('id',''))")
  [ -z "$ID" ] && { echo "mint failed (no stock for '$GPU'? try GPU=\"NVIDIA H100 80GB HBM3\")"; exit 1; }
  echo "$ID" > "$STATE"
  echo "minted $POD_NAME = $ID"
  ;;
status)
  gql "{\"query\":\"query { pod(input: {podId: \\\"$(pod_id)\\\"}) { id name desiredStatus costPerHr runtime { uptimeInSeconds ports { ip publicPort privatePort type } } } }\"}"; echo
  ;;
ssh)
  gql "{\"query\":\"query { pod(input: {podId: \\\"$(pod_id)\\\"}) { runtime { ports { ip publicPort privatePort type } } } }\"}" | python3 -c "
import json, sys
runtime = (((json.load(sys.stdin).get('data') or {}).get('pod') or {}).get('runtime') or {})
for p in runtime.get('ports') or []:
    if p.get('privatePort') == 22:
        print(f\"ssh -o StrictHostKeyChecking=accept-new -p {p['publicPort']} root@{p['ip']}\")
        break
else:
    sys.exit('ssh port not ready yet; try again in a few seconds')
"
  ;;
kill)
  ID=$(pod_id)
  gql "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$ID\\\"}) }\"}"; echo
  sleep 5
  gql '{"query":"query { myself { pods { id name desiredStatus } } }"}' | python3 -c "
import json, sys
pods = ((json.load(sys.stdin).get('data') or {}).get('myself') or {}).get('pods') or []
print('terminated' if all(p['id'] != '$ID' for p in pods) else 'STILL PRESENT, check the RunPod console')
"
  mv "$STATE" "$STATE.terminated"
  ;;
*)
  echo "usage: $0 {mint|status|ssh|kill}"; exit 1
  ;;
esac
