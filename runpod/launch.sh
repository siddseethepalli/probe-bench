#!/usr/bin/env bash
# From the laptop. Adapted from the tutorial recipe (tar over ssh, detached start).
#   runpod/launch.sh setup    # wait for ssh, sync the repo, start remote_setup.sh detached
#   runpod/launch.sh sync     # sync the repo only
#   runpod/launch.sh cache    # copy data/cache/sets into the pod's generation cache
#   runpod/launch.sh start    # sync, then (re)start the server via remote_start.sh
#   runpod/launch.sh logs     # tail the setup and server logs
#   runpod/launch.sh run CMD  # run a command on the pod
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
export RUNPOD_API_KEY="${RUNPOD_API_KEY:-$(grep '^RUNPOD_API_KEY=' "$REPO/.env" | cut -d= -f2-)}"

ssh_cmd() {
  for _ in $(seq 1 60); do
    C=$("$HERE/pod.sh" ssh 2>/dev/null) && case "$C" in ssh*) echo "$C"; return 0;; esac
    sleep 10
  done
  echo "pod ssh never came up" >&2; return 1
}
SSH_CMD=$(ssh_cmd)
PORT=$(echo "$SSH_CMD" | awk '{for (i=1;i<=NF;i++) if ($i=="-p") print $(i+1)}')
HOST=$(echo "$SSH_CMD" | awk '{print $NF}')
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -p $PORT"
for _ in $(seq 1 30); do $SSH -n "$HOST" true 2>/dev/null && break; sleep 10; done   # -n: never consume the caller's stdin

sync_repo() {
  tar -C "$REPO" --exclude .git --exclude node_modules --exclude 'frontend/dist' --exclude .venv \
      --exclude __pycache__ --exclude 'runpod/.pod-id*' --exclude probes --exclude cache -czf - . \
    | $SSH "$HOST" 'mkdir -p /workspace/probe-bench && tar -xzf - --no-same-owner -C /workspace/probe-bench && chmod +x /workspace/probe-bench/runpod/*.sh'
}

case "${1:-}" in
setup)
  sync_repo
  $SSH "$HOST" 'cd /workspace/probe-bench && rm -f /workspace/SETUP_DONE /workspace/FALLBACK_READY && (nohup setsid runpod/remote_setup.sh > /workspace/setup.log 2>&1 < /dev/null &)'
  echo "setup started. follow with: runpod/launch.sh logs" ;;
sync) sync_repo; echo "synced" ;;
sync-head)
  # Ship the committed tree only, so an in-progress edit never reaches the live server.
  git -C "$REPO" archive --format=tar HEAD | gzip | $SSH "$HOST" 'mkdir -p /workspace/probe-bench && tar -xzf - --no-same-owner -C /workspace/probe-bench && chmod +x /workspace/probe-bench/runpod/*.sh'
  echo "synced HEAD ($(git -C "$REPO" rev-parse --short HEAD))" ;;
cache)
  # Ship the reviewed contrast sets into the pod's generation cache, so prebuilt concepts never regenerate.
  tar -C "$REPO/data/cache" -czf - sets | $SSH "$HOST" 'mkdir -p /workspace/data/cache && tar -xzf - --no-same-owner -C /workspace/data/cache && ls /workspace/data/cache/sets | wc -l'
  echo "cache entries on pod (above)" ;;
start) sync_repo; $SSH "$HOST" 'cd /workspace/probe-bench && runpod/remote_start.sh' ;;
logs) $SSH "$HOST" 'tail -n 30 /workspace/setup.log 2>/dev/null; echo ---; tail -n 30 /workspace/server.log 2>/dev/null' ;;
run) shift; $SSH "$HOST" "$@" ;;
*) echo "usage: $0 {setup|sync|start|logs|run CMD}"; exit 1 ;;
esac
