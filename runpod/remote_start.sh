#!/usr/bin/env bash
# Runs ON the pod: (re)start the API server detached, with a watchdog that restarts it if it dies.
# Log: /workspace/server.log. Set MODEL_ID to override (e.g. the fallback while the 27B downloads).
set -euo pipefail
export HF_HOME=/workspace/hf
cd /workspace/probe-bench
set -a; [ -f .env ] && . ./.env; set +a
# The watchdog re-launches the server, so the model choice and data dir are written to a file it sources.
printf 'MODEL_ID=%s\nDATA_DIR=%s\n' "${MODEL_ID:-Qwen/Qwen3.8-27B}" "${DATA_DIR:-/workspace/data}" > /workspace/server.env
pkill -f '[w]atchdog.sh' || true  # bracket pattern: never matches the pkill command itself
pkill -f '[u]vicorn backend.server:app' || true
sleep 1
cat > /workspace/watchdog.sh <<'W'
#!/usr/bin/env bash
# probe-bench-watchdog
export HF_HOME=/workspace/hf
cd /workspace/probe-bench
set -a; [ -f .env ] && . ./.env; [ -f /workspace/server.env ] && . /workspace/server.env; set +a
while true; do
  if ! pgrep -f "uvicorn backend.server:app" >/dev/null; then
    (nohup setsid .venv/bin/uvicorn backend.server:app --host 0.0.0.0 --port 8000 >> /workspace/server.log 2>&1 < /dev/null &)
  fi
  sleep 20
done
W
chmod +x /workspace/watchdog.sh
(nohup setsid /workspace/watchdog.sh > /workspace/watchdog.log 2>&1 < /dev/null &)
echo "server starting; log: /workspace/server.log"
