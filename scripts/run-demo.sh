#!/usr/bin/env bash
# Run the judged demo with the dashboard up (and, optionally, a public read-only link).
#
#   ./scripts/run-demo.sh --max-parts 6            # dashboard + demo
#   ./scripts/run-demo.sh --tunnel --max-parts 6   # also a public read-only link (cloudflared)
#
# - Starts tools/demo_dashboard.py in the background if it isn't already up:
#     http://127.0.0.1:8900  full console (controls)      — this laptop only
#     http://127.0.0.1:8901  read-only (rubric, run, docs, diagnostics) — what the tunnel shares
# - --tunnel: a Cloudflare quick tunnel to 8901; prints the https://…trycloudflare.com link. Closed when the demo exits.
# - Then runs `uv run viamrobot-demo` with the remaining arguments. Each run writes its own runs/judge-<UTC time>.csv.
# - The dashboard stays up after the demo exits, so the finished run can be shown. Stop it with: kill $(cat runs/dashboard.pid)
# The demo does not depend on the dashboard: if the dashboard fails to start, the demo still runs.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p runs

TUNNEL=0
ARGS=()
for a in "$@"; do
  if [ "$a" = "--tunnel" ]; then TUNNEL=1; else ARGS+=("$a"); fi
done

if curl -fs -o /dev/null --max-time 2 http://127.0.0.1:8900/; then
  echo "dashboard: already running on http://127.0.0.1:8900"
else
  VIAMROBOT_DIR="$ROOT" PUBLIC_PORT=8901 nohup python3 tools/demo_dashboard.py >> runs/dashboard.log 2>&1 &
  echo $! > runs/dashboard.pid
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    curl -fs -o /dev/null --max-time 1 http://127.0.0.1:8900/ && break
    sleep 0.5
  done
  if curl -fs -o /dev/null --max-time 1 http://127.0.0.1:8900/; then
    echo "dashboard: http://127.0.0.1:8900 (read-only: http://127.0.0.1:8901), log runs/dashboard.log"
  else
    echo "dashboard: did not start (see runs/dashboard.log); running the demo anyway" >&2
  fi
fi

TUNNEL_PID=""
if [ "$TUNNEL" = 1 ]; then
  if command -v cloudflared >/dev/null; then
    : > runs/tunnel.log
    cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8901 >> runs/tunnel.log 2>&1 &
    TUNNEL_PID=$!
    URL=""
    for _ in $(seq 1 30); do
      URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' runs/tunnel.log | head -1)
      [ -n "$URL" ] && break
      sleep 1
    done
    if [ -n "$URL" ]; then echo "public read-only link: $URL"; else echo "tunnel: no link yet (see runs/tunnel.log)" >&2; fi
  else
    echo "tunnel: cloudflared not installed (brew install cloudflared); skipping" >&2
  fi
fi

cleanup() { [ -n "$TUNNEL_PID" ] && kill "$TUNNEL_PID" 2>/dev/null && echo "tunnel closed"; }
trap cleanup EXIT

uv run viamrobot-demo "${ARGS[@]+"${ARGS[@]}"}"
