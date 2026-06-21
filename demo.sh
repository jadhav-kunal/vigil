#!/usr/bin/env bash
#
# Vigil 90-second demo — starts the proxy + a scripted looping upstream and trips the breaker,
# so you can watch a runaway loop get caught and the cost meter freeze. No API key required.
#
#   ./demo.sh                 # proxy on :8765 (default)
#   PORT=8766 ./demo.sh       # if :8765 is busy (then dashboard: VITE_VIGIL_WS=ws://localhost:8766/ws)
#   STEPS=14 ./demo.sh        # drive more steps
#
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8765}"
MOCK_PORT="${MOCK_PORT:-8799}"
STEPS="${STEPS:-10}"
DB="$(mktemp -t vigil-demo).db"
LOG="$(mktemp -t vigil-demo-proxy).log"

cleanup() { [ -n "${PROXY_PID:-}" ] && kill "$PROXY_PID" 2>/dev/null || true; rm -f "$DB"*; }
trap cleanup EXIT INT TERM

if lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "✗ Port $PORT is already in use."
  echo "  Run on another port:  PORT=8766 ./demo.sh"
  echo "  (and start the dashboard with VITE_VIGIL_WS=ws://localhost:8766/ws npm run dev)"
  exit 1
fi

echo "▶ Starting Vigil on :$PORT, pointed at a looping mock upstream on :$MOCK_PORT ..."
OPENAI_BASE_URL="http://127.0.0.1:${MOCK_PORT}/v1" VIGIL_EMBED_HASHING=true VIGIL_DB_PATH="$DB" \
  uv run uvicorn vigil_proxy.app:app --port "$PORT" --log-level info >"$LOG" 2>&1 &
PROXY_PID=$!

for _ in $(seq 1 60); do curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
curl -sf "localhost:$PORT/health" >/dev/null 2>&1 || { echo "✗ proxy did not start — see $LOG"; tail -5 "$LOG"; exit 1; }
echo "✓ proxy healthy at http://localhost:$PORT   (proxy logs: $LOG)"
echo

echo "▶ Driving a scripted runaway loop (the agent keeps calling the same tool) ..."
node cli/vigil.js demo --proxy "http://localhost:$PORT" --mock-port "$MOCK_PORT" --steps "$STEPS"
echo

echo "▶ Cross-session aggregate (counts only — never prompt content):"
curl -s "localhost:$PORT/metrics/aggregate" | (python3 -m json.tool 2>/dev/null || cat)
echo

echo "▶ Cached-trace replay of the demo session (zero upstream calls):"
REPLAY="$(curl -s -X POST "localhost:$PORT/sessions/vigil-demo/replay")"
echo "$REPLAY" | python3 -c "import sys,json;d=json.load(sys.stdin);print('  replayed',len(d['steps']),'steps; upstream_calls =',d['upstream_calls'],'; trace_hash =',d['trace_hash'][:12]+'...')" 2>/dev/null \
  || echo "  $REPLAY"
echo
echo "Done. For the live dashboard view: in another terminal,"
echo "  cd packages/dashboard && npm run dev      # then re-run ./demo.sh and watch it live"
