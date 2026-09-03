#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROUTER_PY:-$ROOT/.venv/bin/python}"
export ROUTER_CONFIG="${ROUTER_CONFIG:-$ROOT/router/config/router.yaml}"
mkdir -p "$ROOT/runtime/logs" "$ROOT/runtime/state"
if [[ -f "$ROOT/runtime/state/gateway.json" ]]; then
  echo 'Gateway record exists. Check the existing process before starting another.' >&2
  exit 1
fi
# Foreground supervision avoids starting the wrong process when the port is occupied.
# Keep this terminal open or use the documented terminal multiplexer.
cd "$ROOT/router"
echo "Gateway runs in this terminal; logs: $ROOT/runtime/logs/gateway.log"
exec "$PY" "$ROOT/scripts/run_gateway.py" >>"$ROOT/runtime/logs/gateway.log" 2>&1
