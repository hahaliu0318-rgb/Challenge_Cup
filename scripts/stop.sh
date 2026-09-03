#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"${ROUTER_PY:-$ROOT/.venv/bin/python}" "$ROOT/scripts/stop_gateway.py"
