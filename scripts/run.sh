#!/usr/bin/env bash
# Start the YuE2 web UI + API. Environment variables are documented at the top
# of service/serve_yue2.py (YUE2_DTYPE, YUE2_OUT_DIR, YUE2_IDLE_UNLOAD_SECONDS, ...).
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
Usage: scripts/run.sh [PORT]

Starts the service on YUE2_HOST:PORT (default 127.0.0.1:5830) using ./.venv.
Set YUE2_HOST=0.0.0.0 to reach it from other machines on your network (there is no login).
EOF
  exit 0
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${YUE2_VENV:-$HERE/.venv}"
[ -x "$VENV/bin/uvicorn" ] || { echo "no venv at $VENV; run scripts/install.sh first" >&2; exit 1; }

export YUE2_DTYPE="${YUE2_DTYPE:-float16}"
export HF_HUB_ENABLE_HF_TRANSFER=0
PORT="${1:-${YUE2_PORT:-5830}}"

cd "$HERE/service"
exec "$VENV/bin/uvicorn" serve_yue2:app --host "${YUE2_HOST:-127.0.0.1}" --port "$PORT" \
  --workers 1 --timeout-keep-alive 75 --no-access-log
