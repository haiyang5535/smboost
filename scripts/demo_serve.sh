#!/usr/bin/env bash
# Serve frontend/ over http and open the demo visualizer in the default browser.
#
# Usage:  scripts/demo_serve.sh [port]
#
# Default port: 8080. Run scripts/demo_driver.py first to populate
# frontend/demo_trace.jsonl — without it the page shows a placeholder.
set -euo pipefail

PORT="${1:-8080}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="${REPO_ROOT}/frontend"

if [ ! -f "${FRONTEND_DIR}/index.html" ]; then
  echo "error: ${FRONTEND_DIR}/index.html not found" >&2
  echo "       (frontend/ is gitignored — it must exist locally)" >&2
  exit 1
fi

URL="http://localhost:${PORT}/index.html"

echo "[demo_serve] serving ${FRONTEND_DIR} on ${URL}"
echo "[demo_serve] Ctrl-C to stop"

# Open the browser in the background (slight delay so the server is ready)
(
  sleep 0.8
  if command -v open >/dev/null 2>&1; then
    open "${URL}" || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${URL}" || true
  fi
) &

cd "${FRONTEND_DIR}"
exec python3 -m http.server "${PORT}" --bind 127.0.0.1
