#!/usr/bin/env bash
# ============================================================
# run.sh – Start the Panoptic Segmentation Web Application
# ============================================================
set -euo pipefail

APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"
WORKERS="${WORKERS:-1}"

echo "======================================================"
echo "  Panoptic Segmentation Web Application"
echo "======================================================"
echo "  Host      : $APP_HOST"
echo "  Port      : $APP_PORT"
echo "  Log level : $LOG_LEVEL"
echo "======================================================"
echo ""

# Ensure virtual environment packages are on the path if a venv is present
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Create storage directories if they don't exist
mkdir -p uploads outputs

# Start the FastAPI application via uvicorn
exec uvicorn app.main:app \
  --host "$APP_HOST" \
  --port "$APP_PORT" \
  --workers "$WORKERS" \
  --log-level "$LOG_LEVEL"
