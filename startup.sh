#!/bin/sh
# startup.sh
# ─────────────────────────────────────────────────────────────────────────────
# POSIX-compliant pre-flight script for Antigravity deployment.
# Performs environment validation, directory setup, and launches Streamlit.
#
# Environment variables (all optional):
#   PORT       Override the Streamlit server port  (default: 8501)
#   DATA_PATH  Override the path to zomato.csv     (default: data/zomato.csv)
# ─────────────────────────────────────────────────────────────────────────────

set -e   # exit immediately on any error
set -u   # treat unset variables as an error

# ── Resolve defaults ──────────────────────────────────────────────────────────
PORT="${PORT:-8501}"
DATA_PATH="${DATA_PATH:-data/zomato.csv}"

# ── Banner ────────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  Zomato Restaurant Recommender — startup"
echo "========================================================"
echo "  Timestamp : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "  PORT      : ${PORT}"
echo "  DATA_PATH : ${DATA_PATH}"

# ── Python version ────────────────────────────────────────────────────────────
echo ""
echo "--- Python version ---"
python --version

# ── Validate data file ────────────────────────────────────────────────────────
echo ""
echo "--- Validating data file ---"
if [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: Dataset not found at '${DATA_PATH}'."
    echo "       Place zomato.csv there or set the DATA_PATH environment variable."
    exit 1
fi
echo "OK: ${DATA_PATH} found."

# ── Create runtime directories ────────────────────────────────────────────────
echo ""
echo "--- Creating runtime directories ---"
mkdir -p cache
echo "OK: cache/ ready."

# ── Health check ──────────────────────────────────────────────────────────────
echo ""
echo "--- Running health check ---"
python health_check.py
echo "Health check passed."

# ── Launch Streamlit ──────────────────────────────────────────────────────────
echo ""
echo "--- Launching Streamlit on port ${PORT} ---"
exec streamlit run app.py \
    --server.port="${PORT}" \
    --server.address="0.0.0.0" \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
