#!/bin/bash
# pipeline.sh — rebuild panel and recompute decisions
# Runs via cron every 5 minutes.
# Runs at lowest CPU/IO priority so it never starves the trading bot.
set -euo pipefail

REPO="/home/openclaw/exchange-flow-fingerprint"
PYTHON="$REPO/.venv/bin/python"
LOG="$REPO/data/pipeline.log"
SNAPS="$REPO/data/snapshots"
LOCK="$REPO/data/.pipeline.lock"
MAX_SNAPS=300   # ~5h at 1/min; enough for signal quality, caps disk at ~100MB

cd "$REPO"

# ── Single-instance guard ──────────────────────────────────────────────────────
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S') UTC] skipped (previous run still active)" >> "$LOG"
  exit 0
fi

{
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S') UTC] pipeline start"

  # ── Prune old snapshots ────────────────────────────────────────────────────
  SNAP_COUNT=$(ls -1 "$SNAPS"/*.json 2>/dev/null | wc -l)
  if [ "$SNAP_COUNT" -gt "$MAX_SNAPS" ]; then
    TO_DELETE=$(( SNAP_COUNT - MAX_SNAPS ))
    ls -1t "$SNAPS"/*.json | tail -n "$TO_DELETE" | xargs rm -f
    echo "pruned $TO_DELETE snapshots (kept $MAX_SNAPS)"
  fi

  # ── Rebuild panel + decisions at idle priority ─────────────────────────────
  nice -n 19 ionice -c 3 "$PYTHON" build_panel.py
  nice -n 19 ionice -c 3 "$PYTHON" compute_decisions_cli.py

  echo "[$(date -u '+%Y-%m-%d %H:%M:%S') UTC] pipeline done"

  # ── Rotate this log (keep last 300 lines) ─────────────────────────────────
  tail -300 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
} >> "$LOG" 2>&1
