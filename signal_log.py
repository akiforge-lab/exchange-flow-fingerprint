"""
signal_log.py — append-only JSONL signal log with forward outcome attachment.

Each record captures the full scoring snapshot at decision time so that
evaluate_signals.py can later attach forward price outcomes and grade
the signal quality.

Schema (one JSON object per line):
  logged_at       ISO-8601 UTC timestamp of when the record was written
  run_id          ISO date "YYYY-MM-DD HH:MM" of the compute run (group key)
  symbol          e.g. "BTC"
  direction       "LONG" | "SHORT" | "NEUTRAL"
  confidence      "high" | "moderate" | "low"
  opportunity     true | false
  watch_score     float
  rate_zscore     float
  trend_persist   float
  leader_gap      float
  exec_lag        float | null
  data_qual       float
  cons_rate       float   (consensus rate at snapshot time)
  rate_std        float
  venues          list[{venue, edge_norm, edge_dir, recommendation, venue_rate, coverage}]
  best_venue      str | null
  # ── data health fields (set at write time, used by tune_weights.py to filter degraded records) ──
  active_exchanges  int      — exchanges with pct_zero < 0.7 for this symbol
  leader_count      int      — leader-class exchanges active; 0 means use_leader_cons=False
  exec_count        int      — exec venues with ≥1 valid rate in smoothing window
                               CRITICAL: leader_gap=0.0 is ambiguous without this field.
                               When exec_count=0, leader_gap=0 means "missing data", NOT "aligned".
                               tune_weights.py filters out records where exec_count < min_exec_count.
  missing_venues    list[str]— exec exchanges that had no valid rate in window
  use_leader_cons   bool     — whether trend signal used leader consensus (True) or all exchanges
  ref_rate_source   str      — "leaders" if ref_rate came from leader median; "consensus" if fallback
  # — filled in later by evaluate_signals.py —
  price_at_signal float | null
  price_500m      float | null
  return_500m_pct float | null
  direction_correct bool | null   (null until outcome attached)
"""

from __future__ import annotations

import datetime
import json
import pathlib
from typing import Any

LOG_PATH = pathlib.Path("data/signal_log.jsonl")


# ── write helpers ──────────────────────────────────────────────────────────────

def build_signal_record(
    *,
    run_id: str,
    symbol: str,
    direction: str,
    confidence: str,
    opportunity: bool,
    watch_score: float,
    rate_zscore: float,
    trend_persist: float,
    leader_gap: float,
    exec_lag: float | None,
    data_qual: float,
    cons_rate: float,
    rate_std: float,
    venues: list[dict],
    best_venue: str | None,
    # ── data health fields ────────────────────────────────────────────────────
    active_exchanges: int,
    leader_count: int,
    exec_count: int,
    missing_venues: list[str],
    use_leader_cons: bool,
    ref_rate_source: str,
) -> dict[str, Any]:
    """
    Build a signal record dict ready for appending to the log.

    Data health fields are required so tune_weights.py can filter out records
    where key inputs were missing or degraded:
    - exec_count=0  → leader_gap=0.0 is a missing-data zero, not a true-alignment zero
    - leader_count=0 → trend_persist used all exchanges (not leader consensus)
    - ref_rate_source="consensus" → leader_gap computed against consensus fallback
    """
    return {
        "logged_at":        datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "run_id":           run_id,
        "symbol":           symbol,
        "direction":        direction,
        "confidence":       confidence,
        "opportunity":      opportunity,
        "watch_score":      round(watch_score, 4),
        "rate_zscore":      round(rate_zscore, 4),
        "trend_persist":    round(trend_persist, 4),
        "leader_gap":       round(leader_gap, 4),
        "exec_lag":         (round(exec_lag, 4) if exec_lag is not None else None),
        "data_qual":        round(data_qual, 4),
        "cons_rate":        cons_rate,
        "rate_std":         rate_std,
        "venues":           venues,
        "best_venue":       best_venue,
        # ── data health ───────────────────────────────────────────────────────
        "active_exchanges": active_exchanges,
        "leader_count":     leader_count,
        "exec_count":       exec_count,
        "missing_venues":   missing_venues,
        "use_leader_cons":  use_leader_cons,
        "ref_rate_source":  ref_rate_source,
        # ── outcome fields — filled later by evaluate_signals.py ───────────────
        "price_at_signal":  None,
        "price_500m":       None,
        "return_500m_pct":  None,
        "direction_correct": None,
    }


def append_records(records: list[dict[str, Any]], path: pathlib.Path = LOG_PATH) -> None:
    """Atomically append records to the JSONL log (one JSON object per line)."""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(json.dumps(r, allow_nan=False) for r in records) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(lines)


# ── read helpers ───────────────────────────────────────────────────────────────

def load_log(path: pathlib.Path = LOG_PATH) -> list[dict[str, Any]]:
    """
    Read all records from the JSONL log.
    Skips blank and malformed lines.  Returns [] if file does not exist.
    """
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def save_log(records: list[dict[str, Any]], path: pathlib.Path = LOG_PATH) -> None:
    """Overwrite the log with the given records (used by evaluate_signals.py)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(json.dumps(r, allow_nan=False) for r in records)
    path.write_text(lines + "\n" if lines else "", encoding="utf-8")


# ── log stats ─────────────────────────────────────────────────────────────────

def log_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return basic counts for /status endpoint or CLI reporting."""
    total = len(records)
    with_outcome = sum(1 for r in records if r.get("direction_correct") is not None)
    opportunities = sum(1 for r in records if r.get("opportunity"))
    return {
        "total_records": total,
        "with_outcome": with_outcome,
        "pending_outcome": total - with_outcome,
        "opportunities_logged": opportunities,
    }
