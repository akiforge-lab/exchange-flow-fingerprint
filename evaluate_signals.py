"""
evaluate_signals.py — attach forward price outcomes to logged signals and
produce a human-readable evaluation report.

Usage:
    python evaluate_signals.py [--log data/signal_log.jsonl]
                               [--prices data/prices]
                               [--horizon 500]
                               [--report]
                               [--min-outcomes N]

Workflow:
  1. Load signal_log.jsonl.
  2. For each record where direction_correct is null:
       - Locate the price CSV for that symbol (data/prices/<SYM>USDT_1m.csv).
       - Find the 1-minute bar closest to logged_at → price_at_signal.
       - Find the bar at logged_at + horizon minutes → price_500m.
       - Compute return_500m_pct.
       - Set direction_correct based on sign alignment with direction.
  3. Overwrite signal_log.jsonl with updated records.
  4. If --report: print a Markdown summary to stdout.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}\npip install pandas numpy", file=sys.stderr)
    sys.exit(1)

from signal_log import load_log, save_log, log_stats

HORIZON_DEFAULT = 500   # minutes
PRICES_DEFAULT  = pathlib.Path("data/prices")
LOG_DEFAULT     = pathlib.Path("data/signal_log.jsonl")


# ── price helpers ──────────────────────────────────────────────────────────────

def _load_price_index(prices_dir: pathlib.Path, symbol: str) -> pd.Series | None:
    """
    Load close prices indexed by UTC timestamp for a symbol.
    Returns None if file missing or unreadable.
    """
    path = prices_dir / f"{symbol}USDT_1m.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df["ts"] = pd.to_datetime(df["open_time"], unit="s", utc=True)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        return df.set_index("ts")["close"].sort_index().dropna()
    except Exception:
        return None


def _nearest_price(series: pd.Series, ts: pd.Timestamp) -> float | None:
    """Return close price for the bar nearest to ts (within ±5 min)."""
    if series.empty:
        return None
    idx = series.index.get_indexer([ts], method="nearest")[0]
    if idx < 0:
        return None
    bar_ts = series.index[idx]
    if abs((bar_ts - ts).total_seconds()) > 300:
        return None
    val = series.iloc[idx]
    return float(val) if math.isfinite(val) else None


# ── outcome attachment ─────────────────────────────────────────────────────────

def attach_outcomes(
    records: list[dict],
    prices_dir: pathlib.Path,
    horizon_min: int,
) -> tuple[int, int]:
    """
    Mutate records in place, filling outcome fields for pending entries.
    Returns (n_attached, n_skipped).
    """
    price_cache: dict[str, pd.Series | None] = {}
    attached = 0
    skipped = 0

    for rec in records:
        if rec.get("direction_correct") is not None:
            continue   # already evaluated

        sym = rec.get("symbol", "")
        direction = rec.get("direction", "NEUTRAL")
        if direction == "NEUTRAL":
            # Nothing to evaluate — mark as not applicable using False
            # (keeps it out of accuracy metrics but removes from "pending")
            rec["direction_correct"] = None   # leave pending; no price needed
            skipped += 1
            continue

        if sym not in price_cache:
            price_cache[sym] = _load_price_index(prices_dir, sym)
        series = price_cache[sym]
        if series is None:
            skipped += 1
            continue

        try:
            signal_ts = pd.Timestamp(rec["logged_at"]).tz_convert("UTC")
        except Exception:
            skipped += 1
            continue

        exit_ts = signal_ts + pd.Timedelta(minutes=horizon_min)

        p_entry = _nearest_price(series, signal_ts)
        p_exit  = _nearest_price(series, exit_ts)

        if p_entry is None or p_exit is None:
            skipped += 1
            continue

        ret_pct = (p_exit - p_entry) / p_entry * 100.0
        correct: bool | None
        if direction == "LONG":
            correct = ret_pct > 0
        elif direction == "SHORT":
            correct = ret_pct < 0
        else:
            correct = None

        rec["price_at_signal"]  = round(p_entry, 6)
        rec["price_500m"]       = round(p_exit, 6)
        rec["return_500m_pct"]  = round(ret_pct, 4)
        rec["direction_correct"] = correct
        attached += 1

    return attached, skipped


# ── reporting ──────────────────────────────────────────────────────────────────

def _accuracy(records: list[dict], filter_fn=None) -> tuple[int, float | None]:
    """Return (n, accuracy) for records that pass filter_fn."""
    subset = [r for r in records
              if r.get("direction_correct") is not None
              and r.get("direction") != "NEUTRAL"
              and (filter_fn is None or filter_fn(r))]
    n = len(subset)
    if n == 0:
        return 0, None
    acc = sum(1 for r in subset if r["direction_correct"]) / n
    return n, acc


def _fmt_pct(v: float | None) -> str:
    return f"{v*100:.1f}%" if v is not None else "—"


def print_report(records: list[dict], horizon_min: int) -> None:
    print(f"\n## Signal Evaluation Report  (horizon = {horizon_min} min)\n")

    stats = log_stats(records)
    print(f"- Total records : {stats['total_records']}")
    print(f"- With outcome  : {stats['with_outcome']}")
    print(f"- Pending       : {stats['pending_outcome']}")
    print(f"- Opportunities : {stats['opportunities_logged']}")
    print()

    evaluated = [r for r in records if r.get("direction_correct") is not None
                 and r.get("direction") != "NEUTRAL"]
    if not evaluated:
        print("No evaluated records yet.")
        return

    n_all, acc_all = _accuracy(records)
    print(f"### Overall accuracy: {_fmt_pct(acc_all)}  (n={n_all})\n")

    # By confidence
    print("### By confidence")
    for conf in ("high", "moderate", "low"):
        n, acc = _accuracy(records, lambda r, c=conf: r.get("confidence") == c)
        print(f"  {conf:8s}  {_fmt_pct(acc)}  (n={n})")
    print()

    # By direction
    print("### By direction")
    for d in ("LONG", "SHORT"):
        n, acc = _accuracy(records, lambda r, dd=d: r.get("direction") == dd)
        print(f"  {d:5s}  {_fmt_pct(acc)}  (n={n})")
    print()

    # Opportunity-flagged vs unflagged
    print("### Opportunity-flagged vs not")
    for opp, label in [(True, "YES"), (False, "no")]:
        n, acc = _accuracy(records, lambda r, o=opp: bool(r.get("opportunity")) == o)
        print(f"  {label:4s}  {_fmt_pct(acc)}  (n={n})")
    print()

    # Component correlation with direction_correct
    print("### Component correlation with direction_correct (Spearman)")
    try:
        ev_df = pd.DataFrame([
            {
                "direction_correct": int(r["direction_correct"]),
                "watch_score":    r.get("watch_score"),
                "rate_zscore":    abs(r.get("rate_zscore", 0) or 0),
                "trend_persist":  r.get("trend_persist"),
                "leader_gap":     r.get("leader_gap"),
                "exec_lag":       r.get("exec_lag"),
            }
            for r in evaluated
            if r.get("direction_correct") is not None
        ])
        for col in ["watch_score", "rate_zscore", "trend_persist", "leader_gap", "exec_lag"]:
            sub = ev_df[["direction_correct", col]].dropna()
            if len(sub) < 5:
                print(f"  {col:16s}  — (n<5)")
                continue
            rho = float(sub["direction_correct"].corr(sub[col], method="spearman"))
            print(f"  {col:16s}  rho={rho:+.3f}  (n={len(sub)})")
    except Exception as exc:
        print(f"  (correlation failed: {exc})")
    print()

    # Average return by direction
    print("### Average return_500m_pct")
    ret_df = pd.DataFrame([
        {"direction": r["direction"], "ret": r["return_500m_pct"]}
        for r in evaluated
        if r.get("return_500m_pct") is not None
    ])
    if not ret_df.empty:
        for d, grp in ret_df.groupby("direction"):
            print(f"  {d:5s}  mean={grp['ret'].mean():+.2f}%  median={grp['ret'].median():+.2f}%  (n={len(grp)})")
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Attach forward outcomes to signal log.")
    parser.add_argument("--log",     default=str(LOG_DEFAULT),     help="Path to signal_log.jsonl")
    parser.add_argument("--prices",  default=str(PRICES_DEFAULT),  help="Directory with price CSVs")
    parser.add_argument("--horizon", default=HORIZON_DEFAULT, type=int,
                        help=f"Forward horizon in minutes (default {HORIZON_DEFAULT})")
    parser.add_argument("--report",  action="store_true",          help="Print evaluation report after attaching")
    parser.add_argument("--min-outcomes", default=0, type=int,
                        help="Minimum evaluated records required before printing report")
    args = parser.parse_args()

    log_path    = pathlib.Path(args.log)
    prices_dir  = pathlib.Path(args.prices)
    horizon_min = args.horizon

    records = load_log(log_path)
    if not records:
        print("Signal log is empty or missing.")
        if args.report:
            return
        sys.exit(0)

    print(f"Loaded {len(records)} records from {log_path}")

    attached, skipped = attach_outcomes(records, prices_dir, horizon_min)
    print(f"Attached outcomes: {attached}  (skipped/pending: {skipped})")

    if attached > 0:
        save_log(records, log_path)
        print(f"Log updated → {log_path}")

    if args.report:
        stats = log_stats(records)
        if stats["with_outcome"] < args.min_outcomes:
            print(f"Only {stats['with_outcome']} evaluated records (min={args.min_outcomes}). Skipping report.")
        else:
            print_report(records, horizon_min)


if __name__ == "__main__":
    main()
