"""
fetch_prices.py - fetch 1-minute OHLCV from Binance for the same time window
as the collected funding snapshots.

Usage:
    py fetch_prices.py                        # BTC + ETH, auto-detects range from snapshots
    py fetch_prices.py --symbols BTC ETH SOL  # add more symbols
    py fetch_prices.py --snapdir data/snapshots --outdir data/prices
    py fetch_prices.py --since "2026-04-02T00:00:00" --until "2026-04-02T12:00:00"

Output:  data/prices/{SYMBOL}USDT_1m.csv
Columns: open_time (unix seconds UTC), open, high, low, close, volume
"""

import argparse
import csv
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
DEFAULT_SYMBOLS = ["BTC", "ETH"]
FORWARD_PADDING_MINUTES = 70   # extra candles after last snapshot (for forward returns)


# ── helpers ──────────────────────────────────────────────────────────────────

def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def detect_range_from_snapshots(snap_dir: pathlib.Path) -> tuple[datetime, datetime]:
    snaps = sorted(snap_dir.glob("*.json"))
    if not snaps:
        raise ValueError(f"No snapshot files found in {snap_dir}")
    fmt = "%Y%m%d_%H%M%S"
    first = datetime.strptime(snaps[0].stem, fmt).replace(tzinfo=timezone.utc)
    last  = datetime.strptime(snaps[-1].stem, fmt).replace(tzinfo=timezone.utc)
    return first, last


def fetch_klines(pair: str, start_ms: int, end_ms: int) -> list[dict]:
    """Fetch 1m candles from Binance in pages of 1000."""
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        url = (
            f"{BINANCE_KLINES}?symbol={pair}&interval=1m"
            f"&startTime={cursor}&endTime={end_ms}&limit=1000"
        )
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                candles = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            raise RuntimeError(f"HTTP {e.code} for {pair}: {body}") from e

        if not candles:
            break

        for c in candles:
            rows.append({
                "open_time": int(c[0]) // 1000,   # ms -> unix seconds
                "open":  float(c[1]),
                "high":  float(c[2]),
                "low":   float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
            })

        if len(candles) < 1000:
            break                               # fetched everything
        cursor = int(candles[-1][0]) + 60_000   # advance by one minute
        time.sleep(0.05)

    return rows


def save_csv(rows: list[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["open_time", "open", "high", "low", "close", "volume"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch 1m OHLCV from Binance")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        metavar="SYM", help="Base symbols (default: BTC ETH)")
    parser.add_argument("--snapdir", default="data/snapshots", metavar="DIR",
                        help="Snapshot dir used to auto-detect time range")
    parser.add_argument("--outdir",  default="data/prices",    metavar="DIR")
    parser.add_argument("--since",   default=None, metavar="ISO",
                        help="Override start time (e.g. 2026-04-02T00:00:00)")
    parser.add_argument("--until",   default=None, metavar="ISO",
                        help="Override end time")
    args = parser.parse_args()

    # ── time range ────────────────────────────────────────────────────────────
    if args.since and args.until:
        start = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
        end   = datetime.fromisoformat(args.until).replace(tzinfo=timezone.utc)
    else:
        snap_dir = pathlib.Path(args.snapdir)
        if not snap_dir.exists():
            print(f"Snapshot dir not found: {snap_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Detecting range from snapshots in {snap_dir}...", end=" ", flush=True)
        first, last = detect_range_from_snapshots(snap_dir)
        start = first - timedelta(minutes=2)
        end   = last  + timedelta(minutes=FORWARD_PADDING_MINUTES)
        print(f"{first.strftime('%H:%M')} -> {last.strftime('%H:%M')} UTC "
              f"(+{FORWARD_PADDING_MINUTES}m padding)")

    start_ms = to_ms(start)
    end_ms   = to_ms(end)
    span_min = (end - start).total_seconds() / 60
    print(f"Fetching {span_min:.0f} minutes of 1m candles "
          f"({start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%H:%M')} UTC)\n")

    # ── fetch each symbol ─────────────────────────────────────────────────────
    outdir = pathlib.Path(args.outdir)
    for sym in args.symbols:
        pair = f"{sym.upper()}USDT"
        print(f"  {pair:<12}", end=" ", flush=True)
        try:
            rows = fetch_klines(pair, start_ms, end_ms)
            out  = outdir / f"{pair}_1m.csv"
            save_csv(rows, out)
            print(f"{len(rows):>4} candles  ->  {out}")
        except Exception as e:
            print(f"FAILED: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
