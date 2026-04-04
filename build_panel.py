"""
build_panel.py — flatten snapshot JSONs into a single CSV for analysis.

Usage:
    py build_panel.py                          # reads data/snapshots/, writes data/panel.csv
    py build_panel.py --snapdir custom/path    # custom snapshot dir
    py build_panel.py --output data/my.csv     # custom output path
    py build_panel.py --since 20260401         # only snapshots with this prefix or later

Output columns:
    fetched_at   — timestamp string from the API (UTC)
    exchange     — exchange name (lowercase)
    symbol       — normalized symbol (e.g. BTC, ETH)
    rate         — funding rate × 10000, approximately 8h-normalized by the API
    interval     — funding payment interval in hours for this exchange+symbol
    oi_rank      — open-interest rank (lower = more liquid); NaN if unranked

Notes:
    - funding_rates_raw is currently always null in the API; omitted here.
    - interval comes from funding_intervals[exchange][symbol]; falls back to
      the exchange-level default from exchanges.exchange_names.
    - Rates on the API appear to be expressed as: actual_rate × 10000.
      E.g. 3.75 ≈ 0.0375% per 8h.  Verify with a known reference before
      using absolute values for anything quantitative.
"""

import argparse
import csv
import json
import pathlib
import sys
from typing import Iterator


def _exchange_defaults(snap: dict) -> dict[str, float]:
    """Return {exchange_name: default_interval} from exchange_names metadata."""
    return {
        e["name"]: e.get("interval", 8)
        for e in snap.get("exchanges", {}).get("exchange_names", [])
    }


def _parse_snapshot(path: pathlib.Path) -> Iterator[dict]:
    """Yield one row dict per (exchange, symbol) observation in a snapshot."""
    try:
        with open(path, encoding="utf-8") as f:
            snap = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"cannot parse {path.name}: {e}") from e

    fetched_at = snap.get("timestamp", path.stem)
    funding_rates = snap.get("funding_rates", {})
    funding_intervals = snap.get("funding_intervals", {})
    oi_rankings = snap.get("oi_rankings", {})
    ex_defaults = _exchange_defaults(snap)

    for exchange, rates in funding_rates.items():
        if not isinstance(rates, dict):
            continue
        ex_intervals = funding_intervals.get(exchange, {})
        default_interval = ex_defaults.get(exchange, 8)

        for symbol, rate in rates.items():
            if rate is None:
                continue
            yield {
                "fetched_at": fetched_at,
                "exchange": exchange,
                "symbol": symbol,
                "rate": rate,
                "interval": ex_intervals.get(symbol, default_interval),
                "oi_rank": oi_rankings.get(symbol),
            }


FIELDNAMES = ["fetched_at", "exchange", "symbol", "rate", "interval", "oi_rank"]


def build_panel(snap_dir: pathlib.Path, output: pathlib.Path, since: str | None) -> None:
    snaps = sorted(snap_dir.glob("*.json"))
    if since:
        snaps = [s for s in snaps if s.stem >= since]

    if not snaps:
        print(f"No snapshots found in {snap_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(snaps)} snapshots → {output}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    skipped = 0

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for snap_path in snaps:
            try:
                rows = list(_parse_snapshot(snap_path))
                writer.writerows(rows)
                total_rows += len(rows)
            except ValueError as e:
                print(f"  Warning: skipped — {e}", file=sys.stderr)
                skipped += 1

    print(f"Done: {total_rows:,} rows written ({skipped} snapshots skipped)")


def main():
    parser = argparse.ArgumentParser(description="Flatten snapshots into a panel CSV")
    parser.add_argument("--snapdir", default="data/snapshots", metavar="DIR",
                        help="Directory containing snapshot JSONs (default: data/snapshots)")
    parser.add_argument("--output", default="data/panel.csv", metavar="FILE",
                        help="Output CSV path (default: data/panel.csv)")
    parser.add_argument("--since", default=None, metavar="PREFIX",
                        help="Only include snapshots whose filename >= PREFIX "
                             "(e.g. 20260401 for April 1 onward)")
    args = parser.parse_args()

    build_panel(
        snap_dir=pathlib.Path(args.snapdir),
        output=pathlib.Path(args.output),
        since=args.since,
    )


if __name__ == "__main__":
    main()
