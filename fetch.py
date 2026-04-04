"""
fetch.py — download one funding-rate snapshot and save it locally.

Usage:
    py fetch.py                  # fetch once
    py fetch.py --loop 60        # fetch every 60 s (API refreshes ~60 s)
    py fetch.py --outdir mydata  # custom output dir

Each snapshot is saved as data/snapshots/YYYYMMDD_HHMMSS.json
"""

import argparse
import json
import pathlib
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

API_URL = "https://api.loris.tools/funding"
DEFAULT_OUTDIR = pathlib.Path("data/snapshots")


def fetch_snapshot() -> dict:
    req = urllib.request.Request(API_URL, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def save_snapshot(data: dict, outdir: pathlib.Path) -> pathlib.Path:
    outdir.mkdir(parents=True, exist_ok=True)
    # Use the API's own timestamp if available, otherwise use local time
    api_ts = data.get("timestamp", "")
    if api_ts:
        # "2026-04-02 00:37:15" → "20260402_003715"
        ts = api_ts.replace("-", "").replace(" ", "_").replace(":", "")
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = outdir / f"{ts}.json"
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description="Fetch funding rate snapshots from loris.tools")
    parser.add_argument(
        "--loop", type=int, default=0, metavar="SECONDS",
        help="Re-fetch every N seconds; 0 = fetch once (default: 0)"
    )
    parser.add_argument(
        "--outdir", default=str(DEFAULT_OUTDIR), metavar="DIR",
        help=f"Output directory (default: {DEFAULT_OUTDIR})"
    )
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    interval = args.loop

    while True:
        try:
            data = fetch_snapshot()
            path = save_snapshot(data, outdir)
            n_exchanges = len(data.get("funding_rates", {}))
            n_symbols = len(data.get("symbols", []))
            api_ts = data.get("timestamp", "?")
            print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] "
                  f"saved {path.name}  "
                  f"(api_ts={api_ts}, {n_exchanges} exchanges, {n_symbols} symbols)",
                  flush=True)
        except urllib.error.HTTPError as e:
            print(f"HTTP error {e.code}: {e.reason}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}", file=sys.stderr, flush=True)

        if not interval:
            break
        time.sleep(interval)


if __name__ == "__main__":
    main()
