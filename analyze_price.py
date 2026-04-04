"""
analyze_price.py - check whether funding signals relate to forward price returns.

Aligns per-snapshot funding rate changes with 1-minute OHLCV price data and asks:
  1. Does the cross-exchange consensus funding change predict price direction?
  2. Do early-mover exchanges predict price better than followers?
  3. Is there a quartile relationship (stronger signal -> larger return)?

Important caveats up front:
  - With ~200 snapshots the sample is small; treat correlations as directional hints
  - Funding rate changes per 60s step are tiny; most steps are zero or noise
  - This is a first-pass check, not a backtest

Usage:
    py analyze_price.py                           # BTC + ETH, horizons 5/15/30/60 min
    py analyze_price.py --symbol BTC              # one symbol
    py analyze_price.py --horizons 5 15 30        # custom horizons
    py analyze_price.py --panel data/panel.csv --prices-dir data/prices
"""

import argparse
import pathlib
import sys
import warnings
from datetime import timezone

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("pip install pandas numpy", file=sys.stderr)
    sys.exit(1)

DEFAULT_SYMBOLS  = ["BTC", "ETH"]
DEFAULT_HORIZONS = [5, 15, 30, 60]


# ── helpers ──────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)


def load_panel(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    df["rate"]       = pd.to_numeric(df["rate"], errors="coerce")
    return df


def load_prices(prices_dir: pathlib.Path, symbol: str) -> pd.DataFrame | None:
    path = prices_dir / f"{symbol.upper()}USDT_1m.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="s", utc=True)
    df["close"]     = pd.to_numeric(df["close"], errors="coerce")
    return df.set_index("open_time").sort_index()


def build_forward_returns(prices: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """
    For each 1m candle at time T, compute:
        fwd_Xm = (close[T + X minutes] / close[T] - 1) * 100   [percent]
    """
    fwd = pd.DataFrame(index=prices.index)
    for h in horizons:
        fwd[f"fwd_{h}m"] = (prices["close"].shift(-h) / prices["close"] - 1) * 100
    return fwd


def build_aligned(panel: pd.DataFrame, symbol: str,
                  prices: pd.DataFrame, fwd: pd.DataFrame) -> pd.DataFrame:
    """
    For each (snapshot, exchange) observation:
      - Round snapshot timestamp down to the minute
      - Attach close price and forward returns from that candle

    Then compute:
      - rate_delta: change in this exchange's rate vs previous snapshot
      - consensus_delta: cross-exchange median of rate_delta at that timestamp
    """
    sub = panel[panel["symbol"] == symbol].copy()
    if sub.empty:
        return pd.DataFrame()

    # Floor to minute so we can join on price candles
    sub["ts_min"] = sub["fetched_at"].dt.floor("1min")

    # Attach close price + forward returns
    price_data = prices[["close"]].join(fwd)
    sub = sub.join(price_data, on="ts_min")

    # Compute per-exchange rate deltas
    sub = sub.sort_values(["exchange", "fetched_at"])
    sub["rate_delta"] = sub.groupby("exchange")["rate"].diff()

    # Cross-exchange consensus delta at each timestamp
    cons = (
        sub.groupby("fetched_at")["rate_delta"]
        .median()
        .rename("consensus_delta")
    )
    sub = sub.join(cons, on="fetched_at")

    return sub


# ── lead scores (re-derived from this panel subset) ──────────────────────────

def compute_lead_scores(aligned: pd.DataFrame, min_obs: int = 15) -> pd.Series:
    """
    lead_score = corr(ex_delta[t], consensus[t+1]) - corr(ex_delta[t+1], consensus[t])
    Positive = exchange moves before consensus.
    """
    scores = {}
    for ex, grp in aligned.groupby("exchange"):
        both = grp[["rate_delta", "consensus_delta"]].dropna()
        if len(both) < min_obs:
            continue
        fwd = both["rate_delta"].corr(both["consensus_delta"].shift(-1))
        lag = both["rate_delta"].shift(-1).corr(both["consensus_delta"])
        if pd.notna(fwd) and pd.notna(lag):
            scores[ex] = round(fwd - lag, 4)
    return pd.Series(scores, name="lead_score").sort_values(ascending=False)


# ── report 1: consensus signal vs price ──────────────────────────────────────

def report_consensus(aligned: pd.DataFrame, horizons: list[int]) -> None:
    section("CONSENSUS FUNDING CHANGE vs FORWARD PRICE RETURN")

    # One row per timestamp (dedup)
    ts = (
        aligned.drop_duplicates("fetched_at")
        .dropna(subset=["consensus_delta"])
        .copy()
    )
    # Exclude zero-delta snapshots (nothing changed)
    nonzero = ts[ts["consensus_delta"] != 0]

    fwd_cols = [f"fwd_{h}m" for h in horizons if f"fwd_{h}m" in ts.columns]

    print(f"\n  Total snapshots     : {len(ts)}")
    print(f"  Non-zero delta steps: {len(nonzero)}")
    print(f"  (zero = no exchange changed its rate that step)\n")

    print(f"  {'Horizon':<10}  {'Pearson r':>10}  {'Hit rate':>10}  {'N':>6}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}")

    for col in fwd_cols:
        sub = nonzero[["consensus_delta", col]].dropna()
        if len(sub) < 10:
            print(f"  {col:<10}  {'too few':>10}")
            continue
        r    = sub["consensus_delta"].corr(sub[col])
        hit  = (np.sign(sub["consensus_delta"]) == np.sign(sub[col])).mean()
        print(f"  {col:<10}  {r:>10.4f}  {hit*100:>9.1f}%  {len(sub):>6}")

    # Quartile breakdown for the first horizon
    first_col = fwd_cols[0] if fwd_cols else None
    if first_col:
        sub = nonzero[["consensus_delta", first_col]].dropna()
        if len(sub) >= 20:
            print(f"\n  Quartile breakdown -- consensus_delta vs {first_col}:")
            sub = sub.copy()
            sub["q"] = pd.qcut(sub["consensus_delta"], q=4,
                               labels=["Q1 bearish", "Q2", "Q3", "Q4 bullish"])
            qt = sub.groupby("q", observed=True)[first_col].agg(
                avg_return=("mean"), n=("count")
            )
            print(qt.to_string())
            print(f"\n  Monotone Q1<Q2<Q3<Q4 return would confirm a directional relationship.")


# ── report 2: per-exchange predictive correlation ────────────────────────────

def report_per_exchange(aligned: pd.DataFrame, horizons: list[int],
                        lead_scores: pd.Series) -> None:
    section("PER-EXCHANGE: FUNDING DELTA vs FORWARD RETURN")

    fwd_cols = [f"fwd_{h}m" for h in horizons if f"fwd_{h}m" in aligned.columns]
    if not fwd_cols:
        print("  No forward return columns available.")
        return

    main_col = fwd_cols[0]
    rows = []

    for ex, grp in aligned.groupby("exchange"):
        sub = grp[["rate_delta"] + fwd_cols].dropna(subset=["rate_delta"])
        sub = sub[sub["rate_delta"] != 0]
        if len(sub) < 10:
            continue

        row = {"exchange": ex, "n": len(sub)}
        for col in fwd_cols:
            s = sub[["rate_delta", col]].dropna()
            if len(s) < 5:
                continue
            row[f"r_{col}"]   = round(s["rate_delta"].corr(s[col]), 4)
            row[f"hit_{col}"] = round(
                (np.sign(s["rate_delta"]) == np.sign(s[col])).mean() * 100, 1
            )
        rows.append(row)

    if not rows:
        print("  Not enough non-zero observations per exchange.")
        return

    result = (
        pd.DataFrame(rows)
        .set_index("exchange")
        .join(lead_scores)
        .sort_values(f"r_{main_col}", ascending=False, na_position="last")
    )

    print()
    print(f"  r_fwd_Xm    : Pearson r (funding delta -> price return X min later)")
    print(f"  hit_fwd_Xm  : directional hit rate % (50% = coin flip)")
    print(f"  lead_score  : from timing analysis (>0 = early mover, <0 = follower)")
    print()
    print(result.to_string())


# ── report 3: early movers vs followers ──────────────────────────────────────

def report_early_vs_follower(aligned: pd.DataFrame, horizons: list[int],
                              lead_scores: pd.Series) -> None:
    section("EARLY MOVERS vs FOLLOWERS: AGGREGATE SIGNAL QUALITY")

    early_ex    = lead_scores[lead_scores >= 0.10].index.tolist()
    follower_ex = lead_scores[lead_scores <= -0.30].index.tolist()

    fwd_cols = [f"fwd_{h}m" for h in horizons if f"fwd_{h}m" in aligned.columns]

    print(f"\n  Early movers (lead_score >= 0.10): {early_ex or 'none'}")
    print(f"  Followers   (lead_score <= -0.30): {follower_ex or 'none'}")

    for label, exchange_list in [("Early movers", early_ex), ("Followers", follower_ex)]:
        if not exchange_list:
            continue

        sub = aligned[aligned["exchange"].isin(exchange_list)].copy()
        sub = sub[sub["rate_delta"].notna() & (sub["rate_delta"] != 0)]
        if sub.empty:
            continue

        # Average the signal across exchanges in this group, per timestamp
        group_signal = sub.groupby("fetched_at")["rate_delta"].mean().rename("g")
        fwd_data = (
            sub.drop_duplicates("fetched_at")
            .set_index("fetched_at")
            [[c for c in fwd_cols if c in sub.columns]]
        )
        combined = group_signal.to_frame().join(fwd_data).dropna(subset=["g"])
        combined = combined[combined["g"] != 0]

        print(f"\n  --- {label} ---")
        print(f"  {'Horizon':<10}  {'Pearson r':>10}  {'Hit rate':>10}  {'N':>6}")
        print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}")

        for col in fwd_cols:
            s = combined[["g", col]].dropna()
            if len(s) < 5:
                continue
            r   = s["g"].corr(s[col])
            hit = (np.sign(s["g"]) == np.sign(s[col])).mean()
            print(f"  {col:<10}  {r:>10.4f}  {hit*100:>9.1f}%  {len(s):>6}")

    print()
    print("  Hypothesis: if early movers' hit rate > followers', their signal is")
    print("  informative about price direction, not just reflecting delayed updates.")


# ── report 4: large funding moves → price behaviour ──────────────────────────

def report_large_moves(aligned: pd.DataFrame, horizons: list[int],
                       z: float = 2.0) -> None:
    section(f"LARGE FUNDING MOVES (>={z}x std) vs PRICE")

    ts = aligned.drop_duplicates("fetched_at").dropna(subset=["consensus_delta"])
    std = ts["consensus_delta"].std()
    if std == 0 or pd.isna(std):
        print("  Not enough variance in consensus delta.")
        return

    large_up   = ts[ts["consensus_delta"] >  z * std]
    large_down = ts[ts["consensus_delta"] < -z * std]

    fwd_cols = [f"fwd_{h}m" for h in horizons if f"fwd_{h}m" in ts.columns]
    print(f"\n  std of consensus_delta: {std:.5f}")
    print(f"  Large UP   events (>{z}x std): {len(large_up)}")
    print(f"  Large DOWN events (<-{z}x std): {len(large_down)}")

    if len(large_up) + len(large_down) < 5:
        print("\n  Too few large events yet. Collect more snapshots.")
        return

    print()
    print(f"  {'Direction':<15}  {'N':>4}", end="")
    for col in fwd_cols:
        print(f"  {col:>10}", end="")
    print()
    print(f"  {'-'*15}  {'-'*4}", end="")
    for _ in fwd_cols:
        print(f"  {'-'*10}", end="")
    print()

    for label, subset in [("UP", large_up), ("DOWN", large_down)]:
        if subset.empty:
            continue
        print(f"  {'avg_ret % ('+label+')':<15}  {len(subset):>4}", end="")
        for col in fwd_cols:
            vals = subset[col].dropna()
            avg = vals.mean() if len(vals) >= 3 else float("nan")
            print(f"  {avg:>10.4f}", end="")
        print()

    print()
    print("  Positive avg_ret on UP and negative on DOWN = funding leads price.")
    print("  Mixed or reversed = no relationship (or noise at this sample size).")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Correlate funding signals with price returns")
    parser.add_argument("--panel",      default="data/panel.csv")
    parser.add_argument("--prices-dir", default="data/prices")
    parser.add_argument("--symbol",     nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--horizons",   nargs="+", type=int, default=DEFAULT_HORIZONS,
                        help="Forward return windows in minutes (default: 5 15 30 60)")
    args = parser.parse_args()

    panel_path  = pathlib.Path(args.panel)
    prices_dir  = pathlib.Path(args.prices_dir)

    if not panel_path.exists():
        print(f"Panel not found: {panel_path}. Run build_panel.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {panel_path}...", end=" ", flush=True)
    panel = load_panel(panel_path)
    print(f"{len(panel):,} rows, {panel['fetched_at'].nunique()} snapshots")

    for sym in [s.upper() for s in args.symbol]:
        section(f"SYMBOL: {sym}")

        prices = load_prices(prices_dir, sym)
        if prices is None:
            print(f"\n  No price file found. Run:  py fetch_prices.py --symbols {sym}")
            continue

        t0, t1 = prices.index[0], prices.index[-1]
        print(f"\n  Price: {len(prices)} candles  ({t0.strftime('%H:%M')} to {t1.strftime('%H:%M')} UTC)")

        fwd     = build_forward_returns(prices, args.horizons)
        aligned = build_aligned(panel, sym, prices, fwd)

        if aligned.empty or aligned["close"].isna().all():
            print(f"  No overlapping data between panel and prices for {sym}.")
            continue

        n_snap = aligned["fetched_at"].nunique()
        n_ex   = aligned["exchange"].nunique()
        pct_aligned = aligned["close"].notna().mean()
        print(f"  Aligned: {n_snap} snapshots x {n_ex} exchanges  "
              f"({pct_aligned*100:.0f}% of rows matched to a candle)")

        lead_scores = compute_lead_scores(aligned)

        report_consensus(aligned, args.horizons)
        report_per_exchange(aligned, args.horizons, lead_scores)
        report_early_vs_follower(aligned, args.horizons, lead_scores)
        report_large_moves(aligned, args.horizons)

    print("\n" + "="*60)
    print("  Done.")
    print("="*60)


if __name__ == "__main__":
    main()
