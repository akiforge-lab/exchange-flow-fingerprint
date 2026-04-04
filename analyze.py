"""
analyze.py - analyze exchange-level funding rate behavior from the panel CSV.

Usage:
    py analyze.py                        # full report (top 30 OI symbols)
    py analyze.py --top 50               # use top 50 OI symbols
    py analyze.py --symbol BTC ETH SOL   # focus on specific symbols
    py analyze.py --min-snapshots 10     # skip analyses that need more data
    py analyze.py --panel data/panel.csv # custom panel path
    py analyze.py --dynamics-only        # skip static reports, run only time-series reports

Static reports (work with even a few snapshots):
    1. Coverage         - how many symbols / snapshots per exchange
    2. Deviation        - persistent level offsets from cross-exchange median
    3. Outlier freq     - how often each exchange is an extreme reading
    4. Latest spread    - cross-exchange spread in the most recent snapshot

Time-series reports (need 50+ consecutive snapshots):
    5. Change stats     - per-snapshot rate change magnitude, direction, autocorrelation
    6. Lead / lag       - which exchanges move before vs after the consensus
    7. Overshoot        - which exchanges tend to reverse after large moves
    8. Fingerprint      - combined summary: structural bias vs reactive behavior

Run `py fetch.py --loop 60` to collect data continuously.
"""

import argparse
import pathlib
import sys
import warnings
from typing import Optional

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Missing dependencies.  Run:  pip install pandas numpy", file=sys.stderr)
    sys.exit(1)


# ── helpers ──────────────────────────────────────────────────────────────────

def load_panel(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["fetched_at"] = pd.to_datetime(df["fetched_at"])
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df["oi_rank"] = pd.to_numeric(df["oi_rank"], errors="coerce")
    return df


def select_symbols(df: pd.DataFrame, top: int, explicit: list[str]) -> list[str]:
    if explicit:
        return [s.upper() for s in explicit]
    # Use OI rank to pick the most liquid symbols that appear across many exchanges
    ranked = (
        df[df["oi_rank"].notna()]
        .groupby("symbol")["oi_rank"]
        .min()
        .nsmallest(top)
        .index.tolist()
    )
    if ranked:
        return ranked
    # Fallback: symbols with most observations
    return df["symbol"].value_counts().head(top).index.tolist()


def section(title: str) -> None:
    print(f"\n{'='*60}", flush=True)
    print(f"  {title}", flush=True)
    print('='*60, flush=True)


def _safe_print(s: str) -> None:
    """Print with fallback for terminals that can't handle non-ASCII."""
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", errors="replace").decode("ascii"))


# ── report 1: coverage ────────────────────────────────────────────────────────

def report_coverage(df: pd.DataFrame) -> None:
    section("COVERAGE REPORT")
    n_snaps = df["fetched_at"].nunique()
    t_min = df["fetched_at"].min()
    t_max = df["fetched_at"].max()
    span_min = (t_max - t_min).total_seconds() / 60 if n_snaps > 1 else 0

    print(f"  Snapshots : {n_snaps}")
    print(f"  Time range: {t_min} to {t_max}")
    print(f"  Span      : {span_min:.0f} minutes")
    print()

    cov = (
        df.groupby("exchange")
        .agg(
            symbols=("symbol", "nunique"),
            observations=("rate", "count"),
            snapshots=("fetched_at", "nunique"),
        )
        .sort_values("symbols", ascending=False)
    )
    cov["obs_per_snap"] = (cov["observations"] / cov["snapshots"]).round(1)
    print(cov.to_string())


# ── report 2: cross-exchange deviation ───────────────────────────────────────

def report_deviation(df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    suffix = "..." if len(symbols) > 10 else ""
    section(f"CROSS-EXCHANGE DEVIATION  (symbols: {', '.join(symbols[:10])}{suffix})")

    sub = df[df["symbol"].isin(symbols)].copy()

    if sub.empty:
        print("  No data for selected symbols.")
        return pd.DataFrame()

    # Median rate per (timestamp, symbol) across all exchanges at that moment
    group_med = sub.groupby(["fetched_at", "symbol"])["rate"].median().rename("cross_median")
    sub = sub.join(group_med, on=["fetched_at", "symbol"])
    sub["dev"] = sub["rate"] - sub["cross_median"]
    sub["abs_dev"] = sub["dev"].abs()

    ex_stats = (
        sub.groupby("exchange")
        .agg(
            obs=("rate", "count"),
            mean_rate=("rate", "mean"),
            std_rate=("rate", "std"),
            mean_abs_dev=("abs_dev", "mean"),
            median_signed_dev=("dev", "median"),
            pct_above_median=("dev", lambda x: (x > 0).mean()),
        )
        .round(4)
        .sort_values("mean_abs_dev", ascending=False)
    )

    print()
    print("  mean_abs_dev: average |exchange_rate - cross_exchange_median| per snapshot")
    print("  median_signed_dev: persistent bias (+ = this exchange runs hotter than peers)")
    print("  pct_above_median: fraction of observations above cross-exchange median")
    print()
    print(ex_stats.to_string())
    return sub


# ── report 3: outlier frequency ──────────────────────────────────────────────

def report_outliers(sub_with_dev: pd.DataFrame, threshold_multiplier: float = 2.0) -> None:
    section("OUTLIER FREQUENCY")

    if sub_with_dev.empty or "abs_dev" not in sub_with_dev.columns:
        print("  (no deviation data)")
        return

    # For each (timestamp, symbol), flag if abs_dev > threshold * median_abs_dev
    group_mad = (
        sub_with_dev.groupby(["fetched_at", "symbol"])["abs_dev"]
        .median()
        .rename("mad_cross")
    )
    sub = sub_with_dev.join(group_mad, on=["fetched_at", "symbol"])
    sub["is_outlier"] = sub["abs_dev"] > (threshold_multiplier * sub["mad_cross"])

    outlier_stats = (
        sub.groupby("exchange")
        .agg(
            obs=("rate", "count"),
            pct_outlier=("is_outlier", "mean"),
            n_outlier=("is_outlier", "sum"),
        )
        .round(3)
        .sort_values("pct_outlier", ascending=False)
    )
    outlier_stats["pct_outlier"] = (outlier_stats["pct_outlier"] * 100).round(1)
    outlier_stats = outlier_stats.rename(columns={"pct_outlier": "pct_outlier_%"})

    print(f"\n  Threshold: |dev| > {threshold_multiplier}x median absolute deviation at that snapshot")
    print()
    print(outlier_stats.to_string())


# ── report 4: lead-lag via cross-correlation ─────────────────────────────────

def report_lead_lag(df: pd.DataFrame, symbols: list[str], min_snapshots: int) -> None:
    section("LEAD-LAG ANALYSIS")

    n_snaps = df["fetched_at"].nunique()
    if n_snaps < min_snapshots:
        print(f"\n  Skipped: only {n_snaps} snapshot(s) available.")
        print(f"  This analysis needs at least {min_snapshots} consecutive snapshots.")
        print(f"  Collect more data:  py fetch.py --loop 60")
        return

    sub = df[df["symbol"].isin(symbols)].copy()

    # Build a pivot: rows = timestamp, columns = exchange, values = median rate across symbols
    # This gives a per-exchange "aggregate signal" over time
    pivot = (
        sub.groupby(["fetched_at", "exchange"])["rate"]
        .median()
        .unstack("exchange")
        .sort_index()
    )

    # Drop exchanges with too many missing values
    coverage_pct = pivot.notna().mean()
    pivot = pivot.loc[:, coverage_pct >= 0.5]

    if pivot.shape[1] < 2:
        print("  Not enough exchange coverage for lead-lag analysis.")
        return

    # First-difference (rate change per step)
    delta = pivot.diff().dropna()

    if len(delta) < 10:
        print(f"  Not enough time steps ({len(delta)}) for meaningful correlation.")
        return

    print(f"\n  Using {len(delta)} time steps, {pivot.shape[1]} exchanges")
    print(f"  Signal: median rate change across symbols {', '.join(symbols[:5])}")

    # Cross-correlations at lag +1 and -1 to detect who leads
    exchanges = list(delta.columns)
    lead_scores = {}

    for ex in exchanges:
        lag_correlations = []
        for other in exchanges:
            if other == ex:
                continue
            # How well does ex at t predict other at t+1?
            x = delta[ex].iloc[:-1].values
            y = delta[other].iloc[1:].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 5:
                continue
            corr = np.corrcoef(x[mask], y[mask])[0, 1]
            lag_correlations.append(corr)
        if lag_correlations:
            lead_scores[ex] = float(np.mean(lag_correlations))

    if not lead_scores:
        print("  Could not compute lead scores.")
        return

    lead_df = (
        pd.Series(lead_scores, name="avg_lead_corr")
        .sort_values(ascending=False)
        .to_frame()
        .round(4)
    )
    print()
    print("  avg_lead_corr: average correlation of exchange[t] vs others[t+1]")
    print("  Higher = tends to move before peers (caution: needs many snapshots)")
    print()
    print(lead_df.to_string())


# ── report 5: per-symbol snapshot of current divergence ──────────────────────

def report_latest_snapshot(df: pd.DataFrame, symbols: list[str]) -> None:
    section("LATEST SNAPSHOT - CROSS-EXCHANGE SPREAD")

    latest_ts = df["fetched_at"].max()
    sub = df[(df["fetched_at"] == latest_ts) & (df["symbol"].isin(symbols))].copy()

    if sub.empty:
        print(f"  No data at latest timestamp {latest_ts}")
        return

    print(f"\n  Timestamp: {latest_ts}")
    print(f"  Symbols  : {', '.join(symbols[:10])}")
    print()

    group_med = sub.groupby("symbol")["rate"].median().rename("median")
    group_std = sub.groupby("symbol")["rate"].std().rename("std")
    group_n = sub.groupby("symbol")["rate"].count().rename("n_exchanges")
    group_min = sub.groupby("symbol")["rate"].min().rename("min")
    group_max = sub.groupby("symbol")["rate"].max().rename("max")

    spread = pd.concat([group_n, group_min, group_max, group_med, group_std], axis=1)
    spread["spread"] = (spread["max"] - spread["min"]).round(4)
    spread = spread.sort_values("spread", ascending=False)
    print(spread.round(4).to_string())


# ── time-series helpers ──────────────────────────────────────────────────────

def build_delta_panel(
    df: pd.DataFrame,
    symbols: list[str],
    min_coverage: float = 0.6,
) -> dict[str, pd.DataFrame]:
    """
    For each symbol, build a (timestamps x exchanges) DataFrame of per-snapshot
    rate changes (first differences).  Only exchanges with >= min_coverage
    fraction of timestamps present are kept.

    Returns {symbol: delta_df}.  Symbols with fewer than 3 qualifying exchanges
    or fewer than 10 time steps are dropped.
    """
    sub = df[df["symbol"].isin(symbols)].copy()
    panels = {}

    for sym, grp in sub.groupby("symbol"):
        pivot = (
            grp.pivot_table(index="fetched_at", columns="exchange", values="rate", aggfunc="first")
            .sort_index()
        )
        # keep only exchanges with decent coverage
        pivot = pivot.loc[:, pivot.notna().mean() >= min_coverage]
        if pivot.shape[1] < 3 or len(pivot) < 10:
            continue
        panels[sym] = pivot.diff()   # rate change per snapshot step

    return panels


def _consensus(delta: pd.DataFrame) -> pd.Series:
    """Cross-exchange median delta at each timestamp."""
    return delta.median(axis=1)


# ── report 5: change-based signals ───────────────────────────────────────────

def report_change_stats(panels: dict[str, pd.DataFrame], min_obs: int = 20) -> pd.DataFrame:
    """
    Per-exchange statistics on how much and how directionally they move.

    autocorr_lag1:
      negative  = exchange overshoots and snaps back next period
      near zero = changes are essentially independent (noise)
      positive  = moves tend to continue in the same direction (momentum)

    pct_zero: high values suggest a stale/slow-updating exchange.
    """
    section("CHANGE-BASED SIGNALS (per-snapshot rate changes)")

    rows = []
    for sym, delta in panels.items():
        cons = _consensus(delta)
        for ex in delta.columns:
            s = delta[ex].dropna()
            if len(s) < min_obs:
                continue
            rows.append({
                "exchange": ex,
                "symbol": sym,
                "mean_abs_change": s.abs().mean(),
                "std_change": s.std(),
                "pct_positive": (s > 0).mean(),
                "pct_zero": (s == 0).mean(),
                "autocorr_lag1": s.autocorr(lag=1) if len(s) > 10 else np.nan,
            })

    if not rows:
        print("  Not enough consecutive data yet.")
        return pd.DataFrame()

    raw = pd.DataFrame(rows)

    agg = (
        raw.groupby("exchange")
        .agg(
            n_symbols=("symbol", "count"),
            mean_abs_change=("mean_abs_change", "mean"),
            std_change=("std_change", "mean"),
            pct_positive=("pct_positive", "mean"),
            pct_zero=("pct_zero", "mean"),
            autocorr_lag1=("autocorr_lag1", "mean"),
        )
        .round(4)
        .sort_values("mean_abs_change", ascending=False)
    )

    print()
    print("  mean_abs_change : avg |rate[t] - rate[t-1]| per snapshot")
    print("  pct_zero        : fraction of steps with no change (stale data warning if high)")
    print("  autocorr_lag1   : <0 = overshoot/revert  ~0 = noise  >0 = momentum")
    print()
    print(agg.to_string())
    return raw


# ── report 6: lead / lag analysis ────────────────────────────────────────────

def report_lead_lag_v2(panels: dict[str, pd.DataFrame], min_obs: int = 20) -> pd.DataFrame:
    """
    For each symbol: compute the cross-exchange consensus delta at each step.
    Then for each exchange measure:

      lead_corr  = corr(exchange[t],   consensus[t+1])   exchange predicts future consensus
      contemp    = corr(exchange[t],   consensus[t])      exchange moves with consensus
      lag_corr   = corr(exchange[t+1], consensus[t])      consensus predicts future exchange

      lead_score = lead_corr - lag_corr

    lead_score > 0  ->  exchange tends to move BEFORE the consensus (early mover)
    lead_score ~ 0  ->  no systematic timing advantage
    lead_score < 0  ->  exchange tends to move AFTER the consensus (follower)

    Caution: interpreting this correctly requires the exchange to have real
    price discovery, not just a slow update cycle.  Check pct_zero in change
    stats -- a high pct_zero exchange may appear to "lead" simply because it
    rarely updates (its delta is always zero, weakly correlated with everything).
    """
    section("LEAD / LAG  (change-based, per-symbol cross-correlation)")

    rows = []
    for sym, delta in panels.items():
        cons = _consensus(delta)

        for ex in delta.columns:
            aligned = pd.DataFrame({"ex": delta[ex], "cons": cons}).dropna()
            if len(aligned) < min_obs:
                continue

            fwd  = aligned["ex"].corr(aligned["cons"].shift(-1))   # ex[t] -> cons[t+1]
            cont = aligned["ex"].corr(aligned["cons"])              # ex[t] -> cons[t]
            lag  = aligned["ex"].shift(-1).corr(aligned["cons"])    # ex[t+1] -> cons[t]

            if pd.isna(fwd) or pd.isna(lag):
                continue

            rows.append({
                "exchange": ex,
                "symbol": sym,
                "lead_corr": round(fwd, 4),
                "contemp_corr": round(cont, 4),
                "lag_corr": round(lag, 4),
                "lead_score": round(fwd - lag, 4),
                "n_obs": len(aligned),
            })

    if not rows:
        print("  Not enough data to compute lead/lag.")
        return pd.DataFrame()

    raw = pd.DataFrame(rows)

    agg = (
        raw.groupby("exchange")
        .agg(
            n_symbols=("symbol", "count"),
            lead_corr=("lead_corr", "mean"),
            contemp_corr=("contemp_corr", "mean"),
            lag_corr=("lag_corr", "mean"),
            lead_score=("lead_score", "mean"),
        )
        .round(4)
        .sort_values("lead_score", ascending=False)
    )

    print()
    print("  lead_score > 0  : moves BEFORE consensus  (early mover)")
    print("  lead_score ~ 0  : moves WITH consensus     (no timing signal)")
    print("  lead_score < 0  : moves AFTER consensus    (follower)")
    print()
    print(agg.to_string())
    return raw


# ── report 7: overshoot / reversion ──────────────────────────────────────────

def report_overshoot(panels: dict[str, pd.DataFrame], large_move_z: float = 1.5,
                     min_obs: int = 20) -> pd.DataFrame:
    """
    Measures whether an exchange tends to move more than its peers and then reverse.

    excess[t] = exchange_delta[t] - consensus_delta[t]
      (how much more or less it moved vs the median peer)

    reversal_rate   : fraction of steps where sign(excess[t]) != sign(excess[t+1])
                      >50% means it oscillates more than a coin flip -> overshoot signal
    cond_reversal   : same, but only after large excess moves (> large_move_z sigma)
                      this is the more meaningful metric -- random noise will also reverse ~50%
    excess_autocorr : autocorr of excess at lag 1
                      strongly negative = consistent overshoot-and-snap pattern
    mean_excess_abs : average magnitude of the excess move (how far from peers it strays)
    """
    section(f"OVERSHOOT / REVERSION  (excess vs consensus, large_move_z={large_move_z})")

    rows = []
    for sym, delta in panels.items():
        cons = _consensus(delta)

        for ex in delta.columns:
            s = delta[ex].dropna()
            aligned_cons = cons.reindex(s.index).dropna()
            s = s.reindex(aligned_cons.index)

            excess = (s - aligned_cons).dropna()
            if len(excess) < min_obs:
                continue

            ex_std = excess.std()
            if ex_std == 0:
                continue

            # pair each excess[t] with excess[t+1]
            e_now  = excess.iloc[:-1].values
            e_next = excess.iloc[1:].values
            valid  = np.isfinite(e_now) & np.isfinite(e_next)
            e_now, e_next = e_now[valid], e_next[valid]

            if len(e_now) < 10:
                continue

            reversal = (np.sign(e_now) != np.sign(e_next)).mean()

            large_mask = np.abs(e_now) > large_move_z * ex_std
            if large_mask.sum() >= 5:
                cond_rev = (np.sign(e_now[large_mask]) != np.sign(e_next[large_mask])).mean()
            else:
                cond_rev = np.nan

            autocorr = excess.autocorr(lag=1) if len(excess) > 10 else np.nan

            rows.append({
                "exchange": ex,
                "symbol": sym,
                "reversal_rate": reversal,
                "cond_reversal": cond_rev,
                "excess_autocorr": autocorr,
                "mean_excess_abs": excess.abs().mean(),
                "n_large_moves": int(large_mask.sum()),
            })

    if not rows:
        print("  Not enough data.")
        return pd.DataFrame()

    raw = pd.DataFrame(rows)

    agg = (
        raw.groupby("exchange")
        .agg(
            n_symbols=("symbol", "count"),
            reversal_rate=("reversal_rate", "mean"),
            cond_reversal=("cond_reversal", "mean"),
            excess_autocorr=("excess_autocorr", "mean"),
            mean_excess_abs=("mean_excess_abs", "mean"),
        )
        .round(3)
    )
    agg["reversal_%"]      = (agg.pop("reversal_rate") * 100).round(1)
    agg["cond_reversal_%"] = (agg.pop("cond_reversal") * 100).round(1)
    agg = agg.sort_values("cond_reversal_%", ascending=False, na_position="last")

    print()
    print(f"  reversal_%      : % of steps where excess reversed (random baseline ~50%)")
    print(f"  cond_reversal_% : same, only after moves >{large_move_z}x std  (more signal)")
    print(f"  excess_autocorr : <0 = systematic overshoot pattern")
    print(f"  mean_excess_abs : how far this exchange typically strays from peers")
    print()
    print(agg.to_string())
    return raw


# ── report 8: combined fingerprint ───────────────────────────────────────────

def report_fingerprint(
    df: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    change_raw: pd.DataFrame,
    lead_raw: pd.DataFrame,
    overshoot_raw: pd.DataFrame,
    symbols: list[str],
) -> None:
    """
    One table combining all signals.  The goal is to separate:

      Structurally different  : high level_bias + stable over time -> by design, not noise
      Early mover             : positive lead_score + low pct_zero
      Follower / reactive     : negative lead_score, moves after consensus
      Noisy / low quality     : high mean_excess_abs + high cond_reversal + negative excess_autocorr
    """
    section("EXCHANGE FINGERPRINT  (structural bias vs dynamic behavior)")

    sub = df[df["symbol"].isin(symbols)].copy()
    grp_med = sub.groupby(["fetched_at", "symbol"])["rate"].median().rename("cross_median")
    sub = sub.join(grp_med, on=["fetched_at", "symbol"])
    sub["dev"] = sub["rate"] - sub["cross_median"]

    level = sub.groupby("exchange").agg(
        level_bias=("dev", "mean"),
        level_variability=("dev", "std"),
    ).round(3)

    parts = [level]

    if not change_raw.empty:
        parts.append(
            change_raw.groupby("exchange").agg(
                change_mag=("mean_abs_change", "mean"),
                pct_zero=("pct_zero", "mean"),
                autocorr=("autocorr_lag1", "mean"),
            ).round(3)
        )

    if not lead_raw.empty:
        parts.append(
            lead_raw.groupby("exchange")["lead_score"].mean().rename("lead_score").round(3).to_frame()
        )

    if not overshoot_raw.empty:
        parts.append(
            overshoot_raw.groupby("exchange").agg(
                cond_reversal=("cond_reversal", "mean"),
                excess_abs=("mean_excess_abs", "mean"),
            ).round(3)
        )

    fp = pd.concat(parts, axis=1).sort_values(
        "lead_score" if "lead_score" in pd.concat(parts, axis=1).columns else "level_bias",
        ascending=False,
        na_position="last",
    )

    print()
    print("  level_bias      : persistent rate offset from peers (structural if stable)")
    print("  level_variability: std of that offset -- low = truly structural, high = episodic")
    print("  change_mag      : avg move per snapshot")
    print("  pct_zero        : fraction of steps with no change (high = slow/stale feed)")
    print("  autocorr        : change autocorr -- neg = overshoot, pos = momentum")
    print("  lead_score      : >0 early, ~0 with consensus, <0 follower")
    print("  cond_reversal   : probability of reversal after large excess move")
    print("  excess_abs      : how far this exchange typically strays from the median peer")
    print()
    print(fp.to_string())
    print()
    print("  Typical profiles:")
    print("    Structural outlier : |level_bias| high + level_variability low + change_mag low")
    print("    Early mover        : lead_score high + pct_zero low + autocorr near 0")
    print("    Follower           : lead_score negative + contemp_corr high")
    print("    Noisy / reactive   : excess_abs high + cond_reversal high + autocorr negative")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze exchange funding rate behavior")
    parser.add_argument("--panel", default="data/panel.csv", metavar="FILE",
                        help="Panel CSV from build_panel.py (default: data/panel.csv)")
    parser.add_argument("--top", type=int, default=30, metavar="N",
                        help="Top N OI-ranked symbols to use (default: 30)")
    parser.add_argument("--symbol", nargs="+", metavar="SYM",
                        help="Explicit symbol list (overrides --top)")
    parser.add_argument("--min-snapshots", type=int, default=50, metavar="N",
                        help="Min snapshots required for time-series reports (default: 50)")
    parser.add_argument("--outlier-threshold", type=float, default=2.0,
                        help="Outlier = |dev| > N x cross-median-abs-dev (default: 2.0)")
    parser.add_argument("--dynamics-only", action="store_true",
                        help="Skip static reports; run only time-series reports")
    parser.add_argument("--min-coverage", type=float, default=0.6, metavar="F",
                        help="Min fraction of snapshots an exchange must have per symbol (default: 0.6)")
    args = parser.parse_args()

    panel_path = pathlib.Path(args.panel)
    if not panel_path.exists():
        print(f"Panel not found: {panel_path}", file=sys.stderr)
        print("Run:  py build_panel.py", file=sys.stderr)
        sys.exit(1)

    # suppress numpy divide-by-zero warnings that arise from autocorr on constant series
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")

    print(f"Loading {panel_path}...", end=" ", flush=True)
    df = load_panel(panel_path)
    n_snaps = df["fetched_at"].nunique()
    print(f"{len(df):,} rows, {df['exchange'].nunique()} exchanges, "
          f"{df['symbol'].nunique()} symbols, "
          f"{n_snaps} snapshots")

    symbols = select_symbols(df, args.top, args.symbol or [])

    # ── static reports ────────────────────────────────────────────────────────
    if not args.dynamics_only:
        report_coverage(df)
        sub = report_deviation(df, symbols)
        report_outliers(sub, threshold_multiplier=args.outlier_threshold)
        report_latest_snapshot(df, symbols)

    # ── time-series reports ───────────────────────────────────────────────────
    if n_snaps < args.min_snapshots:
        print(f"\n  Time-series reports need >= {args.min_snapshots} snapshots "
              f"(have {n_snaps}).  Run: py fetch.py --loop 60")
    else:
        print(f"\nBuilding delta panels for {len(symbols)} symbols...", end=" ", flush=True)
        panels = build_delta_panel(df, symbols, min_coverage=args.min_coverage)
        print(f"{len(panels)} symbols with enough exchange coverage")

        if not panels:
            print("  No symbols had enough coverage for time-series analysis.")
        else:
            change_raw    = report_change_stats(panels)
            lead_raw      = report_lead_lag_v2(panels)
            overshoot_raw = report_overshoot(panels)
            report_fingerprint(df, panels, change_raw, lead_raw, overshoot_raw, symbols)

    print("\n" + "="*60)
    print("  Done.")
    print("="*60)


if __name__ == "__main__":
    main()
