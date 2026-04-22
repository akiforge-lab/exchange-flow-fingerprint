"""
compute_decisions_cli.py — standalone decision engine for cron use.

Reads data/panel.csv, scores every symbol using the same leader-aware
medium-horizon logic as dashboard.py, and writes data/.decisions.json.

No Streamlit dependency.  Designed to be run by pipeline.sh every 5 min.

Usage:
    python compute_decisions_cli.py [--panel data/panel.csv] [--exec binance hyperliquid]
                                    [--log-signals] [--no-log-signals]
"""

import argparse
import datetime
import json
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

# ── constants ──────────────────────────────────────────────────────────────────

PANEL_DEFAULT   = pathlib.Path("data/panel.csv")
DECISIONS_CACHE = pathlib.Path("data/.decisions.json")
EXEC_DEFAULT    = ["aster", "binance", "hyperliquid"]
MIN_NON_ZERO    = 10

# Churn-reduction constants (fallback if no config file)
SMOOTH_WIN         = 5
ENTRY_THRESHOLD    = 0.38
EXIT_THRESHOLD     = 0.28
ENTRY_NO_EXEC      = 0.45
EXIT_NO_EXEC       = 0.35

# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_panel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    df["rate"]       = pd.to_numeric(df["rate"], errors="coerce")
    df = df.dropna(subset=["symbol", "exchange"])
    df["symbol"]   = df["symbol"].astype(str)
    df["exchange"] = df["exchange"].astype(str)
    return df


def save_decisions(df: pd.DataFrame) -> None:
    DECISIONS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    # Use pandas to_json so NaN → null (json.dumps writes NaN as bare NaN, invalid JSON)
    records = json.loads(df.reset_index().to_json(orient="records"))
    DECISIONS_CACHE.write_text(json.dumps({
        "computed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "records":     records,
    }))


def load_decisions() -> tuple[pd.DataFrame | None, str]:
    """Load last saved decisions for hysteresis.  Returns (df, computed_at) or (None, '')."""
    try:
        obj = json.loads(DECISIONS_CACHE.read_text())
        df  = pd.DataFrame(obj["records"])
        if "rank" in df.columns:
            df = df.set_index("rank")
        return df, obj.get("computed_at", "")
    except Exception:
        return None, ""

# ── scoring primitives ─────────────────────────────────────────────────────────

def _lead_score_for(delta: pd.DataFrame, ex: str, cons: pd.Series) -> dict | None:
    d = delta[ex].dropna()
    nonzero = d[d != 0]
    if len(nonzero) < MIN_NON_ZERO:
        return None
    both = pd.DataFrame({"d": d, "c": cons}).dropna()
    if len(both) < MIN_NON_ZERO:
        return None
    fwd = both["d"].corr(both["c"].shift(-1))
    lag = both["d"].shift(-1).corr(both["c"])
    if not (pd.notna(fwd) and pd.notna(lag)):
        return None
    return {"lead_score": float(fwd - lag),
            "pct_zero":   float((d == 0).mean()),
            "change_mag": float(nonzero.abs().mean()),
            "fwd_corr":   float(fwd), "lag_corr": float(lag)}


def compute_exchange_authority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-symbol exchange authority: average lead score across all symbols.
    Returns DataFrame indexed by exchange with columns: avg_lead, symbol_count, role.
    """
    scores: dict[str, list[float]] = {}
    for sym, sub in df.groupby("symbol"):
        pivot = (sub.pivot_table(index="fetched_at", columns="exchange",
                                 values="rate", aggfunc="first").sort_index())
        if len(pivot) < 15:
            continue
        delta = pivot.diff()
        cons  = delta.median(axis=1)
        for ex in delta.columns:
            m = _lead_score_for(delta, ex, cons)
            if m is None:
                continue
            scores.setdefault(ex, []).append(m["lead_score"])

    if not scores:
        return pd.DataFrame(columns=["avg_lead", "symbol_count", "role"])

    rows = []
    for ex, vals in scores.items():
        avg  = float(np.mean(vals))
        role = "leader" if avg > 0.03 else "follower" if avg < -0.03 else "neutral"
        rows.append({"exchange": ex, "avg_lead": round(avg, 4),
                     "symbol_count": len(vals), "role": role})

    return (pd.DataFrame(rows)
            .set_index("exchange")
            .sort_values("avg_lead", ascending=False))


def _derive_direction(rate_zscore: float, trend_dir: float, trend_persist: float,
                      exec_lag: float, watch_score: float) -> tuple[str, str, str]:
    """
    Medium-horizon direction heuristic (500-1000 min holding).
    Identical logic to dashboard.py _derive_direction.
    """
    parts: list[str] = []
    conf  = 0.15

    if abs(rate_zscore) < 0.5 and trend_persist < 0.30:
        return "NEUTRAL", "low", "funding near mean, no persistent trend"

    if abs(rate_zscore) >= 1.5:
        base_dir = -int(np.sign(rate_zscore))
        side     = "longs" if rate_zscore > 0 else "shorts"
        parts.append(f"funding {rate_zscore:+.1f}σ from mean ({side} crowded)")
        conf += 0.25
        if trend_dir != 0 and np.sign(trend_dir) == np.sign(base_dir):
            parts.append("trend reversing toward mean")
            conf = min(conf + 0.18, 0.85)
        elif trend_dir != 0:
            parts.append("trend still extending -- crowded side building")
            conf = max(conf - 0.05, 0.15)

    elif trend_persist >= 0.45:
        base_dir = int(np.sign(trend_dir)) if trend_dir != 0 else 0
        if base_dir == 0:
            return "NEUTRAL", "low", "no clear trend direction"
        direction_word = "rising" if trend_dir > 0 else "falling"
        parts.append(f"funding {direction_word}, {trend_persist:.0%} consistent over 2h")
        conf += 0.10 + trend_persist * 0.15

    elif abs(rate_zscore) >= 0.8:
        base_dir = -int(np.sign(rate_zscore))
        parts.append(f"mild funding bias ({rate_zscore:+.1f}σ)")
        conf += 0.08

    else:
        return "NEUTRAL", "low", "no structural signal"

    if pd.notna(exec_lag) and exec_lag < -0.05:
        parts.append("exec lagging (window open)")
        conf = min(conf + 0.15, 0.85)

    if watch_score >= 0.50:
        conf = min(conf + 0.05, 0.85)

    direction = "LONG" if base_dir > 0 else "SHORT" if base_dir < 0 else "NEUTRAL"
    label     = "high" if conf >= 0.60 else "moderate" if conf >= 0.40 else "low"
    return direction, label, "; ".join(parts)

# ── main decision computation ──────────────────────────────────────────────────

def compute_decisions(
    panel_path: str,
    exec_exchanges: tuple,
    cfg: dict | None = None,
    log_signals: bool = False,
) -> pd.DataFrame:
    """
    Per-symbol decision table.  Columns:
      symbol, opportunity, direction, confidence, watch_score,
      leader_gap, exec_lag, best_venue, reason
    Sorted: opportunities first, then by confidence, then watch_score, then symbol.

    If log_signals=True and signal_log / venue_ranking modules are available,
    appends a signal snapshot record to data/signal_log.jsonl.
    """
    # ── Config ────────────────────────────────────────────────────────────────
    if cfg is None:
        try:
            from scoring_config import load_config
            cfg = load_config()
        except ImportError:
            cfg = {}

    w_rate_extreme  = float(cfg.get("w_rate_extreme",  0.30))
    w_trend_persist = float(cfg.get("w_trend_persist", 0.30))
    w_leader_gap    = float(cfg.get("w_leader_gap",    0.25))
    w_data_qual     = float(cfg.get("w_data_qual",     0.15))
    smooth_win      = int(cfg.get("smooth_win", SMOOTH_WIN))

    entry_threshold = float(cfg.get("entry_threshold", ENTRY_THRESHOLD))
    exit_threshold  = float(cfg.get("exit_threshold",  EXIT_THRESHOLD))
    entry_no_exec   = float(cfg.get("entry_no_exec",   ENTRY_NO_EXEC))
    exit_no_exec    = float(cfg.get("exit_no_exec",    EXIT_NO_EXEC))

    # ── Venue ranking import (optional, soft dep) ─────────────────────────────
    try:
        from venue_ranking import rank_venues, venues_to_dict, best_venue as _best_venue
        _venue_ranking_ok = True
    except ImportError:
        _venue_ranking_ok = False

    # ── Signal log import (optional, soft dep) ────────────────────────────────
    _sig_append = None
    if log_signals:
        try:
            from signal_log import build_signal_record, append_records
            _sig_append = (build_signal_record, append_records)
        except ImportError:
            pass

    df       = load_panel(panel_path)
    has_exec = len(exec_exchanges) > 0

    # Run ID for grouping signal log records
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")

    # Exchange authority: stable cross-symbol leader / follower classification
    try:
        authority  = compute_exchange_authority(df)
        leader_exs = set(authority[authority["role"] == "leader"].index)
    except Exception:
        authority  = pd.DataFrame()
        leader_exs = set()

    # Hysteresis: symbols that were already YES get a lower exit threshold
    try:
        _prev, _ = load_decisions()
        prev_yes = set(_prev.loc[_prev["opportunity"] == "YES", "symbol"].tolist()) \
                   if _prev is not None and "opportunity" in _prev.columns else set()
    except Exception:
        prev_yes = set()

    rows: list[dict] = []
    signal_records: list[dict] = []

    for sym, sub in df.groupby("symbol"):
        pivot = (sub.pivot_table(index="fetched_at", columns="exchange",
                                 values="rate", aggfunc="first").sort_index())
        if len(pivot) < 15:
            continue

        delta = pivot.diff()
        pz    = (delta == 0).mean()

        # ── Leader consensus ──────────────────────────────────────────────────
        leader_cols     = [ex for ex in delta.columns
                           if ex in leader_exs and pz.get(ex, 1.0) < 0.7]
        use_leader_cons = len(leader_cols) >= 2
        signal_delta    = delta[leader_cols] if use_leader_cons else delta
        signal_cons     = signal_delta.median(axis=1)

        # ── 1. Rate extreme ───────────────────────────────────────────────────
        cons_rate    = pivot.median(axis=1)
        sw           = min(smooth_win, len(cons_rate))
        current_rate = float(cons_rate.iloc[-sw:].mean())
        rate_mean    = float(cons_rate.mean())
        rate_std     = float(cons_rate.std())

        if rate_std > 1e-10:
            rate_zscore  = (current_rate - rate_mean) / rate_std
            rate_extreme = float(min(abs(rate_zscore) / 3.0, 1.0))
        else:
            rate_zscore  = 0.0
            rate_extreme = 0.0

        # ── 2. Trend persistence (leader consensus, 2h window) ────────────────
        TREND_WIN  = min(120, len(signal_cons) - 1)
        recent_sc  = signal_cons.iloc[-TREND_WIN:]
        nonzero_sc = recent_sc[recent_sc != 0]
        if len(nonzero_sc) > 5:
            trend_dir     = float(np.sign(nonzero_sc.mean()))
            frac_same_dir = float((np.sign(nonzero_sc) == trend_dir).mean())
            trend_persist = max(0.0, (frac_same_dir - 0.5) * 2.0)
        else:
            trend_dir     = 0.0
            trend_persist = 0.0

        # ── 3. Per-venue rates + leader→exec gap ──────────────────────────────
        # venue_rates: {exchange: smoothed_rate} for exec exchanges present in data
        venue_rates: dict[str, float] = {}
        exec_active  = 0
        exec_leads   = []
        for ex in exec_exchanges:
            if ex not in pivot.columns:
                continue
            col_slice = pivot[ex].iloc[-sw:]
            if col_slice.notna().any():
                val = float(col_slice.mean())
                venue_rates[ex] = val
            m = _lead_score_for(delta, ex, signal_cons)
            if m:
                exec_leads.append(m["lead_score"])
                if m["pct_zero"] < 0.7:
                    exec_active += 1

        exec_rate_now = float(np.nanmean(list(venue_rates.values()))) if venue_rates else float("nan")
        exec_lag      = float(np.mean(exec_leads)) if exec_leads else float("nan")

        leader_rate_now = (float(np.nanmedian(pivot[leader_cols].iloc[-sw:].values))
                           if leader_cols else float("nan"))
        ref_rate = leader_rate_now if pd.notna(leader_rate_now) else current_rate

        if pd.notna(exec_rate_now) and rate_std > 1e-10:
            leader_gap = float(min(abs(exec_rate_now - ref_rate) / (rate_std + 1e-10) / 2.0, 1.0))
        else:
            leader_gap = 0.0

        # ── 4. Data quality ───────────────────────────────────────────────────
        active_all = sum(1 for ex in delta.columns if pz.get(ex, 1.0) < 0.7)
        data_qual  = active_all / max(len(delta.columns), 1)

        # ── Watch score (config-driven weights) ───────────────────────────────
        watch_score = (
            w_rate_extreme  * rate_extreme  +
            w_trend_persist * trend_persist +
            w_leader_gap    * leader_gap    +
            w_data_qual     * data_qual
        )

        # ── Opportunity flag (with hysteresis) ────────────────────────────────
        has_structural = abs(rate_zscore) > 0.8 or trend_persist > 0.35
        leader_moving  = use_leader_cons and trend_persist > 0.25
        was_yes        = sym in prev_yes
        thresh         = exit_threshold if was_yes else entry_threshold
        thresh_nx      = exit_no_exec   if was_yes else entry_no_exec
        if has_exec:
            opportunity = (
                watch_score >= thresh
                and has_structural
                and exec_active >= 1
                and (leader_gap > 0.10 or (pd.notna(exec_lag) and exec_lag < -0.02))
            )
        else:
            opportunity = (
                watch_score >= thresh_nx
                and has_structural
                and (leader_moving or abs(rate_zscore) > 1.0)
            )

        # ── Direction + reason ────────────────────────────────────────────────
        direction, confidence, reason = _derive_direction(
            rate_zscore, trend_dir, trend_persist, exec_lag, watch_score
        )
        if use_leader_cons and leader_gap > 0.10:
            names  = ", ".join(leader_cols[:2])
            reason = reason + f"; leader flow ({names}) not yet in exec"

        # ── Venue ranking ─────────────────────────────────────────────────────
        venue_scores_list: list[dict] = []
        best_v: str | None = None
        if _venue_ranking_ok and venue_rates and direction != "NEUTRAL":
            scores = rank_venues(direction, venue_rates, ref_rate, rate_std)
            venue_scores_list = venues_to_dict(scores)
            best_v = _best_venue(scores)

        rows.append({
            "symbol":      sym,
            "opportunity": "YES" if opportunity else "",
            "direction":   direction,
            "confidence":  confidence,
            "watch_score": round(watch_score, 3),
            "leader_gap":  round(leader_gap, 3),
            "exec_lag":    round(exec_lag, 3) if pd.notna(exec_lag) else None,
            "best_venue":  best_v,
            "reason":      reason,
        })

        # ── Signal log record ─────────────────────────────────────────────────
        if _sig_append is not None:
            build_fn, _ = _sig_append
            rec = build_fn(
                run_id        = run_id,
                symbol        = sym,
                direction     = direction,
                confidence    = confidence,
                opportunity   = bool(opportunity),
                watch_score   = watch_score,
                rate_zscore   = rate_zscore,
                trend_persist = trend_persist,
                leader_gap    = leader_gap,
                exec_lag      = exec_lag if pd.notna(exec_lag) else None,
                data_qual     = data_qual,
                cons_rate     = float(current_rate),
                rate_std      = float(rate_std),
                venues        = venue_scores_list,
                best_venue    = best_v,
            )
            signal_records.append(rec)

    # Flush signal log once for the whole run
    if signal_records and _sig_append is not None:
        _, append_fn = _sig_append
        try:
            append_fn(signal_records)
        except Exception as e:
            print(f"Warning: could not write signal log: {e}", file=sys.stderr)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["_s1"] = (out["opportunity"] == "YES").astype(int)
    out["_s2"] = out["confidence"].map({"high": 3, "moderate": 2, "low": 1}).fillna(0)
    out = (out.sort_values(["_s1", "_s2", "watch_score", "symbol"],
                            ascending=[False, False, False, True])
              .drop(columns=["_s1", "_s2"])
              .reset_index(drop=True))
    out.index = out.index + 1
    out.index.name = "rank"
    return out

# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    global DECISIONS_CACHE
    parser = argparse.ArgumentParser(description="Compute and save funding-rate decisions.")
    parser.add_argument("--panel", default=str(PANEL_DEFAULT),
                        help="Path to panel CSV (default: data/panel.csv)")
    parser.add_argument("--exec", nargs="*", dest="exec_exchanges",
                        default=EXEC_DEFAULT,
                        help="Execution exchange names (default: aster binance hyperliquid)")
    parser.add_argument("--out", default=str(DECISIONS_CACHE),
                        help="Output path for decisions JSON")
    parser.add_argument("--log-signals", action="store_true", default=False,
                        help="Append signal snapshots to data/signal_log.jsonl")
    parser.add_argument("--no-log-signals", action="store_true", default=False,
                        help="Disable signal logging (overrides --log-signals)")
    args = parser.parse_args()

    DECISIONS_CACHE = pathlib.Path(args.out)

    panel_path = pathlib.Path(args.panel)
    if not panel_path.exists():
        print(f"Panel not found: {panel_path}", file=sys.stderr)
        sys.exit(1)

    exec_exchanges = tuple(args.exec_exchanges or [])
    log_signals    = args.log_signals and not args.no_log_signals

    # Load config (falls back to defaults silently)
    try:
        from scoring_config import load_config, config_summary
        cfg = load_config()
        print(f"Config: {config_summary(cfg)}")
    except ImportError:
        cfg = None

    decisions = compute_decisions(str(panel_path), exec_exchanges, cfg=cfg, log_signals=log_signals)

    if decisions.empty:
        print("No decisions generated (not enough data).", file=sys.stderr)
        sys.exit(0)

    save_decisions(decisions)
    n_opp = int((decisions["opportunity"] == "YES").sum())
    print(f"Saved {len(decisions)} symbols, {n_opp} opportunities → {DECISIONS_CACHE}")
    if log_signals:
        print(f"Signal snapshots appended → data/signal_log.jsonl")


if __name__ == "__main__":
    main()
