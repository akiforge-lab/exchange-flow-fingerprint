"""
dashboard.py  --  Exchange Flow Fingerprint dashboard.

Two tabs:
  Opportunities  (default) -- compact decision table: symbol / opportunity / direction / confidence / reason
  Detail                   -- drill-down for one symbol: price, funding, exchange breakdown

Run:
    streamlit run dashboard.py
"""

import argparse
import datetime
import json
import pathlib
import subprocess
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import pandas as pd
    import numpy as np
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError as e:
    print(f"Missing dependency: {e}\npip install pandas numpy streamlit plotly", file=sys.stderr)
    sys.exit(1)

# ── constants ─────────────────────────────────────────────────────────────────

PANEL_DEFAULT      = pathlib.Path("data/panel.csv")
SNAP_DIR           = pathlib.Path("data/snapshots")
PID_FILE           = pathlib.Path("data/.collector.pid")
PRICES_DIR         = pathlib.Path("data/prices")
AUTO_STATE_FILE    = pathlib.Path("data/.auto.json")
DECISIONS_CACHE    = pathlib.Path("data/.decisions.json")
_NOWND             = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
EXEC_DEFAULT       = ["aster", "binance", "hyperliquid"]
ACTIVE_SYMBOLS     = ["ENA", "ETH", "PEPE", "SOL", "SUI"]
MIN_NON_ZERO       = 10
WATCHLIST_RECENT   = 30
EXEC_COLORS        = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261", "#9b5de5"]
REBUILD_DEBOUNCE_N = 5    # rebuild after this many new snapshots accumulate
REBUILD_COOLDOWN_S = 120  # minimum seconds between rebuilds
AUTO_REFRESH_S     = 60   # page auto-refresh interval when auto mode is on

st.set_page_config(page_title="Exchange Flow", layout="wide", initial_sidebar_state="expanded")

# ── collection helpers ────────────────────────────────────────────────────────

def _collector_pid() -> int | None:
    try:
        return int(PID_FILE.read_text().strip()) if PID_FILE.exists() else None
    except (ValueError, OSError):
        return None


def collector_running() -> bool:
    pid = _collector_pid()
    if pid is None:
        return False
    try:
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                           capture_output=True, text=True, timeout=5,
                           creationflags=_NOWND)
        return str(pid) in r.stdout
    except Exception:
        return False


def start_collector() -> str:
    if collector_running():
        return "already running"
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.Popen(["py", "fetch.py", "--loop", "60"], creationflags=flags)
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(proc.pid))
    return f"started (pid {proc.pid})"


def stop_collector() -> str:
    pid = _collector_pid()
    if pid is None:
        return "not running"
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                       capture_output=True, timeout=5, creationflags=_NOWND)
    except Exception:
        pass
    if PID_FILE.exists():
        PID_FILE.unlink()
    return f"stopped (pid {pid})"


def last_snapshot_info() -> tuple[str, float]:
    if not SNAP_DIR.exists():
        return "none", -1
    files = sorted(SNAP_DIR.glob("*.json"))
    if not files:
        return "none", -1
    try:
        dt  = datetime.datetime.strptime(files[-1].stem, "%Y%m%d_%H%M%S")
        age = (datetime.datetime.now() - dt).total_seconds()
        return f"{dt.strftime('%H:%M:%S')} ({age:.0f}s ago)", age
    except ValueError:
        return files[-1].name, -1

# ── auto mode ─────────────────────────────────────────────────────────────────
#
# State is persisted in AUTO_STATE_FILE so it survives Streamlit reruns.
# A daemon background thread checks every 30s and does the actual work
# (starting the collector, triggering rebuilds).  It never calls Streamlit
# APIs -- it only does subprocess + file I/O.
# The UI reads the state file on each render and injects a JS auto-refresh
# so the page updates without blocking.

_DEFAULT_AUTO_STATE: dict = {
    "auto_mode":           True,
    "last_rebuild_n":      0,
    "last_rebuild_time":   None,
    "rebuild_pid":         None,
    "cache_stale":         False,
    "decisions_stale":     True,   # trigger computation on first run
    "decisions_computing": False,
    "exec_exchanges":      EXEC_DEFAULT,
}

_bg_thread: threading.Thread | None = None


def read_auto_state() -> dict:
    try:
        return {**_DEFAULT_AUTO_STATE, **json.loads(AUTO_STATE_FILE.read_text())}
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(_DEFAULT_AUTO_STATE)


def write_auto_state(updates: dict) -> None:
    state = read_auto_state()
    state.update(updates)
    AUTO_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUTO_STATE_FILE.write_text(json.dumps(state, default=str, indent=2))


def save_decisions(df: pd.DataFrame) -> None:
    """Persist the decisions DataFrame to a JSON file."""
    DECISIONS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    DECISIONS_CACHE.write_text(json.dumps({
        "computed_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "records":     df.reset_index().to_dict(orient="records"),
    }, default=str))


def load_decisions() -> tuple[pd.DataFrame | None, str]:
    """Load the last saved decisions.  Returns (df, computed_at_str) or (None, '')."""
    try:
        obj = json.loads(DECISIONS_CACHE.read_text())
        df  = pd.DataFrame(obj["records"])
        if "rank" in df.columns:
            df = df.set_index("rank")
        return df, obj.get("computed_at", "")
    except Exception:
        return None, ""


def _recompute_decisions_bg() -> None:
    """
    Compute decisions and save to DECISIONS_CACHE.
    Safe to call from the background thread -- uses .__wrapped__ to bypass
    Streamlit's cache (which isn't available outside a script run context).
    """
    try:
        state          = read_auto_state()
        exec_exchanges = tuple(state.get("exec_exchanges", EXEC_DEFAULT))
        df = compute_decisions.__wrapped__(
            str(PANEL_DEFAULT), str(PRICES_DIR), exec_exchanges,
        )
        save_decisions(df)
    except Exception:
        pass
    finally:
        write_auto_state({"decisions_computing": False})


def _n_snapshots_on_disk() -> int:
    return len(list(SNAP_DIR.glob("*.json"))) if SNAP_DIR.exists() else 0


def _rebuild_running(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                           capture_output=True, text=True, timeout=5,
                           creationflags=_NOWND)
        return str(pid) in r.stdout
    except Exception:
        return False


def _bg_worker() -> None:
    """Background thread: runs every 30s, never touches Streamlit."""
    panel_path = str(PANEL_DEFAULT)
    while True:
        try:
            state = read_auto_state()
            if not state.get("auto_mode"):
                time.sleep(30)
                continue

            # 1. Keep collector alive
            if not collector_running():
                start_collector()

            # 2. Check if an in-progress rebuild just finished
            pid = state.get("rebuild_pid")
            if pid and not _rebuild_running(pid):
                # Rebuild done -- clear UI cache and mark decisions stale
                write_auto_state({"rebuild_pid": None, "cache_stale": True,
                                  "decisions_stale": True})
                state = read_auto_state()

            # 3a. Recompute decisions if stale (runs inline; bg thread blocks ~5-10s)
            if state.get("decisions_stale") and not state.get("decisions_computing"):
                write_auto_state({"decisions_stale": False, "decisions_computing": True})
                _recompute_decisions_bg()
                state = read_auto_state()

            # 3. Trigger a new rebuild if enough new snapshots have accumulated
            if not _rebuild_running(state.get("rebuild_pid")):
                n_disk = _n_snapshots_on_disk()
                n_last = state.get("last_rebuild_n", 0)
                last_t = state.get("last_rebuild_time")
                cooldown_ok = True
                if last_t:
                    try:
                        elapsed = (datetime.datetime.now() -
                                   datetime.datetime.fromisoformat(last_t)).total_seconds()
                        cooldown_ok = elapsed >= REBUILD_COOLDOWN_S
                    except (ValueError, TypeError):
                        pass

                if n_disk - n_last >= REBUILD_DEBOUNCE_N and cooldown_ok:
                    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                    proc  = subprocess.Popen(
                        ["py", "build_panel.py", "--output", panel_path],
                        creationflags=flags,
                    )
                    write_auto_state({
                        "last_rebuild_n":   n_disk,
                        "last_rebuild_time": datetime.datetime.now().isoformat(timespec="seconds"),
                        "rebuild_pid":      proc.pid,
                        "cache_stale":      False,
                    })

        except Exception:
            pass  # never crash the bg thread

        time.sleep(30)


def ensure_bg_thread() -> None:
    """Start the background thread if it isn't already running."""
    global _bg_thread
    if _bg_thread is not None and _bg_thread.is_alive():
        return
    _bg_thread = threading.Thread(target=_bg_worker, daemon=True, name="efp-auto")
    _bg_thread.start()


# ── data loading ──────────────────────────────────────────────────────────────

def find_panel() -> pathlib.Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--panel", default=str(PANEL_DEFAULT))
    args, _ = parser.parse_known_args()
    return pathlib.Path(args.panel)


@st.cache_data(show_spinner="Loading panel...")
def load_panel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    df["rate"]       = pd.to_numeric(df["rate"], errors="coerce")
    df = df.dropna(subset=["symbol", "exchange"])
    df["symbol"]   = df["symbol"].astype(str)
    df["exchange"] = df["exchange"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def get_symbols(path: str) -> list[str]:
    return sorted(load_panel(path)["symbol"].unique().tolist())


@st.cache_data(show_spinner=False)
def get_all_exchanges(path: str) -> list[str]:
    return sorted(load_panel(path)["exchange"].unique().tolist())


@st.cache_data(show_spinner=False)
def build_pivot(path: str, symbol: str) -> pd.DataFrame:
    df = load_panel(path)
    return (
        df[df["symbol"] == symbol]
        .pivot_table(index="fetched_at", columns="exchange", values="rate", aggfunc="first")
        .sort_index()
    )


@st.cache_data(show_spinner=False)
def build_delta(path: str, symbol: str) -> pd.DataFrame:
    return build_pivot(path, symbol).diff()

# ── price helpers ─────────────────────────────────────────────────────────────

def prices_available(prices_dir: str) -> list[str]:
    d = pathlib.Path(prices_dir)
    return [p.stem.replace("USDT_1m", "") for p in sorted(d.glob("*USDT_1m.csv"))] if d.exists() else []


@st.cache_data(show_spinner=False)
def load_prices(prices_dir: str, symbol: str) -> pd.DataFrame | None:
    path = pathlib.Path(prices_dir) / f"{symbol}USDT_1m.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["ts"]    = pd.to_datetime(df["open_time"], unit="s", utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.set_index("ts").sort_index()[["close", "volume"]]


@st.cache_data(show_spinner=False)
def align_funding_price(panel_path: str, prices_dir: str, symbol: str) -> pd.DataFrame | None:
    prices = load_prices(prices_dir, symbol)
    if prices is None:
        return None
    for w in (5, 15):
        prices[f"return_{w}m"] = prices["close"].pct_change(w) * 100

    pivot      = build_pivot(panel_path, symbol)
    delta      = build_delta(panel_path, symbol)
    cons_rate  = pivot.median(axis=1)
    cons_delta = delta.median(axis=1)

    snap = pd.DataFrame({"fetched_at": cons_rate.index,
                         "cons_rate":  cons_rate.values,
                         "cons_delta": cons_delta.values})
    snap["ts_min"] = snap["fetched_at"].dt.floor("1min")
    snap = snap.join(prices[["close", "return_5m", "return_15m"]], on="ts_min").dropna(subset=["close"])
    if snap.empty:
        return None

    snap = snap.set_index("fetched_at")
    snap["signal"] = "neutral"
    nonzero  = snap["cons_delta"] != 0
    has_ret  = snap["return_5m"].notna() & (snap["return_5m"] != 0)
    same_dir = np.sign(snap["cons_delta"]) == np.sign(snap["return_5m"])
    snap.loc[nonzero & has_ret &  same_dir, "signal"] = "reinforcing"
    snap.loc[nonzero & has_ret & ~same_dir, "signal"] = "opposing"
    return snap

# ── shared metric helpers ─────────────────────────────────────────────────────

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


@st.cache_data(show_spinner=False)
def compute_metrics(path: str, symbol: str) -> pd.DataFrame:
    delta = build_delta(path, symbol)
    cons  = delta.median(axis=1)
    rows  = []
    for ex in delta.columns:
        m = _lead_score_for(delta, ex, cons)
        if m is None:
            continue
        d      = delta[ex].dropna()
        both   = pd.DataFrame({"d": d, "c": cons}).dropna()
        excess = both["d"] - both["c"]
        rows.append({
            "exchange":   ex,
            "lead_score": round(m["lead_score"], 4),
            "pct_zero":   round(m["pct_zero"], 3),
            "change_mag": round(m["change_mag"], 6),
            "fwd_corr":   round(m["fwd_corr"], 4),
            "lag_corr":   round(m["lag_corr"], 4),
            "excess_abs": round(float(excess.abs().mean()), 6),
        })
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows).set_index("exchange")
            .sort_values("lead_score", ascending=False, na_position="last"))

# ── exchange authority ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=300)
def compute_exchange_authority(panel_path: str) -> pd.DataFrame:
    """
    Stable, cross-symbol exchange authority scores.

    Iterates every symbol in the panel and averages per-exchange lead scores.
    Returns a DataFrame indexed by exchange with columns:
      avg_lead     -- mean lead score across symbols (higher = moves earlier)
      symbol_count -- how many symbols contributed data
      role         -- 'leader' (avg_lead > 0.03) / 'follower' (< -0.03) / 'neutral'

    Leaders: their funding changes tend to *predict* the cross-exchange consensus.
    Followers: they confirm after the fact -- weaker signal weight.
    """
    df = load_panel(panel_path)
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

# ── direction heuristic ───────────────────────────────────────────────────────

def _derive_direction(rate_zscore: float, trend_dir: float, trend_persist: float,
                      exec_lag: float, watch_score: float) -> tuple[str, str, str]:
    """
    Medium-horizon direction heuristic (500-1000 min holding).

    Two regimes:
      Structural extreme  (|rate_zscore| >= 1.5)
        -- lean against the crowded side; mean-reversion is the thesis
        -- if the trend is already reversing toward the mean, confidence rises
      Sustained momentum  (trend_persist > 0.45, rate near neutral)
        -- follow the flow; longs entering = LONG, shorts entering = SHORT

    exec_lag < -0.05 means exec venues are lagging the signal (window still open).
    """
    parts: list[str] = []
    conf  = 0.15

    if abs(rate_zscore) < 0.5 and trend_persist < 0.30:
        return "NEUTRAL", "low", "funding near mean, no persistent trend"

    if abs(rate_zscore) >= 1.5:
        # Structural extreme: crowded side will mean-revert over hours
        base_dir = -int(np.sign(rate_zscore))   # lean against the crowd
        side     = "longs" if rate_zscore > 0 else "shorts"
        parts.append(f"funding {rate_zscore:+.1f}σ from mean ({side} crowded)")
        conf += 0.25
        # Trend toward mean = confirmation; trend extending = caution
        if trend_dir != 0 and np.sign(trend_dir) == np.sign(base_dir):
            parts.append("trend reversing toward mean")
            conf = min(conf + 0.18, 0.85)
        elif trend_dir != 0:
            parts.append("trend still extending -- crowded side building")
            conf = max(conf - 0.05, 0.15)

    elif trend_persist >= 0.45:
        # Sustained momentum: no extreme yet, but flow is consistent
        base_dir = int(np.sign(trend_dir)) if trend_dir != 0 else 0
        if base_dir == 0:
            return "NEUTRAL", "low", "no clear trend direction"
        direction_word = "rising" if trend_dir > 0 else "falling"
        parts.append(f"funding {direction_word}, {trend_persist:.0%} consistent over 2h")
        conf += 0.10 + trend_persist * 0.15

    elif abs(rate_zscore) >= 0.8:
        # Mild bias: light lean against the elevated side
        base_dir = -int(np.sign(rate_zscore))
        parts.append(f"mild funding bias ({rate_zscore:+.1f}σ)")
        conf += 0.08

    else:
        return "NEUTRAL", "low", "no structural signal"

    # Exec lag: window is still open
    if pd.notna(exec_lag) and exec_lag < -0.05:
        parts.append("exec lagging (window open)")
        conf = min(conf + 0.15, 0.85)

    if watch_score >= 0.50:
        conf = min(conf + 0.05, 0.85)

    direction = "LONG" if base_dir > 0 else "SHORT" if base_dir < 0 else "NEUTRAL"
    label     = "high" if conf >= 0.60 else "moderate" if conf >= 0.40 else "low"
    return direction, label, "; ".join(parts)

# ── decision table ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Analyzing opportunities...", ttl=120)
def compute_decisions(panel_path: str, prices_dir: str,
                      exec_exchanges: tuple) -> pd.DataFrame:
    """
    Per-symbol decision table.  Columns:
      symbol, opportunity, direction, confidence, watch_score,
      leader_gap, exec_lag, reason
    Sorted: opportunities first, then by confidence, then watch_score.

    Signal is weighted toward 'leader' exchanges (those with positive
    cross-symbol average lead scores).  'leader_gap' measures how far
    exec venues currently sit from leader-exchange rates -- the core
    "smart flow not yet reflected" signal.
    """
    df       = load_panel(panel_path)
    has_exec = len(exec_exchanges) > 0

    # Exchange authority: stable leader / follower classification.
    # Use .__wrapped__ to bypass the Streamlit cache guard when called
    # from the background thread (no active script context there).
    try:
        authority  = compute_exchange_authority.__wrapped__(panel_path)
        leader_exs = set(authority[authority["role"] == "leader"].index)
    except Exception:
        authority  = pd.DataFrame()
        leader_exs = set()

    rows: list[dict] = []
    for sym, sub in df.groupby("symbol"):
        pivot = (sub.pivot_table(index="fetched_at", columns="exchange",
                                 values="rate", aggfunc="first").sort_index())
        if len(pivot) < 15:
            continue

        delta = pivot.diff()
        pz    = (delta == 0).mean()

        # ── Leader consensus ──────────────────────────────────────────────────
        # If we have ≥ 2 active leader exchanges, use only their deltas as the
        # signal source.  This filters out the follower noise that would dilute
        # the median under equal weighting.
        leader_cols     = [ex for ex in delta.columns
                           if ex in leader_exs and pz.get(ex, 1.0) < 0.7]
        use_leader_cons = len(leader_cols) >= 2
        signal_delta    = delta[leader_cols] if use_leader_cons else delta
        signal_cons     = signal_delta.median(axis=1)

        # Full consensus (all exchanges) is kept for rate level -- leaders
        # may have sparse coverage so the full median is more stable.
        full_cons    = delta.median(axis=1)

        # ── 1. Rate extreme: current rate vs history ──────────────────────────
        cons_rate    = pivot.median(axis=1)
        current_rate = float(cons_rate.iloc[-1])
        rate_mean    = float(cons_rate.mean())
        rate_std     = float(cons_rate.std())

        if rate_std > 1e-10:
            rate_zscore  = (current_rate - rate_mean) / rate_std
            rate_extreme = float(min(abs(rate_zscore) / 3.0, 1.0))
        else:
            rate_zscore  = 0.0
            rate_extreme = 0.0

        # ── 2. Trend persistence: 2h consistency measured on leader consensus ─
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

        # ── 3. Exec position and leader→exec gap ─────────────────────────────
        # exec_lag  : temporal lead score of exec venues vs signal consensus
        #             (negative = exec lags in timing)
        # leader_gap: how far exec rates sit from leader rates right now
        #             (higher = larger window; the core "smart flow" signal)
        exec_rates_now = []
        exec_active    = 0
        exec_leads     = []
        for ex in exec_exchanges:
            if ex not in pivot.columns:
                continue
            val = float(pivot[ex].iloc[-1]) if pd.notna(pivot[ex].iloc[-1]) else float("nan")
            if pd.notna(val):
                exec_rates_now.append(val)
            m = _lead_score_for(delta, ex, signal_cons)
            if m:
                exec_leads.append(m["lead_score"])
                if m["pct_zero"] < 0.7:
                    exec_active += 1
        exec_rate_now = float(np.nanmean(exec_rates_now)) if exec_rates_now else float("nan")
        exec_lag      = float(np.mean(exec_leads))         if exec_leads     else float("nan")

        # Reference point for the gap: leader median rate if available,
        # otherwise full consensus rate.
        leader_rate_now = (float(pivot[leader_cols].iloc[-1].median())
                           if leader_cols else float("nan"))
        ref_rate = leader_rate_now if pd.notna(leader_rate_now) else current_rate
        if pd.notna(exec_rate_now) and rate_std > 1e-10:
            leader_gap = float(min(abs(exec_rate_now - ref_rate) / (rate_std + 1e-10) / 2.0, 1.0))
        else:
            leader_gap = 0.0

        # ── 4. Data quality ───────────────────────────────────────────────────
        active_all = sum(1 for ex in delta.columns if pz.get(ex, 1.0) < 0.7)
        data_qual  = active_all / max(len(delta.columns), 1)

        # ── Watch score (leader-aware, medium-horizon) ────────────────────────
        watch_score = (
            0.30 * rate_extreme   +
            0.30 * trend_persist  +   # measured on leader consensus
            0.25 * leader_gap     +   # exec lagging smart flow
            0.15 * data_qual
        )

        # ── Opportunity flag ──────────────────────────────────────────────────
        has_structural = abs(rate_zscore) > 0.8 or trend_persist > 0.35
        leader_moving  = use_leader_cons and trend_persist > 0.25
        if has_exec:
            opportunity = (
                watch_score >= 0.38
                and has_structural
                and exec_active >= 1
                and (leader_gap > 0.10 or (pd.notna(exec_lag) and exec_lag < -0.02))
            )
        else:
            opportunity = (
                watch_score >= 0.45
                and has_structural
                and (leader_moving or abs(rate_zscore) > 1.0)
            )

        # ── Direction + reason ────────────────────────────────────────────────
        direction, confidence, reason = _derive_direction(
            rate_zscore, trend_dir, trend_persist, exec_lag, watch_score
        )
        # Append leader-gap context when it is the driving factor
        if use_leader_cons and leader_gap > 0.10:
            names = ", ".join(leader_cols[:2])
            reason = reason + f"; leader flow ({names}) not yet in exec"

        rows.append({
            "symbol":      sym,
            "opportunity": "YES" if opportunity else "",
            "direction":   direction,
            "confidence":  confidence,
            "watch_score": round(watch_score, 3),
            "leader_gap":  round(leader_gap, 3),
            "exec_lag":    round(exec_lag, 3) if pd.notna(exec_lag) else float("nan"),
            "reason":      reason,
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["_s1"] = (out["opportunity"] == "YES").astype(int)
    out["_s2"] = out["confidence"].map({"high": 3, "moderate": 2, "low": 1}).fillna(0)
    out = (out.sort_values(["_s1", "_s2", "watch_score"], ascending=[False, False, False])
              .drop(columns=["_s1", "_s2"])
              .reset_index(drop=True))
    out.index = out.index + 1
    out.index.name = "rank"
    return out

# ── sidebar ───────────────────────────────────────────────────────────────────

def sidebar(panel_path: str) -> tuple[list[str], str]:
    """Returns (exec_exchanges, detail_symbol)."""
    st.sidebar.title("Exchange Flow")

    # Load panel data up-front -- needed by symbol/exchange selectors and dataset caption.
    p = pathlib.Path(panel_path)
    if p.exists():
        df      = load_panel(panel_path)
        n_panel = df["fetched_at"].nunique()
    else:
        df      = None
        n_panel = 0

    # ── Auto mode status ──────────────────────────────────────────────────────
    state     = read_auto_state()
    auto_mode = state.get("auto_mode", True)
    running   = collector_running()
    snap_label, snap_age = last_snapshot_info()

    # Toggle button -- prominent, at the top
    toggle_label = "Auto mode: ON  (click to turn off)" if auto_mode else "Auto mode: OFF  (click to turn on)"
    toggle_style = "primary" if auto_mode else "secondary"
    if st.sidebar.button(toggle_label, use_container_width=True, type=toggle_style):
        write_auto_state({"auto_mode": not auto_mode})
        time.sleep(0.2); st.rerun()

    # Status lines
    coll_icon   = "green" if running else "red"
    coll_label  = "Running" if running else "Stopped"
    rebuild_pid = state.get("rebuild_pid")
    rebuild_str = "rebuilding..." if _rebuild_running(rebuild_pid) else (
        state.get("last_rebuild_time") or "never"
    )
    n_disk  = _n_snapshots_on_disk()
    n_last  = state.get("last_rebuild_n", 0)
    pending = max(0, n_disk - n_last)

    st.sidebar.markdown(
        f":{coll_icon}[Collector: **{coll_label}**]  |  "
        f"Last snap: `{snap_label}`"
    )
    st.sidebar.caption(
        f"Panel last rebuilt: {rebuild_str}"
        + (f"  |  **{pending} pending**" if pending >= REBUILD_DEBOUNCE_N else "")
    )

    if running and snap_age > 180:
        st.sidebar.warning("Collector running but no snapshot in >3 min.")

    # Manual controls (secondary, collapsed by default)
    with st.sidebar.expander("Manual controls"):
        ca, cb = st.columns(2)
        if ca.button("Start collector", disabled=running, use_container_width=True):
            st.success(start_collector()); time.sleep(0.5); st.rerun()
        if cb.button("Stop collector", disabled=not running, use_container_width=True):
            st.info(stop_collector()); time.sleep(0.5); st.rerun()
        cc, cd = st.columns(2)
        if cc.button("Force rebuild", use_container_width=True):
            r = subprocess.run(["py", "build_panel.py", "--output", panel_path],
                               capture_output=True, text=True, creationflags=_NOWND)
            if r.returncode == 0:
                write_auto_state({"last_rebuild_n": n_disk,
                                  "last_rebuild_time": datetime.datetime.now().isoformat(timespec="seconds"),
                                  "cache_stale": True})
                st.cache_data.clear(); st.rerun()
            else:
                st.error(r.stderr[-300:])
        if cd.button("Refresh view", use_container_width=True):
            st.cache_data.clear(); st.rerun()

    st.sidebar.markdown("---")

    if df is None:
        st.error(f"Panel not found: `{panel_path}`. Click **Rebuild**.")
        st.stop()

    all_exchanges = get_all_exchanges(panel_path)
    all_symbols   = get_symbols(panel_path)

    # Execution exchanges
    st.sidebar.markdown("### Execution exchanges")
    # Use persisted value as default so the bg thread and UI stay in sync
    persisted_exec = read_auto_state().get("exec_exchanges", EXEC_DEFAULT)
    exec_default   = [e for e in persisted_exec if e in all_exchanges] \
                     or [e for e in EXEC_DEFAULT if e in all_exchanges]
    exec_exchanges = st.sidebar.multiselect(
        "My venues (for opportunity & direction)",
        options=all_exchanges,
        default=exec_default,
    )
    # Persist to auto state and mark decisions stale whenever the selection changes
    if sorted(exec_exchanges) != sorted(persisted_exec):
        write_auto_state({"exec_exchanges": exec_exchanges, "decisions_stale": True})

    st.sidebar.markdown("---")

    # Detail symbol selector
    st.sidebar.markdown("### Detail tab")
    default_sym   = [s for s in ACTIVE_SYMBOLS if s in all_symbols] or all_symbols[:1]
    detail_symbol = st.sidebar.selectbox("Symbol", options=all_symbols,
                                         index=all_symbols.index(default_sym[0]) if default_sym else 0)

    t0   = df["fetched_at"].min()
    t1   = df["fetched_at"].max()
    span = (t1 - t0).total_seconds() / 60
    st.sidebar.caption(
        f"{df['symbol'].nunique()} symbols | {n_panel} snapshots\n"
        f"{t0.strftime('%b %d %H:%M')} -- {t1.strftime('%H:%M')} UTC ({span:.0f} min)"
    )

    return exec_exchanges, detail_symbol

# ── tab: opportunities (main decision view) ────────────────────────────────────

@st.fragment(run_every=AUTO_REFRESH_S)
def _opportunities_live(panel_path: str, prices_dir: str, exec_exchanges: list[str]) -> None:
    """Fragment: reruns every AUTO_REFRESH_S seconds without reloading the full page."""
    has_exec = bool(exec_exchanges)
    exec_str = ", ".join(exec_exchanges) if has_exec else "none -- using watch_score >= 0.45"

    decisions, computed_at = load_decisions()

    if decisions is None:
        with st.spinner("Computing opportunities (first run)..."):
            decisions = compute_decisions(panel_path, prices_dir, tuple(exec_exchanges))
            save_decisions(decisions)
            write_auto_state({"decisions_computing": False, "decisions_stale": False})
        computed_at = "just now"

    if decisions.empty:
        st.warning("Not enough data. Collect more snapshots and rebuild the panel.")
        return

    state      = read_auto_state()
    computing  = state.get("decisions_computing", False)
    status_sfx = "  ·  _updating..._" if computing else ""
    st.caption(
        f"Exec: {exec_str}  ·  computed {computed_at}{status_sfx}"
    )

    col_f1, col_f2, col_f3 = st.columns([2, 2, 6])
    with col_f1:
        show_opp_only = st.checkbox("Opportunities only", value=True)
    with col_f2:
        conf_filter = st.selectbox("Min confidence", ["all", "low", "moderate", "high"],
                                   index=0, label_visibility="collapsed")

    view = decisions.copy()
    if show_opp_only:
        view = view[view["opportunity"] == "YES"]
    if conf_filter != "all":
        order = {"low": 1, "moderate": 2, "high": 3}
        view  = view[view["confidence"].map(order).fillna(0) >= order[conf_filter]]

    if view.empty:
        st.info("No symbols match the current filters.")
        with st.expander("Show all symbols"):
            _render_decision_table(decisions, exec_exchanges)
        return

    _render_decision_table(view, exec_exchanges)

    with st.expander(f"All symbols ({len(decisions)} total)", expanded=False):
        _render_decision_table(decisions, exec_exchanges)


def tab_opportunities(panel_path: str, prices_dir: str, exec_exchanges: list[str]) -> None:
    _opportunities_live(panel_path, prices_dir, exec_exchanges)


def _render_decision_table(df: pd.DataFrame, exec_exchanges: list[str]) -> None:
    """Styled decision table."""
    CONF_ORDER = {"high": 3, "moderate": 2, "low": 1}
    DIR_ICON   = {"LONG": "L", "SHORT": "S", "NEUTRAL": "--"}

    def style_row(row):
        is_opp   = row.get("opportunity") == "YES"
        direction = row.get("direction", "")
        conf      = row.get("confidence", "")

        if is_opp and direction == "LONG":
            base = "background-color: #d4edda"  # green tint
        elif is_opp and direction == "SHORT":
            base = "background-color: #f8d7da"  # red tint
        elif is_opp:
            base = "background-color: #fff3cd"  # yellow tint
        else:
            base = ""

        styles = [base] * len(row)
        cols   = list(row.index)

        # Bold confidence for high
        if "confidence" in cols and conf == "high":
            styles[cols.index("confidence")] = base + "; font-weight: bold"

        # Mute direction for neutral
        if "direction" in cols and direction == "NEUTRAL":
            styles[cols.index("direction")] = "color: #888"

        return styles

    st.dataframe(
        df.style.apply(style_row, axis=1),
        use_container_width=True,
        height=min(600, 36 + len(df) * 35),
        column_config={
            "opportunity": st.column_config.TextColumn("Opp", width="small"),
            "direction":   st.column_config.TextColumn("Direction", width="small"),
            "confidence":  st.column_config.TextColumn("Confidence", width="small"),
            "watch_score": st.column_config.NumberColumn("Score", format="%.3f", width="small"),
            "leader_gap":  st.column_config.NumberColumn("Leader gap", format="%.3f", width="small",
                               help="How far exec venues sit from leader-exchange rates (in rate std-devs). "
                                    "Higher = larger window of smart flow not yet reflected in your venues."),
            "exec_lag":    st.column_config.NumberColumn("Exec lag", format="%.3f", width="small",
                               help="Temporal lead score of exec exchanges vs signal consensus. "
                                    "Negative = exec is lagging in timing."),
            "reason":      st.column_config.TextColumn("Reason", width="large"),
        },
    )

# ── tab: detail ────────────────────────────────────────────────────────────────

def tab_detail(panel_path: str, prices_dir: str, symbol: str,
               exec_exchanges: list[str]) -> None:
    st.subheader(symbol)
    exec_set   = set(exec_exchanges)
    avail_p    = prices_available(prices_dir)

    # ── Price + funding dual-axis (if available) ──────────────────────────────
    if symbol in avail_p:
        aligned = align_funding_price(panel_path, prices_dir, symbol)
        if aligned is not None and not aligned.empty:
            st.markdown("**Price vs Consensus Funding Rate**")
            st.caption("Blue = price (left). Orange = funding rate (right). Divergence = squeeze setup.")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=aligned.index, y=aligned["close"],
                                     name="Price", line=dict(color="#2980b9", width=2), yaxis="y1"))
            fig.add_trace(go.Scatter(x=aligned.index, y=aligned["cons_rate"],
                                     name="Funding (consensus)", line=dict(color="#e67e22", width=1.5, dash="dash"),
                                     yaxis="y2"))
            fig.update_layout(
                height=280, margin=dict(l=0, r=60, t=10, b=0),
                yaxis=dict(title="Price", side="left"),
                yaxis2=dict(title="Funding rate", side="right", overlaying="y", showgrid=False),
                legend=dict(orientation="h", y=1.05),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Signal classification summary
            sig = aligned["signal"].value_counts()
            n_r = int(sig.get("reinforcing", 0))
            n_o = int(sig.get("opposing",    0))
            n_a = n_r + n_o
            if n_a > 0:
                c1, c2, c3 = st.columns(3)
                c1.metric("Reinforcing", n_r, f"{n_r/n_a*100:.0f}%")
                c2.metric("Opposing",    n_o, f"{n_o/n_a*100:.0f}%")
                avg_r = aligned.loc[aligned["signal"] == "reinforcing", "return_5m"].mean()
                avg_o = aligned.loc[aligned["signal"] == "opposing",    "return_5m"].mean()
                c3.metric("Avg 5m return (reinf / opp)",
                           f"{avg_r:+.3f}% / {avg_o:+.3f}%" if pd.notna(avg_r) else "n/a")
        else:
            st.caption(f"Price data for {symbol} doesn't overlap with current panel window.")
    else:
        st.caption(f"No price data for {symbol}. Run `py fetch_prices.py` to add it.")

    st.markdown("---")

    # ── Funding rate time series ───────────────────────────────────────────────
    pivot = build_pivot(panel_path, symbol)
    delta = build_delta(panel_path, symbol)
    cons  = delta.median(axis=1)

    # Show exec exchanges prominently; others muted
    all_ex     = list(pivot.columns)
    exec_avail = [e for e in exec_exchanges if e in pivot.columns]
    other_ex   = [e for e in all_ex if e not in exec_set]

    st.markdown("**Funding Rate Levels**")
    fig2 = go.Figure()
    for ex in other_ex:
        s = pivot[ex].dropna()
        fig2.add_trace(go.Scatter(x=s.index, y=s.values, name=ex, mode="lines",
                                  line=dict(width=0.8, color="rgba(160,160,160,0.3)"),
                                  showlegend=False))
    for i, ex in enumerate(exec_avail):
        s = pivot[ex].dropna()
        fig2.add_trace(go.Scatter(x=s.index, y=s.values, name=f"{ex} (EXEC)", mode="lines",
                                  line=dict(width=2.5, color=EXEC_COLORS[i % len(EXEC_COLORS)])))
    fig2.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="rate (x10000)",
                       legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Rate Changes (delta) vs Consensus**")
    fig3 = go.Figure()
    for ex in other_ex:
        s = delta[ex].dropna()
        fig3.add_trace(go.Scatter(x=s.index, y=s.values, name=ex, mode="lines",
                                  line=dict(width=0.8, color="rgba(160,160,160,0.3)"),
                                  showlegend=False))
    for i, ex in enumerate(exec_avail):
        s = delta[ex].dropna()
        fig3.add_trace(go.Scatter(x=s.index, y=s.values, name=f"{ex} (EXEC)", mode="lines",
                                  line=dict(width=2.5, color=EXEC_COLORS[i % len(EXEC_COLORS)])))
    fig3.add_trace(go.Scatter(x=cons.index, y=cons.values, name="consensus (median)", mode="lines",
                              line=dict(width=2, dash="dash", color="black")))
    fig3.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="delta",
                       legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ── Exchange metrics table ─────────────────────────────────────────────────
    with st.expander("Exchange breakdown", expanded=False):
        # Authority table: cross-symbol leader / follower classification
        try:
            auth = compute_exchange_authority(panel_path)
            if not auth.empty:
                st.caption("Exchange authority  (cross-symbol average lead score)")

                def _style_auth(row):
                    role = row.get("role", "")
                    if role == "leader":
                        return ["background-color: #d4edda"] * len(row)
                    if role == "follower":
                        return ["background-color: #f8d7da"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    auth.style.apply(_style_auth, axis=1),
                    use_container_width=True,
                    column_config={
                        "avg_lead":     st.column_config.NumberColumn(
                            "Avg lead score", format="%.4f",
                            help="Mean of (fwd_corr − lag_corr) across all symbols. "
                                 "Positive = typically moves before the consensus."),
                        "symbol_count": st.column_config.NumberColumn("Symbols", format="%d"),
                        "role":         st.column_config.TextColumn("Role"),
                    },
                )
                st.markdown("---")
        except Exception:
            pass

        metrics = compute_metrics(panel_path, symbol)
        if metrics.empty:
            st.info("Not enough data.")
        else:
            def style_metrics(row):
                is_exec = row.name in exec_set
                base    = "background-color: #dbeafe; " if is_exec else ""
                ls      = row.get("lead_score")
                if pd.notna(ls):
                    lead = "color: #1a7a1a; font-weight: bold" if ls > 0.05 else \
                           "color: #c0392b" if ls < -0.05 else ""
                else:
                    lead = ""
                styles = [base] * len(row)
                if "lead_score" in row.index:
                    styles[list(row.index).index("lead_score")] = (base + lead).strip("; ")
                return styles

            st.dataframe(metrics.style.apply(style_metrics, axis=1), use_container_width=True)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    panel_path = find_panel()

    # Start bg thread (idempotent -- no-op if already alive).
    # The thread handles auto-starting the collector on its first tick (~instant).
    ensure_bg_thread()

    # If bg thread just finished a rebuild, clear the Streamlit data cache
    state = read_auto_state()
    if state.get("cache_stale"):
        st.cache_data.clear()
        write_auto_state({"cache_stale": False})

    exec_exchanges, detail_symbol = sidebar(str(panel_path))

    tab_opp, tab_det = st.tabs(["Opportunities", "Detail"])

    with tab_opp:
        tab_opportunities(str(panel_path), str(PRICES_DIR), exec_exchanges)
    with tab_det:
        tab_detail(str(panel_path), str(PRICES_DIR), detail_symbol, exec_exchanges)

if __name__ == "__main__":
    main()
