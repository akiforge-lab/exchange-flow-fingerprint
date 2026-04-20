from __future__ import annotations

import datetime as dt
import logging
import pathlib
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from data_helpers import load_panel, load_prices
from data_helpers import _lead_score_for  # reuse internal helper

PANEL_PATH = pathlib.Path("data/panel.csv")
PRICES_DIR = pathlib.Path("data/prices")

logger = logging.getLogger("uvicorn.error")

_cache: Dict[str, object] = {
    "panel": None,
    "symbol_df": {},
    "pivot": {},
    "delta": {},
    "metrics": {},
    "authority": None,
    "aligned": {},
    "stats": {"symbols_list": [], "symbols_cached": 0, "panel_mtime": None, "warmed_at": None, "warm_duration": None},
}


def _get_file_mtime(path: pathlib.Path) -> Optional[float]:
    try:
        return path.stat().st_mtime
    except (FileNotFoundError, OSError):
        return None


def cache_stats() -> dict:
    stats = _cache["stats"]
    warmed_at = stats.get("warmed_at")
    return {
        "cache_warm": bool(_cache.get("panel")),
        "symbols_cached": stats.get("symbols_cached", 0),
        "panel_mtime": stats.get("panel_mtime"),
        "warmed_at": warmed_at,
        "warm_duration": stats.get("warm_duration"),
    }


def warm_cache() -> None:
    panel_mtime = _get_file_mtime(PANEL_PATH)
    if panel_mtime is None:
        logger.warning("Panel cache warm skipped: %s not found", PANEL_PATH)
        _cache["panel"] = None
        _cache["stats"].update({"symbols_list": [], "symbols_cached": 0, "panel_mtime": None})
        return
    panel_entry = _cache.get("panel")
    if panel_entry and panel_entry[0] == panel_mtime:
        logger.info(
            "Panel cache already warm: %d symbols (mtime %s)",
            _cache["stats"].get("symbols_cached", 0),
            dt.datetime.fromtimestamp(panel_mtime).isoformat(),
        )
        return
    _rebuild_cache(panel_mtime)


def _rebuild_cache(panel_mtime: float) -> None:
    start = time.perf_counter()
    panel_df = load_panel(str(PANEL_PATH))
    _cache["panel"] = (panel_mtime, panel_df)
    _cache["symbol_df"] = {}
    _cache["pivot"] = {}
    _cache["delta"] = {}
    _cache["metrics"] = {}
    _cache["authority"] = None
    _cache["aligned"] = {}

    symbols = []
    for symbol, sub in panel_df.groupby("symbol"):
        symbols.append(symbol)
        key = (symbol, panel_mtime)
        _cache["symbol_df"][key] = sub
        try:
            pivot = _build_pivot_from_df(sub)
            _cache["pivot"][key] = pivot
            delta = pivot.diff()
            _cache["delta"][key] = delta
            metrics = _compute_metrics_from_delta(delta)
            _cache["metrics"][key] = metrics
        except Exception as exc:
            logger.exception("Failed to precompute symbol %s: %s", symbol, exc)

    try:
        authority = _compute_exchange_authority_from_df(panel_df)
        _cache["authority"] = (panel_mtime, authority)
    except Exception as exc:
        logger.exception("Failed to compute exchange authority: %s", exc)
        _cache["authority"] = None

    duration = time.perf_counter() - start
    _cache["stats"].update(
        {
            "symbols_list": symbols,
            "symbols_cached": len(symbols),
            "panel_mtime": panel_mtime,
            "warmed_at": dt.datetime.now(dt.timezone.utc),
            "warm_duration": duration,
        }
    )
    logger.info(
        "Panel cache warm complete: %d symbols in %.2fs (mtime %s)",
        len(symbols),
        duration,
        dt.datetime.fromtimestamp(panel_mtime).isoformat(),
    )


def _ensure_panel_loaded() -> Optional[float]:
    panel_mtime = _get_file_mtime(PANEL_PATH)
    if panel_mtime is None:
        return None
    panel_entry = _cache.get("panel")
    if not panel_entry or panel_entry[0] != panel_mtime:
        warm_cache()
        panel_entry = _cache.get("panel")
        if not panel_entry:
            return None
    return panel_entry[0]


def get_panel_df() -> Optional[pd.DataFrame]:
    panel_mtime = _ensure_panel_loaded()
    if panel_mtime is None:
        return None
    panel_entry = _cache.get("panel")
    return panel_entry[1] if panel_entry else None


def list_symbols() -> list[str]:
    panel_mtime = _ensure_panel_loaded()
    if panel_mtime is None:
        return []
    symbols = _cache["stats"].get("symbols_list", [])
    return list(symbols)


def get_symbol_df(symbol: str) -> Optional[pd.DataFrame]:
    panel_mtime = _ensure_panel_loaded()
    if panel_mtime is None:
        return None
    key = (symbol, panel_mtime)
    entry = _cache["symbol_df"].get(key)
    if entry is not None:
        return entry
    panel_entry = _cache.get("panel")
    if not panel_entry:
        return None
    panel_df = panel_entry[1]
    subset = panel_df[panel_df["symbol"] == symbol]
    if subset.empty:
        return None
    _cache["symbol_df"][key] = subset
    return subset


def get_pivot(symbol: str) -> Optional[pd.DataFrame]:
    panel_mtime = _ensure_panel_loaded()
    if panel_mtime is None:
        return None
    key = (symbol, panel_mtime)
    pivot = _cache["pivot"].get(key)
    if pivot is not None:
        return pivot
    symbol_df = get_symbol_df(symbol)
    if symbol_df is None:
        return None
    pivot = _build_pivot_from_df(symbol_df)
    _cache["pivot"][key] = pivot
    _cache["delta"].pop(key, None)
    return pivot


def get_delta(symbol: str) -> Optional[pd.DataFrame]:
    panel_mtime = _ensure_panel_loaded()
    if panel_mtime is None:
        return None
    key = (symbol, panel_mtime)
    delta = _cache["delta"].get(key)
    if delta is not None:
        return delta
    pivot = get_pivot(symbol)
    if pivot is None:
        return None
    delta = pivot.diff()
    _cache["delta"][key] = delta
    return delta


def get_metrics(symbol: str) -> Optional[pd.DataFrame]:
    panel_mtime = _ensure_panel_loaded()
    if panel_mtime is None:
        return None
    key = (symbol, panel_mtime)
    metrics = _cache["metrics"].get(key)
    if metrics is not None:
        return metrics
    delta = get_delta(symbol)
    if delta is None:
        return None
    metrics = _compute_metrics_from_delta(delta)
    _cache["metrics"][key] = metrics
    return metrics


def get_authority() -> Optional[pd.DataFrame]:
    panel_mtime = _ensure_panel_loaded()
    if panel_mtime is None:
        return None
    entry = _cache["authority"]
    if entry and entry[0] == panel_mtime:
        return entry[1]
    panel_df = get_panel_df()
    if panel_df is None:
        return None
    authority = _compute_exchange_authority_from_df(panel_df)
    _cache["authority"] = (panel_mtime, authority)
    return authority


def get_aligned_snapshot(symbol: str) -> Optional[pd.DataFrame]:
    panel_mtime = _ensure_panel_loaded()
    if panel_mtime is None:
        return None
    price_path = PRICES_DIR / f"{symbol}USDT_1m.csv"
    price_mtime = _get_file_mtime(price_path) or 0.0
    key = (symbol, panel_mtime, price_mtime)
    entry = _cache["aligned"].get(key)
    if entry is not None:
        return entry
    pivot = get_pivot(symbol)
    delta = get_delta(symbol)
    if pivot is None or delta is None:
        _cache["aligned"][key] = None
        return None
    prices = load_prices(str(PRICES_DIR), symbol)
    if prices is None:
        _cache["aligned"][key] = None
        return None
    aligned = _build_aligned_snapshot(pivot, delta, prices)
    _cache["aligned"][key] = aligned
    return aligned


def _build_pivot_from_df(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pivot_table(index="fetched_at", columns="exchange", values="rate", aggfunc="first")
        .sort_index()
    )


def _compute_metrics_from_delta(delta: pd.DataFrame) -> pd.DataFrame:
    cons = delta.median(axis=1)
    rows = []
    for ex in delta.columns:
        m = _lead_score_for(delta, ex, cons)
        if m is None:
            continue
        d = delta[ex].dropna()
        both = pd.DataFrame({"d": d, "c": cons}).dropna()
        excess = both["d"] - both["c"]
        rows.append(
            {
                "exchange": ex,
                "lead_score": round(m["lead_score"], 4),
                "pct_zero": round(m["pct_zero"], 3),
                "change_mag": round(m["change_mag"], 6),
                "fwd_corr": round(m["fwd_corr"], 4),
                "lag_corr": round(m["lag_corr"], 4),
                "excess_abs": round(float(excess.abs().mean()), 6),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("exchange").sort_values("lead_score", ascending=False, na_position="last")


def _compute_exchange_authority_from_df(df: pd.DataFrame) -> pd.DataFrame:
    scores: Dict[str, list[float]] = {}
    for _, sub in df.groupby("symbol"):
        pivot = (
            sub.pivot_table(index="fetched_at", columns="exchange", values="rate", aggfunc="first").sort_index()
        )
        if len(pivot) < 15:
            continue
        delta = pivot.diff()
        cons = delta.median(axis=1)
        for ex in delta.columns:
            m = _lead_score_for(delta, ex, cons)
            if m is None:
                continue
            scores.setdefault(ex, []).append(m["lead_score"])
    if not scores:
        return pd.DataFrame(columns=["avg_lead", "symbol_count", "role"])
    rows = []
    for ex, vals in scores.items():
        avg = float(np.mean(vals))
        role = "leader" if avg > 0.03 else "follower" if avg < -0.03 else "neutral"
        rows.append({"exchange": ex, "avg_lead": round(avg, 4), "symbol_count": len(vals), "role": role})
    return pd.DataFrame(rows).set_index("exchange").sort_values("avg_lead", ascending=False)


def _build_aligned_snapshot(pivot: pd.DataFrame, delta: pd.DataFrame, prices: pd.DataFrame) -> Optional[pd.DataFrame]:
    cons_rate = pivot.median(axis=1)
    cons_delta = delta.median(axis=1)
    snap = pd.DataFrame(
        {"fetched_at": cons_rate.index, "cons_rate": cons_rate.values, "cons_delta": cons_delta.values}
    )
    snap["ts_min"] = snap["fetched_at"].dt.floor("1min")
    prices = prices.copy()
    for w in (5, 15):
        prices[f"return_{w}m"] = prices["close"].pct_change(w) * 100
    snap = snap.join(prices[["close", "return_5m", "return_15m"]], on="ts_min").dropna(subset=["close"])
    if snap.empty:
        return None
    snap = snap.set_index("fetched_at")
    snap["signal"] = "neutral"
    nonzero = snap["cons_delta"] != 0
    has_ret = snap["return_5m"].notna() & (snap["return_5m"] != 0)
    same_dir = np.sign(snap["cons_delta"]) == np.sign(snap["return_5m"])
    snap.loc[nonzero & has_ret & same_dir, "signal"] = "reinforcing"
    snap.loc[nonzero & has_ret & ~same_dir, "signal"] = "opposing"
    return snap
