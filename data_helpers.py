from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pandas as pd

PANEL_DEFAULT = pathlib.Path("data/panel.csv")
PRICES_DIR = pathlib.Path("data/prices")
MIN_NON_ZERO = 10


def load_panel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df = df.dropna(subset=["symbol", "exchange"])
    df["symbol"] = df["symbol"].astype(str)
    df["exchange"] = df["exchange"].astype(str)
    return df


def build_pivot(path: str, symbol: str) -> pd.DataFrame:
    df = load_panel(path)
    return (
        df[df["symbol"] == symbol]
        .pivot_table(index="fetched_at", columns="exchange", values="rate", aggfunc="first")
        .sort_index()
    )


def build_delta(path: str, symbol: str) -> pd.DataFrame:
    return build_pivot(path, symbol).diff()


def prices_available(prices_dir: str) -> list[str]:
    d = pathlib.Path(prices_dir)
    return [p.stem.replace("USDT_1m", "") for p in sorted(d.glob("*USDT_1m.csv"))] if d.exists() else []


def load_prices(prices_dir: str, symbol: str) -> pd.DataFrame | None:
    path = pathlib.Path(prices_dir) / f"{symbol}USDT_1m.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["open_time"], unit="s", utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.set_index("ts").sort_index()[["close", "volume"]]


def align_funding_price(panel_path: str, prices_dir: str, symbol: str) -> pd.DataFrame | None:
    prices = load_prices(prices_dir, symbol)
    if prices is None:
        return None
    for w in (5, 15):
        prices[f"return_{w}m"] = prices["close"].pct_change(w) * 100

    pivot = build_pivot(panel_path, symbol)
    delta = build_delta(panel_path, symbol)
    cons_rate = pivot.median(axis=1)
    cons_delta = delta.median(axis=1)

    snap = pd.DataFrame(
        {"fetched_at": cons_rate.index, "cons_rate": cons_rate.values, "cons_delta": cons_delta.values}
    )
    snap["ts_min"] = snap["fetched_at"].dt.floor("1min")
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


def _lead_score_for(delta: pd.DataFrame, ex: str, cons: pd.Series) -> dict[str, Any] | None:
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
    return {
        "lead_score": float(fwd - lag),
        "pct_zero": float((d == 0).mean()),
        "change_mag": float(nonzero.abs().mean()),
        "fwd_corr": float(fwd),
        "lag_corr": float(lag),
    }


def compute_metrics(path: str, symbol: str) -> pd.DataFrame:
    delta = build_delta(path, symbol)
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


def compute_exchange_authority(panel_path: str) -> pd.DataFrame:
    df = load_panel(panel_path)
    scores: dict[str, list[float]] = {}
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
