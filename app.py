from __future__ import annotations

import datetime as dt
import json
import pathlib
from contextlib import asynccontextmanager
from typing import List, Optional

import plotly.graph_objects as go
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from plotly.utils import PlotlyJSONEncoder

from panel_cache import (
    cache_stats,
    get_aligned_snapshot,
    get_authority,
    get_delta,
    get_metrics,
    get_pivot,
    list_symbols,
    warm_cache,
)

DECISIONS_CACHE = pathlib.Path("data/.decisions.json")
AUTO_STATE_FILE = pathlib.Path("data/.auto.json")
SNAP_DIR = pathlib.Path("data/snapshots")

DEFAULT_AUTO_STATE = {
    "auto_mode": False,
    "collector_running": False,
    "last_rebuild_time": None,
    "exec_exchanges": [],
}

CONF_RANK = {"low": 1, "moderate": 2, "high": 3}
EXEC_COLORS = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261", "#9b5de5"]


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[override]
    warm_cache()
    yield


app = FastAPI(title="Exchange Flow Dashboard", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


def _render(name: str, context: dict) -> HTMLResponse:
    template = templates.get_template(name)
    return HTMLResponse(template.render(context))

STATIC_DIR = pathlib.Path("static")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _read_json(path: pathlib.Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def _load_auto_state() -> dict:
    data = _read_json(AUTO_STATE_FILE) or {}
    state = {**DEFAULT_AUTO_STATE, **data}
    if not isinstance(state.get("exec_exchanges"), list):
        state["exec_exchanges"] = []
    return state


def _humanize_relative(ts: dt.datetime | None) -> str:
    if not ts:
        return "never"
    now = dt.datetime.now(dt.timezone.utc)
    delta = now - ts.replace(tzinfo=dt.timezone.utc) if ts.tzinfo is None else now - ts.astimezone(dt.timezone.utc)
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def _snapshot_status() -> dict:
    if not SNAP_DIR.exists():
        return {"label": "no snapshots", "age": None}
    files = sorted(SNAP_DIR.glob("*.json"))
    if not files:
        return {"label": "no snapshots", "age": None}
    latest = files[-1]
    try:
        dt_utc = dt.datetime.strptime(latest.stem, "%Y%m%d_%H%M%S").replace(tzinfo=dt.timezone.utc)
        age = (dt.datetime.now(dt.timezone.utc) - dt_utc).total_seconds()
        label = f"{dt_utc.strftime('%H:%M:%S')} UTC ({_humanize_relative(dt_utc)})"
        return {"label": label, "age": age}
    except ValueError:
        return {"label": latest.name, "age": None}


def _load_decisions() -> tuple[List[dict], Optional[str], Optional[str]]:
    if not DECISIONS_CACHE.exists():
        return [], None, "Decisions cache not found."
    try:
        payload = json.loads(DECISIONS_CACHE.read_text())
        records = payload.get("records", [])
        computed_at = payload.get("computed_at")
        return records, computed_at, None
    except json.JSONDecodeError as exc:
        return [], None, f"Invalid JSON in decisions cache ({exc})."
    except Exception as exc:  # pragma: no cover - defensive
        return [], None, f"Unable to read decisions cache ({exc})."


def _filter_decisions(
    decisions: List[dict], opp_only: bool, confidence: str
) -> List[dict]:
    filtered = []
    for idx, row in enumerate(decisions, start=1):
        opp = row.get("opportunity") == "YES"
        conf = row.get("confidence", "").lower()
        if opp_only and not opp:
            continue
        if confidence != "all" and CONF_RANK.get(conf, 0) < CONF_RANK.get(confidence, 0):
            continue
        style = ""
        direction = row.get("direction", "").upper()
        if opp:
            if direction == "LONG":
                style = "background-color: #d4edda;"
            elif direction == "SHORT":
                style = "background-color: #f8d7da;"
            else:
                style = "background-color: #fff3cd;"
        filtered.append(
            {
                "rank": idx,
                "symbol": row.get("symbol", ""),
                "opportunity": row.get("opportunity", ""),
                "direction": direction,
                "confidence": row.get("confidence", ""),
                "watch_score": row.get("watch_score"),
                "leader_gap": row.get("leader_gap"),
                "exec_lag": row.get("exec_lag"),
                "reason": row.get("reason", ""),
                "row_style": style,
            }
        )
    return filtered


def _fig_to_json(fig: Optional[go.Figure]) -> Optional[str]:
    if fig is None:
        return None
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def _build_level_chart(pivot, exec_exchanges) -> Optional[go.Figure]:
    if pivot is None or pivot.empty:
        return None
    fig = go.Figure()
    for ex in pivot.columns:
        series = pivot[ex].dropna()
        if series.empty:
            continue
        is_exec = ex in exec_exchanges
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=f"{ex}{' (EXEC)' if is_exec else ''}",
                line=dict(
                    width=2.5 if is_exec else 0.8,
                    color=EXEC_COLORS[exec_exchanges.index(ex) % len(EXEC_COLORS)]
                    if is_exec and exec_exchanges
                    else "rgba(160,160,160,0.4)",
                ),
                showlegend=is_exec,
            )
        )
    if not fig.data:
        return None
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0), yaxis_title="Rate (x10000)")
    return fig


def _build_delta_chart(delta, exec_exchanges) -> Optional[go.Figure]:
    if delta is None or delta.empty:
        return None
    fig = go.Figure()
    cons = delta.median(axis=1)
    for ex in delta.columns:
        series = delta[ex].dropna()
        if series.empty:
            continue
        is_exec = ex in exec_exchanges
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=f"{ex}{' (EXEC)' if is_exec else ''}",
                line=dict(
                    width=2.5 if is_exec else 0.8,
                    color=EXEC_COLORS[exec_exchanges.index(ex) % len(EXEC_COLORS)]
                    if is_exec and exec_exchanges
                    else "rgba(160,160,160,0.3)",
                ),
                showlegend=is_exec,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=cons.index,
            y=cons.values,
            mode="lines",
            name="Consensus (median)",
            line=dict(color="black", dash="dash"),
        )
    )
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=0), yaxis_title="Delta")
    return fig


def _build_price_vs_funding(aligned) -> Optional[go.Figure]:
    if aligned is None or aligned.empty:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=aligned.index,
            y=aligned["close"],
            mode="lines",
            name="Price",
            line=dict(color="#2980b9", width=2),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=aligned.index,
            y=aligned["cons_rate"],
            mode="lines",
            name="Funding (consensus)",
            line=dict(color="#e67e22", width=1.5, dash="dash"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=40, t=20, b=0),
        yaxis=dict(title="Price", side="left"),
        yaxis2=dict(title="Funding", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def _status_payload(auto_state: dict) -> dict:
    snapshot = _snapshot_status()
    rebuild_str = "never"
    last_rebuild = auto_state.get("last_rebuild_time")
    if last_rebuild:
        try:
            parsed = dt.datetime.fromisoformat(str(last_rebuild))
            rebuild_str = f"{parsed.strftime('%Y-%m-%d %H:%M')} ({_humanize_relative(parsed)})"
        except (ValueError, TypeError):
            rebuild_str = str(last_rebuild)
    return {
        "collector_running": bool(auto_state.get("collector_running")),
        "auto_mode": bool(auto_state.get("auto_mode")),
        "snapshot_label": snapshot["label"],
        "rebuild_label": rebuild_str,
    }


@app.get("/", response_class=HTMLResponse)
async def opportunities(request: Request):
    params = request.query_params
    opp_values = params.getlist("opp_only")
    opp_only = True if not opp_values else opp_values[-1] != "0"
    confidence = params.get("confidence", "all")
    if confidence not in {"all", "low", "moderate", "high"}:
        confidence = "all"
    auto_state = _load_auto_state()
    decisions, computed_at, warn = _load_decisions()
    view = _filter_decisions(
        decisions,
        opp_only,
        confidence,
    )
    context = {
        "request": request,
        "active_page": "opportunities",
        "filters": {"opp_only": opp_only, "confidence": confidence},
        "decisions": view,
        "warning": warn if warn or not view else None,
        "computed_at": computed_at,
        "exec_exchanges": auto_state.get("exec_exchanges", []),
        "status": _status_payload(auto_state),
        "static_exists": STATIC_DIR.exists(),
    }
    return _render("opportunities.html", context)


@app.get("/detail", response_class=HTMLResponse)
async def detail(request: Request, symbol: Optional[str] = None):
    auto_state = _load_auto_state()
    panel_error = None
    stats = cache_stats()
    if not stats.get("panel_mtime"):
        warm_cache()
        stats = cache_stats()

    symbols = sorted(list_symbols())
    if not symbols:
        panel_error = "Panel not available."
    selected_symbol = symbol if symbol in symbols else (symbols[0] if symbols else None)

    pivot = delta = None
    funding_levels = funding_delta = price_fig = None
    metrics_rows: List[dict] = []
    authority_rows: List[dict] = []
    detail_warning = None

    if selected_symbol:
        pivot = get_pivot(selected_symbol)
        delta = get_delta(selected_symbol)
        if pivot is None or delta is None:
            detail_warning = "Funding data unavailable for this symbol."
        exec_ex = auto_state.get("exec_exchanges", [])
        funding_levels = _fig_to_json(_build_level_chart(pivot, exec_ex))
        funding_delta = _fig_to_json(_build_delta_chart(delta, exec_ex))
        aligned = get_aligned_snapshot(selected_symbol)
        price_fig = _fig_to_json(_build_price_vs_funding(aligned)) if aligned is not None else None
        metrics = get_metrics(selected_symbol)
        if metrics is not None and not metrics.empty:
            metrics_rows = metrics.reset_index().to_dict(orient="records")
        authority = get_authority()
        if authority is not None and not authority.empty:
            authority_rows = authority.reset_index().to_dict(orient="records")
    elif not panel_error:
        detail_warning = "No symbols available."

    context = {
        "request": request,
        "active_page": "detail",
        "status": _status_payload(auto_state),
        "symbols": symbols,
        "selected_symbol": selected_symbol,
        "panel_error": panel_error,
        "detail_warning": detail_warning,
        "funding_levels_json": funding_levels,
        "funding_delta_json": funding_delta,
        "price_funding_json": price_fig,
        "metrics_rows": metrics_rows,
        "authority_rows": authority_rows,
        "static_exists": STATIC_DIR.exists(),
    }
    return _render("detail.html", context)


@app.get("/status", response_class=PlainTextResponse)
async def status() -> PlainTextResponse:
    stats = cache_stats()
    warmed = "yes" if stats.get("panel_mtime") else "no"
    warmed_at = stats.get("warmed_at")
    warmed_ts = warmed_at.isoformat() if isinstance(warmed_at, dt.datetime) else "n/a"
    panel_mtime = stats.get("panel_mtime")
    panel_ts = dt.datetime.fromtimestamp(panel_mtime).isoformat() if panel_mtime else "n/a"
    lines = [
        f"cache_warm: {warmed}",
        f"symbols_cached: {stats.get('symbols_cached', 0)}",
        f"panel_mtime: {panel_ts}",
        f"warmed_at: {warmed_ts}",
        f"warm_duration_s: {stats.get('warm_duration') or 'n/a'}",
    ]
    return PlainTextResponse("\n".join(lines))
