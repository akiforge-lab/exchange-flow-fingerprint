"""
scoring_config.py — load / save scoring weights and thresholds from
data/scoring_config.json with safe fallback to hardcoded defaults.

All callers should import `load_config()` and read keys from the returned dict.
`save_config()` is called only by tune_weights.py (--apply mode).
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

CONFIG_PATH = pathlib.Path("data/scoring_config.json")

# ── Hardcoded defaults (must match compute_decisions_cli.py constants) ─────────

_DEFAULTS: dict[str, Any] = {
    # Composite watch-score weights (must sum to 1.0)
    "w_rate_extreme":  0.30,
    "w_trend_persist": 0.30,
    "w_leader_gap":    0.25,
    "w_data_qual":     0.15,

    # Opportunity thresholds
    "entry_threshold":  0.38,
    "exit_threshold":   0.28,
    "entry_no_exec":    0.45,
    "exit_no_exec":     0.35,

    # Smoothing window (snapshots)
    "smooth_win": 5,

    # Exchange authority role boundaries
    "leader_threshold":   0.03,
    "follower_threshold": -0.03,
}

# Per-weight bounds to prevent degenerate solutions
_WEIGHT_FLOOR   = 0.05
_WEIGHT_CEILING = 0.60
_WEIGHT_KEYS    = ["w_rate_extreme", "w_trend_persist", "w_leader_gap", "w_data_qual"]


def _validate(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of cfg with all keys present and weights normalised."""
    out = dict(_DEFAULTS)
    out.update({k: v for k, v in cfg.items() if k in _DEFAULTS})

    # Clamp weights to bounds
    for k in _WEIGHT_KEYS:
        out[k] = max(_WEIGHT_FLOOR, min(_WEIGHT_CEILING, float(out[k])))

    # Normalise weights so they sum to exactly 1.0
    total = sum(out[k] for k in _WEIGHT_KEYS)
    if total > 0:
        for k in _WEIGHT_KEYS:
            out[k] = round(out[k] / total, 6)

    return out


def load_config() -> dict[str, Any]:
    """
    Load scoring configuration.  Falls back to defaults on any error.
    Always returns a fully populated, validated dict.
    """
    if CONFIG_PATH.exists():
        try:
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return _validate(raw)
        except Exception:
            pass
    return dict(_DEFAULTS)


def save_config(cfg: dict[str, Any]) -> None:
    """
    Validate then persist cfg to CONFIG_PATH.
    Raises ValueError if the config cannot be validated.
    """
    validated = _validate(cfg)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(validated, indent=2) + "\n",
        encoding="utf-8",
    )


def config_summary(cfg: dict[str, Any]) -> str:
    """One-line human-readable weight summary for logging."""
    return (
        f"w=[re={cfg['w_rate_extreme']:.2f} tp={cfg['w_trend_persist']:.2f} "
        f"lg={cfg['w_leader_gap']:.2f} dq={cfg['w_data_qual']:.2f}] "
        f"thresh=[entry={cfg['entry_threshold']:.2f} exit={cfg['exit_threshold']:.2f}]"
    )
