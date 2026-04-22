"""
tune_weights.py — cautious self-tuning for scoring weights.

Reads signal_log.jsonl, uses Spearman correlation of each component against
direction_correct to derive a suggested weight adjustment, then optionally
writes the new weights to data/scoring_config.json.

Safety constraints
------------------
- Max ±0.05 change per component per update (DELTA_CAP).
- Per-component floor / ceiling (5% – 60%).
- Weights always re-normalised to sum to 1.0.
- Minimum 30 evaluated records required before suggesting any change.
- Report always printed; --apply flag required to persist.

Usage:
    python tune_weights.py [--log data/signal_log.jsonl]
                           [--min-records 30]
                           [--apply]
                           [--dry-run]    (same as omitting --apply, just clearer)
"""

from __future__ import annotations

import argparse
import math
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

from signal_log import load_log
from scoring_config import load_config, save_config, config_summary, _WEIGHT_KEYS, _WEIGHT_FLOOR, _WEIGHT_CEILING

LOG_DEFAULT  = pathlib.Path("data/signal_log.jsonl")
MIN_RECORDS  = 30
DELTA_CAP    = 0.05   # max absolute change per component per update

# Map scoring_config weight key → signal_log column
_COMPONENT_MAP: dict[str, str] = {
    "w_rate_extreme":  "rate_zscore",    # we use abs(rate_zscore) as proxy for rate_extreme
    "w_trend_persist": "trend_persist",
    "w_leader_gap":    "leader_gap",
    "w_data_qual":     "data_qual",
}


def _spearman_rho(series_a: pd.Series, series_b: pd.Series) -> float | None:
    """Return Spearman rho between two series, or None if < 5 paired values."""
    both = pd.concat([series_a, series_b], axis=1).dropna()
    if len(both) < 5:
        return None
    return float(both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman"))


def compute_weight_suggestions(
    records: list[dict],
    current_cfg: dict,
    min_records: int,
) -> dict[str, dict]:
    """
    For each weight key, compute Spearman rho of its component against
    direction_correct, then suggest a new weight (bounded, normalised).

    Returns a dict keyed by weight name:
      {
        "current": float,
        "rho": float | None,
        "raw_delta": float | None,
        "capped_delta": float | None,
        "suggested": float,
        "n": int,
      }
    """
    # Filter to records with evaluated outcomes and non-NEUTRAL direction
    evaluated = [
        r for r in records
        if r.get("direction_correct") is not None
        and r.get("direction") not in ("NEUTRAL", None)
    ]

    result: dict[str, dict] = {}

    if len(evaluated) < min_records:
        for wk in _WEIGHT_KEYS:
            result[wk] = {
                "current":     current_cfg[wk],
                "rho":         None,
                "raw_delta":   None,
                "capped_delta": None,
                "suggested":   current_cfg[wk],
                "n":           len(evaluated),
                "note":        f"insufficient data ({len(evaluated)} < {min_records})",
            }
        return result

    df = pd.DataFrame(evaluated)
    # Binarise outcome
    outcome = df["direction_correct"].astype(float)

    # Compute raw rhos
    rhos: dict[str, float | None] = {}
    ns: dict[str, int] = {}
    for wk, col in _COMPONENT_MAP.items():
        if col not in df.columns:
            rhos[wk] = None
            ns[wk] = 0
            continue
        component = df[col].copy()
        # rate_zscore: use absolute value (larger abs = more extreme = more informative)
        if col == "rate_zscore":
            component = component.abs()
        rho = _spearman_rho(outcome, component)
        rhos[wk] = rho
        ns[wk] = int(pd.concat([outcome, component], axis=1).dropna().shape[0])

    # Derive proportional target weights from |rho| (higher correlation → more weight)
    rho_abs = {wk: abs(r) if r is not None else None for wk, r in rhos.items()}
    total_rho = sum(v for v in rho_abs.values() if v is not None and math.isfinite(v))

    suggested_raw: dict[str, float] = {}
    for wk in _WEIGHT_KEYS:
        cur = current_cfg[wk]
        ra = rho_abs.get(wk)
        if ra is None or not math.isfinite(ra) or total_rho < 1e-6:
            # No data: keep current weight
            suggested_raw[wk] = cur
        else:
            # Target proportional to |rho|
            target = ra / total_rho
            raw_delta = target - cur
            # Cap delta
            capped = max(-DELTA_CAP, min(DELTA_CAP, raw_delta))
            suggested_raw[wk] = cur + capped

    # Clamp to floor/ceiling
    for wk in _WEIGHT_KEYS:
        suggested_raw[wk] = max(_WEIGHT_FLOOR, min(_WEIGHT_CEILING, suggested_raw[wk]))

    # Normalise
    total = sum(suggested_raw.values())
    suggested_norm = {wk: suggested_raw[wk] / total for wk in _WEIGHT_KEYS}

    # Build result
    for wk in _WEIGHT_KEYS:
        cur = current_cfg[wk]
        ra = rho_abs.get(wk)
        total_rho2 = sum(v for v in rho_abs.values() if v is not None and math.isfinite(v))
        if ra is not None and math.isfinite(ra) and total_rho2 > 1e-6:
            target = ra / total_rho2
            raw_delta = target - cur
            capped = max(-DELTA_CAP, min(DELTA_CAP, raw_delta))
        else:
            raw_delta = None
            capped = None

        result[wk] = {
            "current":      round(cur, 4),
            "rho":          round(rhos[wk], 4) if rhos[wk] is not None else None,
            "raw_delta":    round(raw_delta, 4) if raw_delta is not None else None,
            "capped_delta": round(capped, 4) if capped is not None else None,
            "suggested":    round(suggested_norm[wk], 4),
            "n":            ns.get(wk, 0),
        }

    return result


def print_tuning_report(suggestions: dict[str, dict], current_cfg: dict, apply: bool) -> None:
    print("\n## Weight Tuning Report\n")
    print(f"Current config: {config_summary(current_cfg)}\n")

    changed = False
    print(f"{'Component':18s}  {'current':>8s}  {'rho':>7s}  {'delta':>7s}  {'suggested':>9s}  n")
    print("-" * 68)
    for wk, info in suggestions.items():
        col = _COMPONENT_MAP.get(wk, wk)
        rho_s  = f"{info['rho']:+.3f}"  if info["rho"]         is not None else "    —"
        dlt_s  = f"{info['capped_delta']:+.3f}" if info["capped_delta"] is not None else "    —"
        sug    = info["suggested"]
        cur    = info["current"]
        marker = " ←" if abs(sug - cur) >= 0.001 else ""
        if marker:
            changed = True
        print(f"  {col:16s}  {cur:8.4f}  {rho_s:>7s}  {dlt_s:>7s}  {sug:9.4f}  {info['n']}{marker}")
        if "note" in info:
            print(f"    note: {info['note']}")

    print()
    if not changed:
        print("No weight changes suggested (all within noise threshold).")
    elif apply:
        print("Weights applied → data/scoring_config.json")
    else:
        print("Dry run — use --apply to persist suggested weights.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Suggest or apply cautious scoring weight updates.")
    parser.add_argument("--log",         default=str(LOG_DEFAULT),  help="Path to signal_log.jsonl")
    parser.add_argument("--min-records", default=MIN_RECORDS, type=int,
                        help=f"Minimum evaluated records required (default {MIN_RECORDS})")
    parser.add_argument("--apply",       action="store_true",
                        help="Persist suggested weights to data/scoring_config.json")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Show suggestions without applying (default behaviour)")
    args = parser.parse_args()

    log_path = pathlib.Path(args.log)
    records  = load_log(log_path)
    if not records:
        print("Signal log is empty or missing.  Nothing to tune.")
        sys.exit(0)

    current_cfg = load_config()
    suggestions = compute_weight_suggestions(records, current_cfg, args.min_records)

    print_tuning_report(suggestions, current_cfg, apply=args.apply)

    if args.apply:
        new_cfg = dict(current_cfg)
        for wk, info in suggestions.items():
            new_cfg[wk] = info["suggested"]
        try:
            save_config(new_cfg)
        except Exception as exc:
            print(f"Failed to save config: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
