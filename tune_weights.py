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
    min_exec_count: int = 1,
    min_active_exchanges: int = 3,
) -> dict[str, dict]:
    """
    For each weight key, compute Spearman rho of its component against
    direction_correct, then suggest a new weight (bounded, normalised).

    Records are filtered before correlation:
    - Only evaluated (direction_correct not null) and non-NEUTRAL direction.
    - min_exec_count (default 1): exclude records where exec_count < N.
      This prevents records where leader_gap=0.0 due to missing exec data
      from diluting the leader_gap↔outcome correlation.
    - min_active_exchanges (default 3): exclude thin-panel runs.

    Records that pre-date the health fields (no "exec_count" key) are treated
    as exec_count=999 so they are NOT excluded — backward compatible with
    logs written before this change.

    Returns a dict keyed by weight name plus a "_summary" meta key:
      {
        "w_rate_extreme": {"current", "rho", "raw_delta", "capped_delta", "suggested", "n"},
        ...
        "_summary": {"n_total_evaluated", "n_after_filter", "n_excluded_degraded",
                     "min_exec_count", "min_active_exchanges"},
      }
    """
    # All evaluated, non-NEUTRAL records (before degradation filter)
    all_evaluated = [
        r for r in records
        if r.get("direction_correct") is not None
        and r.get("direction") not in ("NEUTRAL", None)
    ]

    # Apply data-health filter.
    # Records lacking health fields (written before this change) pass through
    # unfiltered (treated as 999) to preserve backward compatibility.
    evaluated = [
        r for r in all_evaluated
        if r.get("exec_count", 999) >= min_exec_count
        and r.get("active_exchanges", 999) >= min_active_exchanges
    ]

    n_excluded = len(all_evaluated) - len(evaluated)

    result: dict[str, dict] = {
        "_summary": {
            "n_total_evaluated":      len(all_evaluated),
            "n_after_filter":         len(evaluated),
            "n_excluded_degraded":    n_excluded,
            "min_exec_count":         min_exec_count,
            "min_active_exchanges":   min_active_exchanges,
        }
    }

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

    # Print data-health filter summary if present
    meta = suggestions.get("_summary", {})
    if meta:
        n_total  = meta.get("n_total_evaluated", "?")
        n_after  = meta.get("n_after_filter", "?")
        n_excl   = meta.get("n_excluded_degraded", 0)
        min_exc  = meta.get("min_exec_count", "?")
        min_act  = meta.get("min_active_exchanges", "?")
        print(f"Records: {n_total} evaluated, {n_after} used for correlation "
              f"({n_excl} excluded: exec_count<{min_exc} or active_exchanges<{min_act})")
        if n_excl > 0:
            frac = n_excl / max(n_total, 1)
            if frac > 0.20:
                print(f"  ⚠  {frac:.0%} of evaluated records excluded as degraded — "
                      f"weight suggestions may be unreliable. "
                      f"Investigate data gaps before applying.")
        print()

    changed = False
    print(f"{'Component':18s}  {'current':>8s}  {'rho':>7s}  {'delta':>7s}  {'suggested':>9s}  n")
    print("-" * 68)
    for wk, info in suggestions.items():
        if wk == "_summary":
            continue
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
    parser.add_argument("--min-exec-count", default=1, type=int,
                        help="Exclude records where exec_count < N from correlation (default 1). "
                             "Prevents NaN-zeroed leader_gap from diluting weight suggestions.")
    parser.add_argument("--min-active-exchanges", default=3, type=int,
                        help="Exclude records where active_exchanges < N (default 3). "
                             "Prevents thin-panel runs from skewing correlation.")
    args = parser.parse_args()

    log_path = pathlib.Path(args.log)
    records  = load_log(log_path)
    if not records:
        print("Signal log is empty or missing.  Nothing to tune.")
        sys.exit(0)

    current_cfg = load_config()
    suggestions = compute_weight_suggestions(
        records, current_cfg, args.min_records,
        min_exec_count=args.min_exec_count,
        min_active_exchanges=args.min_active_exchanges,
    )

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
