"""
venue_ranking.py — rank execution venues by directional funding-rate edge.

For each venue we compute:

    venue_gap_raw   = ref_rate - venue_rate
        (+) means venue is BELOW the reference → room to move up
        (-) means venue is ABOVE the reference → crowded relative to leaders

    venue_edge_dir  = venue_gap_raw * direction_sign
        Flip sign so that "positive edge" always means
        "this venue gives us an advantage in the signalled direction".
        e.g. LONG signal + venue below ref → positive edge (cheaper entry)
             SHORT signal + venue above ref → positive edge (more expensive to borrow)

    venue_edge_norm = venue_edge_dir / max(rate_std, EPS)
        Normalised by historical rate volatility so scores are comparable
        across symbols with very different rate magnitudes.

The caller passes:
  - direction   : "LONG" | "SHORT" | "NEUTRAL"
  - venue_rates : {venue_name: smoothed_rate}  (NaN venues should be excluded)
  - ref_rate    : leader consensus rate (or overall consensus if no leaders)
  - rate_std    : historical std of consensus rate

Returns a list of dicts sorted best→worst, ready to JSON-serialise.
"""

from __future__ import annotations

import math
from typing import NamedTuple

EPS = 1e-10


class VenueScore(NamedTuple):
    venue: str
    edge_norm: float          # normalised directional edge (higher = better)
    edge_dir: float           # raw directional edge (bps equivalent)
    gap_raw: float            # ref_rate - venue_rate (unsigned direction)
    venue_rate: float         # smoothed venue rate
    recommendation: str       # "preferred" | "neutral" | "avoid"


def rank_venues(
    direction: str,
    venue_rates: dict[str, float],
    ref_rate: float,
    rate_std: float,
) -> list[VenueScore]:
    """
    Rank venues by directional edge.  Returns [] for NEUTRAL direction
    or when venue_rates is empty.

    Parameters
    ----------
    direction   : "LONG", "SHORT", or "NEUTRAL"
    venue_rates : mapping of venue name → smoothed rate (NaN already dropped)
    ref_rate    : leader consensus rate (or consensus rate if no leaders)
    rate_std    : historical std of consensus rate for normalisation
    """
    if direction == "NEUTRAL" or not venue_rates:
        return []

    direction_sign = 1.0 if direction == "LONG" else -1.0
    std_safe = max(rate_std, EPS)

    scores: list[VenueScore] = []
    for venue, vrate in venue_rates.items():
        if not math.isfinite(vrate):
            continue
        gap_raw    = ref_rate - vrate
        edge_dir   = gap_raw * direction_sign
        edge_norm  = edge_dir / std_safe

        if edge_norm >= 0.30:
            rec = "preferred"
        elif edge_norm <= -0.30:
            rec = "avoid"
        else:
            rec = "neutral"

        scores.append(VenueScore(
            venue        = venue,
            edge_norm    = round(edge_norm, 4),
            edge_dir     = round(edge_dir, 8),
            gap_raw      = round(gap_raw, 8),
            venue_rate   = round(vrate, 8),
            recommendation = rec,
        ))

    # Best edge first; stable sort on venue name for ties
    scores.sort(key=lambda s: (-s.edge_norm, s.venue))
    return scores


def venues_to_dict(scores: list[VenueScore]) -> list[dict]:
    """Serialise VenueScore list to JSON-safe list of dicts."""
    return [
        {
            "venue":          s.venue,
            "edge_norm":      s.edge_norm,
            "edge_dir":       s.edge_dir,
            "gap_raw":        s.gap_raw,
            "venue_rate":     s.venue_rate,
            "recommendation": s.recommendation,
        }
        for s in scores
    ]


def best_venue(scores: list[VenueScore]) -> str | None:
    """Return name of the top-ranked venue, or None if list is empty."""
    return scores[0].venue if scores else None
