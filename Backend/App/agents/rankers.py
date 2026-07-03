# agents/ranker.py
"""
Priority-based trial ranking.
Responsibility: Take scored EligibilityResults and sort them into
                priority tiers for final recommendation surfacing.

Depends on: core/state (EligibilityResult)
Called by:  recommend_trials_node (Node 4) in agents/nodes.py

Tier System:
    Tier 1 🥇 — score >= 10, hard_filter_pass, no red flags
    Tier 2 🥈 — score 5–9,  hard_filter_pass, no red flags
    Tier 3 🥉 — score 1–4,  hard_filter_pass, no red flags
    Excluded  — score < 1 OR hard_filter_pass=False OR red flags present
"""

import logging
from typing import List, NamedTuple

from core.state import EligibilityResult

logger = logging.getLogger("uvicorn.error")

__all__ = [
    "rank_trials_by_priority",
    "RankedTrial",
    "PriorityTier",
]


# -----------------------------------------------------------
# TIER THRESHOLDS
# -----------------------------------------------------------

_TIER_1_MIN_SCORE = 10   # Strong match
_TIER_2_MIN_SCORE = 5    # Moderate match
_TIER_3_MIN_SCORE = 1    # Weak but plausible match


# -----------------------------------------------------------
# RED FLAG PHRASES
# Phrases in assessment or score_reasons that
# disqualify a trial from any tier regardless of score.
# Kept in sync with is_reasonably_strong_trial() in scoring.py
# -----------------------------------------------------------

_RED_FLAG_PHRASES = frozenset([
    "early-stage while trial targets advanced",
    "advanced/metastatic while trial targets early-stage",
    "treatment-naive patients but patient has prior",
    "exceed trial's maximum",
    "explicitly excluded by this trial",
    "not found in trial's extracted signals",
    "match status: no match",
])


# -----------------------------------------------------------
# PRIORITY TIER ENUM-LIKE CONSTANTS
# Using plain strings for JSON-serializable output
# -----------------------------------------------------------

class PriorityTier:
    TIER_1 = "TIER_1"   # 🥇 Strong match
    TIER_2 = "TIER_2"   # 🥈 Moderate match
    TIER_3 = "TIER_3"   # 🥉 Weak but plausible
    EXCLUDED = "EXCLUDED"


# -----------------------------------------------------------
# RANKED TRIAL CONTAINER
# Wraps EligibilityResult with tier + rank metadata
# -----------------------------------------------------------

class RankedTrial(NamedTuple):
    """
    Wraps an EligibilityResult with priority tier and rank metadata.

    Fields:
        result      — original EligibilityResult
        tier        — PriorityTier constant
        tier_rank   — position within the tier (1-indexed)
        overall_rank — position across all tiers (1-indexed)
    """
    result: EligibilityResult
    tier: str
    tier_rank: int
    overall_rank: int


# -----------------------------------------------------------
# INTERNAL HELPERS
# -----------------------------------------------------------

def _has_red_flags(result: EligibilityResult) -> bool:
    """
    Returns True if the result contains any disqualifying phrase
    in assessment or score_reasons.
    """
    assessment_lower = (result.get("assessment") or "").lower()
    score_reasons_lower = " ".join(
        result.get("score_reasons") or []
    ).lower()

    combined = assessment_lower + " " + score_reasons_lower
    return any(phrase in combined for phrase in _RED_FLAG_PHRASES)


def _assign_tier(result: EligibilityResult) ->str:
    """
    Assigns a PriorityTier to a single EligibilityResult.

    Rules (evaluated top-to-bottom):
        1. hard_filter_pass must be True
        2. No red flag phrases in assessment or score_reasons
        3. Score >= 10  → TIER_1
        4. Score 5–9    → TIER_2
        5. Score 1–4    → TIER_3
        6. Anything else → EXCLUDED
    """
    if not result.get("hard_filter_pass"):
        return PriorityTier.EXCLUDED

    if _has_red_flags(result):
        return PriorityTier.EXCLUDED

    score = result.get("score") or 0

    if score >= _TIER_1_MIN_SCORE:
        return PriorityTier.TIER_1
    if score >= _TIER_2_MIN_SCORE:
        return PriorityTier.TIER_2
    if score >= _TIER_3_MIN_SCORE:
        return PriorityTier.TIER_3

    return PriorityTier.EXCLUDED


# -----------------------------------------------------------
# MAIN RANKER
# -----------------------------------------------------------

def rank_trials_by_priority(
    results: List[EligibilityResult],
    include_excluded: bool = False,
) -> List[RankedTrial]:
    """
    Sorts EligibilityResults into priority tiers and returns
    a fully ranked list of RankedTrial objects.

    Ranking logic:
        1. Assign each result a PriorityTier via _assign_tier()
        2. Within each tier → sort by score DESC
        3. Final order: TIER_1 → TIER_2 → TIER_3 → EXCLUDED (optional)
        4. Assign overall_rank and tier_rank

    Args:
        results:          List of EligibilityResult dicts from evaluate_trials_node
        include_excluded: If True, EXCLUDED trials are appended at the end.
                          Default False — excluded trials are dropped.

    Returns:
        List[RankedTrial] — ordered by priority tier then score DESC
    """
    if not results:
        return []

    # Step 1 — Bucket results by tier
    buckets: dict[str, List[EligibilityResult]] = {
        PriorityTier.TIER_1: [],
        PriorityTier.TIER_2: [],
        PriorityTier.TIER_3: [],
        PriorityTier.EXCLUDED: [],
    }

    for result in results:
        tier = _assign_tier(result)
        buckets[tier].append(result)

    # Step 2 — Sort each bucket by score DESC
    for tier in buckets:
        buckets[tier].sort(
            key=lambda r: r.get("score") or 0,
            reverse=True,
        )

    # Step 3 — Log tier summary for observability
    logger.info(
        "Priority ranking complete — "
        "Tier 1: %d | Tier 2: %d | Tier 3: %d | Excluded: %d",
        len(buckets[PriorityTier.TIER_1]),
        len(buckets[PriorityTier.TIER_2]),
        len(buckets[PriorityTier.TIER_3]),
        len(buckets[PriorityTier.EXCLUDED]),
    )

    # Step 4 — Build ordered RankedTrial list
    ordered_tiers = [
        PriorityTier.TIER_1,
        PriorityTier.TIER_2,
        PriorityTier.TIER_3,
    ]
    if include_excluded:
        ordered_tiers.append(PriorityTier.EXCLUDED)

    ranked: List[RankedTrial] = []
    overall_rank = 1

    for tier in ordered_tiers:
        for tier_rank, result in enumerate(buckets[tier], start=1):
            ranked.append(RankedTrial(
                result=result,
                tier=tier,
                tier_rank=tier_rank,
                overall_rank=overall_rank,
            ))
            overall_rank += 1

    return ranked


# -----------------------------------------------------------
# CONVENIENCE ACCESSORS
# -----------------------------------------------------------

def get_top_trials(
    ranked: List[RankedTrial],
    n: int = 3,
    min_tier: str = PriorityTier.TIER_3,
) -> List[RankedTrial]:
    """
    Returns the top N ranked trials filtered by minimum tier.

    Args:
        ranked:   Output of rank_trials_by_priority()
        n:        Maximum number of trials to return (default 3)
        min_tier: Minimum acceptable tier (default TIER_3)
                  Use PriorityTier.TIER_1 to get only strong matches.

    Returns:
        List[RankedTrial] — top N at or above min_tier
    """
    _TIER_ORDER = {
        PriorityTier.TIER_1: 1,
        PriorityTier.TIER_2: 2,
        PriorityTier.TIER_3: 3,
        PriorityTier.EXCLUDED: 4,
    }
    min_tier_value = _TIER_ORDER.get(min_tier, 3)

    filtered = [
        rt for rt in ranked
        if _TIER_ORDER.get(rt.tier, 4) <= min_tier_value
    ]
    return filtered[:n]


def get_tier_summary(ranked: List[RankedTrial]) -> dict:
    """
    Returns a summary dict of tier counts and top trial per tier.
    Useful for logging, API responses, and frontend display.

    Returns:
    {
        "total": int,
        "tier_1_count": int,
        "tier_2_count": int,
        "tier_3_count": int,
        "excluded_count": int,
        "top_trial": { nct_id, title, score, tier } | None
    }
    """
    counts = {
        PriorityTier.TIER_1: 0,
        PriorityTier.TIER_2: 0,
        PriorityTier.TIER_3: 0,
        PriorityTier.EXCLUDED: 0,
    }
    for rt in ranked:
        counts[rt.tier] = counts.get(rt.tier, 0) + 1

    top = ranked[0] if ranked else None

    return {
        "total": len(ranked),
        "tier_1_count": counts[PriorityTier.TIER_1],
        "tier_2_count": counts[PriorityTier.TIER_2],
        "tier_3_count": counts[PriorityTier.TIER_3],
        "excluded_count": counts[PriorityTier.EXCLUDED],
        "top_trial": {
            "nct_id": top.result.get("nct_id"),
            "title": top.result.get("title"),
            "score": top.result.get("score"),
            "tier": top.tier,
        } if top else None,
    }
