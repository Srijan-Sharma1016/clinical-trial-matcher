"""
Output formatting helpers.
Responsibility: Format data structures into human-readable
strings for LLM prompts and fallback reports.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from core.state import EligibilityResult
    from schemas import PatientProfile

__all__ = [
    "format_eligibility_assessment_for_prompt",
    "format_hard_filter_no_match_assessment",
    "format_not_assessed_assessment",
    "format_fallback_recommendation",
]

# -----------------------------------------------------------
# INTERNAL HELPERS
# -----------------------------------------------------------

def _format_reasons(reasons: List[str]) -> str:
    """Formats a list of reasons as bullet points, or '- None' if empty."""
    return "\n".join(f"- {r}" for r in reasons) if reasons else "- None"


# -----------------------------------------------------------
# PROMPT FORMATTERS
# -----------------------------------------------------------

def format_eligibility_assessment_for_prompt(
    result: "EligibilityResult",
) -> str:
    """
    Formats a single EligibilityResult into a structured
    plain-text block for the recommendation LLM prompt.
    """
    score_reasons = result.get("score_reasons") or []
    hard_filter_reasons = result.get("hard_filter_reasons") or []

    return (
        f"Trial ID: {result.get('nct_id')}\n"
        f"Title: {result.get('title')}\n"
        f"Hard Filter Pass: {result.get('hard_filter_pass')}\n"
        f"Hard Filter Reasons:\n"
        f"{_format_reasons(hard_filter_reasons)}\n"
        f"Deterministic Score: {result.get('score')}\n"
        f"Deterministic Reasons:\n"
        f"{_format_reasons(score_reasons)}\n"
        f"Biomarker Check:\n{result.get('biomarker_check') or 'Not available.'}\n"
        f"Treatment Check:\n{result.get('treatment_check') or 'Not available.'}\n"
        f"Assessment:\n{result.get('assessment') or 'Not available.'}"
    )


# -----------------------------------------------------------
# ASSESSMENT BUILDERS
# -----------------------------------------------------------

def format_hard_filter_no_match_assessment(reasons: List[str]) -> str:
    """
    Builds a structured NO MATCH assessment string
    for trials that failed hard eligibility filters.
    """
    against_text = _format_reasons(
        reasons or ["The trial failed one or more hard eligibility filters."]
    )
    return (
        "MATCH STATUS: NO MATCH\n"
        "REASONS FOR MATCH:\n"
        "- No strong supporting match can be claimed because hard filters failed.\n"
        "REASONS AGAINST MATCH:\n"
        f"{against_text}\n"
        "MISSING INFORMATION:\n"
        "- A clinician may still review edge cases if any profile data is incomplete.\n"
        "PATIENT BENEFITS:\n"
        "- Not applicable based on current hard-filter mismatch.\n"
        "CONCERNS:\n"
        "- Enrollment is unlikely unless key eligibility facts differ from the current profile.\n"
        "DISCLAIMER: This assessment is AI-assisted and for informational purposes only. "
        "A qualified oncologist must review before any enrollment decision."
    )


def format_not_assessed_assessment(score: int, reasons: List[str]) -> str:
    """
    Builds a structured NEEDS FURTHER REVIEW assessment string
    for trials that passed hard filters but weren't sent to the LLM.
    """
    reasons_text = _format_reasons(
        reasons or ["Ranked lower than the top trials selected for LLM review."]
    )
    return (
        "MATCH STATUS: NEEDS FURTHER REVIEW\n"
        "REASONS FOR MATCH:\n"
        f"- Deterministic score suggests possible relevance (score: {score}).\n"
        "REASONS AGAINST MATCH:\n"
        f"{reasons_text}\n"
        "MISSING INFORMATION:\n"
        "- Full LLM assessment was skipped because this trial ranked below the evaluation limit.\n"
        "PATIENT BENEFITS:\n"
        "- Could still warrant clinician review if higher-ranked options are unsuitable.\n"
        "CONCERNS:\n"
        "- Match confidence is limited because no full narrative eligibility assessment was generated.\n"
        "DISCLAIMER: This assessment is AI-assisted and for informational purposes only. "
        "A qualified oncologist must review before any enrollment decision."
    )


# -----------------------------------------------------------
# FALLBACK REPORT
# -----------------------------------------------------------

def format_fallback_recommendation(
    patient_data: "PatientProfile",
    eligibility_results: List["EligibilityResult"],
) -> str:
    """
    Generates a plain-text fallback recommendation report
    when no strong trial matches are found or the LLM fails.
    """
    from agents.scoring import _is_reasonably_strong_trial  # lazy import

    strong_trials = sorted(
        [r for r in eligibility_results if _is_reasonably_strong_trial(r)],
        key=lambda r: r.get("score", 0),
        reverse=True,
    )

    cancer_label = (
        patient_data.cancer_type
        or patient_data.diagnosis
        or "Not available"
    )

    lines = [
        "Final Trial Matching Summary",
        "",
        f"Patient cancer type: {cancer_label}",
        "",
    ]

    if not strong_trials:
        lines.extend([
            "No strong matches were found among the evaluated trials.",
            "",
            "Why:",
            "- The reviewed trials did not show strong enough alignment "
            "on the core rule-based checks.",
            "- Some options may still deserve clinician review if important "
            "patient details are incomplete or changing.",
            "",
            "Suggested next steps:",
            "- Reconfirm stage, biomarker profile, and prior treatment history.",
            "- Expand the search to more trials, locations, or related disease terms.",
            "- Ask an oncologist or trial coordinator to review borderline options.",
        ])
        return "\n".join(lines)

    lines.append("Top potentially relevant trials:")
    for idx, result in enumerate(strong_trials[:2], start=1):
        lines.extend([
            "",
            f"{idx}. {result.get('title')} ({result.get('nct_id')})",
            f"   - Deterministic score: {result.get('score')}",
            f"   - Key reasons: "
            f"{', '.join((result.get('score_reasons') or [])[:3]) or 'No reasons available'}",
        ])

    lines.extend([
        "",
        "Important cautions:",
        "- These results are screening support only, not a final eligibility decision.",
        "- Final trial fit depends on full protocol review, clinician judgment, "
        "and site-specific screening.",
    ])

    return "\n".join(lines)
