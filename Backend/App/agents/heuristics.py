"""
Heuristic pre-checks for LLM context enrichment.
Responsibility: Provide biomarker and treatment compatibility signals.
Depends on: Nothing external — pure string logic.
"""

import re
from typing import List

__all__ = [
    "heuristic_biomarker_check",
    "heuristic_treatment_history_check",
]

_EXCLUSION_MARKERS = frozenset([
    "no prior",
    "no previous",
    "must not have received",
    "must not have been treated with",
    "excluded",
    "exclusion",
    "prior treatment with",
    "previous treatment with",
    "refractory to",
    "progressed on",
    "intolerant to",
])

_BIOMARKER_ALIASES = {
    "her2+": ["her2+", "her2 positive", "erbb2", "erbb2 amplified", "erbb2 amplification"],
    "her2": ["her2", "erbb2"],
    "egfr": ["egfr", "epidermal growth factor receptor"],
    "alk": ["alk", "anaplastic lymphoma kinase"],
    "ros1": ["ros1"],
    "braf": ["braf", "braf v600e", "braf v600"],
    "kras": ["kras", "kras g12c"],
    "nras": ["nras"],
    "msi-h": ["msi-h", "microsatellite instability-high", "microsatellite instability high"],
    "dmmr": ["dmmr", "mismatch repair deficient", "mismatch-repair deficient"],
    "pdl1": ["pd-l1", "pdl1", "programmed death-ligand 1"],
    "er+": ["er+", "estrogen receptor positive", "er positive"],
    "pr+": ["pr+", "progesterone receptor positive", "pr positive"],
    "triple negative": ["triple negative", "tnbc"],
}

def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _regex_phrase_in_text(phrase: str, text: str) -> bool:
    pattern = r"\b" + re.escape(phrase.lower()) + r"\b"
    return bool(re.search(pattern, text))


def _get_biomarker_terms(biomarker: str) -> List[str]:
    biomarker_norm = _normalize_text(biomarker)
    if biomarker_norm in _BIOMARKER_ALIASES:
        return _BIOMARKER_ALIASES[biomarker_norm]
    return [biomarker_norm]


def _biomarker_in_text(biomarker: str, text: str) -> bool:
    terms = _get_biomarker_terms(biomarker)
    return any(_regex_phrase_in_text(term, text) for term in terms if term)


def _find_treatment_mentions(treatment: str, criteria_lower: str) -> List[int]:
    treatment = treatment.lower().strip()
    if not treatment:
        return []
    return [m.start() for m in re.finditer(re.escape(treatment), criteria_lower)]


def _check_treatment_conflict(treatment: str, criteria_lower: str) -> bool:
    """
    Checks whether a treatment appears in an exclusion-like context.
    Evaluates all mentions, not just the first.
    """
    positions = _find_treatment_mentions(treatment, criteria_lower)
    if not positions:
        return False

    for idx in positions:
        window_start = max(0, idx - 200)
        window_end = min(len(criteria_lower), idx + 200)
        context = criteria_lower[window_start:window_end]
        if any(marker in context for marker in _EXCLUSION_MARKERS):
            return True

    return False


def heuristic_biomarker_check(
    biomarkers: List[str],
    trial_criteria: str,
) -> str:
    """
    Checks which patient biomarkers appear in trial eligibility criteria.
    Returns a plain-text compatibility signal for LLM context enrichment.
    """
    if not biomarkers:
        return "No patient biomarkers provided."

    criteria_lower = _normalize_text(trial_criteria)
    matches, not_found = [], []

    for biomarker in biomarkers:
        biomarker_clean = (biomarker or "").strip()
        if not biomarker_clean:
            continue

        if _biomarker_in_text(biomarker_clean, criteria_lower):
            matches.append(biomarker_clean)
        else:
            not_found.append(biomarker_clean)

    if not matches and not not_found:
        return "No valid biomarkers could be evaluated."

    result = []
    if matches:
        result.append(f"Biomarkers FOUND in criteria: {', '.join(matches)}")
    if not_found:
        result.append(f"Biomarkers NOT explicitly mentioned: {', '.join(not_found)}")

    if matches and not not_found:
        compatibility = "HIGH"
    elif matches:
        compatibility = "MEDIUM"
    else:
        compatibility = "LOW"

    result.append(f"Biomarker Compatibility Signal: {compatibility}")
    return "\n".join(result)


def heuristic_treatment_history_check(
    previous_treatments: List[str],
    trial_criteria: str,
) -> str:
    """
    Checks patient treatment history against trial eligibility criteria.
    Returns a plain-text compatibility signal for LLM context enrichment.
    """
    if not previous_treatments:
        return "No previous treatment history provided."

    criteria_lower = _normalize_text(trial_criteria)
    compatible, potential_conflicts, not_mentioned = [], [], []

    for treatment in previous_treatments:
        treatment_clean = (treatment or "").strip()
        if not treatment_clean:
            continue

        mentions = _find_treatment_mentions(treatment_clean, criteria_lower)

        if _check_treatment_conflict(treatment_clean, criteria_lower):
            potential_conflicts.append(treatment_clean)
        elif mentions:
            compatible.append(treatment_clean)
        else:
            not_mentioned.append(treatment_clean)

    if not compatible and not potential_conflicts and not not_mentioned:
        return "No valid treatments could be evaluated."

    result = []
    if compatible:
        result.append(
            f"Prior treatments explicitly referenced without obvious conflict: {', '.join(compatible)}"
        )
    if potential_conflicts:
        result.append(
            f"Potential treatment-history conflicts: {', '.join(potential_conflicts)}"
        )
    if not_mentioned:
        result.append(
            f"Prior treatments not explicitly mentioned in criteria: {', '.join(not_mentioned)}"
        )

    if potential_conflicts:
        compatibility = "LOW" if not compatible else "MEDIUM"
    elif compatible:
        compatibility = "HIGH"
    else:
        compatibility = "MEDIUM"

    result.append(f"Treatment History Compatibility Signal: {compatibility}")
    return "\n".join(result)
