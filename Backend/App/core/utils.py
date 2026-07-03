# core/utils.py
from __future__ import annotations

"""
Shared utility and helper functions.
Responsibility: Pure, stateless helper logic only — no LLM, no DB, no API calls.
"""

import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.state import EligibilityResult

from schemas import PatientProfile, TrialProfile
from config.settings import (
    SEARCH_TERM_MAP,
    MAX_SCORE_REASONS,
    MAX_LLM_SUMMARY_CHARS,
    MAX_LLM_DESCRIPTION_CHARS,
    MAX_LLM_CRITERIA_CHARS,
    MAX_LLM_STUDY_POPULATION_CHARS,
    MAX_LLM_LOCATIONS,
    MAX_LLM_MESH_TERMS,
)

__all__ = [
    "normalize_free_text",
    "truncate_text",
    "safe_join_text",
    "limit_score_reasons",
    "normalize_sex",
    "parse_age_to_years",
    "get_patient_age",
    "get_patient_sex",
    "get_patient_country",
    "build_patient_summary",
    "resolve_cancer_type_from_structured_data",
    "get_api_search_term",
    "get_trial_searchable_text",
    "get_trial_key",
    "build_trial_llm_context",
    "serialize_trial_results",
    "serialize_trial_summaries",
]

# -----------------------------------------------------------
# TEXT NORMALIZATION HELPERS
# -----------------------------------------------------------

def normalize_free_text(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def truncate_text(value: Optional[str], max_chars: int) -> Optional[str]:
    if not value:
        return None
    value = value.strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "..."


def safe_join_text(parts: List[Optional[str]]) -> str:
    return " ".join([str(p).strip() for p in parts if p and str(p).strip()])


def limit_score_reasons(
    reasons: List[str],
    limit: int = MAX_SCORE_REASONS,
) -> List[str]:
    return [r for r in reasons if r][:limit]


# -----------------------------------------------------------
# PATIENT DATA HELPERS
# -----------------------------------------------------------

def normalize_sex(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    val = str(value).strip().upper()
    if val in {"ALL", "ANY"}:
        return "ALL"
    if val in {"MALE", "M"}:
        return "MALE"
    if val in {"FEMALE", "F"}:
        return "FEMALE"
    return val


def parse_age_to_years(age_str: Optional[str]) -> Optional[float]:
    if not age_str:
        return None
    raw = age_str.strip().lower()
    if raw in {"n/a", "not specified", "na"}:
        return None
    match = re.match(
        r"(\d+)\s+(year|years|month|months|week|weeks|day|days)", raw
    )
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if "year" in unit:
        return value
    if "month" in unit:
        return value / 12
    if "week" in unit:
        return value / 52
    if "day" in unit:
        return value / 365
    return None


def get_patient_age(patient_data: PatientProfile) -> Optional[float]:
    if patient_data.age is None:
        return None
    return float(patient_data.age)


def get_patient_sex(patient_data: PatientProfile) -> Optional[str]:
    return normalize_sex(patient_data.gender)


def get_patient_country(patient_data: PatientProfile) -> Optional[str]:
    if not patient_data.country:
        return None
    return patient_data.country.strip().lower()


def _is_meaningful(value: Any) -> bool:
    """Returns True if a value is non-empty and worth showing in a summary."""
    if value is None or value == "" or value == {}:
        return False
    if isinstance(value, list):
        return any(str(i).strip() for i in value)
    return True


def build_patient_summary(patient_data: PatientProfile) -> str:
    data = patient_data.model_dump()
    return "\n".join(
        f"- {k.replace('_', ' ').title()}: {v}"
        for k, v in data.items()
        if _is_meaningful(v)
    )


def resolve_cancer_type_from_structured_data(
    patient_data: PatientProfile,
) -> str:
    for value in [patient_data.cancer_type, patient_data.diagnosis]:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


# -----------------------------------------------------------
# TRIAL DATA HELPERS
# -----------------------------------------------------------

def get_api_search_term(cancer_type: str) -> str:
    clean = re.sub(r"\(.*?\)", "", cancer_type or "").strip()
    clean_lower = clean.lower()

    # Exact match first (case-insensitive)
    for mesh_term, simple_term in SEARCH_TERM_MAP.items():
        if mesh_term.lower() == clean_lower:
            return simple_term

    # Partial match fallback
    for mesh_term, simple_term in SEARCH_TERM_MAP.items():
        if mesh_term.lower() in clean_lower:
            return simple_term

    return clean_lower


def get_trial_searchable_text(trial: TrialProfile) -> str:
    location_text = " ".join(
        safe_join_text([loc.facility, loc.city, loc.country])
        for loc in trial.locations
    )
    return normalize_free_text(
        safe_join_text([
            trial.title,
            trial.official_title,
            " ".join(trial.conditions),
            " ".join(trial.mesh_terms),
            trial.brief_summary,
            trial.detailed_description,
            trial.eligibility.criteria_text,
            trial.eligibility.study_population,
            location_text,
        ])
    )


def get_trial_key(trial: TrialProfile) -> str:
    """Returns a stable unique key for a trial — used for deduplication."""
    return (
        trial.trial_id
        or normalize_free_text(trial.title)
        or f"trial_{id(trial)}"
    )


def build_trial_llm_context(trial: TrialProfile) -> Dict[str, Any]:
    compact_locations = [
        {
            "facility": loc.facility,
            "city": loc.city,
            "country": loc.country,
        }
        for loc in trial.locations[:MAX_LLM_LOCATIONS]
    ]
    return {
        "trial_id": trial.trial_id,
        "title": trial.title,
        "official_title": trial.official_title,
        "status": trial.status,
        "study_type": trial.study_type,
        "phases": trial.phases[:5],
        "conditions": trial.conditions[:8],
        "brief_summary": truncate_text(
            trial.brief_summary, MAX_LLM_SUMMARY_CHARS
        ),
        "detailed_description": truncate_text(
            trial.detailed_description, MAX_LLM_DESCRIPTION_CHARS
        ),
        "eligibility": {
            "criteria_text": truncate_text(
                trial.eligibility.criteria_text, MAX_LLM_CRITERIA_CHARS
            ),
            "healthy_volunteers": trial.eligibility.healthy_volunteers,
            "sex": trial.eligibility.sex,
            "minimum_age": trial.eligibility.minimum_age,
            "maximum_age": trial.eligibility.maximum_age,
            "age_groups": trial.eligibility.age_groups[:5],
            "study_population": truncate_text(
                trial.eligibility.study_population,
                MAX_LLM_STUDY_POPULATION_CHARS,
            ),
        },
        "locations": compact_locations,
        "sponsor_name": trial.sponsor_name,
        "sponsor_class": trial.sponsor_class,
        "mesh_terms": trial.mesh_terms[:MAX_LLM_MESH_TERMS],
        "has_results": trial.has_results,
    }


# -----------------------------------------------------------
# RESULT SERIALIZERS
# -----------------------------------------------------------

def serialize_trial_results(
    results: List["EligibilityResult"],
) -> List[Dict[str, Any]]:
    """Serialize EligibilityResult list to plain dicts for DB persistence."""
    return [
        {
            "nct_id": r.get("nct_id"),
            "title": r.get("title"),
            "hard_filter_pass": r.get("hard_filter_pass"),
            "hard_filter_reasons": r.get("hard_filter_reasons", []),
            "score": r.get("score"),
            "score_reasons": r.get("score_reasons", []),
            "biomarker_check": r.get("biomarker_check"),
            "treatment_check": r.get("treatment_check"),
            "assessment": r.get("assessment"),
        }
        for r in results
    ]


def serialize_trial_summaries(
    trials: List[TrialProfile],
) -> List[Dict[str, Any]]:
    """Serialize TrialProfile list to compact summary dicts."""
    return [
        {
            "trial_id": trial.trial_id,
            "title": trial.title,
            "status": trial.status,
            "study_type": trial.study_type,
            "phases": trial.phases[:3],
            "conditions": trial.conditions[:5],
            "locations": [
                {
                    "facility": loc.facility,
                    "city": loc.city,
                    "country": loc.country,
                }
                for loc in trial.locations[:5]
            ],
        }
        for trial in trials
    ]
