"""
Hard eligibility filter logic.
Responsibility: Rule-based binary pass/fail checks before LLM evaluation.
Depends on: config/settings.py, core/utils.py, schemas
"""

from typing import List, Tuple

from schemas import PatientProfile, TrialProfile
from config.settings import ALLOWED_ACTIVE_STATUSES
from core.utils import (
    get_patient_age,
    get_patient_sex,
    parse_age_to_years,
)

__all__ = ["hard_filter_trial"]

_NEGATIVE_CANCER_TERMS = frozenset({
    "none", "healthy", "n/a", "not applicable", "unknown", ""
})


def _normalize_text(value: str | None) -> str:
    return (value or "").strip().upper()


def _patient_has_cancer(patient_data: PatientProfile) -> bool:
    """Returns True if patient profile clearly indicates a cancer diagnosis."""
    cancer_value = (
        patient_data.cancer_type or patient_data.diagnosis or ""
    ).strip().lower()
    return bool(cancer_value) and cancer_value not in _NEGATIVE_CANCER_TERMS


def hard_filter_trial(
    patient_data: PatientProfile,
    trial: TrialProfile,
) -> Tuple[bool, List[str]]:
    """
    Applies hard eligibility filters to a trial.

    Filters applied in order:
        1. Trial status — must be actively recruiting
        2. Sex eligibility — must match or be ALL
        3. Age eligibility — must be within min/max bounds

    Returns:
        (passes: bool, reasons: List[str])
        reasons is populated with ALL failure reasons, not just the first.
    """
    reasons: List[str] = []
    passes = True

    eligibility = getattr(trial, "eligibility", None)

    # --- 1. Status Check ---
    status = (trial.status or "UNKNOWN").strip()
    normalized_status = status.upper()
    allowed_statuses = {s.strip().upper() for s in ALLOWED_ACTIVE_STATUSES}

    if normalized_status not in allowed_statuses:
        return False, [
            f"Trial status is '{status}', not currently active for enrollment."
        ]

    # If eligibility is missing, do not hard-fail the trial.
    # Let downstream evaluation handle it.
    if eligibility is None:
        return True, []

    # --- 2. Sex Check ---
    patient_sex = _normalize_text(get_patient_sex(patient_data))
    trial_sex = _normalize_text(getattr(eligibility, "sex", None))

    if patient_sex and trial_sex and trial_sex != "ALL" and patient_sex != trial_sex:
        passes = False
        reasons.append(
            f"Patient sex '{patient_sex}' does not match "
            f"trial eligibility '{trial_sex}'."
        )

    # --- 3. Age Check ---
    patient_age = get_patient_age(patient_data)
    if patient_age is not None:
        min_age = parse_age_to_years(getattr(eligibility, "minimum_age", None))
        max_age = parse_age_to_years(getattr(eligibility, "maximum_age", None))

        if min_age is not None and patient_age < min_age:
            passes = False
            reasons.append(
                f"Patient age {patient_age:g} is below "
                f"minimum eligible age {min_age:g}."
            )

        if max_age is not None and patient_age > max_age:
            passes = False
            reasons.append(
                f"Patient age {patient_age:g} is above "
                f"maximum eligible age {max_age:g}."
            )

    return passes, reasons
