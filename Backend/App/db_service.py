# db_service.py
"""
Database service layer.
Responsibility: All DB read/write operations — no business logic.
Depends on: models.py, database.py
"""

import json
import logging
import unicodedata
from typing import Any, Dict, List, Union

from sqlmodel import Session

from models import PatientProfileTable, TrialMatchTable

logger = logging.getLogger("uvicorn.error")

__all__ = [
    "create_patient_profile_record",
    "create_trial_match_run",
    "save_trial_match_results",
    "update_trial_match_run_success",
    "update_trial_match_run_failed",
    "update_trial_match_run_partial",
]


# -----------------------------------------------------------
# INTERNAL HELPERS
# -----------------------------------------------------------

def _to_dict(payload: Union[dict, Any]) -> Dict[str, Any]:
    """
    Converts various input types to a plain dict.
    Handles: dict, Pydantic/SQLModel models, or any object.
    """
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return dict(payload)


def _safe_text(value: Any) -> str:
    """
    Converts a value to a clean ASCII-safe string.
    Replaces problematic Unicode characters for DB storage.
    """
    if value is None:
        return ""

    text = str(value)

    replacements = {
        "\u2160": "I",       # Roman numeral I
        "\u2161": "II",      # Roman numeral II
        "\u2162": "III",     # Roman numeral III
        "\u2163": "IV",      # Roman numeral IV
        "\u2164": "V",       # Roman numeral V
        "\u2013": "-",       # en-dash
        "\u2014": "-",       # em-dash
        "\u2018": "'",       # left single quote
        "\u2019": "'",       # right single quote
        "\u201c": '"',       # left double quote
        "\u201d": '"',       # right double quote
        "\u2264": "&lt;=",      # ≤ — actual characters, not HTML entities
        "\u2265": "&gt;=",      # ≥ — actual characters, not HTML entities
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    text = unicodedata.normalize("NFKD", text)
    return text


# -----------------------------------------------------------
# PATIENT PROFILE
# -----------------------------------------------------------

def create_patient_profile_record(
    session: Session,
    patient_profile: Union[dict, Any],
) -> PatientProfileTable:
    """
    Creates and flushes a PatientProfileTable record.
    Returns the record with DB-assigned ID.
    Does NOT commit — caller is responsible for commit.
    """
    data = _to_dict(patient_profile)

    record = PatientProfileTable(
        age=int(data["age"]) if data.get("age") else None,
        gender=_safe_text(data.get("gender", "")) or None,
        cancer_type=_safe_text(data.get("cancer_type", "")) or None,
        cancer_stage=_safe_text(data.get("cancer_stage", "")) or None,
        biomarkers=[
            _safe_text(x)
            for x in (data.get("biomarkers") or [])
        ],
        previous_treatments=[
            _safe_text(x)
            for x in (data.get("previous_treatments") or [])
        ],
    )

    session.add(record)
    session.flush()

    logger.info(
        "Patient profile record created | id=%s | cancer_type=%s | age=%s",
        record.id,
        record.cancer_type,
        record.age,
    )

    return record


# -----------------------------------------------------------
# TRIAL MATCH RUN
# -----------------------------------------------------------

def create_trial_match_run(
    session: Session,
    patient_profile_id: int,
) -> Dict[str, Any]:
    """
    Creates a trial match run record linked to a patient profile.

    TODO: Replace with real TrialMatchRunTable insert once
    TrialMatchRunTable model is added to models.py.
    Currently returns a mock run record.
    """
    logger.warning(
        "create_trial_match_run() is a stub — no DB write performed. "
        "patient_profile_id=%s",
        patient_profile_id,
    )
    return {
        "id": patient_profile_id,
        "patient_profile_id": patient_profile_id,
    }


# -----------------------------------------------------------
# TRIAL MATCH RESULTS
# -----------------------------------------------------------

def save_trial_match_results(
    session: Session,
    trial_match_run_id: int,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Saves individual trial match results to TrialMatchTable.
    match_run_id removed — TrialMatchRunTable not yet implemented.

    TODO: When TrialMatchRunTable is added, look up actual
    patient_profile_id from the run record instead of using
    trial_match_run_id directly as alias.
    """
    # Temporary alias — valid only while create_trial_match_run is a stub
    patient_profile_id = trial_match_run_id

    saved = 0
    for result in results:
        explanation = (
            result.get("assessment")
            or json.dumps(result, indent=2, ensure_ascii=False)
        )

        row = TrialMatchTable(
            patient_profile_id=patient_profile_id,
            nct_id=_safe_text(result.get("nct_id", "")) or None,
            title=_safe_text(result.get("title", "")) or None,
            match_explanation=_safe_text(explanation),
            # match_run_id intentionally omitted
            # until TrialMatchRunTable is implemented
        )
        session.add(row)
        saved += 1

    session.flush()

    logger.info(
        "Trial match results saved | patient_profile_id=%s | count=%d",
        patient_profile_id,
        saved,
    )

    return {"saved_count": saved}


# -----------------------------------------------------------
# TRIAL MATCH RUN STATUS UPDATES
# -----------------------------------------------------------

def update_trial_match_run_success(
    session: Session,
    trial_match_run_id: int,
    final_recommendations: str = "",
) -> Dict[str, Any]:
    """
    Marks a trial match run as successfully completed.
    TODO: Update actual TrialMatchRunTable row when model exists.
    """
    logger.warning(
        "update_trial_match_run_success() is a stub — no DB write. "
        "run_id=%s",
        trial_match_run_id,
    )
    return {
        "id": trial_match_run_id,
        "status": "success",
        "final_recommendations": _safe_text(final_recommendations),
    }


def update_trial_match_run_failed(
    session: Session,
    trial_match_run_id: int,
    error: str = "",
) -> Dict[str, Any]:
    """
    Marks a trial match run as failed.
    TODO: Update actual TrialMatchRunTable row when model exists.
    """
    logger.warning(
        "update_trial_match_run_failed() is a stub — no DB write. "
        "run_id=%s | error=%s",
        trial_match_run_id,
        error,
    )
    return {
        "id": trial_match_run_id,
        "status": "failed",
        "error": _safe_text(error),
    }


def update_trial_match_run_partial(
    session: Session,
    trial_match_run_id: int,
    final_recommendations: str = "",
) -> Dict[str, Any]:
    """
    Marks a trial match run as partially completed.
    TODO: Update actual TrialMatchRunTable row when model exists.
    """
    logger.warning(
        "update_trial_match_run_partial() is a stub — no DB write. "
        "run_id=%s",
        trial_match_run_id,
    )
    return {
        "id": trial_match_run_id,
        "status": "partial",
        "final_recommendations": _safe_text(final_recommendations),
    }
