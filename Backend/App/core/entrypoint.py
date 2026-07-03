# core/entrypoint.py
"""
Public entrypoint for trial matching workflow.
Responsibility: Orchestrate DB setup, run workflow, persist results.
Depends on: pipeline, state, utils, db_service, callbacks
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from sqlmodel import Session

from schemas import PatientProfile
from normalizer import normalize_patient_payload
from database import engine
from db_service import (
    create_patient_profile_record,
    create_trial_match_run,
    save_trial_match_results,
    update_trial_match_run_success,
    update_trial_match_run_failed,
    update_trial_match_run_partial,
)
from core.pipeline import get_trial_matching_app
from core.state import AgentState
from core.utils import serialize_trial_results, serialize_trial_summaries
from core.callbacks import default_callback

logger = logging.getLogger("uvicorn.error")

__all__ = [
    "run_trial_matching",
    "process_patient_trial_matching",
    "run_trial_matcher_agent",
]


# -----------------------------------------------------------
# DATABASE SAFETY HELPERS
# -----------------------------------------------------------

def _extract_record_id(record: Any) -> Optional[Any]:
    """
    Extracts the primary key ID from a DB record.
    Handles both dict and ORM model responses.
    """
    if record is None:
        return None
    if isinstance(record, dict):
        for key in (
            "id", "run_id", "match_run_id",
            "trial_match_run_id", "patient_profile_id", "patient_id"
        ):
            if key in record and record[key] is not None:
                return record[key]
    for attr in (
        "id", "run_id", "match_run_id",
        "trial_match_run_id", "patient_profile_id", "patient_id"
    ):
        if hasattr(record, attr):
            value = getattr(record, attr)
            if value is not None:
                return value
    return None


def _safe_service_call(
    fn: Any,
    attempts: List[Tuple[Tuple[Any, ...], Dict[str, Any]]],
) -> Any:
    """
    Tries multiple call signatures until one succeeds.
    TODO: Remove once db_service.py function signatures are finalized.
    Replace with direct calls e.g. create_patient_profile_record(session, patient_data)
    """
    last_error: Optional[Exception] = None
    for args, kwargs in attempts:
        try:
            return fn(*args, **kwargs)
        except TypeError as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    return None


# -----------------------------------------------------------
# DB PHASE HELPERS
# -----------------------------------------------------------

def _setup_db_records(
    session: Session,
    patient_data: PatientProfile,
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Creates patient profile and trial match run records.
    Returns (patient_profile_id, trial_match_run_id).
    Non-fatal — caller continues even if this fails.
    """
    patient_profile_id = None
    trial_match_run_id = None

    try:
        patient_record = _safe_service_call(
            create_patient_profile_record,
            [
                ((session, patient_data), {}),
                ((session, patient_data.model_dump()), {}),
                ((session,), {"patient_profile": patient_data}),
                ((session,), {"patient_data": patient_data.model_dump()}),
                ((session,), {"payload": patient_data.model_dump()}),
            ],
        )
        patient_profile_id = _extract_record_id(patient_record)
    except Exception:
        logger.exception("Failed to create patient profile record")
        session.rollback()

    try:
        if patient_profile_id is not None:
            trial_match_run = _safe_service_call(
                create_trial_match_run,
                [
                    ((session, patient_profile_id), {}),
                    ((session,), {"patient_profile_id": patient_profile_id}),
                    ((session,), {"patient_id": patient_profile_id}),
                    ((session,), {"profile_id": patient_profile_id}),
                ],
            )
            trial_match_run_id = _extract_record_id(trial_match_run)
    except Exception:
        logger.exception("Failed to create trial match run record")
        session.rollback()

    return patient_profile_id, trial_match_run_id


def _persist_results(
    session: Session,
    trial_match_run_id: Any,
    final_state: Dict[str, Any],
    serialized_results: List[Dict[str, Any]],
) -> None:
    """
    Saves trial match results and updates run status.
    Three status paths: success, failed, partial.
    """
    try:
        _safe_service_call(
            save_trial_match_results,
            [
                ((session, trial_match_run_id, serialized_results), {}),
                ((session,), {
                    "match_run_id": trial_match_run_id,
                    "results": serialized_results,
                }),
                ((session,), {
                    "run_id": trial_match_run_id,
                    "results": serialized_results,
                }),
            ],
        )
    except Exception:
        logger.exception("Failed to save trial match results")
        session.rollback()

    try:
        final_recs = final_state.get("final_recommendations", "")

        if final_state.get("error"):
            _safe_service_call(
                update_trial_match_run_failed,
                [
                    ((session, trial_match_run_id, final_state["error"]), {}),
                    ((session,), {
                        "match_run_id": trial_match_run_id,
                        "error": final_state["error"],
                    }),
                ],
            )
        elif serialized_results:
            _safe_service_call(
                update_trial_match_run_success,
                [
                    ((session, trial_match_run_id, final_recs), {}),
                    ((session,), {
                        "match_run_id": trial_match_run_id,
                        "final_recommendations": final_recs,
                    }),
                ],
            )
        else:
            _safe_service_call(
                update_trial_match_run_partial,
                [
                    ((session, trial_match_run_id, final_recs), {}),
                    ((session,), {
                        "match_run_id": trial_match_run_id,
                        "final_recommendations": final_recs,
                    }),
                ],
            )
    except Exception:
        logger.exception("Failed to update trial match run status")
        session.rollback()


# -----------------------------------------------------------
# PUBLIC ENTRYPOINT
# -----------------------------------------------------------

async def run_trial_matching(
    patient_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Main entrypoint for the clinical trial matching pipeline.

    Phases:
        1. Normalize + validate patient payload
        2. DB setup — create patient + run records (non-blocking)
        3. Run LangGraph workflow
        4. Persist results (non-blocking)
        5. Return rich result dict
    """
    patient_profile_id: Optional[Any] = None
    trial_match_run_id: Optional[Any] = None

    # --- Phase 1: Validate only — do NOT normalize again ---
    try:
        if isinstance(patient_payload, PatientProfile):
            patient_data = patient_payload
        else:
            patient_data = PatientProfile.model_validate(patient_payload)
    except Exception as exc:
        logger.exception("Patient payload validation failed")
        return {
            "success": False,
            "error": f"Invalid patient payload: {str(exc)}",
            "patient_profile_id": None,
            "trial_match_run_id": None,
            "cancer_type": "",
            "trial_count": 0,
            "trials": [],
            "eligibility_results": [],
            "final_recommendations": "",
        }


    # --- Phase 2: DB Setup ---
    try:
        with Session(engine) as session:
            patient_profile_id, trial_match_run_id = _setup_db_records(
                session, patient_data
            )
            try:
                session.commit()
            except Exception:
                logger.exception("Commit failed while creating DB records")
                session.rollback()
    except Exception:
        logger.exception("Database setup failed before workflow execution")

    # --- Phase 3: Run Workflow ---
    initial_state: AgentState = {
        "patient_data": patient_data,
        "cancer_type": None,
        "trials_raw": [],
        "eligibility_results": [],
        "final_recommendations": "",
        "error": None,
        "patient_profile_id": patient_profile_id,
        "trial_match_run_id": trial_match_run_id,
    }

    start_time = default_callback.on_pipeline_start(payload={})

    try:
        final_state = await get_trial_matching_app().ainvoke(initial_state)
        default_callback.on_pipeline_end(start_time, success=True)
    except Exception:
        logger.exception("Trial matching workflow execution failed")
        default_callback.on_pipeline_end(start_time, success=False)
        final_state = {
            **initial_state,
            "error": "Workflow execution failed unexpectedly.",
            "final_recommendations": (
                "Unable to complete clinical trial matching at this time."
            ),
        }

    # --- Serialize ---
    serialized_trials = serialize_trial_summaries(
        final_state.get("trials_raw") or []
    )
    serialized_results = serialize_trial_results(
        final_state.get("eligibility_results") or []
    )

    # --- Phase 4: Persist Results ---
    try:
        if trial_match_run_id is not None:
            with Session(engine) as session:
                _persist_results(
                    session,
                    trial_match_run_id,
                    final_state,
                    serialized_results,
                )
                try:
                    session.commit()
                except Exception:
                    logger.exception("Commit failed while saving workflow results")
                    session.rollback()
    except Exception:
        logger.exception("Database persistence failed after workflow execution")

    # --- Phase 5: Return ---
    return {
        "success": final_state.get("error") is None,
        "error": final_state.get("error"),
        "patient_profile_id": patient_profile_id,
        "trial_match_run_id": trial_match_run_id,
        "cancer_type": final_state.get("cancer_type", ""),
        "trial_count": len(serialized_trials),
        "trials": serialized_trials,
        "eligibility_results": serialized_results,
        "final_recommendations": final_state.get("final_recommendations", ""),
    }


# --- Aliases for backward compatibility ---
process_patient_trial_matching = run_trial_matching
run_trial_matcher_agent = run_trial_matching
