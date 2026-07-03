# core/state.py
"""
Typed state models for the LangGraph agent workflow.
Responsibility: Define shared state contracts — nothing else.
"""

from typing import TypedDict, List, Optional
from schemas import PatientProfile, TrialProfile


class EligibilityResult(TypedDict, total=False):
    nct_id: str
    title: str
    hard_filter_pass: bool
    hard_filter_reasons: List[str]
    score: int
    score_reasons: List[str]
    biomarker_check: str
    treatment_check: str
    assessment: str


class AgentState(TypedDict):
    patient_data: PatientProfile
    cancer_type: str
    trials_raw: List[TrialProfile]
    eligibility_results: List[EligibilityResult]
    final_recommendations: str
    error: Optional[str]
