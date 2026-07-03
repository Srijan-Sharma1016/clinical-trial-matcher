import os
import re
import json
import asyncio
import requests
import logging
from typing import TypedDict, List, Optional, Dict, Any, Tuple, Union

from dotenv import load_dotenv
from sqlmodel import Session
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from schemas import PatientProfile, TrialProfile
from normalizer import normalize_patient_payload, normalize_trial_study
from database import engine
from db_service import (
    create_patient_profile_record,
    create_trial_match_run,
    save_trial_match_results,
    update_trial_match_run_success,
    update_trial_match_run_failed,
    update_trial_match_run_partial,
)

# -----------------------------------------------------------
# ENVIRONMENT CONFIGURATION
# -----------------------------------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("System Error: GROQ_API_KEY is missing from environment.")

logger = logging.getLogger("uvicorn.error")

# -----------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------

ALLOWED_ACTIVE_STATUSES = {
    "RECRUITING",
    "NOT_YET_RECRUITING",
    "ENROLLING_BY_INVITATION",
}

SEARCH_TERM_MAP = {
    "Carcinoma, Non-Small-Cell Lung": "non small cell lung cancer",
    "Carcinoma, Small Cell": "small cell lung cancer",
    "Carcinoma, Hepatocellular": "hepatocellular carcinoma",
    "Leukemia, Myeloid, Acute": "acute myeloid leukemia",
    "Leukemia, Myelogenous, Chronic, BCR-ABL Positive": "chronic myeloid leukemia",
    "Precursor Cell Lymphoblastic Leukemia-Lymphoma": "acute lymphoblastic leukemia",
    "Triple Negative Breast Neoplasms": "triple negative breast cancer",
    "Carcinoma, Renal Cell": "renal cell carcinoma",
    "Prostatic Neoplasms": "prostate cancer",
    "Colorectal Neoplasms": "colorectal cancer",
    "Colonic Neoplasms": "colon cancer",
    "Rectal Neoplasms": "rectal cancer",
    "Breast Neoplasms": "breast cancer",
    "Lung Neoplasms": "lung cancer",
    "Ovarian Neoplasms": "ovarian cancer",
    "Stomach Neoplasms": "gastric cancer",
    "Pancreatic Neoplasms": "pancreatic cancer",
    "Brain Neoplasms": "brain cancer",
    "Skin Neoplasms": "skin cancer",
    "Thyroid Neoplasms": "thyroid cancer",
    "Urinary Bladder Neoplasms": "bladder cancer",
    "Kidney Neoplasms": "kidney cancer",
    "Head and Neck Neoplasms": "head and neck cancer",
    "Uterine Cervical Neoplasms": "cervical cancer",
    "Endometrial Neoplasms": "endometrial cancer",
    "Uterine Neoplasms": "uterine cancer",
    "Hodgkin Disease": "hodgkin lymphoma",
    "Lymphoma, Non-Hodgkin": "non hodgkin lymphoma",
    "Multiple Myeloma": "multiple myeloma",
    "Glioblastoma": "glioblastoma",
    "Glioma": "glioma",
    "Melanoma": "melanoma",
    "Sarcoma": "sarcoma",
    "Mesothelioma": "mesothelioma",
    "Neuroblastoma": "neuroblastoma",
    "Carcinoma, Basal Cell": "basal cell carcinoma",
    "Carcinoma, Squamous Cell": "squamous cell carcinoma",
    "Esophageal Neoplasms": "esophageal cancer",
    "Cholangiocarcinoma": "cholangiocarcinoma",
    "Liver Neoplasms": "liver cancer",
}

CLINICALTRIALS_BASE_URL = os.getenv(
    "CLINICALTRIALS_BASE_URL",
    "https://clinicaltrials.gov/api/v2"
)

CLINICALTRIALS_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "clinical-trial-service/1.0",
}

# Demo-safe limits
MAX_FETCH_RESULTS = 6
MAX_LLM_EVALUATION_TRIALS = 2

# Compact payload limits for LLM
MAX_LLM_SUMMARY_CHARS = 700
MAX_LLM_DESCRIPTION_CHARS = 700
MAX_LLM_CRITERIA_CHARS = 1200
MAX_LLM_STUDY_POPULATION_CHARS = 300
MAX_LLM_LOCATIONS = 2
MAX_LLM_MESH_TERMS = 6
MAX_SCORE_REASONS = 5

# -----------------------------------------------------------
# TYPED STATE MODELS
# -----------------------------------------------------------

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

# -----------------------------------------------------------
# BASIC HELPERS
# -----------------------------------------------------------

def get_api_search_term(cancer_type: str) -> str:
    clean = re.sub(r"\(.*?\)", "", cancer_type or "").strip()

    if clean in SEARCH_TERM_MAP:
        return SEARCH_TERM_MAP[clean]

    for mesh_term, simple_term in SEARCH_TERM_MAP.items():
        if mesh_term.lower() in clean.lower():
            return simple_term

    return clean.lower()


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

    match = re.match(r"(\d+)\s+(year|years|month|months|week|weeks|day|days)", raw)
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


def build_patient_summary(patient_data: PatientProfile) -> str:
    data = patient_data.model_dump()
    return "\n".join(
        f"- {k.replace('_', ' ').title()}: {v}"
        for k, v in data.items()
        if v not in (None, "", [], {})
    )


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


def resolve_cancer_type_from_structured_data(patient_data: PatientProfile) -> str:
    for value in [patient_data.cancer_type, patient_data.diagnosis]:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


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


def limit_score_reasons(reasons: List[str], limit: int = MAX_SCORE_REASONS) -> List[str]:
    cleaned = [r for r in reasons if r]
    return cleaned[:limit]

# -----------------------------------------------------------
# HARD FILTERS
# -----------------------------------------------------------

def hard_filter_trial(patient_data: PatientProfile, trial: TrialProfile) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    passes = True

    status = trial.status
    if status not in ALLOWED_ACTIVE_STATUSES:
        passes = False
        reasons.append(f"Trial status is {status}, not currently active for enrollment.")

    patient_sex = get_patient_sex(patient_data)
    trial_sex = trial.eligibility.sex
    if patient_sex and trial_sex and trial_sex != "ALL" and patient_sex != trial_sex:
        passes = False
        reasons.append(
            f"Patient sex '{patient_sex}' does not match trial eligibility '{trial_sex}'."
        )

    patient_age = get_patient_age(patient_data)
    if patient_age is not None:
        min_age = parse_age_to_years(trial.eligibility.minimum_age)
        max_age = parse_age_to_years(trial.eligibility.maximum_age)

        if min_age is not None and patient_age < min_age:
            passes = False
            reasons.append(f"Patient age {patient_age} is below minimum eligible age {min_age}.")
        if max_age is not None and patient_age > max_age:
            passes = False
            reasons.append(f"Patient age {patient_age} is above maximum eligible age {max_age}.")

    has_cancer = bool(patient_data.cancer_type or patient_data.diagnosis)
    if trial.eligibility.healthy_volunteers is True and has_cancer:
        passes = False
        reasons.append("Trial is intended for healthy volunteers only.")

    return passes, reasons

# -----------------------------------------------------------
# ACCURACY / DETERMINISTIC RANKING
# -----------------------------------------------------------

def infer_patient_disease_setting(patient_data: PatientProfile) -> str:
    text = normalize_free_text(
        safe_join_text([
            patient_data.cancer_stage,
            patient_data.cancer_type,
            patient_data.diagnosis,
        ])
    )

    if not text:
        return "unknown"

    if any(k in text for k in ["stage iv", "stage 4", "metastatic", "advanced", "unresectable"]):
        return "advanced"

    if any(k in text for k in ["stage iii", "stage 3", "locally advanced"]):
        return "locally_advanced"

    if any(k in text for k in ["stage i", "stage 1", "stage ii", "stage 2", "early", "localized"]):
        return "early"

    return "unknown"


def infer_trial_disease_setting(trial: TrialProfile) -> str:
    priority_text = normalize_free_text(
        safe_join_text([
            trial.title,
            trial.official_title,
            " ".join(trial.conditions),
            trial.brief_summary,
        ])
    )

    secondary_text = normalize_free_text(
        safe_join_text([
            trial.detailed_description,
            trial.eligibility.study_population,
            trial.eligibility.criteria_text,
        ])
    )

    early_markers = [
        "early stage",
        "early-stage",
        "localized",
        "resectable",
        "adjuvant",
        "neoadjuvant",
        "stage i",
        "stage 1",
        "stage ii",
        "stage 2",
        "locally confined",
    ]

    locally_advanced_markers = [
        "locally advanced",
        "stage iii",
        "stage 3",
    ]

    advanced_markers = [
        "metastatic",
        "advanced breast cancer",
        "advanced solid tumors",
        "unresectable",
        "recurrent",
        "relapsed",
        "stage iv",
        "stage 4",
    ]

    # Priority section first
    if any(k in priority_text for k in early_markers):
        return "early"

    if any(k in priority_text for k in locally_advanced_markers):
        return "locally_advanced"

    if any(k in priority_text for k in advanced_markers):
        return "advanced"

    # Secondary section only if priority text was unclear
    if any(k in secondary_text for k in early_markers):
        return "early"

    if any(k in secondary_text for k in locally_advanced_markers):
        return "locally_advanced"

    if any(k in secondary_text for k in advanced_markers):
        return "advanced"

    return "unknown"


def score_cancer_match(patient_data: PatientProfile, trial: TrialProfile) -> Tuple[int, List[str]]:
    reasons: List[str] = []

    patient_cancer = patient_data.cancer_type or patient_data.diagnosis
    if not patient_cancer:
        reasons.append("Cancer type is missing from patient profile.")
        return 0, reasons

    simple_term = get_api_search_term(patient_cancer)
    trial_text = get_trial_searchable_text(trial)
    simple_term_normalized = normalize_free_text(simple_term)

    if simple_term_normalized and simple_term_normalized in trial_text:
        reasons.append(f"Strong cancer type match for '{simple_term}'.")
        return 5, reasons

    patient_tokens = [t for t in simple_term_normalized.split() if len(t) > 3]
    overlap = [t for t in patient_tokens if t in trial_text]

    if overlap:
        reasons.append(f"Partial cancer-type overlap found: {', '.join(overlap)}.")
        return 2, reasons

    reasons.append("Cancer type match appears weak or unclear.")
    return -2, reasons


def score_stage_match(patient_data: PatientProfile, trial: TrialProfile) -> Tuple[int, List[str]]:
    reasons: List[str] = []

    patient_setting = infer_patient_disease_setting(patient_data)
    trial_setting = infer_trial_disease_setting(trial)

    if patient_setting == "unknown" or trial_setting == "unknown":
        reasons.append("Disease setting is unclear for patient or trial.")
        return 0, reasons

    if patient_setting == trial_setting:
        reasons.append(f"Disease setting aligns: '{patient_setting}'.")
        return 4, reasons

    if patient_setting == "early" and trial_setting == "advanced":
        reasons.append("Patient appears early-stage while trial targets advanced/metastatic disease.")
        return -6, reasons

    if patient_setting == "advanced" and trial_setting == "early":
        reasons.append("Patient appears advanced/metastatic while trial targets early-stage disease.")
        return -6, reasons

    if patient_setting == "locally_advanced" and trial_setting == "advanced":
        reasons.append("Patient may be locally advanced while trial appears metastatic-focused.")
        return -2, reasons

    if patient_setting == "advanced" and trial_setting == "locally_advanced":
        reasons.append("Patient may be metastatic while trial appears locally-advanced focused.")
        return -2, reasons

    reasons.append(f"Disease setting mismatch: patient='{patient_setting}', trial='{trial_setting}'.")
    return -1, reasons


def score_biomarker_match(patient_data: PatientProfile, trial: TrialProfile) -> Tuple[int, List[str]]:
    reasons: List[str] = []
    biomarkers = patient_data.biomarkers or []

    if not biomarkers:
        reasons.append("No patient biomarkers available for deterministic scoring.")
        return 0, reasons

    trial_text = get_trial_searchable_text(trial)
    total = 0

    for biomarker in biomarkers:
        marker = normalize_free_text(biomarker)

        if marker in {"her2", "her2+", "her2 positive"}:
            if "her2-positive" in trial_text or "her2 positive" in trial_text:
                total += 4
                reasons.append("HER2-positive biomarker strongly aligns.")
            elif "her2-low" in trial_text or "her2 low" in trial_text:
                total += 1
                reasons.append("HER2-low found; only partial HER2 alignment.")
            elif "her2" in trial_text:
                total += 2
                reasons.append("Generic HER2 mention found.")
            else:
                total -= 1
                reasons.append("HER2 not clearly supported in trial text.")
            continue

        if marker in {"her2-low", "her2 low"}:
            if "her2-low" in trial_text or "her2 low" in trial_text:
                total += 4
                reasons.append("HER2-low biomarker strongly aligns.")
            elif "her2-positive" in trial_text or "her2 positive" in trial_text:
                total -= 2
                reasons.append("Trial targets HER2-positive, not HER2-low.")
            elif "her2" in trial_text:
                total += 1
                reasons.append("Generic HER2 mention found; specificity unclear.")
            else:
                total -= 1
                reasons.append("HER2-low not clearly matched.")
            continue

        if marker in {"triple negative", "tnbc"}:
            if "triple negative" in trial_text or "tnbc" in trial_text:
                total += 4
                reasons.append("Triple-negative breast cancer alignment found.")
            else:
                total -= 2
                reasons.append("TNBC not clearly supported in trial text.")
            continue

        if marker in {"pd-l1", "pdl1"}:
            if "pd-l1" in trial_text or "pd l1" in trial_text or "pdl1" in trial_text:
                total += 2
                reasons.append("PD-L1 mentioned in trial text.")
            else:
                total -= 1
                reasons.append("PD-L1 not clearly matched.")
            continue

        if marker in trial_text:
            total += 2
            reasons.append(f"Biomarker '{biomarker}' found in trial text.")
        else:
            reasons.append(f"Biomarker '{biomarker}' not explicitly found.")

    total = max(min(total, 5), -4)
    if total == 0 and reasons:
        return 0, reasons
    return total, reasons


def score_treatment_history_match(patient_data: PatientProfile, trial: TrialProfile) -> Tuple[int, List[str]]:
    reasons: List[str] = []
    treatments = patient_data.previous_treatments or []

    if not treatments:
        reasons.append("No previous treatment history available for scoring.")
        return 0, reasons

    trial_text = get_trial_searchable_text(trial)

    untreated_markers = [
        "previously untreated",
        "untreated",
        "treatment naive",
        "treatment-naive",
        "no prior treatment",
        "no previous treatment",
    ]

    prior_treatment_positive_markers = [
        "previously treated",
        "received prior",
        "after prior",
        "progressed on",
        "1 prior line",
        "one prior line",
        "≤1 prior line",
        "<=1 prior line",
        "prior chemotherapy",
    ]

    exclusion_markers = [
        "must not have received",
        "excluded",
        "exclusion",
        "prior treatment with",
        "previous treatment with",
    ]

    if any(marker in trial_text for marker in untreated_markers):
        reasons.append("Trial appears to prefer untreated / treatment-naive patients.")
        return -5, reasons

    score = 0

    if any(marker in trial_text for marker in prior_treatment_positive_markers):
        score += 2
        reasons.append("Trial text appears compatible with prior treatment exposure.")

    normalized_treatments = [normalize_free_text(t) for t in treatments]

    for treatment in normalized_treatments:
        if treatment in trial_text and any(marker in trial_text for marker in exclusion_markers):
            score -= 3
            reasons.append(f"Prior treatment '{treatment}' may conflict with exclusion criteria.")

    if score == 0:
        reasons.append("Prior treatment compatibility remains unclear.")
    return score, reasons


def score_location_match(patient_data: PatientProfile, trial: TrialProfile) -> Tuple[int, List[str]]:
    reasons: List[str] = []

    patient_country = get_patient_country(patient_data)
    if not patient_country:
        reasons.append("Patient country is not available.")
        return 0, reasons

    if not trial.locations:
        reasons.append("Trial location data is unavailable.")
        return 0, reasons

    trial_countries = {
        (loc.country or "").strip().lower()
        for loc in trial.locations
        if loc.country
    }

    if not trial_countries:
        reasons.append("Trial countries are unavailable.")
        return 0, reasons

    if patient_country in trial_countries:
        reasons.append("Trial includes the patient's country.")
        return 2, reasons

    reasons.append(f"Trial does not list patient's country '{patient_country}'.")
    return -2, reasons


def score_study_type(trial: TrialProfile) -> Tuple[int, List[str]]:
    reasons: List[str] = []

    if trial.study_type == "INTERVENTIONAL":
        reasons.append("Interventional trial preferred for treatment relevance.")
        return 2, reasons

    if trial.study_type == "OBSERVATIONAL":
        reasons.append("Observational study is less suitable for treatment matching.")
        return -3, reasons

    reasons.append("Study type has neutral effect on ranking.")
    return 0, reasons


def score_trial_match(patient_data: PatientProfile, trial: TrialProfile) -> Tuple[int, List[str]]:
    total_score = 0
    all_reasons: List[str] = []

    scoring_functions = [
        score_cancer_match,
        score_stage_match,
        score_biomarker_match,
        score_treatment_history_match,
        score_location_match,
    ]

    for fn in scoring_functions:
        score, reasons = fn(patient_data, trial)
        total_score += score
        all_reasons.extend(reasons)

    study_type_score, study_type_reasons = score_study_type(trial)
    total_score += study_type_score
    all_reasons.extend(study_type_reasons)

    return total_score, limit_score_reasons(all_reasons)

# -----------------------------------------------------------
# COMPACT TRIAL CONTEXT FOR LLM
# -----------------------------------------------------------

def build_trial_llm_context(trial: TrialProfile) -> Dict[str, Any]:
    compact_locations = []
    for loc in trial.locations[:MAX_LLM_LOCATIONS]:
        compact_locations.append({
            "facility": loc.facility,
            "city": loc.city,
            "country": loc.country,
        })

    return {
        "trial_id": trial.trial_id,
        "title": trial.title,
        "official_title": trial.official_title,
        "status": trial.status,
        "study_type": trial.study_type,
        "phases": trial.phases[:5],
        "conditions": trial.conditions[:8],
        "brief_summary": truncate_text(trial.brief_summary, MAX_LLM_SUMMARY_CHARS),
        "detailed_description": truncate_text(trial.detailed_description, MAX_LLM_DESCRIPTION_CHARS),
        "eligibility": {
            "criteria_text": truncate_text(trial.eligibility.criteria_text, MAX_LLM_CRITERIA_CHARS),
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
# HEURISTIC SUPPORT TEXT FOR LLM
# -----------------------------------------------------------

def heuristic_biomarker_check(biomarkers: List[str], trial_criteria: str) -> str:
    if not biomarkers:
        return "No patient biomarkers provided."

    criteria_lower = (trial_criteria or "").lower()
    matches = []
    not_found = []

    for biomarker in biomarkers:
        biomarker_clean = biomarker.strip()
        if not biomarker_clean:
            continue
        if biomarker_clean.lower() in criteria_lower:
            matches.append(biomarker_clean)
        else:
            not_found.append(biomarker_clean)

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


def heuristic_treatment_history_check(previous_treatments: List[str], trial_criteria: str) -> str:
    if not previous_treatments:
        return "No previous treatment history provided."

    criteria_lower = (trial_criteria or "").lower()
    compatible = []
    potential_conflicts = []

    exclusion_markers = [
        "no prior",
        "no previous",
        "must not have received",
        "excluded",
        "exclusion",
        "prior treatment with",
        "previous treatment with",
    ]

    for treatment in previous_treatments:
        treatment_clean = treatment.strip()
        if not treatment_clean:
            continue

        treatment_lower = treatment_clean.lower()
        found_treatment = treatment_lower in criteria_lower
        found_exclusion_context = any(marker in criteria_lower for marker in exclusion_markers)

        if found_treatment and found_exclusion_context:
            potential_conflicts.append(treatment_clean)
        else:
            compatible.append(treatment_clean)

    result = []
    if compatible:
        result.append(f"Potentially compatible prior treatments: {', '.join(compatible)}")
    if potential_conflicts:
        result.append(f"Potential treatment-history conflicts: {', '.join(potential_conflicts)}")

    compatibility = "LOW" if potential_conflicts else "MEDIUM"
    if compatible and not potential_conflicts:
        compatibility = "HIGH"

    result.append(f"Treatment History Compatibility Signal: {compatibility}")
    return "\n".join(result)

# -----------------------------------------------------------
# LLM ENGINE + PROMPT CHAINS
# -----------------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0,
    max_retries=3,
)

cancer_type_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a clinical data parser. Extract ONLY the main cancer type "
        "from the patient profile. Return just the cancer type string and nothing else."
    ),
    (
        "user",
        "Patient Profile:\n{patient_data}\n\nExtract the main cancer type:"
    )
])
cancer_type_chain = cancer_type_prompt | llm | StrOutputParser()

eligibility_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert oncology clinical trial eligibility assessor.\n"
        "Use only the provided patient profile and compact trial data.\n"
        "Do not invent facts.\n"
        "If information is missing, say it is missing.\n"
        "Use the deterministic score and short score reasons as supporting signals.\n"
        "Do not overstate certainty.\n\n"
        "Return the answer in exactly this format:\n"
        "MATCH STATUS: <strong>\n"
        "REASONS FOR MATCH:\n"
        "- ...\n"
        "REASONS AGAINST MATCH:\n"
        "- ...\n"
        "MISSING INFORMATION:\n"
        "- ...\n"
        "PATIENT BENEFITS:\n"
        "- ...\n"
        "CONCERNS:\n"
        "- ...\n"
    ),
    (
        "user",
        "Patient Profile:\n{patient_profile}\n\n"
        "Compact Trial Details JSON:\n{trial_details}\n\n"
        "Deterministic Match Score: {trial_score}\n\n"
        "Deterministic Ranking Reasons:\n{score_reasons}\n\n"
        "Biomarker Check:\n{biomarker_check}\n\n"
        "Treatment History Check:\n{treatment_check}\n"
    ),
])
eligibility_chain = eligibility_prompt | llm | StrOutputParser()

def _is_reasonably_strong_trial(result: EligibilityResult) -> bool:
    if not result.get("hard_filter_pass"):
        return False

    assessment = (result.get("assessment") or "").upper()
    if "MATCH STATUS: NO MATCH" in assessment:
        return False

    score = result.get("score", 0)
    if score is None or score < 1:
        return False

    negative_phrases = [
        "early-stage while trial targets advanced",
        "patient appears early-stage while trial targets advanced",
        "advanced/metastatic while trial targets early-stage",
        "patient appears advanced/metastatic while trial targets early-stage",
        "treatment-naive patients",
        "treatment naive patients",
        "may conflict with exclusion criteria",
        "not clearly matched",
        "not clearly supported",
        "targets her2-positive, not her2-low",
        "targets her2-low, not her2-positive",
        "tnbc not clearly supported",
    ]

    assessment_lower = (result.get("assessment") or "").lower()
    score_reasons_lower = " ".join(result.get("score_reasons") or []).lower()

    if any(p in assessment_lower for p in negative_phrases):
        return False
    if any(p in score_reasons_lower for p in negative_phrases):
        return False

    return True
recommendation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a careful oncology trial summarizer.\n"
        "Use the trial assessments exactly as provided.\n"
        "Higher deterministic scores indicate stronger rule-based alignment.\n"
        "Do not claim certainty where data is incomplete.\n"
        "Do not recommend a trial as a good fit if it was clearly marked as NO MATCH.\n"
        "Do not present weak, conflicting, or negative-score trials as strong matches.\n"
        "If all available trials have major conflicts, say clearly that no strong matches were found.\n"
        "A trial should only be presented as a top match if:\n"
        "- hard_filter_pass is true\n"
        "- it is not described as NO MATCH\n"
        "- its deterministic score is positive\n"
        "- and its assessment does not contain major contradictions with patient stage, biomarker status, or treatment history.\n"
        "If the best available trials are still weak or borderline, explicitly label them as tentative or clinician-review-only options.\n"
        "Use plain, patient-friendly language.\n"
        "Never overrule the structured eligibility assessment."
    ),
    (
        "user",
        "Patient Profile:\n{patient_profile}\n\n"
        "Trial Eligibility Assessments:\n{eligibility_assessments}\n\n"
        "Write a final recommendation report that:\n"
        "1. Identifies up to 2 strongest trials only if they are reasonably aligned\n"
        "2. Clearly says 'No strong matches found' if the available trials are weak, conflicting, or borderline\n"
        "3. Explains in plain language why each trial is or is not a fit\n"
        "4. Includes important cautions or next steps\n"
        "5. Keeps the tone clear, calm, and supportive\n"
    ),
])
recommendation_chain = recommendation_prompt | llm | StrOutputParser()


async def recommend_trials_node(state: AgentState) -> AgentState:
    logger.info("Node 4: Generating final recommendations...")

    if state.get("error"):
        return {
            **state,
            "final_recommendations": f"Trial matching could not be completed: {state['error']}",
        }

    try:
        patient_data = state["patient_data"]
        eligibility_results = state.get("eligibility_results") or []

        if not eligibility_results:
            no_match_text = (
                "No recruiting clinical trials were found or no trials could be evaluated "
                "for the current patient profile. Consider broadening the search terms, "
                "locations, or reviewing missing patient details."
            )
            return {
                **state,
                "final_recommendations": no_match_text,
            }

        strong_results = sorted(
            [r for r in eligibility_results if _is_reasonably_strong_trial(r)],
            key=lambda r: r.get("score", 0),
            reverse=True,
        )

        if not strong_results:
            final_recommendations = _fallback_recommendation_text(
                patient_data,
                eligibility_results,
            )
            return {
                **state,
                "final_recommendations": final_recommendations,
            }

        results_for_recommendation = strong_results[:2]

        assessment_text = "\n\n" + ("\n\n".join(
            _format_eligibility_assessment_for_prompt(result)
            for result in results_for_recommendation
        ))

        try:
            recommendation = await recommendation_chain.ainvoke({
                "patient_profile": build_patient_summary(patient_data),
                "eligibility_assessments": assessment_text.strip(),
            })
        except Exception:
            logger.exception("LLM recommendation generation failed")
            recommendation = _fallback_recommendation_text(
                patient_data,
                results_for_recommendation,
            )

        return {
            **state,
            "final_recommendations": recommendation.strip(),
        }

    except Exception:
        logger.exception("Failed in recommend_trials_node")
        return {
            **state,
            "error": "Unexpected error while generating final recommendations.",
            "final_recommendations": "Unable to generate a recommendation report.",
        }
# -----------------------------------------------------------
# API FUNCTIONS
# -----------------------------------------------------------

def _clinicaltrials_get_json(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{CLINICALTRIALS_BASE_URL.rstrip('/')}/{path.lstrip('/')}"

    logger.info("ClinicalTrials.gov request | url=%s | params=%s", url, params)

    response = requests.get(
        url,
        params=params,
        headers=CLINICALTRIALS_HEADERS,
        timeout=30,
    )

    logger.info(
        "ClinicalTrials.gov response | status=%s | final_url=%s",
        response.status_code,
        response.url,
    )

    response.raise_for_status()
    return response.json()


async def search_clinical_trials(cancer_type: str, max_results: int = MAX_FETCH_RESULTS) -> List[TrialProfile]:
    max_results = min(max_results, 10)
    api_search_term = get_api_search_term(cancer_type)

    logger.info(
        "Searching ClinicalTrials.gov — original='%s' api_term='%s'",
        cancer_type,
        api_search_term,
    )

    params = {
        "query.cond": api_search_term,
        "filter.overallStatus": "RECRUITING",
        "pageSize": max_results,
    }

    try:
        data = await asyncio.to_thread(
            _clinicaltrials_get_json,
            "/studies",
            params,
        )

        studies = data.get("studies", [])
        logger.info("ClinicalTrials.gov search returned %d studies.", len(studies))

        return [normalize_trial_study(study) for study in studies]

    except requests.Timeout:
        logger.warning("ClinicalTrials.gov search timed out.")
        return []
    except requests.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "unknown"
        body_preview = e.response.text[:1000] if e.response is not None else ""
        logger.error(
            "ClinicalTrials.gov search failed | status=%s | body=%s",
            status_code,
            body_preview,
        )
        return []
    except requests.RequestException as e:
        logger.error("ClinicalTrials.gov search request error: %s", str(e))
        return []
    except Exception:
        logger.exception("Unexpected error during ClinicalTrials.gov search")
        return []


async def get_trial_details(nct_id: str) -> Union[TrialProfile, Dict[str, Any]]:
    logger.info("Fetching trial details for '%s'", nct_id)

    try:
        data = await asyncio.to_thread(
            _clinicaltrials_get_json,
            f"/studies/{nct_id}",
            None,
        )
        return normalize_trial_study(data)

    except requests.Timeout:
        return {"error": f"Timeout fetching details for trial {nct_id}."}
    except requests.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "unknown"
        body_preview = e.response.text[:500] if e.response is not None else ""
        return {"error": f"API error fetching trial {nct_id}: {status_code} body={body_preview}"}
    except requests.RequestException as e:
        return {"error": f"Request error fetching trial {nct_id}: {str(e)}"}
    except Exception as e:
        logger.exception("Unexpected error in get_trial_details")
        return {"error": f"Unexpected error: {str(e)}"}

# -----------------------------------------------------------
# LANGGRAPH NODE DEFINITIONS
# -----------------------------------------------------------

async def extract_cancer_type_node(state: AgentState) -> AgentState:
    logger.info("Node 1: Extracting cancer type...")

    try:
        patient_data = state["patient_data"]

        structured_value = resolve_cancer_type_from_structured_data(patient_data)
        if structured_value:
            logger.info("Cancer type resolved from structured patient data: '%s'", structured_value)
            updated_patient = patient_data.model_copy(update={"cancer_type": structured_value})
            return {
                **state,
                "patient_data": updated_patient,
                "cancer_type": structured_value,
            }

        patient_summary = build_patient_summary(patient_data)
        cancer_type = await cancer_type_chain.ainvoke({
            "patient_data": patient_summary
        })

        cancer_type = cancer_type.strip()
        if not cancer_type:
            return {**state, "error": "Unable to determine cancer type from patient profile."}

        updated_patient = patient_data.model_copy(update={"cancer_type": cancer_type})

        logger.info("Cancer type extracted by LLM: '%s'", cancer_type)
        return {
            **state,
                        "patient_data": updated_patient,
            "cancer_type": cancer_type,
        }

    except Exception:
        logger.exception("Failed in extract_cancer_type_node")
        return {
            **state,
            "error": "Unexpected error while extracting cancer type.",
        }


def _trial_key(trial: TrialProfile) -> str:
    return trial.trial_id or normalize_free_text(trial.title) or f"trial_{id(trial)}"


def _build_hard_filter_no_match_assessment(reasons: List[str]) -> str:
    against = reasons or ["The trial failed one or more hard eligibility filters."]
    against_text = "\n".join(f"- {r}" for r in against)

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
    )


def _build_not_assessed_assessment(score: int, reasons: List[str]) -> str:
    reasons_text = "\n".join(f"- {r}" for r in (reasons or ["Ranked lower than the top trials selected for LLM review."]))

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
    )


async def _evaluate_trial_with_llm(
    patient_data: PatientProfile,
    trial: TrialProfile,
    hard_filter_pass: bool,
    hard_filter_reasons: List[str],
    score: int,
    score_reasons: List[str],
) -> EligibilityResult:
    criteria_text = trial.eligibility.criteria_text or ""

    biomarker_check = heuristic_biomarker_check(
        patient_data.biomarkers or [],
        criteria_text,
    )
    treatment_check = heuristic_treatment_history_check(
        patient_data.previous_treatments or [],
        criteria_text,
    )

    try:
        assessment = await eligibility_chain.ainvoke({
            "patient_profile": build_patient_summary(patient_data),
            "trial_details": json.dumps(
                build_trial_llm_context(trial),
                indent=2,
                ensure_ascii=False,
            ),
            "trial_score": score,
            "score_reasons": "\n".join(f"- {r}" for r in score_reasons) or "- No deterministic reasons available.",
            "biomarker_check": biomarker_check,
            "treatment_check": treatment_check,
        })
    except Exception:
        logger.exception("LLM eligibility evaluation failed for trial '%s'", trial.trial_id)
        assessment = (
            "MATCH STATUS: NEEDS FURTHER REVIEW\n"
            "REASONS FOR MATCH:\n"
            f"- Deterministic score: {score}\n"
            "REASONS AGAINST MATCH:\n"
            + ("\n".join(f"- {r}" for r in score_reasons) or "- Insufficient scoring detail available.") + "\n"
            "MISSING INFORMATION:\n"
            "- LLM assessment failed, so clinician review is needed.\n"
            "PATIENT BENEFITS:\n"
            "- Trial may still be relevant if key criteria align.\n"
            "CONCERNS:\n"
            "- Automated narrative review could not be completed.\n"
        )

    return EligibilityResult(
        nct_id=trial.trial_id,
        title=trial.title,
        hard_filter_pass=hard_filter_pass,
        hard_filter_reasons=hard_filter_reasons,
        score=score,
        score_reasons=limit_score_reasons(score_reasons),
        biomarker_check=biomarker_check,
        treatment_check=treatment_check,
        assessment=assessment.strip(),
    )


async def search_trials_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    logger.info("Node 2: Searching trials...")

    try:
        cancer_type = (state.get("cancer_type") or "").strip()
        if not cancer_type:
            return {
                **state,
                "error": "Cancer type is missing. Cannot search for trials.",
            }

        trials = await search_clinical_trials(cancer_type, MAX_FETCH_RESULTS)

        if not trials:
            logger.info("No trials found for cancer type '%s'", cancer_type)
            return {
                **state,
                "trials_raw": [],
            }

        detail_tasks = [
            get_trial_details(trial.trial_id)
            for trial in trials
            if trial.trial_id
        ]

        enriched_trials: List[TrialProfile] = []

        if detail_tasks:
            detail_results = await asyncio.gather(*detail_tasks, return_exceptions=True)

            fallback_trials = [trial for trial in trials if trial.trial_id]
            for fallback_trial, detail_result in zip(fallback_trials, detail_results):
                if isinstance(detail_result, TrialProfile):
                    enriched_trials.append(detail_result)
                else:
                    if isinstance(detail_result, Exception):
                        logger.warning(
                            "Trial detail fetch raised exception for '%s': %s",
                            fallback_trial.trial_id,
                            str(detail_result),
                        )
                    elif isinstance(detail_result, dict) and detail_result.get("error"):
                        logger.warning(
                            "Trial detail fetch returned error for '%s': %s",
                            fallback_trial.trial_id,
                            detail_result["error"],
                        )
                    enriched_trials.append(fallback_trial)

            no_id_trials = [trial for trial in trials if not trial.trial_id]
            enriched_trials.extend(no_id_trials)
        else:
            enriched_trials = trials

        deduped: Dict[str, TrialProfile] = {}
        for trial in enriched_trials:
            deduped[_trial_key(trial)] = trial

        final_trials = list(deduped.values())
        logger.info("Prepared %d trial records for evaluation.", len(final_trials))

        return {
            **state,
            "trials_raw": final_trials,
        }

    except Exception:
        logger.exception("Failed in search_trials_node")
        return {
            **state,
            "error": "Unexpected error while searching clinical trials.",
            "trials_raw": [],
        }


async def evaluate_trials_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    logger.info("Node 3: Evaluating trials...")

    try:
        patient_data = state["patient_data"]
        trials = state.get("trials_raw") or []

        if not trials:
            return {
                **state,
                "eligibility_results": [],
            }

        ranked_trials: List[Dict[str, Any]] = []

        for trial in trials:
            hard_filter_pass, hard_filter_reasons = hard_filter_trial(patient_data, trial)
            score, score_reasons = score_trial_match(patient_data, trial)

            ranked_trials.append({
                "trial": trial,
                "hard_filter_pass": hard_filter_pass,
                "hard_filter_reasons": hard_filter_reasons,
                "score": score,
                "score_reasons": limit_score_reasons(score_reasons),
            })

        ranked_trials.sort(
            key=lambda item: (
                1 if item["hard_filter_pass"] else 0,
                item["score"],
            ),
            reverse=True,
        )

        llm_candidates = [
            item for item in ranked_trials
            if item["hard_filter_pass"]
        ][:MAX_LLM_EVALUATION_TRIALS]

        llm_results_by_key: Dict[str, EligibilityResult] = {}

        if llm_candidates:
            llm_tasks = [
                _evaluate_trial_with_llm(
                    patient_data=patient_data,
                    trial=item["trial"],
                    hard_filter_pass=item["hard_filter_pass"],
                    hard_filter_reasons=item["hard_filter_reasons"],
                    score=item["score"],
                    score_reasons=item["score_reasons"],
                )
                for item in llm_candidates
            ]

            llm_outputs = await asyncio.gather(*llm_tasks, return_exceptions=True)

            for item, llm_output in zip(llm_candidates, llm_outputs):
                trial = item["trial"]
                key = _trial_key(trial)

                if isinstance(llm_output, Exception):
                    logger.exception(
                        "Unhandled exception during LLM evaluation for trial '%s'",
                        trial.trial_id,
                    )
                    llm_results_by_key[key] = EligibilityResult(
                        nct_id=trial.trial_id,
                        title=trial.title,
                        hard_filter_pass=item["hard_filter_pass"],
                        hard_filter_reasons=item["hard_filter_reasons"],
                        score=item["score"],
                        score_reasons=item["score_reasons"],
                        biomarker_check=heuristic_biomarker_check(
                            patient_data.biomarkers or [],
                            trial.eligibility.criteria_text or "",
                        ),
                        treatment_check=heuristic_treatment_history_check(
                            patient_data.previous_treatments or [],
                            trial.eligibility.criteria_text or "",
                        ),
                        assessment=_build_not_assessed_assessment(
                            item["score"],
                            item["score_reasons"],
                        ),
                    )
                else:
                    llm_results_by_key[key] = llm_output

        final_results: List[EligibilityResult] = []

        for item in ranked_trials:
            trial = item["trial"]
            key = _trial_key(trial)

            if not item["hard_filter_pass"]:
                final_results.append(EligibilityResult(
                    nct_id=trial.trial_id,
                    title=trial.title,
                    hard_filter_pass=False,
                    hard_filter_reasons=item["hard_filter_reasons"],
                    score=item["score"],
                    score_reasons=item["score_reasons"],
                    biomarker_check=heuristic_biomarker_check(
                        patient_data.biomarkers or [],
                        trial.eligibility.criteria_text or "",
                    ),
                    treatment_check=heuristic_treatment_history_check(
                        patient_data.previous_treatments or [],
                        trial.eligibility.criteria_text or "",
                    ),
                    assessment=_build_hard_filter_no_match_assessment(
                        item["hard_filter_reasons"],
                    ),
                ))
                continue

            if key in llm_results_by_key:
                final_results.append(llm_results_by_key[key])
            else:
                final_results.append(EligibilityResult(
                    nct_id=trial.trial_id,
                    title=trial.title,
                    hard_filter_pass=item["hard_filter_pass"],
                    hard_filter_reasons=item["hard_filter_reasons"],
                    score=item["score"],
                    score_reasons=item["score_reasons"],
                    biomarker_check=heuristic_biomarker_check(
                        patient_data.biomarkers or [],
                        trial.eligibility.criteria_text or "",
                    ),
                    treatment_check=heuristic_treatment_history_check(
                        patient_data.previous_treatments or [],
                        trial.eligibility.criteria_text or "",
                    ),
                    assessment=_build_not_assessed_assessment(
                        item["score"],
                        item["score_reasons"],
                    ),
                ))

        logger.info("Generated %d eligibility results.", len(final_results))

        return {
            **state,
            "eligibility_results": final_results,
        }

    except Exception:
        logger.exception("Failed in evaluate_trials_node")
        return {
            **state,
            "error": "Unexpected error while evaluating trial eligibility.",
            "eligibility_results": [],
        }


def _format_eligibility_assessment_for_prompt(result: EligibilityResult) -> str:
    score_reasons = result.get("score_reasons") or []
    hard_filter_reasons = result.get("hard_filter_reasons") or []

    return (
        f"Trial ID: {result.get('nct_id')}\n"
        f"Title: {result.get('title')}\n"
        f"Hard Filter Pass: {result.get('hard_filter_pass')}\n"
        f"Hard Filter Reasons:\n"
        f"{chr(10).join(f'- {r}' for r in hard_filter_reasons) or '- None'}\n"
        f"Deterministic Score: {result.get('score')}\n"
        f"Deterministic Reasons:\n"
        f"{chr(10).join(f'- {r}' for r in score_reasons) or '- None'}\n"
        f"Biomarker Check:\n{result.get('biomarker_check') or 'Not available.'}\n"
        f"Treatment Check:\n{result.get('treatment_check') or 'Not available.'}\n"
        f"Assessment:\n{result.get('assessment') or 'Not available.'}"
    )


def _fallback_recommendation_text(
    patient_data: PatientProfile,
    eligibility_results: List[EligibilityResult],
) -> str:
    strong_trials = sorted(
        [r for r in eligibility_results if _is_reasonably_strong_trial(r)],
        key=lambda r: r.get("score", 0),
        reverse=True,
    )

    lines = [
        "Final Trial Matching Summary",
        "",
        f"Patient cancer type: {patient_data.cancer_type or patient_data.diagnosis or 'Not available'}",
        "",
    ]

    if not strong_trials:
        lines.extend([
            "No strong matches were found among the evaluated trials.",
            "",
            "Why:",
            "- The reviewed trials did not show strong enough alignment on the core rule-based checks.",
            "- Some options may still deserve clinician review if important patient details are incomplete or changing.",
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
            f"   - Key reasons: {', '.join((result.get('score_reasons') or [])[:3]) or 'No reasons available'}",
        ])

    lines.extend([
        "",
        "Important cautions:",
        "- These results are screening support only, not a final eligibility decision.",
        "- Final trial fit depends on full protocol review, clinician judgment, and site-specific screening.",
    ])

    return "\n".join(lines)


# -----------------------------------------------------------
# LANGGRAPH WORKFLOW
# -----------------------------------------------------------

trial_matching_graph = StateGraph(AgentState)

trial_matching_graph.add_node("extract_cancer_type", extract_cancer_type_node)
trial_matching_graph.add_node("search_trials", search_trials_node)
trial_matching_graph.add_node("evaluate_trials", evaluate_trials_node)
trial_matching_graph.add_node("recommend_trials", recommend_trials_node)

trial_matching_graph.set_entry_point("extract_cancer_type")
trial_matching_graph.add_edge("extract_cancer_type", "search_trials")
trial_matching_graph.add_edge("search_trials", "evaluate_trials")
trial_matching_graph.add_edge("evaluate_trials", "recommend_trials")
trial_matching_graph.add_edge("recommend_trials", END)

trial_matching_app = trial_matching_graph.compile()


# -----------------------------------------------------------
# DATABASE SAFETY HELPERS
# -----------------------------------------------------------

def _extract_record_id(record: Any) -> Optional[Any]:
    if record is None:
        return None

    if isinstance(record, dict):
        for key in ("id", "run_id", "match_run_id", "trial_match_run_id", "patient_profile_id", "patient_id"):
            if key in record and record[key] is not None:
                return record[key]

    for attr in ("id", "run_id", "match_run_id", "trial_match_run_id", "patient_profile_id", "patient_id"):
        if hasattr(record, attr):
            value = getattr(record, attr)
            if value is not None:
                return value

    return None


def _safe_service_call(
    fn: Any,
    attempts: List[Tuple[Tuple[Any, ...], Dict[str, Any]]],
) -> Any:
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


def _serialize_trial_results(results: List[EligibilityResult]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []

    for result in results:
        serialized.append({
            "nct_id": result.get("nct_id"),
            "title": result.get("title"),
            "hard_filter_pass": result.get("hard_filter_pass"),
            "hard_filter_reasons": result.get("hard_filter_reasons", []),
            "score": result.get("score"),
            "score_reasons": result.get("score_reasons", []),
            "biomarker_check": result.get("biomarker_check"),
            "treatment_check": result.get("treatment_check"),
            "assessment": result.get("assessment"),
        })

    return serialized


def _serialize_trial_summaries(trials: List[TrialProfile]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []

    for trial in trials:
        summaries.append({
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
        })

    return summaries


# -----------------------------------------------------------
# PUBLIC ENTRYPOINT
# -----------------------------------------------------------
async def run_trial_matching(patient_payload: Dict[str, Any]) -> Dict[str, Any]:
    patient_profile_id: Optional[Any] = None
    trial_match_run_id: Optional[Any] = None

    try:
        normalized_payload = normalize_patient_payload(patient_payload)
        patient_data = PatientProfile.model_validate(normalized_payload)
    except Exception as exc:
        logger.exception("Patient payload normalization/validation failed")
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

    try:
        with Session(engine) as session:
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

            try:
                session.commit()
            except Exception:
                logger.exception("Commit failed while creating DB records")
                session.rollback()
    except Exception:
        logger.exception("Database setup failed before workflow execution")

    initial_state: AgentState = {
        "patient_data": patient_data,
        "cancer_type": "",
        "trials_raw": [],
        "eligibility_results": [],
        "final_recommendations": "",
        "error": None,
    }

    try:
        final_state = await trial_matching_app.ainvoke(initial_state)
    except Exception:
        logger.exception("Trial matching workflow execution failed")
        final_state = {
            **initial_state,
            "error": "Workflow execution failed unexpectedly.",
            "final_recommendations": "Unable to complete clinical trial matching at this time.",
        }

    serialized_trials = _serialize_trial_summaries(
        final_state.get("trials_raw") or []
    )
    serialized_results = _serialize_trial_results(
        final_state.get("eligibility_results") or []
    )

    try:
        if trial_match_run_id is not None:
            with Session(engine) as session:
                try:
                    _safe_service_call(
                        save_trial_match_results,
                        [
                            ((session, trial_match_run_id, serialized_results), {}),
                            ((session,), {"match_run_id": trial_match_run_id, "results": serialized_results}),
                            ((session,), {"trial_match_run_id": trial_match_run_id, "results": serialized_results}),
                            ((session,), {"run_id": trial_match_run_id, "results": serialized_results}),
                        ],
                    )
                except Exception:
                    logger.exception("Failed to save trial match results")
                    session.rollback()

                try:
                    if final_state.get("error"):
                        _safe_service_call(
                            update_trial_match_run_failed,
                            [
                                ((session, trial_match_run_id, final_state["error"]), {}),
                                ((session,), {"match_run_id": trial_match_run_id, "error": final_state["error"]}),
                                ((session,), {"run_id": trial_match_run_id, "error": final_state["error"]}),
                                ((session,), {"trial_match_run_id": trial_match_run_id, "error": final_state["error"]}),
                                ((session,), {"match_run_id": trial_match_run_id, "message": final_state["error"]}),
                            ],
                        )
                    elif serialized_results:
                        _safe_service_call(
                            update_trial_match_run_success,
                            [
                                ((session, trial_match_run_id, final_state.get("final_recommendations", "")), {}),
                                ((session,), {
                                    "match_run_id": trial_match_run_id,
                                    "final_recommendations": final_state.get("final_recommendations", ""),
                                }),
                                ((session,), {
                                    "run_id": trial_match_run_id,
                                    "final_recommendations": final_state.get("final_recommendations", ""),
                                }),
                                ((session,), {
                                    "trial_match_run_id": trial_match_run_id,
                                    "final_recommendations": final_state.get("final_recommendations", ""),
                                }),
                            ],
                        )
                    else:
                        _safe_service_call(
                            update_trial_match_run_partial,
                            [
                                ((session, trial_match_run_id, final_state.get("final_recommendations", "")), {}),
                                ((session,), {
                                    "match_run_id": trial_match_run_id,
                                    "final_recommendations": final_state.get("final_recommendations", ""),
                                }),
                                ((session,), {
                                    "run_id": trial_match_run_id,
                                    "final_recommendations": final_state.get("final_recommendations", ""),
                                }),
                                ((session,), {
                                    "trial_match_run_id": trial_match_run_id,
                                    "final_recommendations": final_state.get("final_recommendations", ""),
                                }),
                            ],
                        )
                except Exception:
                    logger.exception("Failed to update trial match run status")
                    session.rollback()

                try:
                    session.commit()
                except Exception:
                    logger.exception("Commit failed while saving workflow results")
                    session.rollback()
    except Exception:
        logger.exception("Database persistence failed after workflow execution")

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


async def process_patient_trial_matching(patient_payload: Dict[str, Any]) -> Dict[str, Any]:
    return await run_trial_matching(patient_payload)


run_trial_matcher_agent = run_trial_matching
