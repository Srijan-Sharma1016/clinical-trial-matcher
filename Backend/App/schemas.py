# schemas.py
"""
Pydantic schema definitions.
Responsibility: Data contracts for patient profiles,
trial data, and API responses.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


# -----------------------------------------------------------
# SHARED UTILITIES
# -----------------------------------------------------------

_PLACEHOLDER_STRINGS = {
    "",
    "unknown",
    "n/a",
    "na",
    "not applicable",
    "null",
    "none",
}


def _normalize_optional_string(value: Any) -> Optional[str]:
    """
    Convert blank/placeholder strings to None,
    otherwise strip whitespace.
    """
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.lower() in _PLACEHOLDER_STRINGS:
            return None
        return cleaned
    return value


def _clean_string_list(value: Any) -> List[str]:
    """
    Normalize string lists — remove blanks/placeholders,
    preserve order, de-duplicate.
    Handles comma-separated strings AND proper arrays.
    """
    if value is None:
        return []

    if isinstance(value, str):
        # Split "EGFR, ALK, PD-L1" → ["EGFR", "ALK", "PD-L1"]
        raw_items = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]

    cleaned_items: List[str] = []
    seen: set = set()

    for item in raw_items:
        if item is None:
            continue
        item_str = str(item).strip()
        if not item_str or item_str.lower() in _PLACEHOLDER_STRINGS:
            continue
        if item_str not in seen:
            seen.add(item_str)
            cleaned_items.append(item_str)

    return cleaned_items


# -----------------------------------------------------------
# PATIENT PROFILE
# -----------------------------------------------------------

class PatientProfile(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    age: Optional[int] = None
    gender: Optional[str] = None
    cancer_type: Optional[str] = None
    cancer_stage: Optional[str] = None
    biomarkers: List[str] = Field(default_factory=list)
    previous_treatments: List[str] = Field(default_factory=list)
    country: Optional[str] = None
    diagnosis: Optional[str] = None

    @field_validator("age", mode="before")
    @classmethod
    def parse_age(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            cleaned = v.strip()

            # Empty or placeholder → None (no error)
            if cleaned == "" or cleaned.lower() in _PLACEHOLDER_STRINGS:
                return None

            # Try integer parse first
            if cleaned.isdigit():
                return int(cleaned)

            # Try float string like "62.0"
            try:
                parsed = int(float(cleaned))
                return parsed
            except (ValueError, TypeError):
                raise ValueError(
                    f"'{cleaned}' is not a valid age. "
                    f"Please enter a number between 1 and 120."
                )

        if isinstance(v, float):
            return int(v)

        return v

    @field_validator("age")
    @classmethod
    def validate_age(cls, v):
        if v is not None and (v <= 0 or v > 120):
            raise ValueError(
                f"Age {v} is not valid. "
                f"Please enter a value between 1 and 120."
            )
        return v

    @field_validator(
        "gender",
        "cancer_type",
        "cancer_stage",
        "country",
        "diagnosis",
        mode="before",
    )
    @classmethod
    def normalize_optional_strings(cls, v):
        return _normalize_optional_string(v)

    @field_validator(
        "biomarkers",
        "previous_treatments",
        mode="before",
    )
    @classmethod
    def clean_string_lists(cls, v):
        return _clean_string_list(v)


# -----------------------------------------------------------
# TRIAL LOCATION
# -----------------------------------------------------------

class TrialLocation(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    facility: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

    @field_validator("facility", "city", "country", mode="before")
    @classmethod
    def normalize_optional_strings(cls, v):
        return _normalize_optional_string(v)

    def __str__(self) -> str:
        parts = [self.facility, self.city, self.country]
        return ", ".join(p for p in parts if p)
# -----------------------------------------------------------
# TRIAL ELIGIBILITY
# -----------------------------------------------------------

class TrialEligibility(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    criteria_text: Optional[str] = None
    healthy_volunteers: Optional[bool] = None
    sex: Optional[Literal["ALL", "MALE", "FEMALE"]] = None
    minimum_age: Optional[str] = None
    maximum_age: Optional[str] = None
    age_groups: List[str] = Field(default_factory=list)
    study_population: Optional[str] = None

    @field_validator(
        "criteria_text",
        "minimum_age",
        "maximum_age",
        "study_population",
        mode="before",
    )
    @classmethod
    def normalize_optional_strings(cls, v):
        return _normalize_optional_string(v)

    @field_validator("sex", mode="before")
    @classmethod
    def normalize_sex(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            cleaned = v.strip().upper()
            if cleaned in _PLACEHOLDER_STRINGS or cleaned == "":
                return None
            return cleaned
        return v

    @field_validator("age_groups", mode="before")
    @classmethod
    def clean_age_groups(cls, v):
        return _clean_string_list(v)


# -----------------------------------------------------------
# TRIAL SCORING SIGNALS
# -----------------------------------------------------------

class TrialScoringSignals(BaseModel):
    """
    Structured signals extracted from a TrialProfile's
    eligibility criteria. Used by scoring.py for
    data-driven comparison instead of hardcoded
    keyword scanning. Extracted at Node 2 (search_trials)
    via extract_trial_scoring_signals().
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    required_biomarkers: List[str] = Field(
        default_factory=list,
        description=(
            "Biomarkers explicitly required for eligibility "
            "(e.g. ['HER2', 'BRCA1'])"
        ),
    )
    excluded_biomarkers: List[str] = Field(
        default_factory=list,
        description=(
            "Biomarkers that explicitly disqualify a patient "
            "(e.g. ['HER2-positive'])"
        ),
    )
    excluded_treatments: List[str] = Field(
        default_factory=list,
        description=(
            "Prior treatments that disqualify a patient "
            "per exclusion criteria"
        ),
    )
    min_prior_lines: Optional[int] = Field(
        default=None,
        description=(
            "Minimum prior treatment lines required "
            "(e.g. 1 = 'at least 1 prior line')"
        ),
    )
    max_prior_lines: Optional[int] = Field(
        default=None,
        description=(
            "Maximum prior treatment lines allowed "
            "(e.g. 2 = '≤2 prior lines')"
        ),
    )
    requires_treatment_naive: bool = Field(
        default=False,
        description=(
            "True if trial explicitly requires "
            "untreated / treatment-naive patients"
        ),
    )
    target_setting: str = Field(
        default="unknown",
        description=(
            "Inferred disease setting the trial targets: "
            "'early' | 'locally_advanced' | 'advanced' | 'unknown'"
        ),
    )
    required_cancer_types: List[str] = Field(
        default_factory=list,
        description=(
            "Normalized cancer type terms extracted from "
            "trial title, conditions, and eligibility text"
        ),
    )
    trial_phase: Optional[str] = Field(
        default=None,
        description=(
            "Trial phase from trial data "
            "(e.g. 'PHASE1', 'PHASE2', 'PHASE3')"
        ),
    )
    is_interventional: bool = Field(
        default=False,
        description="True if study_type is INTERVENTIONAL",
    )

    @field_validator("target_setting", mode="before")
    @classmethod
    def validate_target_setting(cls, v: Any) -> str:
        allowed = {"early", "locally_advanced", "advanced", "unknown"}
        normalized = str(v or "").strip().lower()
        return normalized if normalized in allowed else "unknown"

    @field_validator(
        "required_biomarkers",
        "excluded_biomarkers",
        "excluded_treatments",
        "required_cancer_types",
        mode="before",
    )
    @classmethod
    def clean_signal_lists(cls, v):
        return [item.lower() for item in _clean_string_list(v)]

    @field_validator("trial_phase", mode="before")
    @classmethod
    def normalize_trial_phase(cls, v):
        normalized = _normalize_optional_string(v)
        if isinstance(normalized, str):
            return normalized.upper()
        return normalized

    @field_validator("min_prior_lines", "max_prior_lines")
    @classmethod
    def validate_prior_lines(cls, v):
        if v is not None and v < 0:
            raise ValueError(
                "Prior treatment line counts cannot be negative."
            )
        return v


# -----------------------------------------------------------
# TRIAL PROFILE
# -----------------------------------------------------------

class TrialProfile(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    trial_id: str
    title: Optional[str] = None
    official_title: Optional[str] = None
    status: Optional[str] = None
    study_type: Optional[str] = None
    phases: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    brief_summary: Optional[str] = None
    detailed_description: Optional[str] = None
    eligibility: TrialEligibility = Field(
        default_factory=TrialEligibility
    )
    locations: List[TrialLocation] = Field(default_factory=list)
    sponsor_name: Optional[str] = None
    sponsor_class: Optional[str] = None
    mesh_terms: List[str] = Field(default_factory=list)
    has_results: bool = False

    # Pre-computed scoring signals — populated at Node 2
    scoring_signals: Optional[TrialScoringSignals] = Field(
        default=None,
        description=(
            "Pre-extracted scoring signals. Populated by "
            "extract_trial_scoring_signals() at Node 2."
        ),
    )

    @field_validator("trial_id")
    @classmethod
    def validate_trial_id(cls, v):
        if not v or not str(v).strip():
            raise ValueError("trial_id cannot be empty")
        return str(v).strip().upper()

    @field_validator(
        "title",
        "official_title",
        "status",
        "study_type",
        "brief_summary",
        "detailed_description",
        "sponsor_name",
        "sponsor_class",
        mode="before",
    )
    @classmethod
    def normalize_optional_strings(cls, v):
        return _normalize_optional_string(v)

    @field_validator(
        "phases", "conditions", "mesh_terms",
        mode="before",
    )
    @classmethod
    def clean_string_lists(cls, v):
        return _clean_string_list(v)


# -----------------------------------------------------------
# TRIAL MATCH RESULT
# -----------------------------------------------------------

class TrialMatchResult(BaseModel):
    """Structured result from the trial matching workflow."""
    final_recommendations: str = ""
    eligibility_results: List[Dict[str, Any]] = Field(
        default_factory=list
    )
    trials: List[Dict[str, Any]] = Field(default_factory=list)
    trial_count: int = 0
    cancer_type: str = ""
    success: bool = True
    error: Optional[str] = None


# -----------------------------------------------------------
# PROFILE ANALYSIS RESPONSE
# -----------------------------------------------------------

class ProfileAnalysisResponse(BaseModel):
    profile: PatientProfile
    status: Literal[
        "PROFILE_READY",
        "NEEDS_CLARIFICATION",
        "MATCHING_FAILED",
    ]
    is_complete: bool
    missing_fields: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)
    agent_suggestions: List[str] = Field(
        default_factory=list,
        description=(
            "Deprecated compatibility field. "
            "Prefer improvement_suggestions."
        ),
    )
    trial_matches: Optional[TrialMatchResult] = None


# -----------------------------------------------------------
# MANUAL PROFILE REQUEST
# -----------------------------------------------------------

class ManualProfileRequest(BaseModel):
    """
    Request model for manual patient profile entry.

    Frontend sends:
    {
        "profile": {
            "age": 62,
            "gender": "Male",
            "cancer_type": "Non-Small Cell Lung Cancer",
            "cancer_stage": "Stage IV",
            "biomarkers": ["EGFR", "PD-L1"],
            "previous_treatments": ["Carboplatin"],
            "country": "India",
            "diagnosis": null
        }
    }

    Notes:
    - age must be integer between 1 and 120
    - empty optional fields must be null not ""
    - biomarkers and previous_treatments accept
      arrays OR comma-separated strings
    """
    profile: PatientProfile


# -----------------------------------------------------------
# CHAT REQUEST / RESPONSE
# -----------------------------------------------------------

class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""
    session_id: Optional[str] = None
    message: str
    patient_profile: Optional[Dict[str, Any]] = None
    trial_matches: Optional[Dict[str, Any]] = None

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message cannot be empty.")
        return v.strip()


class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""
    session_id: str
    response: str
    message_count: int
