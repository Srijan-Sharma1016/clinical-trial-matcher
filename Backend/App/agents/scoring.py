# agents/scoring.py
"""
Deterministic trial scoring logic.
Responsibility: Score patient-trial alignment across multiple dimensions.
Depends on: schemas, core/utils, TrialScoringSignals (via trial.scoring_signals)

Key fixes:
- Handles negative biomarkers correctly, e.g. ALK- does NOT match ALK-positive trials.
- Parses patient biomarkers into structured positive/negative/unknown signals.
- Penalizes biomarker-specific trials when patient is known negative for that marker.
- Recognizes EGFR Exon 19 deletion and similar variants as positive biomarker evidence.
- Downgrades diagnostic/imaging-only studies for treatment matching.
- Avoids treating "biomarker not mentioned" as an automatic hard rejection.
"""
import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from schemas import PatientProfile, TrialProfile, TrialScoringSignals
from core.utils import (
    normalize_free_text,
    safe_join_text,
    get_patient_country,
    get_api_search_term,
    get_trial_searchable_text,
    limit_score_reasons,
)

__all__ = [
    "score_trial_match",
    "score_cancer_match",
    "score_stage_match",
    "score_biomarker_match",
    "score_treatment_history_match",
    "score_location_match",
    "score_study_type",
    "score_phase_bonus",
    "infer_patient_disease_setting",
    "is_reasonably_strong_trial",
]


# -----------------------------------------------------------
# MODULE-LEVEL CONSTANTS
# -----------------------------------------------------------

_ACTIONABLE_DRIVER_BIOMARKERS = {
    "egfr",
    "alk",
    "ros1",
    "kras",
    "braf",
    "met",
    "ret",
    "ntrk",
    "her2",
    "erbb2",
}

_FALLBACK_EARLY_MARKERS = frozenset([
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
])

_FALLBACK_LOCALLY_ADVANCED_MARKERS = frozenset([
    "locally advanced",
    "stage iii",
    "stage 3",
])

_FALLBACK_ADVANCED_MARKERS = frozenset([
    "metastatic",
    "advanced",
    "unresectable",
    "recurrent",
    "relapsed",
    "stage iv",
    "stage 4",
])

_FALLBACK_UNTREATED_MARKERS = frozenset([
    "previously untreated",
    "untreated",
    "treatment naive",
    "treatment-naive",
    "no prior treatment",
    "no previous treatment",
    "first line",
    "first-line",
    "1st line",
])

_FALLBACK_PRIOR_TREATMENT_MARKERS = frozenset([
    "previously treated",
    "received prior",
    "after prior",
    "progressed on",
    "1 prior line",
    "one prior line",
    "prior chemotherapy",
])

_FALLBACK_EXCLUSION_MARKERS = frozenset([
    "must not have received",
    "excluded",
    "exclusion",
    "prior treatment with",
    "previous treatment with",
])

_PHASE_BONUS_MAP = {
    "PHASE3": 2,
    "PHASE2": 1,
    "PHASE1": 0,
    "PHASE4": 2,
}

# Biomarkers we want to reason about explicitly.
# This does not limit raw text scoring, but helps parse common oncology markers.
_KNOWN_BIOMARKERS = [
    "egfr",
    "alk",
    "ros1",
    "kras",
    "braf",
    "met",
    "ret",
    "ntrk",
    "her2",
    "erbb2",
    "pd-l1",
    "pdl1",
    "msi",
    "mmr",
    "brca1",
    "brca2",
    "nras",
]

_BIOMARKER_ALIASES = {
    "pdl1": "pd-l1",
    "erbb2": "her2",
}

_POSITIVE_SIGNAL_WORDS = [
    "positive",
    "pos",
    "+",
    "mutation",
    "mutated",
    "mutant",
    "variant",
    "exon",
    "del",
    "deletion",
    "ins",
    "insertion",
    "fusion",
    "rearranged",
    "rearrangement",
    "amplification",
    "amplified",
    "overexpression",
    "expressing",
    "expression",
    "g12c",
    "g12d",
    "v600e",
]

_NEGATIVE_SIGNAL_WORDS = [
    "negative",
    "neg",
    "wild type",
    "wild-type",
    "wt",
    "not detected",
    "undetected",
    "absent",
]

_DIAGNOSTIC_OR_SCREENING_MARKERS = [
    "diagnosis",
    "diagnostic",
    "screening",
    "imaging",
    "biopsy",
    "detection",
    "detect",
    "endomicroscopy",
    "confocal",
    "real-time imaging",
    "radiomics",
    "ct scan",
    "pet scan",
]

_TREATMENT_INTENT_MARKERS = [
    "treatment",
    "therapy",
    "chemotherapy",
    "immunotherapy",
    "targeted therapy",
    "radiotherapy",
    "radiation therapy",
    "drug",
    "combination",
    "osimertinib",
    "erlotinib",
    "gefitinib",
    "afatinib",
    "lazertinib",
    "amivantamab",
    "lorlatinib",
    "alectinib",
    "brigatinib",
    "repotrectinib",
    "pembrolizumab",
    "nivolumab",
    "atezolizumab",
]


# -----------------------------------------------------------
# SAFE HELPERS
# -----------------------------------------------------------

def _as_text(value: Any) -> str:
    """
    Converts arbitrary schema values to text safely.
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        return " ".join(_as_text(v) for v in value.values())

    if isinstance(value, list):
        return " ".join(_as_text(v) for v in value)

    return str(value)


def _normalize_dash_text(value: Any) -> str:
    """
    Normalizes text while preserving enough dash information to detect ALK-, KRAS-, etc.
    """
    text = _as_text(value).lower()
    text = (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("＋", "+")
    )
    return text.strip()


def _canonical_marker_name(marker: str) -> str:
    marker = marker.lower().strip()
    return _BIOMARKER_ALIASES.get(marker, marker)


def _normalized_list(values: Optional[List[Any]]) -> List[str]:
    if not values:
        return []

    result = []
    for value in values:
        text = normalize_free_text(_as_text(value))
        if text:
            result.append(text)

    return result


def _get_signals(trial: TrialProfile) -> Optional[TrialScoringSignals]:
    """
    Safely retrieves scoring signals from a trial.
    Returns None if signals were not extracted.
    """
    return trial.scoring_signals if trial.scoring_signals is not None else None


def _get_trial_full_text(trial: TrialProfile) -> str:
    """
    Defensive trial text builder.
    Avoids errors if eligibility or nested fields are missing.
    """
    eligibility = getattr(trial, "eligibility", None)

    return normalize_free_text(
        safe_join_text([
            getattr(trial, "title", ""),
            getattr(trial, "official_title", ""),
            " ".join(getattr(trial, "conditions", []) or []),
            getattr(trial, "brief_summary", ""),
            getattr(trial, "detailed_description", ""),
            getattr(eligibility, "study_population", "") if eligibility else "",
            getattr(eligibility, "criteria_text", "") if eligibility else "",
        ])
    )


def _extract_marker_names(text: str) -> List[str]:
    """
    Finds known biomarker names in normalized/free text.
    """
    raw = _normalize_dash_text(text)
    normalized = normalize_free_text(raw)

    found = []

    for marker in _KNOWN_BIOMARKERS:
        canonical = _canonical_marker_name(marker)

        marker_variants = {marker, canonical}

        if canonical == "pd-l1":
            marker_variants.update({"pd l1", "pdl1", "pd-l1"})

        if canonical == "her2":
            marker_variants.update({"her2", "erbb2"})

        for variant in marker_variants:
            if variant in raw or variant in normalized:
                if canonical not in found:
                    found.append(canonical)
                break

    return found


def _has_marker_negative_syntax(raw_text: str, marker: str) -> bool:
    """
    Detects marker-negative syntax like:
    - ALK-
    - ALK negative
    - ALK neg
    - EGFR wild type
    """
    raw = _normalize_dash_text(raw_text)
    marker = _canonical_marker_name(marker)

    variants = {marker}
    if marker == "pd-l1":
        variants.update({"pd l1", "pdl1", "pd-l1"})
    if marker == "her2":
        variants.update({"her2", "erbb2"})

    for variant in variants:
        if f"{variant}-" in raw:
            return True

        if f"{variant} -" in raw:
            return True

        for word in _NEGATIVE_SIGNAL_WORDS:
            if f"{variant} {word}" in raw or f"{word} {variant}" in raw:
                return True

    return False


def _has_marker_positive_syntax(raw_text: str, marker: str) -> bool:
    """
    Detects marker-positive/variant syntax like:
    - EGFR Exon 19 del
    - ALK+
    - ROS1 rearranged
    - KRAS G12C
    """
    raw = _normalize_dash_text(raw_text)
    normalized = normalize_free_text(raw)
    marker = _canonical_marker_name(marker)

    variants = {marker}
    if marker == "pd-l1":
        variants.update({"pd l1", "pdl1", "pd-l1"})
    if marker == "her2":
        variants.update({"her2", "erbb2"})

    for variant in variants:
        if f"{variant}+" in raw or f"{variant} +" in raw:
            return True

        if variant in raw or variant in normalized:
            if any(word in raw or word in normalized for word in _POSITIVE_SIGNAL_WORDS):
                return True

    return False


def _parse_patient_biomarker(value: Any) -> Dict[str, Any]:
    """
    Converts patient biomarker text into structured form.

    Examples:
        ALK-              -> {"name": "alk", "status": "negative"}
        KRAS-             -> {"name": "kras", "status": "negative"}
        EGFR Exon 19 del  -> {"name": "egfr", "status": "positive"}
        ROS1              -> {"name": "ros1", "status": "unknown"}
    """
    raw = _as_text(value)
    normalized = normalize_free_text(raw)
    names = _extract_marker_names(raw)

    if not names:
        return {
            "raw": raw,
            "normalized": normalized,
            "name": normalized,
            "status": "unknown",
        }

    # Prefer first detected marker.
    name = names[0]

    if _has_marker_negative_syntax(raw, name):
        status = "negative"
    elif _has_marker_positive_syntax(raw, name):
        status = "positive"
    else:
        status = "unknown"

    return {
        "raw": raw,
        "normalized": normalized,
        "name": name,
        "status": status,
    }


def _trial_requirement_mentions_marker(requirements: List[str], marker: str) -> List[str]:
    """
    Returns requirement strings that mention the marker.
    """
    marker = _canonical_marker_name(marker)
    matches = []

    for req in requirements:
        req_text = _normalize_dash_text(req)
        req_norm = normalize_free_text(req_text)
        names = _extract_marker_names(req_text)

        if marker in names:
            matches.append(req)
            continue

        if marker in req_text or marker in req_norm:
            matches.append(req)

    return matches


def _trial_text_mentions_marker(trial_text: str, marker: str) -> bool:
    marker = _canonical_marker_name(marker)
    names = _extract_marker_names(trial_text)

    if marker in names:
        return True

    trial_text = _normalize_dash_text(trial_text)
    trial_norm = normalize_free_text(trial_text)

    return marker in trial_text or marker in trial_norm


# -----------------------------------------------------------
# PATIENT DISEASE SETTING INFERENCE
# -----------------------------------------------------------

def infer_patient_disease_setting(patient_data: PatientProfile) -> str:
    """
    Infers disease setting from patient profile fields.
    Returns: 'early' | 'locally_advanced' | 'advanced' | 'unknown'
    """
    text = normalize_free_text(
        safe_join_text([
            patient_data.cancer_stage,
            patient_data.cancer_type,
            patient_data.diagnosis,
        ])
    )

    if not text:
        return "unknown"

        # Use regex boundaries to avoid bugs like:
    # "stage i" matching inside "stage iv"
    advanced_re = re.compile(
        r"\b(stage\s*(iv|4)|metastatic|unresectable|recurrent|relapsed|advanced)\b",
        re.IGNORECASE,
    )

    locally_advanced_re = re.compile(
        r"\b(locally\s+advanced|stage\s*(iii|3))\b",
        re.IGNORECASE,
    )

    early_re = re.compile(
        r"\b("
        r"early[-\s]?stage|localized|resectable|adjuvant|neoadjuvant|"
        r"stage\s*(i|1|ii|2)|locally\s+confined"
        r")\b",
        re.IGNORECASE,
    )

    if advanced_re.search(text):
        return "advanced"

    if locally_advanced_re.search(text):
        return "locally_advanced"

    if early_re.search(text):
        return "early"

    return "unknown"
# -----------------------------------------------------------
# INDIVIDUAL SCORING FUNCTIONS
# -----------------------------------------------------------

def score_cancer_match(
    patient_data: PatientProfile,
    trial: TrialProfile,
) -> Tuple[int, List[str]]:
    """
    Scores cancer type alignment between patient and trial.
    """
    reasons: List[str] = []

    patient_cancer = patient_data.cancer_type or patient_data.diagnosis

    if not patient_cancer:
        reasons.append("Cancer type is missing from patient profile.")
        return 0, reasons

    signals = _get_signals(trial)
    patient_term = normalize_free_text(get_api_search_term(patient_cancer))

    if signals is not None and signals.required_cancer_types:
        required_cancer_types = _normalized_list(signals.required_cancer_types)

        if any(
            patient_term in cancer_type or cancer_type in patient_term
            for cancer_type in required_cancer_types
        ):
            reasons.append(
                f"Strong cancer type match via structured trial conditions "
                f"for '{patient_cancer}'."
            )
            return 5, reasons

        patient_tokens = [t for t in patient_term.split() if len(t) > 3]
        matched_tokens = [
            t
            for t in patient_tokens
            if any(t in cancer_type for cancer_type in required_cancer_types)
        ]

        if matched_tokens:
            reasons.append(
                f"Partial cancer type overlap in trial conditions: "
                f"{', '.join(matched_tokens)}."
            )
            return 2, reasons

        reasons.append(
            f"Patient cancer type '{patient_cancer}' not found in "
            f"trial's structured conditions."
        )
        return -2, reasons

    trial_text = get_trial_searchable_text(trial)

    if patient_term and patient_term in trial_text:
        reasons.append(
            f"Cancer type match via raw text scan for '{patient_cancer}'."
        )
        return 5, reasons

    patient_tokens = [t for t in patient_term.split() if len(t) > 3]
    overlap = [t for t in patient_tokens if t in trial_text]

    if overlap:
        reasons.append(
            f"Partial cancer type overlap in raw text: {', '.join(overlap)}."
        )
        return 2, reasons

    reasons.append("Cancer type match appears weak or unclear.")
    return -2, reasons


def score_stage_match(
    patient_data: PatientProfile,
    trial: TrialProfile,
) -> Tuple[int, List[str]]:
    """
    Scores disease setting alignment between patient and trial.
    """
    reasons: List[str] = []

    patient_setting = infer_patient_disease_setting(patient_data)
    signals = _get_signals(trial)

    if signals is not None and signals.target_setting != "unknown":
        trial_setting = signals.target_setting
    else:
        trial_text = _get_trial_full_text(trial)

        if any(k in trial_text for k in _FALLBACK_EARLY_MARKERS):
            trial_setting = "early"
        elif any(k in trial_text for k in _FALLBACK_LOCALLY_ADVANCED_MARKERS):
            trial_setting = "locally_advanced"
        elif any(k in trial_text for k in _FALLBACK_ADVANCED_MARKERS):
            trial_setting = "advanced"
        else:
            trial_setting = "unknown"

    if patient_setting == "unknown" or trial_setting == "unknown":
        reasons.append("Disease setting is unclear for patient or trial.")
        return 0, reasons

    if patient_setting == trial_setting:
        reasons.append(f"Disease setting aligns: '{patient_setting}'.")
        return 4, reasons

    if patient_setting == "early" and trial_setting == "advanced":
        reasons.append(
            "Stage mismatch: patient appears early-stage while trial targets "
            "advanced/metastatic disease."
        )
        return -6, reasons

    if patient_setting == "advanced" and trial_setting == "early":
        reasons.append(
            "Stage mismatch: patient appears advanced/metastatic while trial "
            "targets early-stage disease."
        )
        return -6, reasons

    mismatch_map = {
        ("early", "locally_advanced"): (
            -2,
            "Stage caution: patient appears early-stage while trial targets "
            "locally advanced disease.",
        ),
        ("locally_advanced", "early"): (
            -2,
            "Stage caution: patient appears locally advanced while trial "
            "targets early-stage disease.",
        ),
        ("locally_advanced", "advanced"): (
            -2,
            "Stage caution: patient may be locally advanced while trial "
            "appears metastatic-focused.",
        ),
        ("advanced", "locally_advanced"): (
            -2,
            "Stage caution: patient may be metastatic while trial appears "
            "locally advanced-focused.",
        ),
    }

    if (patient_setting, trial_setting) in mismatch_map:
        pts, msg = mismatch_map[(patient_setting, trial_setting)]
        reasons.append(msg)
        return pts, reasons

    reasons.append(
        f"Disease setting mismatch: patient='{patient_setting}', "
        f"trial='{trial_setting}'."
    )
    return -1, reasons

def score_treatment_history_match(
    patient_data: PatientProfile,
    trial: TrialProfile,
) -> Tuple[int, List[str]]:
    """
    Scores treatment history compatibility.
    """
    reasons: List[str] = []

    treatments = patient_data.previous_treatments or []
    signals = _get_signals(trial)
    score = 0

    patient_prior_lines = len(treatments)

    if signals is not None:
        if signals.requires_treatment_naive:
            if patient_prior_lines == 0:
                score += 3
                reasons.append(
                    "Trial requires treatment-naive patients, and patient has "
                    "no prior treatments recorded."
                )
            else:
                score -= 5
                reasons.append(
                    "Treatment mismatch: trial requires treatment-naive patients "
                    "but patient has prior treatment history."
                )
            return score, reasons

        if signals.max_prior_lines is not None:
            if patient_prior_lines <= signals.max_prior_lines:
                score += 2
                reasons.append(
                    f"Patient's prior treatment lines ({patient_prior_lines}) "
                    f"are within trial's maximum ({signals.max_prior_lines})."
                )
            else:
                score -= 3
                reasons.append(
                    f"Treatment mismatch: patient's prior treatment lines "
                    f"({patient_prior_lines}) exceed trial's maximum "
                    f"({signals.max_prior_lines})."
                )

        if signals.min_prior_lines is not None:
            if patient_prior_lines >= signals.min_prior_lines:
                score += 1
                reasons.append(
                    f"Patient's prior treatment lines ({patient_prior_lines}) "
                    f"meet trial's minimum ({signals.min_prior_lines})."
                )
            else:
                score -= 2
                reasons.append(
                    f"Treatment caution: patient's prior treatment lines "
                    f"({patient_prior_lines}) are below trial's minimum "
                    f"({signals.min_prior_lines})."
                )

        if signals.excluded_treatments and treatments:
            excluded_treatments = _normalized_list(signals.excluded_treatments)
            normalized_treatments = [
                normalize_free_text(_as_text(t))
                for t in treatments
            ]

            for treatment in normalized_treatments:
                if treatment in excluded_treatments:
                    score -= 3
                    reasons.append(
                        f"Treatment mismatch: prior treatment '{treatment}' is "
                        f"explicitly excluded by this trial."
                    )

        if score == 0:
            reasons.append(
                "Prior treatment history appears compatible or not specifically restricted."
            )

        return score, reasons

    if not treatments:
        reasons.append("No previous treatment history available for scoring.")
        return 0, reasons

    trial_text = get_trial_searchable_text(trial)

    if any(marker in trial_text for marker in _FALLBACK_UNTREATED_MARKERS):
        reasons.append(
            "Treatment mismatch: trial appears to prefer untreated / "
            "treatment-naive patients."
        )
        return -5, reasons

    if any(marker in trial_text for marker in _FALLBACK_PRIOR_TREATMENT_MARKERS):
        score += 2
        reasons.append(
            "Trial text appears compatible with prior treatment exposure."
        )

    normalized_treatments = [
        normalize_free_text(_as_text(t))
        for t in treatments
    ]

    for treatment in normalized_treatments:
        if treatment in trial_text and any(
            marker in trial_text for marker in _FALLBACK_EXCLUSION_MARKERS
        ):
            score -= 3
            reasons.append(
                f"Treatment caution: prior treatment '{treatment}' may conflict "
                f"with exclusion criteria."
            )

    if score == 0:
        reasons.append("Prior treatment compatibility remains unclear.")

    return score, reasons


def score_location_match(
    patient_data: PatientProfile,
    trial: TrialProfile,
) -> Tuple[int, List[str]]:
    """
    Scores location alignment.
    """
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

    reasons.append(
        f"Trial does not list patient's country '{patient_country}'."
    )
    return -2, reasons


def score_study_type(
    patient_data: PatientProfile,
    trial: TrialProfile,
) -> Tuple[int, List[str]]:
    """
    Scores study type and treatment relevance.

    Also downgrades diagnostic/imaging-only trials for treatment matching.
    """
    reasons: List[str] = []

    signals = _get_signals(trial)
    trial_text = _get_trial_full_text(trial)

    diagnostic_hits = [
        marker
        for marker in _DIAGNOSTIC_OR_SCREENING_MARKERS
        if marker in trial_text
    ]

    treatment_hits = [
        marker
        for marker in _TREATMENT_INTENT_MARKERS
        if marker in trial_text
    ]

    # Diagnostic-only trials should not rank as strong treatment matches.
    if diagnostic_hits and not treatment_hits:
        reasons.append(
            "Study relevance caution: trial appears primarily diagnostic, "
            "screening, imaging, or biopsy-focused rather than treatment-focused."
        )
        return -4, reasons

    if signals is not None:
        if signals.is_interventional:
            reasons.append(
                "Interventional trial preferred for treatment relevance."
            )
            return 2, reasons

        reasons.append(
            "Non-interventional study is less suitable for treatment matching."
        )
        return -3, reasons

    if trial.study_type == "INTERVENTIONAL":
        reasons.append(
            "Interventional trial preferred for treatment relevance."
        )
        return 2, reasons

    if trial.study_type == "OBSERVATIONAL":
        reasons.append(
            "Observational study is less suitable for treatment matching."
        )
        return -3, reasons

    reasons.append("Study type has neutral effect on ranking.")
    return 0, reasons


def score_phase_bonus(
    patient_data: PatientProfile,
    trial: TrialProfile,
) -> Tuple[int, List[str]]:
    """
    Awards a bonus based on trial phase.
    """
    reasons: List[str] = []

    signals = _get_signals(trial)

    if signals is not None and signals.trial_phase:
        phase = signals.trial_phase.upper()
        bonus = _PHASE_BONUS_MAP.get(phase, 0)

        if bonus > 0:
            reasons.append(
                f"Phase bonus: {signals.trial_phase} trial "
                f"(+{bonus} for clinical maturity)."
            )
        else:
            reasons.append(
                f"No phase bonus applied for {signals.trial_phase}."
            )

        return bonus, reasons

    if trial.phases:
        phase_order = {
            "PHASE4": 4,
            "PHASE3": 3,
            "PHASE2": 2,
            "PHASE1": 1,
        }

        normalized = [
            _as_text(p).upper().replace(" ", "")
            for p in trial.phases
        ]

        ranked = [p for p in normalized if p in phase_order]

        if ranked:
            highest = max(ranked, key=lambda p: phase_order[p])
            bonus = _PHASE_BONUS_MAP.get(highest, 0)

            if bonus > 0:
                reasons.append(
                    f"Phase bonus: {highest} trial "
                    f"(+{bonus} for clinical maturity)."
                )
            else:
                reasons.append(f"No phase bonus for {highest}.")

            return bonus, reasons

    reasons.append("Trial phase not available — no bonus applied.")
    return 0, reasons


# -----------------------------------------------------------
# COMPOSITE SCORER
# -----------------------------------------------------------

def score_biomarker_match(
    patient_data: PatientProfile,
    trial: TrialProfile,
) -> Tuple[int, List[str]]:
    """
    Scores biomarker alignment.

    Important behavior:
    - ALK- does NOT partially match ALK-positive trials.
    - KRAS- does NOT partially match KRAS-specific trials.
    - EGFR Exon 19 del is treated as EGFR-positive variant evidence.
    - Positive actionable biomarkers that are not addressed by a trial
      reduce confidence, preventing broad non-biomarker trials from
      ranking too strongly.
    """
    reasons: List[str] = []

    raw_biomarkers = patient_data.biomarkers or []

    if not raw_biomarkers:
        reasons.append("No patient biomarkers available for deterministic scoring.")
        return 0, reasons

    patient_markers = [
        _parse_patient_biomarker(marker)
        for marker in raw_biomarkers
        if _as_text(marker).strip()
    ]

    if not patient_markers:
        reasons.append("No usable patient biomarkers available for scoring.")
        return 0, reasons

    signals = _get_signals(trial)
    total = 0

    # -------------------------------------------------------
    # PRIMARY PATH: structured trial scoring signals available
    # -------------------------------------------------------
    if signals is not None and (
        signals.required_biomarkers or signals.excluded_biomarkers
    ):
        required = _normalized_list(signals.required_biomarkers)
        excluded = _normalized_list(signals.excluded_biomarkers)

        for patient_marker in patient_markers:
            marker_name = patient_marker["name"]
            marker_status = patient_marker["status"]
            marker_raw = patient_marker["raw"]

            required_matches = _trial_requirement_mentions_marker(
                required,
                marker_name,
            )
            excluded_matches = _trial_requirement_mentions_marker(
                excluded,
                marker_name,
            )

            # ---------------------------------------------------
            # Exclusion rules first.
            # ---------------------------------------------------
            if excluded_matches:
                if marker_status == "negative":
                    total += 1
                    reasons.append(
                        f"Patient is {marker_name.upper()}-negative, which does "
                        f"not conflict with trial exclusion signal: "
                        f"{', '.join(excluded_matches)}."
                    )
                else:
                    total -= 5
                    reasons.append(
                        f"Biomarker mismatch: patient biomarker '{marker_raw}' "
                        f"conflicts with trial exclusion signal: "
                        f"{', '.join(excluded_matches)}."
                    )
                continue

            # ---------------------------------------------------
            # Required biomarker rules.
            # ---------------------------------------------------
            if required_matches:
                if marker_status == "negative":
                    total -= 7
                    reasons.append(
                        f"Biomarker mismatch: patient is {marker_name.upper()}-negative, "
                        f"but this trial appears to require {marker_name.upper()} "
                        f"positivity/signaling: {', '.join(required_matches)}."
                    )
                    continue

                if marker_status == "positive":
                    total += 5
                    reasons.append(
                        f"Biomarker match: patient biomarker '{marker_raw}' aligns "
                        f"with trial requirement: {', '.join(required_matches)}."
                    )
                    continue

                # Unknown/bare marker, e.g. "ROS1" without positive/negative.
                total += 1
                reasons.append(
                    f"Biomarker uncertainty: patient lists '{marker_raw}', and "
                    f"trial references {marker_name.upper()}, but positivity/status "
                    f"is not explicit in the patient profile."
                )
                continue

            # ---------------------------------------------------
            # Patient has this biomarker, but the trial does not
            # specifically require/exclude it.
            # ---------------------------------------------------
            if (
                marker_status == "positive"
                and marker_name in _ACTIONABLE_DRIVER_BIOMARKERS
            ):
                total -= 2
                reasons.append(
                    f"Actionable biomarker caution: patient has '{marker_raw}', "
                    f"but this trial does not appear to specifically address "
                    f"{marker_name.upper()}."
                )

            elif marker_status == "negative":
                reasons.append(
                    f"Patient marker '{marker_raw}' is negative and does not "
                    f"appear to conflict with this trial."
                )

            else:
                reasons.append(
                    f"Trial does not appear biomarker-specific for patient marker "
                    f"'{marker_raw}'."
                )

        raw_total = total
        total = max(min(total, 6), -8)

        if raw_total != total:
            reasons.append(
                f"Biomarker score clamped from {raw_total} to {total}."
            )

        return total, reasons

    # -------------------------------------------------------
    # FALLBACK PATH: structured biomarker signals unavailable
    # -------------------------------------------------------
    trial_text = get_trial_searchable_text(trial)

    for patient_marker in patient_markers:
        marker_name = patient_marker["name"]
        marker_status = patient_marker["status"]
        marker_raw = patient_marker["raw"]

        trial_mentions_marker = _trial_text_mentions_marker(
            trial_text,
            marker_name,
        )

        if not trial_mentions_marker:
            if (
                marker_status == "positive"
                and marker_name in _ACTIONABLE_DRIVER_BIOMARKERS
            ):
                total -= 2
                reasons.append(
                    f"Actionable biomarker caution: patient has '{marker_raw}', "
                    f"but this trial text does not specifically reference "
                    f"{marker_name.upper()}."
                )
            elif marker_status == "negative":
                reasons.append(
                    f"Patient marker '{marker_raw}' is negative and does not "
                    f"appear to conflict with this trial text."
                )
            else:
                reasons.append(
                    f"Biomarker '{marker_raw}' is not specifically referenced in "
                    f"trial text."
                )
            continue

        if marker_status == "negative":
            total -= 4
            reasons.append(
                f"Biomarker caution: patient is {marker_name.upper()}-negative, "
                f"while trial text references {marker_name.upper()}."
            )

        elif marker_status == "positive":
            total += 3
            reasons.append(
                f"Biomarker '{marker_raw}' found in trial text."
            )

        else:
            total += 1
            reasons.append(
                f"Trial text references {marker_name.upper()}, but patient "
                f"biomarker status is unclear."
            )

    raw_total = total
    total = max(min(total, 6), -8)

    if raw_total != total:
        reasons.append(f"Biomarker score clamped from {raw_total} to {total}.")

    return total, reasons


# -----------------------------------------------------------
# GATE FUNCTION
# -----------------------------------------------------------

if TYPE_CHECKING:
    from core.state import EligibilityResult


def is_reasonably_strong_trial(result: "EligibilityResult") -> bool:
    """
    Gate function — determines if a trial result is strong enough
    to surface in final recommendation report.

    Important:
    Generic "biomarker not mentioned" should NOT automatically reject a trial.
    True mismatches like ALK-negative patient vs ALK-positive trial should reject.
    """
    if not result.get("hard_filter_pass"):
        return False

    assessment = (result.get("assessment") or "").upper()

    if "MATCH STATUS: NO MATCH" in assessment:
        return False

    if (result.get("score") or 0) < 1:
        return False

    negative_phrases = [
        "stage mismatch",
        "biomarker mismatch",
        "treatment mismatch",
        "explicitly excluded by this trial",
        "requires treatment-naive patients but patient has prior",
        "exceed trial's maximum",
        "diagnostic, screening, imaging, or biopsy-focused",
        "not clearly matched",
        "not clearly supported",
    ]

    assessment_lower = (result.get("assessment") or "").lower()
    score_reasons_lower = " ".join(result.get("score_reasons") or []).lower()

    if any(p in assessment_lower for p in negative_phrases):
        return False

    if any(p in score_reasons_lower for p in negative_phrases):
        return False

    return True
# -----------------------------------------------------------
# MAIN ORCHESTRATOR — called by evaluate_trials_node
# -----------------------------------------------------------
def score_trial_match(
    patient_data: PatientProfile,
    trial: TrialProfile,
) -> Tuple[int, List[str]]:
    """
    Orchestrates all individual scoring functions and returns a
    combined (score, score_reasons) tuple.

    Called by evaluate_trials_node in agents/nodes.py as:
        score, score_reasons = score_trial_match(patient_data, trial)
    """
    total_score = 0
    all_reasons: List[str] = []

    # 1. Cancer type match
    cancer_score, cancer_reasons = score_cancer_match(patient_data, trial)
    total_score += cancer_score
    all_reasons.extend(cancer_reasons)

    # 2. Stage match
    stage_score, stage_reasons = score_stage_match(patient_data, trial)
    total_score += stage_score
    all_reasons.extend(stage_reasons)

    # 3. Biomarker match
    biomarker_score, biomarker_reasons = score_biomarker_match(patient_data, trial)
    total_score += biomarker_score
    all_reasons.extend(biomarker_reasons)

    # 4. Treatment history match
    treatment_score, treatment_reasons = score_treatment_history_match(patient_data, trial)
    total_score += treatment_score
    all_reasons.extend(treatment_reasons)

    # 5. Location match
    location_score, location_reasons = score_location_match(patient_data, trial)
    total_score += location_score
    all_reasons.extend(location_reasons)

    # 6. Study type
    study_score, study_reasons = score_study_type(patient_data, trial)
    total_score += study_score
    all_reasons.extend(study_reasons)

    # 7. Phase bonus
    phase_score, phase_reasons = score_phase_bonus(patient_data, trial)
    total_score += phase_score
    all_reasons.extend(phase_reasons)

    return total_score, limit_score_reasons(all_reasons)

