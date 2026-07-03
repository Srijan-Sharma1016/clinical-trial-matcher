# agents/nodes.py

import asyncio
import json
import logging
from operator import gt
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from schemas import PatientProfile, TrialProfile, TrialScoringSignals
from core.state import AgentState, EligibilityResult
from core.utils import (
    build_patient_summary,
    build_trial_llm_context,
    resolve_cancer_type_from_structured_data,
    limit_score_reasons,
    get_trial_key,
    normalize_free_text,
    safe_join_text,
)
from core.formatters import (
    format_eligibility_assessment_for_prompt,
    format_hard_filter_no_match_assessment,
    format_not_assessed_assessment,
)
from agents.chains import (
    get_cancer_type_chain,
    get_eligibility_chain,
    get_recommendation_chain,
)
from agents.filters import hard_filter_trial
from agents.scoring import score_trial_match, is_reasonably_strong_trial
from agents.heuristics import (
    heuristic_biomarker_check,
    heuristic_treatment_history_check,
)
# ✅ NO signal_extractor import — extractor lives in this file
from tools.clinicaltrials_tools import search_clinical_trials, get_trial_details
from config.settings import MAX_FETCH_RESULTS, MAX_LLM_EVALUATION_TRIALS

logger = logging.getLogger("uvicorn.error")

__all__ = [
    "extract_cancer_type_node",
    "search_trials_node",
    "evaluate_trials_node",
    "recommend_trials_node",
]

# -----------------------------------------------------------
# SIGNAL EXTRACTION CONSTANTS
# -----------------------------------------------------------

_SIGNAL_EARLY_MARKERS = frozenset([
    "early stage", "early-stage", "localized", "resectable",
    "adjuvant", "neoadjuvant", "stage i", "stage 1",
    "stage ii", "stage 2", "locally confined",
])
_SIGNAL_LOCALLY_ADVANCED_MARKERS = frozenset([
    "locally advanced", "stage iii", "stage 3",
])
_SIGNAL_ADVANCED_MARKERS = frozenset([
    "metastatic", "advanced", "unresectable",
    "recurrent", "relapsed", "stage iv", "stage 4",
])
_SIGNAL_TREATMENT_NAIVE_MARKERS = frozenset([
    "previously untreated", "untreated", "treatment naive",
    "treatment-naive", "no prior treatment", "no previous treatment",
    "first line", "first-line", "1st line",
])

_KNOWN_BIOMARKERS = [
    "her2-positive", "her2 positive", "her2+",
    "her2-low", "her2 low",
    "her2-negative", "her2 negative", "her2-",
    "her2",
    "er-positive", "er positive", "er+",
    "pr-positive", "pr positive", "pr+",
    "hr-positive", "hr positive",
    "triple negative", "tnbc",
    "pd-l1", "pdl1", "pd-1", "pd1",
    "msi-h", "msi high", "microsatellite instability",
    "mmr deficient", "dmmr",
    "brca1", "brca2", "brca",
    "kras", "nras", "braf v600e", "braf",
    "egfr", "alk", "ros1", "met", "ret", "ntrk",
    "idh1", "idh2",
    "fgfr1", "fgfr2", "fgfr3",
    "pik3ca", "pten", "cdh1", "tp53",
    "atm", "chek2", "palb2",
    "bcr-abl", "bcr abl", "philadelphia chromosome",
    "jak2", "npm1", "flt3",
    "cd20", "cd19", "cd30", "cd38",
    "tumor mutational burden", "tmb",
    "homologous recombination deficiency", "hrd",
]
_KNOWN_BIOMARKERS_SORTED = sorted(_KNOWN_BIOMARKERS, key=len, reverse=True)

_PRIOR_LINES_PATTERNS = [
    (r"(?:≤|<=|no more than|at most)\s*(\d+)\s*prior", "max"),
    (r"(?:≥|>=|at least)\s*(\d+)\s*prior", "min"),
    (r"(\d+)\s*or\s*(\d+)\s*prior", "range"),
    (r"^(\d+)\s*prior\s*line", "exact"),
]
_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6,
}
_TREATMENT_TERMS = re.compile(
    r"\b(chemotherapy|immunotherapy|radiotherapy|radiation|"
    r"hormone therapy|endocrine therapy|targeted therapy|"
    r"checkpoint inhibitor|anti-pd-1|anti-pd-l1|"
    r"trastuzumab|pertuzumab|cdk4/6 inhibitor|"
    r"capecitabine|paclitaxel|docetaxel|carboplatin|"
    r"cisplatin|oxaliplatin|bevacizumab|pembrolizumab|"
    r"nivolumab|atezolizumab|olaparib|niraparib)\b",
    re.IGNORECASE,
)


# -----------------------------------------------------------
# SIGNAL EXTRACTION HELPERS
# -----------------------------------------------------------

def _parse_prior_lines(
    text: str,
) -> Tuple[Optional[int], Optional[int]]:
    min_lines: Optional[int] = None
    max_lines: Optional[int] = None
    for word, num in _WORD_TO_NUM.items():
        text = text.replace(word, str(num))
    for pattern, kind in _PRIOR_LINES_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        if kind == "max":
            max_lines = int(match.group(1))
        elif kind == "min":
            min_lines = int(match.group(1))
        elif kind == "range":
            min_lines = int(match.group(1))
            max_lines = int(match.group(2))
        elif kind == "exact":
            max_lines = int(match.group(1))
    return min_lines, max_lines


def _canonical_biomarker_marker(marker: str) -> str:
    """
    Normalizes biomarker aliases used by the signal extractor.
    """
    marker = (marker or "").strip().lower()

    aliases = {
        "pdl1": "pd-l1",
        "pd l1": "pd-l1",
        "erbb2": "her2",
    }

    return aliases.get(marker, marker)


def _sentence_has_negative_marker_context(sentence: str, marker: str) -> bool:
    """
    Detects whether a biomarker mention is negative/exclusionary.

    Handles patterns like:
        ALK-
        ALK negative
        ALK-negative
        negative for ALK
        without ALK
        absence of ALK
        no EGFR mutation
    """
    sentence = sentence.lower()
    marker = marker.lower()

    negative_patterns = [
        rf"\b{re.escape(marker)}\s*-\b",
        rf"\b{re.escape(marker)}[-\s]?negative\b",
        rf"\bnegative\s+for\s+{re.escape(marker)}\b",
        rf"\bwithout\s+{re.escape(marker)}\b",
        rf"\babsence\s+of\s+{re.escape(marker)}\b",
        rf"\bno\s+{re.escape(marker)}\b",
        rf"\bnot\s+{re.escape(marker)}\b",
        rf"\bexclude[s]?\s+.*\b{re.escape(marker)}\b",
        rf"\bexcluded\s+.*\b{re.escape(marker)}\b",
        rf"\bmust\s+not\s+have\s+.*\b{re.escape(marker)}\b",
        rf"\bnon[-\s]?{re.escape(marker)}\b",
        rf"\b{re.escape(marker)}\s+wild[-\s]?type\b",
        rf"\bwild[-\s]?type\s+{re.escape(marker)}\b",
    ]

    return any(re.search(pattern, sentence) for pattern in negative_patterns)


def _sentence_has_positive_marker_context(sentence: str, marker: str) -> bool:
    """
    Detects whether a biomarker mention is positive/required.
    """
    sentence = sentence.lower()
    marker = marker.lower()

    positive_patterns = [
        rf"\b{re.escape(marker)}\s*\+\b",
        rf"\b{re.escape(marker)}[-\s]?positive\b",
        rf"\bpositive\s+for\s+{re.escape(marker)}\b",
        rf"\b{re.escape(marker)}\s+mutation\b",
        rf"\b{re.escape(marker)}[-\s]?mutated\b",
        rf"\b{re.escape(marker)}\s+exon\b",
        rf"\b{re.escape(marker)}\s+fusion\b",
        rf"\b{re.escape(marker)}[-\s]?rearranged\b",
        rf"\b{re.escape(marker)}\s+rearrangement\b",
        rf"\b{re.escape(marker)}\s+amplification\b",
        rf"\b{re.escape(marker)}[-\s]?amplified\b",
    ]

    return any(re.search(pattern, sentence) for pattern in positive_patterns)


def _extract_biomarkers_from_text(
    text: str,
) -> Tuple[List[str], List[str]]:
    """
    Extracts required/excluded biomarker signals from trial text.

    Fixes previous issue where:
        ALK-negative / without ALK / no ALK
    could be incorrectly classified as a required ALK biomarker.
    """
    required: List[str] = []
    excluded: List[str] = []

    if not text:
        return required, excluded

    normalized_text = normalize_free_text(text)

    # Split into rough local contexts so negation is evaluated near the marker.
    sentences = re.split(
        r"[\n\r.;:•]+|\b(?:inclusion criteria|exclusion criteria)\b",
        normalized_text,
    )

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        for marker in _KNOWN_BIOMARKERS_SORTED:
            marker_norm = normalize_free_text(marker)
            canonical = _canonical_biomarker_marker(marker_norm)

            if marker_norm not in sentence:
                continue

            is_negative = _sentence_has_negative_marker_context(
                sentence,
                marker_norm,
            )
            is_positive = _sentence_has_positive_marker_context(
                sentence,
                marker_norm,
            )

            if is_negative:
                if canonical not in excluded:
                    excluded.append(canonical)
                continue

            if is_positive:
                if canonical not in required:
                    required.append(canonical)
                continue

            # Default: if biomarker is mentioned in trial title/criteria
            # without negation, treat as required/biomarker-relevant.
            if canonical not in required:
                required.append(canonical)

    return required, excluded


def _extract_cancer_types(trial: TrialProfile) -> List[str]:
    cancer_types: List[str] = []
    for condition in trial.conditions:
        normalized = normalize_free_text(condition)
        if normalized and normalized not in cancer_types:
            cancer_types.append(normalized)
    for mesh in trial.mesh_terms:
        normalized = normalize_free_text(mesh)
        if normalized and normalized not in cancer_types:
            cancer_types.append(normalized)
    title_text = normalize_free_text(
        safe_join_text([trial.title, trial.official_title])
    )
    cancer_noun_pattern = re.compile(
        r"([\w\s\-]+(?:cancer|carcinoma|tumor|tumour|lymphoma|"
        r"leukemia|leukaemia|melanoma|sarcoma|glioma|myeloma|"
        r"adenocarcinoma|blastoma|mesothelioma))",
        re.IGNORECASE,
    )
    for match in cancer_noun_pattern.finditer(title_text):
        term = match.group(1).strip().lower()
        if term and term not in cancer_types:
            cancer_types.append(term)
    return cancer_types


def _infer_target_setting_from_trial(trial: TrialProfile) -> str:
    """
    Infers trial disease setting safely.

    Important fix:
    Avoid substring bugs like:
        "stage i" matching inside "stage iv"

    Priority:
        1. Strong advanced/metastatic signals
        2. Locally advanced signals
        3. Generic advanced signals
        4. Early-stage signals
    """
    eligibility = getattr(trial, "eligibility", None)

    priority_text = normalize_free_text(
        safe_join_text([
            trial.title,
            trial.official_title,
            " ".join(trial.conditions or []),
            trial.brief_summary,
        ])
    )

    secondary_text = normalize_free_text(
        safe_join_text([
            trial.detailed_description,
            getattr(eligibility, "study_population", "") if eligibility else "",
            getattr(eligibility, "criteria_text", "") if eligibility else "",
        ])
    )

    advanced_strong_re = re.compile(
        r"\b(stage\s*(iv|4)|metastatic|unresectable|recurrent|relapsed)\b",
        re.IGNORECASE,
    )

    locally_advanced_re = re.compile(
        r"\b(locally\s+advanced|stage\s*(iii|3))\b",
        re.IGNORECASE,
    )

    generic_advanced_re = re.compile(
        r"\badvanced\b",
        re.IGNORECASE,
    )

    early_re = re.compile(
        r"\b("
        r"early[-\s]?stage|localized|resectable|adjuvant|neoadjuvant|"
        r"stage\s*(i|1|ii|2)|locally\s+confined"
        r")\b",
        re.IGNORECASE,
    )

    for text in [priority_text, secondary_text]:
        if not text:
            continue

        if advanced_strong_re.search(text):
            return "advanced"

        if locally_advanced_re.search(text):
            return "locally_advanced"

        if generic_advanced_re.search(text):
            return "advanced"

        if early_re.search(text):
            return "early"

    return "unknown"




def _normalize_phase(phases: List[str]) -> Optional[str]:
    if not phases:
        return None
    phase_order = {"PHASE4": 4, "PHASE3": 3, "PHASE2": 2, "PHASE1": 1}
    normalized = [p.upper().replace(" ", "") for p in phases]
    ranked = [p for p in normalized if p in phase_order]
    if ranked:
        return max(ranked, key=lambda p: phase_order[p])
    return normalized[0] if normalized else None


def extract_trial_scoring_signals(trial: TrialProfile) -> TrialScoringSignals:
    """
    Parses a TrialProfile into a structured TrialScoringSignals object.
    Called by _attach_scoring_signals() inside search_trials_node.
    """
    eligibility_text = normalize_free_text(
        trial.eligibility.criteria_text or ""
    )
    full_trial_text = normalize_free_text(
        safe_join_text([
            trial.title,
            trial.official_title,
            " ".join(trial.conditions),
            trial.brief_summary,
            trial.detailed_description,
            trial.eligibility.criteria_text,
            trial.eligibility.study_population,
            " ".join(trial.mesh_terms),
        ])
    )
    required_biomarkers, excluded_biomarkers = _extract_biomarkers_from_text(
        full_trial_text
    )
    min_prior_lines, max_prior_lines = _parse_prior_lines(eligibility_text)
    requires_treatment_naive = any(
        marker in eligibility_text
        for marker in _SIGNAL_TREATMENT_NAIVE_MARKERS
    )
    if requires_treatment_naive:
        max_prior_lines = 0

    excluded_treatments: List[str] = []
    exclusion_section = ""
    if "exclusion criteria" in eligibility_text:
        idx = eligibility_text.find("exclusion criteria")
        exclusion_section = eligibility_text[idx:]
    if exclusion_section:
        for match in _TREATMENT_TERMS.finditer(exclusion_section):
            term = match.group(1).strip().lower()
            if term not in excluded_treatments:
                excluded_treatments.append(term)

    return TrialScoringSignals(
        required_biomarkers=required_biomarkers,
        excluded_biomarkers=excluded_biomarkers,
        excluded_treatments=excluded_treatments,
        min_prior_lines=min_prior_lines,
        max_prior_lines=max_prior_lines,
        requires_treatment_naive=requires_treatment_naive,
        target_setting=_infer_target_setting_from_trial(trial),
        required_cancer_types=_extract_cancer_types(trial),
        trial_phase=_normalize_phase(trial.phases),
        is_interventional=(trial.study_type or "").upper() == "INTERVENTIONAL",
    )


# -----------------------------------------------------------
# INTERNAL HELPERS
# -----------------------------------------------------------

def _compute_heuristics(
    patient_data: PatientProfile,
    trial: TrialProfile,
) -> Tuple[str, str]:
    """
    Computes biomarker and treatment heuristic checks for a trial.
    Returns (biomarker_check, treatment_check).
    """
    criteria = trial.eligibility.criteria_text or ""
    return (
        heuristic_biomarker_check(patient_data.biomarkers or [], criteria),
        heuristic_treatment_history_check(
            patient_data.previous_treatments or [], criteria
        ),
    )


async def _fetch_trial_with_delay(
    trial_id: str,
    delay: float,
) -> Union[TrialProfile, Dict[str, Any]]:
    """
    Fetches trial details with a staggered delay.
    Prevents ConnectionAbortedError from parallel requests
    overwhelming the ClinicalTrials.gov API.
    """
    await asyncio.sleep(delay)
    return await get_trial_details(trial_id)


def _attach_scoring_signals(trials: List[TrialProfile]) -> List[TrialProfile]:
    """
    Runs extract_trial_scoring_signals() on every trial in the list
    and attaches the result to trial.scoring_signals.

    - Runs synchronously (signal extraction is pure CPU/regex — no I/O).
    - Skips trials that already have signals attached (idempotent).
    - Logs a warning and continues on any per-trial failure so one
      bad trial never aborts the whole batch.

    Called once in search_trials_node after deduplication,
    before trials enter evaluate_trials_node.
    """
    for trial in trials:
        # Idempotency guard — don't re-extract if already populated
        if trial.scoring_signals is not None:
            continue
        try:
            trial.scoring_signals = extract_trial_scoring_signals(trial)
        except Exception:
            logger.warning(
                "Signal extraction failed for trial '%s' — "
                "scoring will fall back to raw text scanning.",
                trial.trial_id,
                exc_info=True,
            )
            # Leave scoring_signals as None — scoring.py handles None gracefully
    return trials
def _match_tier(score: int) -> str:
    """
    Converts deterministic score into a display tier.
    Keep these thresholds aligned with frontend labels.
    """
    if score >= 8:
        return "Strong"
    if score >= 5:
        return "Moderate"
    if score >= 1:
        return "Possible"
    return "Weak"


def _is_displayable_recommendation_result(result: EligibilityResult) -> bool:
    """
    Determines whether a result should be considered in the final recommendation
    summary.

    This is intentionally less strict than is_reasonably_strong_trial().
    The final summary should match the displayed ranked cards, including
    Possible matches, while still excluding obvious no-match items.
    """
    if not result.get("hard_filter_pass"):
        return False

    score = result.get("score") or 0
    if score < 1:
        return False

    assessment = (result.get("assessment") or "").upper()
    if "MATCH STATUS: NO MATCH" in assessment:
        return False

    return True


def _rank_recommendation_results(
    eligibility_results: List[EligibilityResult],
) -> List[EligibilityResult]:
    """
    Returns the same kind of ranked list the UI should summarize:
    hard-filter passing, score >= 1, sorted by score descending.
    """
    return sorted(
        [
            r for r in eligibility_results
            if _is_displayable_recommendation_result(r)
        ],
        key=lambda r: r.get("score", 0),
        reverse=True,
    )


def _recommendation_mentions_unknown_trial(
    recommendation: str,
    allowed_results: List[EligibilityResult],
) -> bool:
    """
    Guards against LLM summaries mentioning trials outside the ranked list.
    """
    if not recommendation:
        return False

    mentioned_ids = set(re.findall(r"NCT\d{8}", recommendation.upper()))
    if not mentioned_ids:
        return False

    allowed_ids = {
        (r.get("nct_id") or "").upper()
        for r in allowed_results
        if r.get("nct_id")
    }

    return bool(mentioned_ids - allowed_ids)


def _recommendation_contradicts_strong_matches(
    recommendation: str,
    ranked_results: List[EligibilityResult],
) -> bool:
    """
    Guards against summaries saying 'no strong matches' when strong matches exist.
    """
    if not recommendation:
        return False

    strong_count = sum(
        1 for r in ranked_results
        if (r.get("score") or 0) >= 8
    )

    if strong_count <= 0:
        return False

    text = recommendation.lower()

    contradiction_phrases = [
        "no strong matches",
        "no strong match",
        "no strong matches were found",
        "no strong match was found",
    ]

    return any(phrase in text for phrase in contradiction_phrases)


def _fallback_recommendation_text(
    patient_data: PatientProfile,
    eligibility_results: List[EligibilityResult],
) -> str:
    """
    Deterministic recommendation summary.

    This is used when:
    - no LLM recommendation is needed,
    - LLM recommendation fails,
    - LLM recommendation contradicts deterministic ranked results.

    It summarizes the same ranked results shown to the user.
    """
    ranked_results = _rank_recommendation_results(eligibility_results)

    cancer_label = (
        patient_data.cancer_type
        or patient_data.diagnosis
        or "Not available"
    )

    strong_count = sum(
        1 for r in ranked_results
        if (r.get("score") or 0) >= 8
    )
    moderate_count = sum(
        1 for r in ranked_results
        if 5 <= (r.get("score") or 0) < 8
    )
    possible_count = sum(
        1 for r in ranked_results
        if 1 <= (r.get("score") or 0) < 5
    )

    lines = [
        "SUMMARY:",
        (
            f"The patient profile was evaluated against available clinical trials "
            f"for {cancer_label}. "
            f"The ranked results include {strong_count} strong, "
            f"{moderate_count} moderate, and {possible_count} possible matches."
        ),
        "",
    ]

    if not ranked_results:
        lines.extend([
            "TOP MATCHES:",
            "No suitable trial matches were identified among the evaluated trials.",
            "",
            "NEXT STEPS:",
            "- Reconfirm cancer stage, biomarker status, prior treatments, and location.",
            "- Consider broadening the search terms or trial locations.",
            "- Ask a qualified oncologist or trial coordinator to review borderline options.",
            "",
            "IMPORTANT DISCLAIMER:",
            "These results are AI-assisted screening support only and do not constitute medical advice. "
            "A qualified oncologist should review the full protocol before any clinical decision.",
        ])
        return "\n".join(lines)

    lines.append("TOP MATCHES:")

    for idx, result in enumerate(ranked_results[:4], start=1):
        score = result.get("score") or 0
        tier = _match_tier(score)
        title = result.get("title") or "Unknown trial"
        nct_id = result.get("nct_id") or "N/A"
        reasons = result.get("score_reasons") or []

        key_reasons = reasons[:3]

        lines.extend([
            "",
            f"{idx}. {title} ({nct_id})",
            f"   - Match tier: {tier}",
            f"   - Deterministic score: {score}",
        ])

        if key_reasons:
            lines.append("   - Key reasons:")
            for reason in key_reasons:
                lines.append(f"     • {reason}")
        else:
            lines.append("   - Key reasons: No deterministic reasons available.")

    lines.extend([
        "",
        "NEXT STEPS:",
        "- Review the top-ranked trials with a qualified oncologist.",
        "- Confirm biomarker status, stage, prior treatment history, performance status, and site availability.",
        "- Contact trial sites or coordinators to verify final eligibility.",
        "",
        "IMPORTANT DISCLAIMER:",
        "These results are AI-generated for informational and screening support only. "
        "They do not constitute medical advice or a final eligibility determination. "
        "Always consult a qualified oncologist before making treatment decisions.",
    ])

    return "\n".join(lines)



# -----------------------------------------------------------
# INTERNAL LLM EVALUATOR
# -----------------------------------------------------------

async def _evaluate_trial_with_llm(
    patient_data: PatientProfile,
    trial: TrialProfile,
    hard_filter_pass: bool,
    hard_filter_reasons: List[str],
    score: int,
    score_reasons: List[str],
) -> EligibilityResult:
    """
    Calls the eligibility LLM chain for a single trial.
    Falls back to format_not_assessed_assessment() on failure.
    """
    biomarker_check, treatment_check = _compute_heuristics(patient_data, trial)

    try:
        assessment = await get_eligibility_chain().ainvoke({
            "patient_profile": build_patient_summary(patient_data),
            "trial_details": json.dumps(
                build_trial_llm_context(trial),
                indent=2,
                ensure_ascii=False,
            ),
            "trial_score": score,
            "score_reasons": (
                "\n".join(f"- {r}" for r in score_reasons)
                or "- No deterministic reasons available."
            ),
            "biomarker_check": biomarker_check,
            "treatment_check": treatment_check,
        })
    except Exception:
        logger.exception(
            "LLM eligibility evaluation failed for trial '%s'", trial.trial_id
        )
        assessment = format_not_assessed_assessment(score, score_reasons)

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


# -----------------------------------------------------------
# GRAPH NODES
# -----------------------------------------------------------

async def extract_cancer_type_node(state: AgentState) -> AgentState:
    logger.info("Node 1: Extracting cancer type...")
    try:
        patient_data = state["patient_data"]
        structured_value = resolve_cancer_type_from_structured_data(patient_data)

        if structured_value:
            logger.info(
                "Cancer type resolved from structured data: '%s'",
                structured_value,
            )
            updated_patient = patient_data.model_copy(
                update={"cancer_type": structured_value}
            )
            return {
                **state,
                "patient_data": updated_patient,
                "cancer_type": structured_value,
            }

        patient_summary = build_patient_summary(patient_data)
        cancer_type = await get_cancer_type_chain().ainvoke(
            {"patient_data": patient_summary}
        )
        cancer_type = cancer_type.strip()

        if not cancer_type or cancer_type.upper() == "UNKNOWN":
            return {
                **state,
                "error": "Unable to determine cancer type from patient profile.",
            }

        updated_patient = patient_data.model_copy(
            update={"cancer_type": cancer_type}
        )
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
            return {**state, "trials_raw": []}

        # Staggered detail fetches — 0.5s apart to avoid ConnectionAbortedError
        detail_tasks = [
            _fetch_trial_with_delay(trial.trial_id, i * 0.5)
            for i, trial in enumerate(trials)
            if trial.trial_id
        ]

        enriched_trials: List[TrialProfile] = []

        if detail_tasks:
            detail_results = await asyncio.gather(
                *detail_tasks, return_exceptions=True
            )
            fallback_trials = [t for t in trials if t.trial_id]

            for fallback_trial, detail_result in zip(fallback_trials, detail_results):
                if isinstance(detail_result, TrialProfile):
                    enriched_trials.append(detail_result)
                else:
                    if isinstance(detail_result, Exception):
                        logger.warning(
                            "Trial detail fetch exception for '%s': %s",
                            fallback_trial.trial_id,
                            str(detail_result),
                        )
                    elif (
                        isinstance(detail_result, dict)
                        and detail_result.get("error")
                    ):
                        logger.warning(
                            "Trial detail fetch error for '%s': %s",
                            fallback_trial.trial_id,
                            detail_result["error"],
                        )
                    enriched_trials.append(fallback_trial)

            enriched_trials.extend([t for t in trials if not t.trial_id])
        else:
            enriched_trials = trials

        # Deduplicate by trial key
        deduped: Dict[str, TrialProfile] = {}
        for trial in enriched_trials:
            deduped[get_trial_key(trial)] = trial

        final_trials = list(deduped.values())

        # -------------------------------------------------------
        # NEW — Phase 2: Attach scoring signals to every trial
        # Must happen AFTER deduplication and BEFORE evaluate_trials_node
        # so scoring.py always has structured signals available.
        # -------------------------------------------------------
        final_trials = _attach_scoring_signals(final_trials)

        logger.info(
            "Prepared %d trial records for evaluation "
            "(%d with scoring signals attached).",
            len(final_trials),
            sum(1 for t in final_trials if t.scoring_signals is not None),
        )
        return {**state, "trials_raw": final_trials}

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
            return {**state, "eligibility_results": []}

        # Score and rank all trials
        ranked_trials: List[Dict[str, Any]] = []
        for trial in trials:
            hard_filter_pass, hard_filter_reasons = hard_filter_trial(
                patient_data, trial
            )
            # NEW — score_trial_match now receives trial.scoring_signals
            # via the trial object itself — no API change needed here
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

        # Select top N passing trials for LLM evaluation
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
                key = get_trial_key(trial)

                if isinstance(llm_output, Exception):
                    logger.exception(
                        "Unhandled LLM exception for trial '%s'",
                        trial.trial_id,
                    )
                    biomarker_check, treatment_check = _compute_heuristics(
                        patient_data, trial
                    )
                    llm_results_by_key[key] = EligibilityResult(
                        nct_id=trial.trial_id,
                        title=trial.trial_id,
                        hard_filter_pass=item["hard_filter_pass"],
                        hard_filter_reasons=item["hard_filter_reasons"],
                        score=item["score"],
                        score_reasons=item["score_reasons"],
                        biomarker_check=biomarker_check,
                        treatment_check=treatment_check,
                        assessment=format_not_assessed_assessment(
                            item["score"], item["score_reasons"]
                        ),
                    )
                else:
                    llm_results_by_key[key] = llm_output

        # Build final results list preserving rank order
        final_results: List[EligibilityResult] = []
        for item in ranked_trials:
            trial = item["trial"]
            key = get_trial_key(trial)
            biomarker_check, treatment_check = _compute_heuristics(
                patient_data, trial
            )

            if not item["hard_filter_pass"]:
                final_results.append(EligibilityResult(
                    nct_id=trial.trial_id,
                    title=trial.title,
                    hard_filter_pass=False,
                    hard_filter_reasons=item["hard_filter_reasons"],
                    score=item["score"],
                    score_reasons=item["score_reasons"],
                    biomarker_check=biomarker_check,
                    treatment_check=treatment_check,
                    assessment=format_hard_filter_no_match_assessment(
                        item["hard_filter_reasons"]
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
                    biomarker_check=biomarker_check,
                    treatment_check=treatment_check,
                    assessment=format_not_assessed_assessment(
                        item["score"], item["score_reasons"]
                    ),
                ))

        logger.info("Generated %d eligibility results.", len(final_results))
        return {**state, "eligibility_results": final_results}

    except Exception:
        logger.exception("Failed in evaluate_trials_node")
        return {
            **state,
            "error": "Unexpected error while evaluating trial eligibility.",
            "eligibility_results": [],
        }


async def recommend_trials_node(state: AgentState) -> AgentState:
    logger.info("Node 4: Generating final recommendations...")

    if state.get("error"):
        return {
            **state,
            "final_recommendations": (
                f"Trial matching could not be completed: {state['error']}"
            ),
        }

    try:
        patient_data = state["patient_data"]
        eligibility_results = state.get("eligibility_results") or []

        if not eligibility_results:
            return {
                **state,
                "final_recommendations": (
                    "No recruiting clinical trials were found or no trials could "
                    "be evaluated for the current patient profile. Consider broadening "
                    "the search terms, locations, or reviewing missing patient details."
                ),
            }

        ranked_results = _rank_recommendation_results(eligibility_results)

        if not ranked_results:
            return {
                **state,
                "final_recommendations": _fallback_recommendation_text(
                    patient_data,
                    eligibility_results,
                ),
            }

        # Deterministic summary is our safety net.
        deterministic_summary = _fallback_recommendation_text(
            patient_data,
            ranked_results,
        )

        # Use top 4 ranked/displayable trials so recommendation aligns with UI cards.
        top_results = ranked_results[:4]

        assessment_text = "\n\n".join(
            format_eligibility_assessment_for_prompt(r)
            for r in top_results
        )

        try:
            recommendation = await get_recommendation_chain().ainvoke({
                "patient_profile": build_patient_summary(patient_data),
                "eligibility_assessments": assessment_text.strip(),
            })

            recommendation = (recommendation or "").strip()

            # ---------------------------------------------------
            # Guardrails against inconsistent LLM summaries
            # ---------------------------------------------------
            if not recommendation:
                logger.warning(
                    "LLM recommendation was empty; using deterministic summary."
                )
                recommendation = deterministic_summary

            elif _recommendation_mentions_unknown_trial(
                recommendation,
                top_results,
            ):
                logger.warning(
                    "LLM recommendation mentioned trials outside top ranked results; "
                    "using deterministic summary."
                )
                recommendation = deterministic_summary

            elif _recommendation_contradicts_strong_matches(
                recommendation,
                top_results,
            ):
                logger.warning(
                    "LLM recommendation contradicted strong deterministic matches; "
                    "using deterministic summary."
                )
                recommendation = deterministic_summary

            else:
                # Extra soft guard: if top result has an NCT ID, the summary should mention it.
                top_nct_id = (top_results[0].get("nct_id") or "").upper()
                if top_nct_id and top_nct_id not in recommendation.upper():
                    logger.warning(
                        "LLM recommendation did not mention top-ranked trial %s; "
                        "using deterministic summary.",
                        top_nct_id,
                    )
                    recommendation = deterministic_summary

        except Exception:
            logger.exception("LLM recommendation generation failed")
            recommendation = deterministic_summary

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
