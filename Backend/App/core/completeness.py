"""
Profile completeness helpers.
Responsibility: Detect missing fields and generate user-facing suggestions.
"""

from typing import List

from schemas import PatientProfile

_PLACEHOLDER_VALUES = {
    "",
    "unknown",
    "n/a",
    "na",
    "not applicable",
    "none",
    "null",
}

_FIELD_PRIORITY = [
    "cancer_type",
    "cancer_stage",
    "biomarkers",
    "previous_treatments",
    "age",
    "gender",
    "country",
]

_FIELD_SUGGESTIONS = {
    "cancer_type": (
        "Add the exact cancer type, for example non-small cell lung cancer, breast cancer, or colorectal cancer."
    ),
    "cancer_stage": (
        "Add the cancer stage if known, such as Stage II, Stage III, Stage IV, metastatic, recurrent, or locally advanced."
    ),
    "biomarkers": (
        "Include biomarker or mutation results if available, such as EGFR, ALK, ROS1, HER2, BRAF, KRAS, PD-L1, MSI-H, or dMMR."
    ),
    "previous_treatments": (
        "List previous cancer treatments, such as chemotherapy, immunotherapy, targeted therapy, radiation, surgery, or stem cell transplant."
    ),
    "age": (
        "Include the patient's age or date of birth, since many trials have minimum and maximum age requirements."
    ),
    "gender": (
        "Include the patient's sex or gender if known, since some trials have sex-specific eligibility."
    ),
    "country": (
        "Include the patient's country or treatment location preference to improve trial availability matching."
    ),
}


def _is_missing_value(value) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        return value.strip().lower() in _PLACEHOLDER_VALUES

    if isinstance(value, list):
        cleaned = [str(v).strip() for v in value if v is not None and str(v).strip()]
        return len(cleaned) == 0

    return False


def get_missing_fields_in_order(profile: PatientProfile) -> List[str]:
    missing_fields: List[str] = []

    for field in _FIELD_PRIORITY:
        value = getattr(profile, field, None)
        if _is_missing_value(value):
            missing_fields.append(field)

    return missing_fields


def build_improvement_suggestions(missing_fields: List[str]) -> List[str]:
    return [
        _FIELD_SUGGESTIONS[field]
        for field in missing_fields
        if field in _FIELD_SUGGESTIONS
    ]
