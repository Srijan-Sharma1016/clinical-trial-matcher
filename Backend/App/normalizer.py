import logging
import re
from typing import Any, Dict, List, Optional

from schemas import PatientProfile, TrialProfile, TrialEligibility, TrialLocation

logger = logging.getLogger("uvicorn.error")

# -----------------------------------------------------------
# GENERIC TEXT HELPERS
# -----------------------------------------------------------

def canonicalize_key(value: str) -> str:
    """
    Normalize text for dictionary lookup.

    Examples:
    - 'HER-2 Positive' -> 'her 2 positive'
    - 'PD-L1' -> 'pd l1'
    - ' Non-Small Cell Lung Cancer ' -> 'non small cell lung cancer'
    """
    value = value.strip().lower()
    value = value.replace("_", " ")
    value = value.replace("-", " ")
    value = re.sub(r"[^\w\s/+]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def clean_text(value: Optional[str]) -> Optional[str]:
    """
    Strip whitespace and return None for empty values.
    """
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned if cleaned else None


def dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []

    for value in values:
        key = value.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(value)

    return result


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


# -----------------------------------------------------------
# NORMALIZATION MAPS
# -----------------------------------------------------------

CANCER_TYPE_MAP = {
    canonicalize_key("nsclc"): "Carcinoma, Non-Small-Cell Lung",
    canonicalize_key("non small cell lung cancer"): "Carcinoma, Non-Small-Cell Lung",
    canonicalize_key("nsclc"): "Carcinoma, Non-Small-Cell Lung",
    canonicalize_key("non small cell lung cancer"): "Carcinoma, Non-Small-Cell Lung",
    canonicalize_key("non-small cell lung cancer"): "Carcinoma, Non-Small-Cell Lung",
    canonicalize_key("non small cell lung carcinoma"): "Carcinoma, Non-Small-Cell Lung",
    canonicalize_key("sclc"): "Carcinoma, Small Cell",
    canonicalize_key("small cell lung cancer"): "Carcinoma, Small Cell",
    canonicalize_key("small cell carcinoma"): "Carcinoma, Small Cell",
    canonicalize_key("lung cancer"): "Lung Neoplasms",
    canonicalize_key("lung carcinoma"): "Lung Neoplasms",
    canonicalize_key("ca lung"): "Lung Neoplasms",
    canonicalize_key("lung tumor"): "Lung Neoplasms",
    canonicalize_key("breast cancer"): "Breast Neoplasms",
    canonicalize_key("ca breast"): "Breast Neoplasms",
    canonicalize_key("breast carcinoma"): "Breast Neoplasms",
    canonicalize_key("breast tumor"): "Breast Neoplasms",
    canonicalize_key("her2 positive breast cancer"): "Breast Neoplasms",
    canonicalize_key("triple negative breast cancer"): "Triple Negative Breast Neoplasms",
    canonicalize_key("tnbc"): "Triple Negative Breast Neoplasms",
    canonicalize_key("colorectal cancer"): "Colorectal Neoplasms",
    canonicalize_key("colon cancer"): "Colonic Neoplasms",
    canonicalize_key("rectal cancer"): "Rectal Neoplasms",
    canonicalize_key("crc"): "Colorectal Neoplasms",
    canonicalize_key("bowel cancer"): "Colorectal Neoplasms",

    canonicalize_key("leukemia"): "Leukemia",
    canonicalize_key("aml"): "Leukemia, Myeloid, Acute",
    canonicalize_key("acute myeloid leukemia"): "Leukemia, Myeloid, Acute",
    canonicalize_key("cml"): "Leukemia, Myelogenous, Chronic, BCR-ABL Positive",
    canonicalize_key("chronic myeloid leukemia"): "Leukemia, Myelogenous, Chronic, BCR-ABL Positive",
    canonicalize_key("all"): "Precursor Cell Lymphoblastic Leukemia-Lymphoma",
    canonicalize_key("acute lymphoblastic leukemia"): "Precursor Cell Lymphoblastic Leukemia-Lymphoma",
    canonicalize_key("lymphoma"): "Lymphoma",
    canonicalize_key("hodgkin lymphoma"): "Hodgkin Disease",
    canonicalize_key("hodgkins lymphoma"): "Hodgkin Disease",
    canonicalize_key("non hodgkin lymphoma"): "Lymphoma, Non-Hodgkin",
    canonicalize_key("nhl"): "Lymphoma, Non-Hodgkin",
    canonicalize_key("multiple myeloma"): "Multiple Myeloma",
    canonicalize_key("mm"): "Multiple Myeloma",

    canonicalize_key("glioblastoma"): "Glioblastoma",
    canonicalize_key("gbm"): "Glioblastoma",
    canonicalize_key("glioma"): "Glioma",
    canonicalize_key("brain cancer"): "Brain Neoplasms",
    canonicalize_key("brain tumor"): "Brain Neoplasms",
    canonicalize_key("meningioma"): "Meningioma",

    canonicalize_key("gastric cancer"): "Stomach Neoplasms",
    canonicalize_key("stomach cancer"): "Stomach Neoplasms",
    canonicalize_key("pancreatic cancer"): "Pancreatic Neoplasms",
    canonicalize_key("pancreas cancer"): "Pancreatic Neoplasms",
    canonicalize_key("liver cancer"): "Liver Neoplasms",
    canonicalize_key("hepatocellular carcinoma"): "Carcinoma, Hepatocellular",
    canonicalize_key("hcc"): "Carcinoma, Hepatocellular",
    canonicalize_key("esophageal cancer"): "Esophageal Neoplasms",
    canonicalize_key("cholangiocarcinoma"): "Cholangiocarcinoma",
    canonicalize_key("bile duct cancer"): "Cholangiocarcinoma",

    canonicalize_key("prostate cancer"): "Prostatic Neoplasms",
    canonicalize_key("ovarian cancer"): "Ovarian Neoplasms",
    canonicalize_key("cervical cancer"): "Uterine Cervical Neoplasms",
    canonicalize_key("endometrial cancer"): "Endometrial Neoplasms",
    canonicalize_key("uterine cancer"): "Uterine Neoplasms",

    canonicalize_key("melanoma"): "Melanoma",
    canonicalize_key("skin cancer"): "Skin Neoplasms",
    canonicalize_key("basal cell carcinoma"): "Carcinoma, Basal Cell",
    canonicalize_key("squamous cell carcinoma"): "Carcinoma, Squamous Cell",
    canonicalize_key("scc"): "Carcinoma, Squamous Cell",

    canonicalize_key("thyroid cancer"): "Thyroid Neoplasms",
    canonicalize_key("bladder cancer"): "Urinary Bladder Neoplasms",
    canonicalize_key("kidney cancer"): "Kidney Neoplasms",
    canonicalize_key("renal cell carcinoma"): "Carcinoma, Renal Cell",
    canonicalize_key("rcc"): "Carcinoma, Renal Cell",
    canonicalize_key("head and neck cancer"): "Head and Neck Neoplasms",
    canonicalize_key("sarcoma"): "Sarcoma",
    canonicalize_key("mesothelioma"): "Mesothelioma",
    canonicalize_key("neuroblastoma"): "Neuroblastoma",
    canonicalize_key("retinoblastoma"): "Retinoblastoma",
    canonicalize_key("non small cell lung cancer nsclc"): "Carcinoma, Non-Small-Cell Lung",

     # --- ADD THIS SECTION AT THE BOTTOM OF CANCER_TYPE_MAP ---
    # Idempotency entries — MeSH terms map to themselves
    # Prevents double-normalization warnings when already-normalized
    # values are passed through the normalizer a second time

    canonicalize_key("Carcinoma, Non-Small-Cell Lung"): "Carcinoma, Non-Small-Cell Lung",
    canonicalize_key("Carcinoma, Small Cell"): "Carcinoma, Small Cell",
    canonicalize_key("Carcinoma, Hepatocellular"): "Carcinoma, Hepatocellular",
    canonicalize_key("Carcinoma, Renal Cell"): "Carcinoma, Renal Cell",
    canonicalize_key("Carcinoma, Basal Cell"): "Carcinoma, Basal Cell",
    canonicalize_key("Carcinoma, Squamous Cell"): "Carcinoma, Squamous Cell",
    canonicalize_key("Breast Neoplasms"): "Breast Neoplasms",
    canonicalize_key("Triple Negative Breast Neoplasms"): "Triple Negative Breast Neoplasms",
    canonicalize_key("Lung Neoplasms"): "Lung Neoplasms",
    canonicalize_key("Colorectal Neoplasms"): "Colorectal Neoplasms",
    canonicalize_key("Colonic Neoplasms"): "Colonic Neoplasms",
    canonicalize_key("Rectal Neoplasms"): "Rectal Neoplasms",
    canonicalize_key("Prostatic Neoplasms"): "Prostatic Neoplasms",
    canonicalize_key("Ovarian Neoplasms"): "Ovarian Neoplasms",
    canonicalize_key("Stomach Neoplasms"): "Stomach Neoplasms",
    canonicalize_key("Pancreatic Neoplasms"): "Pancreatic Neoplasms",
    canonicalize_key("Brain Neoplasms"): "Brain Neoplasms",
    canonicalize_key("Liver Neoplasms"): "Liver Neoplasms",
    canonicalize_key("Skin Neoplasms"): "Skin Neoplasms",
    canonicalize_key("Thyroid Neoplasms"): "Thyroid Neoplasms",
    canonicalize_key("Kidney Neoplasms"): "Kidney Neoplasms",
    canonicalize_key("Urinary Bladder Neoplasms"): "Urinary Bladder Neoplasms",
    canonicalize_key("Head and Neck Neoplasms"): "Head and Neck Neoplasms",
    canonicalize_key("Uterine Cervical Neoplasms"): "Uterine Cervical Neoplasms",
    canonicalize_key("Endometrial Neoplasms"): "Endometrial Neoplasms",
    canonicalize_key("Uterine Neoplasms"): "Uterine Neoplasms",
    canonicalize_key("Esophageal Neoplasms"): "Esophageal Neoplasms",
    canonicalize_key("Leukemia"): "Leukemia",
    canonicalize_key("Leukemia, Myeloid, Acute"): "Leukemia, Myeloid, Acute",
    canonicalize_key("Leukemia, Myelogenous, Chronic, BCR-ABL Positive"): "Leukemia, Myelogenous, Chronic, BCR-ABL Positive",
    canonicalize_key("Precursor Cell Lymphoblastic Leukemia-Lymphoma"): "Precursor Cell Lymphoblastic Leukemia-Lymphoma",
    canonicalize_key("Lymphoma"): "Lymphoma",
    canonicalize_key("Hodgkin Disease"): "Hodgkin Disease",
    canonicalize_key("Lymphoma, Non-Hodgkin"): "Lymphoma, Non-Hodgkin",
    canonicalize_key("Multiple Myeloma"): "Multiple Myeloma",
    canonicalize_key("Glioblastoma"): "Glioblastoma",
    canonicalize_key("Glioma"): "Glioma",
    canonicalize_key("Melanoma"): "Melanoma",
    canonicalize_key("Sarcoma"): "Sarcoma",
    canonicalize_key("Mesothelioma"): "Mesothelioma",
    canonicalize_key("Neuroblastoma"): "Neuroblastoma",
    canonicalize_key("Retinoblastoma"): "Retinoblastoma",
    canonicalize_key("Meningioma"): "Meningioma",
    canonicalize_key("Cholangiocarcinoma"): "Cholangiocarcinoma",
}
BIOMARKER_MAP = {
    canonicalize_key("egfr+"): "EGFR",
    canonicalize_key("egfr positive"): "EGFR",
    canonicalize_key("egfr mutation"): "EGFR",
    canonicalize_key("egfr mut"): "EGFR",
    canonicalize_key("egfr"): "EGFR",

    canonicalize_key("her2+"): "HER2",
    canonicalize_key("her2 positive"): "HER2",
    canonicalize_key("her-2"): "HER2",
    canonicalize_key("her2"): "HER2",
    canonicalize_key("erbb2"): "HER2",

    canonicalize_key("pdl1"): "PD-L1",
    canonicalize_key("pd-l1"): "PD-L1",
    canonicalize_key("pd l1"): "PD-L1",
    canonicalize_key("pd-l1 positive"): "PD-L1",
    canonicalize_key("programmed death ligand 1"): "PD-L1",

    canonicalize_key("braf"): "BRAF",
    canonicalize_key("braf v600e"): "BRAF V600E",
    canonicalize_key("braf mutation"): "BRAF",

    canonicalize_key("alk"): "ALK",
    canonicalize_key("alk positive"): "ALK",
    canonicalize_key("alk rearrangement"): "ALK",
    canonicalize_key("alk fusion"): "ALK",

    canonicalize_key("kras"): "KRAS",
    canonicalize_key("kras g12c"): "KRAS G12C",
    canonicalize_key("kras mutation"): "KRAS",

    canonicalize_key("brca1"): "BRCA1",
    canonicalize_key("brca2"): "BRCA2",
    canonicalize_key("brca"): "BRCA1/2",
    canonicalize_key("brca mutation"): "BRCA1/2",

    canonicalize_key("ros1"): "ROS1",
    canonicalize_key("ros1 fusion"): "ROS1",
    canonicalize_key("ros1 rearrangement"): "ROS1",

    canonicalize_key("met"): "MET",
    canonicalize_key("met amplification"): "MET",
    canonicalize_key("met exon 14"): "MET Exon 14",
    canonicalize_key("c-met"): "MET",

    canonicalize_key("ntrk"): "NTRK",
    canonicalize_key("ntrk fusion"): "NTRK",
    canonicalize_key("ntrk1"): "NTRK1",
    canonicalize_key("ntrk2"): "NTRK2",
    canonicalize_key("ntrk3"): "NTRK3",

    canonicalize_key("ret"): "RET",
    canonicalize_key("ret fusion"): "RET",
    canonicalize_key("ret rearrangement"): "RET",

    canonicalize_key("msi"): "MSI-H",
    canonicalize_key("msi-h"): "MSI-H",
    canonicalize_key("microsatellite instability"): "MSI-H",
    canonicalize_key("mismatch repair deficient"): "dMMR",
    canonicalize_key("dmmr"): "dMMR",

    canonicalize_key("tmb"): "TMB-H",
    canonicalize_key("tmb-h"): "TMB-H",
    canonicalize_key("high tumor mutational burden"): "TMB-H",

    canonicalize_key("pik3ca"): "PIK3CA",
    canonicalize_key("pik3ca mutation"): "PIK3CA",

    canonicalize_key("tp53"): "TP53",
    canonicalize_key("p53"): "TP53",
    canonicalize_key("tp53 mutation"): "TP53",
    canonicalize_key("egfr positive"): "EGFR",
    canonicalize_key("alk negative"): "ALK-",
    canonicalize_key("egfr positive exon 19 deletion detected"): "EGFR Exon 19 del",
    canonicalize_key("alk negative for translocations"): "ALK-",
    canonicalize_key("ros1 wild type"): "ROS1-",
    canonicalize_key("ros1 wild type negative"): "ROS1-",
    canonicalize_key("kras negative"): "KRAS-",
    canonicalize_key("pd l1 expression"): "PD-L1",
    canonicalize_key("pd l1 expression tumor proportion score tps is 65 high expression"): "PD-L1 High",
    canonicalize_key("pd l1 high expression"): "PD-L1 High",
    canonicalize_key("EGFR"): "EGFR",
canonicalize_key("EGFR Exon 19 del"): "EGFR Exon 19 del",
canonicalize_key("HER2"): "HER2",
canonicalize_key("ALK"): "ALK",
canonicalize_key("ALK-"): "ALK",
canonicalize_key("ROS1"): "ROS1",
canonicalize_key("ROS1-"): "ROS1",
canonicalize_key("KRAS"): "KRAS",
canonicalize_key("KRAS-"): "KRAS",
canonicalize_key("KRAS G12C"): "KRAS G12C",
canonicalize_key("BRAF"): "BRAF",
canonicalize_key("BRAF V600E"): "BRAF V600E",
canonicalize_key("PD-L1"): "PD-L1",
canonicalize_key("PD-L1 High"): "PD-L1 High",
canonicalize_key("BRCA1"): "BRCA1",
canonicalize_key("BRCA2"): "BRCA2",
canonicalize_key("BRCA1/2"): "BRCA1/2",
canonicalize_key("MSI-H"): "MSI-H",
canonicalize_key("TMB-H"): "TMB-H",
canonicalize_key("MET"): "MET",
canonicalize_key("NTRK"): "NTRK",
canonicalize_key("RET"): "RET",
}

TREATMENT_MAP = {
    canonicalize_key("chemo"): "Chemotherapy",
    canonicalize_key("chemotherapy"): "Chemotherapy",
    canonicalize_key("chemo therapy"): "Chemotherapy",
    canonicalize_key("cytotoxic therapy"): "Chemotherapy",

    canonicalize_key("radiation"): "Radiation Therapy",
    canonicalize_key("radiotherapy"): "Radiation Therapy",
    canonicalize_key("rt"): "Radiation Therapy",
    canonicalize_key("radiation therapy"): "Radiation Therapy",
    canonicalize_key("xrt"): "Radiation Therapy",
    canonicalize_key("stereotactic radiosurgery"): "Stereotactic Radiosurgery",
    canonicalize_key("sbrt"): "Stereotactic Body Radiation Therapy",

    canonicalize_key("immuno"): "Immunotherapy",
    canonicalize_key("immunotherapy"): "Immunotherapy",
    canonicalize_key("io"): "Immunotherapy",
    canonicalize_key("checkpoint inhibitor"): "Immunotherapy",
    canonicalize_key("pembrolizumab"): "Pembrolizumab",
    canonicalize_key("keytruda"): "Pembrolizumab",
    canonicalize_key("nivolumab"): "Nivolumab",
    canonicalize_key("opdivo"): "Nivolumab",
    canonicalize_key("atezolizumab"): "Atezolizumab",
    canonicalize_key("durvalumab"): "Durvalumab",

    canonicalize_key("targeted therapy"): "Targeted Therapy",
    canonicalize_key("target therapy"): "Targeted Therapy",
    canonicalize_key("erlotinib"): "Erlotinib",
    canonicalize_key("osimertinib"): "Osimertinib",
    canonicalize_key("tagrisso"): "Osimertinib",
    canonicalize_key("gefitinib"): "Gefitinib",
    canonicalize_key("imatinib"): "Imatinib",
    canonicalize_key("gleevec"): "Imatinib",
    canonicalize_key("crizotinib"): "Crizotinib",
    canonicalize_key("vemurafenib"): "Vemurafenib",

    canonicalize_key("hormone therapy"): "Hormone Therapy",
    canonicalize_key("hormonal therapy"): "Hormone Therapy",
    canonicalize_key("endocrine therapy"): "Hormone Therapy",
    canonicalize_key("tamoxifen"): "Tamoxifen",
    canonicalize_key("letrozole"): "Letrozole",
    canonicalize_key("anastrozole"): "Anastrozole",

    canonicalize_key("surgery"): "Surgery",
    canonicalize_key("surgical resection"): "Surgery",
    canonicalize_key("resection"): "Surgery",
    canonicalize_key("mastectomy"): "Mastectomy",
    canonicalize_key("lumpectomy"): "Lumpectomy",

    canonicalize_key("stem cell transplant"): "Stem Cell Transplant",
    canonicalize_key("bone marrow transplant"): "Stem Cell Transplant",
    canonicalize_key("bmt"): "Stem Cell Transplant",
    canonicalize_key("hsct"): "Stem Cell Transplant",
    canonicalize_key("autologous transplant"): "Autologous Stem Cell Transplant",
    canonicalize_key("allogeneic transplant"): "Allogeneic Stem Cell Transplant",

    canonicalize_key("car-t"): "CAR-T Cell Therapy",
    canonicalize_key("car t"): "CAR-T Cell Therapy",
    canonicalize_key("chimeric antigen receptor"): "CAR-T Cell Therapy",

    canonicalize_key("cisplatin combined with pemetrexed"): "Cisplatin + Pemetrexed",
    canonicalize_key("cisplatin"): "Cisplatin",
    canonicalize_key("pemetrexed"): "Pemetrexed",
    canonicalize_key("platinum based doublet chemotherapy"): "Platinum-based Chemotherapy",
    canonicalize_key("carboplatin"): "Carboplatin",
    canonicalize_key("paclitaxel"): "Paclitaxel",
    canonicalize_key("docetaxel"): "Docetaxel",
    canonicalize_key("bevacizumab"): "Bevacizumab",
    canonicalize_key("avastin"): "Bevacizumab",
    canonicalize_key("Chemotherapy"): "Chemotherapy",
canonicalize_key("Immunotherapy"): "Immunotherapy",
canonicalize_key("Radiation Therapy"): "Radiation Therapy",
canonicalize_key("Surgery"): "Surgery",
canonicalize_key("Targeted Therapy"): "Targeted Therapy",
canonicalize_key("Cisplatin"): "Cisplatin",
canonicalize_key("Cisplatin + Pemetrexed"): "Cisplatin + Pemetrexed",
canonicalize_key("Pemetrexed"): "Pemetrexed",
canonicalize_key("Carboplatin"): "Carboplatin",
canonicalize_key("Paclitaxel"): "Paclitaxel",
canonicalize_key("Pembrolizumab"): "Pembrolizumab",
canonicalize_key("Nivolumab"): "Nivolumab",
canonicalize_key("Osimertinib"): "Osimertinib",
canonicalize_key("Erlotinib"): "Erlotinib",
canonicalize_key("CAR-T Cell Therapy"): "CAR-T Cell Therapy",
}

STAGE_MAP = {
    canonicalize_key("stage 1"): "Stage I",
    canonicalize_key("stage i"): "Stage I",
    canonicalize_key("stage 1a"): "Stage IA",
    canonicalize_key("stage ia"): "Stage IA",
    canonicalize_key("stage 1b"): "Stage IB",
    canonicalize_key("stage ib"): "Stage IB",
    canonicalize_key("stage 2"): "Stage II",
    canonicalize_key("stage ii"): "Stage II",
    canonicalize_key("stage 2a"): "Stage IIA",
    canonicalize_key("stage iia"): "Stage IIA",
    canonicalize_key("stage 2b"): "Stage IIB",
    canonicalize_key("stage iib"): "Stage IIB",
    canonicalize_key("stage 3"): "Stage III",
    canonicalize_key("stage iii"): "Stage III",
    canonicalize_key("stage 3a"): "Stage IIIA",
    canonicalize_key("stage iiia"): "Stage IIIA",
    canonicalize_key("stage 3b"): "Stage IIIB",
    canonicalize_key("stage iiib"): "Stage IIIB",
    canonicalize_key("stage 3c"): "Stage IIIC",
    canonicalize_key("stage iiic"): "Stage IIIC",
    canonicalize_key("stage 4"): "Stage IV",
    canonicalize_key("stage iv"): "Stage IV",
    canonicalize_key("stage 4a"): "Stage IVA",
    canonicalize_key("stage iva"): "Stage IVA",
    canonicalize_key("stage 4b"): "Stage IVB",
    canonicalize_key("stage ivb"): "Stage IVB",

    canonicalize_key("advanced"): "Stage IV",
    canonicalize_key("metastatic"): "Stage IV",
    canonicalize_key("locally advanced"): "Stage III",
    canonicalize_key("early stage"): "Stage I",
    canonicalize_key("early"): "Stage I",
    canonicalize_key("localized"): "Stage I",
}

GENDER_MAP = {
    canonicalize_key("m"): "Male",
    canonicalize_key("male"): "Male",
    canonicalize_key("man"): "Male",
    canonicalize_key("f"): "Female",
    canonicalize_key("female"): "Female",
    canonicalize_key("woman"): "Female",
    canonicalize_key("other"): "Other",
    canonicalize_key("non-binary"): "Other",
    canonicalize_key("prefer not to say"): "Other",
}

COUNTRY_MAP = {
    canonicalize_key("india"): "India",
    canonicalize_key("ind"): "India",
    canonicalize_key("usa"): "United States",
    canonicalize_key("us"): "United States",
    canonicalize_key("u s"): "United States",
    canonicalize_key("united states of america"): "United States",
    canonicalize_key("uk"): "United Kingdom",
    canonicalize_key("u k"): "United Kingdom",
    canonicalize_key("england"): "United Kingdom",
    canonicalize_key("japan"): "Japan",
}

TRIAL_SEX_MAP = {
    canonicalize_key("all"): "ALL",
    canonicalize_key("any"): "ALL",
    canonicalize_key("male"): "MALE",
    canonicalize_key("m"): "MALE",
    canonicalize_key("female"): "FEMALE",
    canonicalize_key("f"): "FEMALE",
}

# -----------------------------------------------------------
# PATIENT NORMALIZATION HELPERS
# -----------------------------------------------------------

def normalize_text(
    value: Optional[str],
    lookup_map: Dict[str, str],
    field_name: str,
) -> Optional[str]:
    raw = clean_text(value)
    if raw is None:
        return None
    key = canonicalize_key(raw)
    normalized = lookup_map.get(key)
    if normalized is None:
        stripped = re.sub(r'\(.*?\)', '', raw).strip()
        if stripped != raw:
            key2 = canonicalize_key(stripped)
            normalized = lookup_map.get(key2)

    if normalized is None:
        logger.warning(
            "Normalization fallback for %s: '%s' not found in map; "
            "preserving raw value.",
            field_name,
            raw,
        )
        return raw

    return normalized


def normalize_list(
    values: Optional[List[str]],
    lookup_map: Dict[str, str],
    field_name: str,
) -> List[str]:
    if not values:
        return []

    result: List[str] = []
    seen = set()

    for value in values:
        raw = clean_text(value)
        if raw is None:
            continue

        # Try 1: Direct lookup
        key = canonicalize_key(raw)
        normalized = lookup_map.get(key)

        # Try 2: Strip parentheticals
        # 'EGFR: Positive (Exon 19 deletion detected)' → 'EGFR: Positive'
        if normalized is None:
            stripped = re.sub(r'\(.*?\)', '', raw).strip()
            if stripped != raw:
                normalized = lookup_map.get(canonicalize_key(stripped))

        # Try 3: Take only the part before the colon
        # 'EGFR: Positive (Exon 19 deletion detected)' → 'EGFR'
        # 'ALK: Negative for translocations' → 'ALK'
        # 'KRAS: Negative' → 'KRAS'
        if normalized is None and ':' in raw:
            before_colon = raw.split(':')[0].strip()
            normalized = lookup_map.get(canonicalize_key(before_colon))

        # Try 4: Strip 'combined with' patterns for treatments
        # 'Cisplatin combined with Pemetrexed' → 'Cisplatin'
        if normalized is None and field_name == "previous_treatments":
            simplified = re.split(
                r'\s+(?:combined with|plus|and|with)\s+',
                raw,
                flags=re.IGNORECASE,
            )[0].strip()
            if simplified != raw:
                normalized = lookup_map.get(canonicalize_key(simplified))

        if normalized is None:
            logger.warning(
                "Normalization fallback for %s item: '%s' not found in map; "
                "preserving raw value.",
                field_name,
                raw,
            )
            normalized = raw

        dedupe_key = normalized.lower()
        if dedupe_key not in seen:
            seen.add(dedupe_key)
            result.append(normalized)

    return result

def validate_age(age: Optional[int]) -> Optional[int]:
    if age is None:
        return None

    if age <= 0 or age > 120:
        raise ValueError(
            f"Invalid age value extracted: {age}. Age must be between 1 and 120."
        )

    return age


def coerce_age(age: Any) -> Optional[int]:
    if age is None:
        return None

    if isinstance(age, int):
        return validate_age(age)

    if isinstance(age, float):
        return validate_age(int(age))

    if isinstance(age, str):
        match = re.search(r"\d+", age)
        if match:
            return validate_age(int(match.group()))

    raise ValueError(f"Unable to parse age from value: {age}")


def normalize_country(country: Optional[str]) -> Optional[str]:
    return normalize_text(country, COUNTRY_MAP, "country")


# -----------------------------------------------------------
# PATIENT NORMALIZERS
# -----------------------------------------------------------

def normalize_patient_profile(profile: PatientProfile) -> PatientProfile:
    logger.info("Starting patient profile normalization...")

    normalized = PatientProfile(
        age=coerce_age(profile.age),
        gender=normalize_text(profile.gender, GENDER_MAP, "gender"),
        cancer_type=normalize_text(profile.cancer_type, CANCER_TYPE_MAP, "cancer_type"),
        cancer_stage=normalize_text(profile.cancer_stage, STAGE_MAP, "cancer_stage"),
        biomarkers=normalize_list(profile.biomarkers, BIOMARKER_MAP, "biomarkers"),
        previous_treatments=normalize_list(
            profile.previous_treatments,
            TREATMENT_MAP,
            "previous_treatments",
        ),
        country=normalize_country(profile.country),
        diagnosis=clean_text(profile.diagnosis),
    )

    logger.info(
        "Patient normalization complete | cancer_type='%s' | cancer_stage='%s' | gender='%s'",
        normalized.cancer_type,
        normalized.cancer_stage,
        normalized.gender,
    )

    return normalized


def normalize_patient_payload(payload: Dict[str, Any]) -> PatientProfile:
    """
    Normalize raw patient input dict into PatientProfile.
    Useful when data arrives from API/UI before schema validation.
    """
    raw_profile = PatientProfile(
        age=payload.get("age"),
        gender=payload.get("gender") or payload.get("sex"),
        cancer_type=payload.get("cancer_type") or payload.get("primary_condition") or payload.get("condition"),
        cancer_stage=payload.get("cancer_stage") or payload.get("stage"),
        biomarkers=payload.get("biomarkers") or [],
        previous_treatments=payload.get("previous_treatments") or payload.get("prior_treatments") or [],
        country=payload.get("country"),
        diagnosis=payload.get("diagnosis"),
    )
    return normalize_patient_profile(raw_profile)


# -----------------------------------------------------------
# TRIAL/API NORMALIZATION HELPERS
# -----------------------------------------------------------

def normalize_trial_sex(value: Optional[str]) -> Optional[str]:
    return normalize_text(value, TRIAL_SEX_MAP, "trial_sex")


def normalize_string_list(values: Any) -> List[str]:
    cleaned = [clean_text(v) for v in ensure_list(values)]
    return dedupe_preserve_order([v for v in cleaned if v])


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# -----------------------------------------------------------
# TRIAL/API NORMALIZER
# -----------------------------------------------------------

def normalize_trial_study(study: Dict[str, Any]) -> TrialProfile:
    """
    Normalize a raw ClinicalTrials.gov study record into TrialProfile.
    """
    protocol = study.get("protocolSection", {}) or {}
    identification = protocol.get("identificationModule", {}) or {}
    status = protocol.get("statusModule", {}) or {}
    description = protocol.get("descriptionModule", {}) or {}
    conditions = protocol.get("conditionsModule", {}) or {}
    design = protocol.get("designModule", {}) or {}
    eligibility = protocol.get("eligibilityModule", {}) or {}
    contacts = protocol.get("contactsLocationsModule", {}) or {}
    sponsors = protocol.get("sponsorCollaboratorsModule", {}) or {}
    derived = study.get("derivedSection", {}) or {}

    locations: List[TrialLocation] = []
    for loc in contacts.get("locations", []) or []:
        geo = loc.get("geoPoint", {}) or {}
        locations.append(
            TrialLocation(
                facility=clean_text(loc.get("facility")),
                city=clean_text(loc.get("city")),
                country=clean_text(loc.get("country")),
                lat=safe_float(geo.get("lat")),
                lon=safe_float(geo.get("lon")),
            )
        )

    mesh_terms = [
        clean_text(mesh.get("term"))
        for mesh in (derived.get("conditionBrowseModule", {}) or {}).get("meshes", []) or []
        if clean_text(mesh.get("term"))
    ]

    trial = TrialProfile(
        trial_id=clean_text(identification.get("nctId")) or "",
        title=clean_text(identification.get("briefTitle")),
        official_title=clean_text(identification.get("officialTitle")),
        status=clean_text(status.get("overallStatus")),
        study_type=clean_text(design.get("studyType")),
        phases=normalize_string_list(design.get("phases")),
        conditions=normalize_string_list(conditions.get("conditions")),
        brief_summary=clean_text(description.get("briefSummary")),
        detailed_description=clean_text(description.get("detailedDescription")),
        eligibility=TrialEligibility(
            criteria_text=clean_text(eligibility.get("eligibilityCriteria")),
            healthy_volunteers=eligibility.get("healthyVolunteers"),
            sex=normalize_trial_sex(eligibility.get("sex")),
            minimum_age=clean_text(eligibility.get("minimumAge")),
            maximum_age=clean_text(eligibility.get("maximumAge")),
            age_groups=normalize_string_list(eligibility.get("stdAges")),
            study_population=clean_text(eligibility.get("studyPopulation")),
        ),
        locations=locations,
        sponsor_name=clean_text((sponsors.get("leadSponsor", {}) or {}).get("name")),
        sponsor_class=clean_text((sponsors.get("leadSponsor", {}) or {}).get("class")),
        mesh_terms=dedupe_preserve_order([m for m in mesh_terms if m]),
        has_results=bool(study.get("hasResults", False)),
    )

    logger.info(
        "Trial normalization complete | trial_id='%s' | title='%s' | status='%s'",
        trial.trial_id,
        trial.title,
        trial.status,
    )
    return trial
