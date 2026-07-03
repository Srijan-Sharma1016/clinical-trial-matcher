# config/settings.py
"""
Central configuration module.
Responsibility: Environment variables, API constants, and LLM limits.
All environment-driven config lives here — nothing else should call os.getenv().
"""

import os
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------
# LLM CONFIGURATION
# -----------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("System Error: GROQ_API_KEY is missing from environment.")

GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0"))
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "3"))

# -----------------------------------------------------------
# DATABASE
# -----------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("System Error: DATABASE_URL is missing from environment.")

DB_ECHO = os.getenv("DB_ECHO", "false").lower() == "true"

# -----------------------------------------------------------
# REDIS / MEMORY
# -----------------------------------------------------------

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# -----------------------------------------------------------
# CORS
# -----------------------------------------------------------

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]

# -----------------------------------------------------------
# FILE UPLOAD LIMITS
# -----------------------------------------------------------

MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_BYTES", str(10 * 1024 * 1024)))  # 10MB
MAX_EXTRACTED_TEXT_CHARS = int(os.getenv("MAX_EXTRACTED_TEXT_CHARS", "30000"))

# -----------------------------------------------------------
# CLINICALTRIALS API
# -----------------------------------------------------------

CLINICALTRIALS_BASE_URL = os.getenv(
    "CLINICALTRIALS_BASE_URL",
    "https://clinicaltrials.gov/api/v2"
)

CLINICALTRIALS_HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}
# -----------------------------------------------------------
# TRIAL FETCH LIMITS
# -----------------------------------------------------------

MAX_FETCH_RESULTS = int(os.getenv("MAX_FETCH_RESULTS", "6"))

# Controls LLM evaluation cost — top N hard-filter-passing trials sent to LLM.
# Higher = better coverage but more API calls + latency. Keep at 2-3 for cost control.
MAX_LLM_EVALUATION_TRIALS = int(os.getenv("MAX_LLM_EVALUATION_TRIALS", "2"))

# -----------------------------------------------------------
# LLM PAYLOAD LIMITS
# -----------------------------------------------------------

MAX_LLM_SUMMARY_CHARS = 700
MAX_LLM_DESCRIPTION_CHARS = 700
MAX_LLM_CRITERIA_CHARS = 1200
MAX_LLM_STUDY_POPULATION_CHARS = 300
MAX_LLM_LOCATIONS = 2
MAX_LLM_MESH_TERMS = 6
MAX_SCORE_REASONS = 5

# -----------------------------------------------------------
# TRIAL STATUS ALLOWLIST
# -----------------------------------------------------------

ALLOWED_ACTIVE_STATUSES = {
    "RECRUITING",
    "NOT_YET_RECRUITING",
    "ENROLLING_BY_INVITATION",
}

# -----------------------------------------------------------
# APP METADATA
# -----------------------------------------------------------

APP_TITLE = "Eisai Trial Matcher API"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "AI-powered clinical trial matching from patient profiles."

# -----------------------------------------------------------
# CANCER TYPE SEARCH TERM MAP
# -----------------------------------------------------------

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
    "Liver Neoplasms": "liver cancer",
}
