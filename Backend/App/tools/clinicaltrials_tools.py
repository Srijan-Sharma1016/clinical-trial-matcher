# tools/clinicaltrials_tools.py
"""
ClinicalTrials.gov API integration.
Responsibility: Fetch and return raw trial data only.
Depends on: config/settings.py, normalizer, schemas
"""

import asyncio
import logging
import requests
from typing import Dict, Any, List, Optional, Union

from schemas import TrialProfile
from normalizer import normalize_trial_study
from config.settings import (
    CLINICALTRIALS_BASE_URL,
    CLINICALTRIALS_HEADERS,
    MAX_FETCH_RESULTS,
)
from core.utils import get_api_search_term

logger = logging.getLogger("uvicorn.error")

__all__ = ["search_clinical_trials", "get_trial_details"]

_MAX_API_PAGE_SIZE = 10


# -----------------------------------------------------------
# INTERNAL REQUEST HELPER — sync, wrapped in asyncio.to_thread
# -----------------------------------------------------------

def _clinicaltrials_get_json(
    path: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Synchronous GET request to ClinicalTrials.gov API v2.
    Called via asyncio.to_thread() to avoid blocking the event loop.
    Uses requests library — confirmed working with ClinicalTrials.gov.
    """
    url = f"{CLINICALTRIALS_BASE_URL.rstrip('/')}/{path.lstrip('/')}"

    logger.info(
        "ClinicalTrials.gov request | url=%s | condition=%s",
        url,
        (params or {}).get("query.cond", "N/A"),
    )

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


# -----------------------------------------------------------
# PUBLIC API FUNCTIONS
# -----------------------------------------------------------

async def search_clinical_trials(
    cancer_type: str,
    max_results: int = MAX_FETCH_RESULTS,
) -> List[TrialProfile]:
    max_results = min(max_results, _MAX_API_PAGE_SIZE)
    api_search_term = get_api_search_term(cancer_type)

    logger.info(
        "Searching ClinicalTrials.gov | original='%s' | api_term='%s'",
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
        logger.info(
            "ClinicalTrials.gov returned %d studies for '%s'.",
            len(studies),
            api_search_term,
        )
        return [normalize_trial_study(study) for study in studies]

    except requests.Timeout:
        logger.warning("ClinicalTrials.gov search timed out for '%s'.", cancer_type)
        return []
    except requests.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "unknown"
        body_preview = e.response.text[:500] if e.response is not None else ""
        logger.error(
            "ClinicalTrials.gov HTTP error | status=%s | body=%s",
            status_code, body_preview,
        )
        return []
    except requests.RequestException as e:
        logger.error("ClinicalTrials.gov request error: %s", str(e))
        return []
    except Exception:
        logger.exception("Unexpected error during ClinicalTrials.gov search")
        return []

async def get_trial_details(
    nct_id: str,
) -> Union[TrialProfile, Dict[str, Any]]:
    """
    Fetches full trial details for a given NCT ID.
    Returns error dict on failure — safe for asyncio.gather().
    """
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
        return {
            "error": f"HTTP {status_code} fetching trial {nct_id}: {body_preview}"
        }
    except requests.RequestException as e:
        return {"error": f"Request error fetching trial {nct_id}: {str(e)}"}
    except Exception as e:
        logger.exception(
            "Unexpected error in get_trial_details for '%s'", nct_id
        )
        return {"error": f"Unexpected error fetching trial {nct_id}: {str(e)}"}
