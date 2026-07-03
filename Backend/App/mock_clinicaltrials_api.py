from fastapi import FastAPI, HTTPException, Query
from dotenv import load_dotenv
app = FastAPI()

MOCK_TRIALS = [
    {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT00000001",
                "briefTitle": "Recruiting Study for Breast Cancer"
            },
            "statusModule": {
                "overallStatus": "RECRUITING"
            },
            "conditionsModule": {
                "conditions": ["Breast Cancer"]
            },
            "descriptionModule": {
                "briefSummary": "A temporary mock trial for breast cancer."
            }
        }
    },
    {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT00000002",
                "briefTitle": "Recruiting Study for Lung Cancer"
            },
            "statusModule": {
                "overallStatus": "RECRUITING"
            },
            "conditionsModule": {
                "conditions": ["Lung Cancer"]
            },
            "descriptionModule": {
                "briefSummary": "A temporary mock trial for lung cancer."
            }
        }
    },
    {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT00000003",
                "briefTitle": "Completed Study for Breast Cancer"
            },
            "statusModule": {
                "overallStatus": "COMPLETED"
            },
            "conditionsModule": {
                "conditions": ["Breast Cancer"]
            },
            "descriptionModule": {
                "briefSummary": "A completed breast cancer study."
            }
        }
    },
]

@app.get("/api/v2/studies")
def search_studies(
    query_cond: str = Query("", alias="query.cond"),
    overall_status: str | None = Query(None, alias="filter.overallStatus"),
    page_size: int = Query(5, alias="pageSize"),
):
    q = query_cond.lower().strip()

    filtered = []
    for trial in MOCK_TRIALS:
        protocol = trial.get("protocolSection", {})
        title = protocol.get("identificationModule", {}).get("briefTitle", "").lower()
        conditions = protocol.get("conditionsModule", {}).get("conditions", [])
        status = protocol.get("statusModule", {}).get("overallStatus")

        matches_query = (
            q in title or any(q in c.lower() for c in conditions)
        ) if q else True

        matches_status = (status == overall_status) if overall_status else True

        if matches_query and matches_status:
            filtered.append(trial)

    return {
        "studies": filtered[:page_size],
        "nextPageToken": None
    }

@app.get("/api/v2/studies/{nct_id}")
def get_study(nct_id: str):
    for trial in MOCK_TRIALS:
        trial_id = (
            trial.get("protocolSection", {})
            .get("identificationModule", {})
            .get("nctId")
        )
        if trial_id == nct_id:
            return trial

    raise HTTPException(status_code=404, detail="Study not found")
