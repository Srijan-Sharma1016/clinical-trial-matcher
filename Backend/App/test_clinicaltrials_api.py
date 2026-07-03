import asyncio
import json
from Backend.App.agents_old_depriciated import run_trial_matcher_agent

mock_payload = {
    "age": 52,
    "gender": "female",
    "cancer_type": "breast cancer",
    "cancer_stage": "stage ii",
    "biomarkers": ["HER2", "PD-L1"],
    "previous_treatments": ["Chemotherapy"],
}

async def main():
    result = await run_trial_matcher_agent(mock_payload)

    print("\n=== SUMMARY ===")
    print("success:", result.get("success"))
    print("error:", result.get("error"))
    print("cancer_type:", result.get("cancer_type"))
    print("trial_count:", result.get("trial_count"))

    print("\n=== ELIGIBILITY RESULTS ===")
    for item in result.get("eligibility_results", [])[:5]:
        print(json.dumps({
            "nct_id": item.get("nct_id"),
            "title": item.get("title"),
            "hard_filter_pass": item.get("hard_filter_pass"),
            "score": item.get("score"),
            "score_reasons": item.get("score_reasons"),
        }, indent=2))

    print("\n=== FINAL RECOMMENDATION ===")
    print(result.get("final_recommendations", ""))

if __name__ == "__main__":
    asyncio.run(main())
