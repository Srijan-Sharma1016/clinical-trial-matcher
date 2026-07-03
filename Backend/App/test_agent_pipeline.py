import asyncio
from Backend.App.agents_old_depriciated import run_trial_matcher_agent

sample_patient = {
    "age": 52,
    "gender": "female",
    "cancer_type": "breast cancer",
    "cancer_stage": "stage 2",
    "biomarkers": ["HER2", "PD-L1"],
    "previous_treatments": ["chemotherapy"],
    "country": "India",
    "diagnosis": "breast carcinoma",
}

async def main():
    result = await run_trial_matcher_agent(sample_patient)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
