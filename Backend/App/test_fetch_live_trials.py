import asyncio
from Backend.App.agents_old_depriciated import search_clinical_trials, get_trial_details

async def main():
    trials = await search_clinical_trials("breast cancer", max_results=3)
    print("Trials found:", len(trials))
    for t in trials:
        print(t["trial_id"], "-", t["title"])

    details = await get_trial_details("NCT04280705")
    print("\nTrial details:")
    print(details)

asyncio.run(main())
