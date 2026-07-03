# test_pipeline.py
import sys
import asyncio

sys.path.insert(0, ".")

from core.entrypoint import run_trial_matching

# -----------------------------------------------------------
# SAMPLE PATIENT PAYLOAD
# -----------------------------------------------------------

sample_patient = {
    "name": "Test Patient",
    "age": 52,
    "gender": "Female",
    "country": "India",
    "cancer_type": "Breast Neoplasms",
    "cancer_stage": "Stage IV",
    "diagnosis": "Metastatic Breast Cancer",
    "biomarkers": ["HER2", "PD-L1"],
    "previous_treatments": ["Chemotherapy", "Trastuzumab"],
}


# -----------------------------------------------------------
# PIPELINE RUNNER
# -----------------------------------------------------------

async def main():
    print("\n" + "=" * 60)
    print("🚀 CLINICAL TRIAL MATCHING — PIPELINE TEST")
    print("=" * 60)
    print("\n📋 Patient Payload:")
    for k, v in sample_patient.items():
        print(f"   {k:<25}: {v}")
    print("\n" + "-" * 60)
    print("⏳ Running pipeline... please wait")
    print("-" * 60 + "\n")

    try:
        result = await run_trial_matching(sample_patient)
    except Exception as e:
        print(f"❌ Pipeline crashed unexpectedly: {str(e)}")
        return

    # -----------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------
    print("=" * 60)
    print("📊 PIPELINE RESULT SUMMARY")
    print("=" * 60)
    print(f"  ✅ Success           : {result.get('success')}")
    print(f"  ❌ Error             : {result.get('error') or 'None'}")
    print(f"  🧬 Cancer Type       : {result.get('cancer_type') or 'Not resolved'}")
    print(f"  🔢 Trial Count       : {result.get('trial_count', 0)}")
    print(f"  🆔 Patient Profile ID: {result.get('patient_profile_id') or 'Not saved'}")
    print(f"  🏃 Match Run ID      : {result.get('trial_match_run_id') or 'Not saved'}")

    # -----------------------------------------------------------
    # TRIALS FETCHED
    # -----------------------------------------------------------
    trials = result.get("trials") or []
    print(f"\n{'=' * 60}")
    print(f"🔬 TRIALS FETCHED ({len(trials)})")
    print("=" * 60)

    if trials:
        for idx, trial in enumerate(trials, start=1):
            print(f"\n  {idx}. {trial.get('title', 'No title')}")
            print(f"     Trial ID   : {trial.get('trial_id', 'N/A')}")
            print(f"     Status     : {trial.get('status', 'N/A')}")
            print(f"     Study Type : {trial.get('study_type', 'N/A')}")
            print(f"     Phases     : {', '.join(trial.get('phases', [])) or 'N/A'}")
            print(f"     Conditions : {', '.join(trial.get('conditions', [])) or 'N/A'}")
    else:
        print("  ⚠️  No trials were fetched.")

    # -----------------------------------------------------------
    # ELIGIBILITY RESULTS
    # -----------------------------------------------------------
    eligibility_results = result.get("eligibility_results") or []
    print(f"\n{'=' * 60}")
    print(f"🏆 ELIGIBILITY RESULTS ({len(eligibility_results)})")
    print("=" * 60)

    if eligibility_results:
        for idx, r in enumerate(eligibility_results, start=1):
            print(f"\n  {idx}. {r.get('title', 'No title')}")
            print(f"     NCT ID          : {r.get('nct_id', 'N/A')}")
            print(f"     Hard Filter Pass: {r.get('hard_filter_pass')}")
            print(f"     Score           : {r.get('score', 'N/A')}")

            score_reasons = r.get("score_reasons") or []
            if score_reasons:
                print(f"     Score Reasons   :")
                for reason in score_reasons:
                    print(f"       - {reason}")

            hard_filter_reasons = r.get("hard_filter_reasons") or []
            if hard_filter_reasons:
                print(f"     Hard Filter Reasons:")
                for reason in hard_filter_reasons:
                    print(f"       - {reason}")

            print(f"     Biomarker Check :")
            for line in (r.get("biomarker_check") or "Not available.").splitlines():
                print(f"       {line}")

            print(f"     Treatment Check :")
            for line in (r.get("treatment_check") or "Not available.").splitlines():
                print(f"       {line}")
    else:
        print("  ⚠️  No eligibility results generated.")

    # -----------------------------------------------------------
    # FINAL RECOMMENDATIONS
    # -----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("📝 FINAL RECOMMENDATIONS")
    print("=" * 60)
    recommendations = result.get("final_recommendations") or "No recommendations generated."
    for line in recommendations.splitlines():
        print(f"  {line}")

    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE")
    print("=" * 60 + "\n")


# -----------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
