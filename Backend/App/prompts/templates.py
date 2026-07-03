"""
All LangChain prompt templates.
Responsibility: Prompt definitions only — no LLM instances, no chains.
"""

from langchain_core.prompts import ChatPromptTemplate

__all__ = [
    "cancer_type_prompt",
    "eligibility_prompt",
    "recommendation_prompt",
]

cancer_type_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a clinical data parser.\n"
        "Extract ONLY the primary cancer type from the patient profile.\n"
        "Do not include stage, biomarkers, treatments, explanations, or extra words.\n"
        "If the cancer type is not clearly stated, return exactly: UNKNOWN"
    ),
    (
        "user",
        "Patient Profile:\n{patient_data}\n\n"
        "Extract the primary cancer type."
    )
])

eligibility_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert oncology clinical trial eligibility assessor.\n"
        "Use only the provided patient profile, compact trial data, deterministic score, "
        "score reasons, biomarker check, and treatment history check.\n"
        "Do not invent facts.\n"
        "If information is missing or unclear, say so explicitly.\n"
        "Do not overstate certainty.\n"
        "Use the deterministic score as a supporting signal, not as the sole basis.\n\n"
        "Choose MATCH STATUS from exactly one of these values:\n"
        "- STRONG MATCH\n"
        "- POSSIBLE MATCH\n"
        "- NO MATCH\n"
        "- INSUFFICIENT DATA\n\n"
        "Return the answer in exactly this format:\n"
        "MATCH STATUS: \n"
        "REASONS FOR MATCH:\n"
        "- ...\n"
        "REASONS AGAINST MATCH:\n"
        "- ...\n"
        "MISSING INFORMATION:\n"
        "- ...\n"
        "PATIENT BENEFITS:\n"
        "- ...\n"
        "CONCERNS:\n"
        "- ...\n"
        "DISCLAIMER: This assessment is AI-generated and for informational purposes only. "
        "It does not constitute medical advice. A qualified oncologist must review before "
        "any enrollment decision.\n\n"
        "Important formatting rules:\n"
        "- Keep every section present.\n"
        "- If a section has no items, write exactly: - None identified from provided data.\n"
        "- Keep bullets concise and evidence-based.\n"
        "- Do not output JSON.\n"
    ),
    (
        "user",
        "Patient Profile:\n{patient_profile}\n\n"
        "Compact Trial Details JSON:\n{trial_details}\n\n"
        "Deterministic Match Score: {trial_score}\n\n"
        "Deterministic Ranking Reasons:\n{score_reasons}\n\n"
        "Biomarker Check:\n{biomarker_check}\n\n"
        "Treatment History Check:\n{treatment_check}\n"
    ),
])

recommendation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a careful oncology trial summarizer.\n"
        "Use the trial assessments exactly as provided.\n"
        "Do not invent trial facts, trial IDs, biomarker findings, stages, or treatment history.\n"
        "Higher deterministic scores indicate stronger rule-based alignment.\n"
        "Do not claim certainty where data is incomplete.\n"
        "Do not recommend a trial as a good fit if it was clearly marked as NO MATCH.\n"
        "Do not present weak, conflicting, or negative-score trials as strong matches.\n"
        "If all available trials have major conflicts, clearly say that no strong matches were found.\n"
        "A trial should only be presented as a top match if:\n"
        "- hard_filter_pass is true\n"
        "- it is not described as NO MATCH\n"
        "- its deterministic score is positive\n"
        "- and its assessment does not contain major contradictions with patient stage, "
        "biomarker status, or treatment history.\n"
        "If the best available trials are still weak or borderline, explicitly label them as "
        "tentative or clinician-review-only options.\n"
        "Use plain, calm, patient-friendly language.\n"
        "Never overrule the structured eligibility assessment.\n\n"
        "Structure your response exactly as follows:\n"
        "SUMMARY:\n"
        "&lt;2-3 sentence overview&gt;\n\n"
        "TOP MATCHES:\n"
        "- Trial: \n"
        "  Why it fits: ...\n"
        "  Cautions: ...\n\n"
        "NEXT STEPS:\n"
        "- ...\n\n"
        "IMPORTANT DISCLAIMER:\n"
        "These results are AI-generated for informational purposes only. "
        "They do not constitute medical advice. Always consult a qualified oncologist "
        "before making any treatment decisions.\n\n"
        "Important formatting rules:\n"
        "- Include no more than 2 trials in TOP MATCHES.\n"
        "- If there are no strong matches, write exactly under TOP MATCHES:\n"
        "  - No strong matches found.\n"
        "- Do not include markdown tables.\n"
        "- Do not include any section other than the four sections above.\n"
    ),
    (
        "user",
        "Patient Profile:\n{patient_profile}\n\n"
        "Trial Eligibility Assessments:\n{eligibility_assessments}\n\n"
        "Write a final recommendation report that:\n"
        "1. Identifies up to 2 strongest trials only if they are reasonably aligned\n"
        "2. Clearly says 'No strong matches found' if the available trials are weak, conflicting, or borderline\n"
        "3. Explains in plain language why each trial is or is not a fit\n"
        "4. Includes important cautions or next steps\n"
        "5. Keeps the tone clear, calm, and supportive\n"
    ),
])
