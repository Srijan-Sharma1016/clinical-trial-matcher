# Clinical Trial Matcher

AI-powered oncology trial matching platform that transforms unstructured patient records into structured clinical profiles and maps them to recruiting clinical trials using deterministic eligibility logic and LLM-assisted reasoning.

---

## Overview

Clinical Trial Matcher is a full-stack clinical decision-support application designed to streamline oncology trial discovery from patient data. The platform supports both **PDF upload** and **manual profile entry**, extracts key clinical attributes, evaluates relevant recruiting trials, and generates plain-language match explanations.

The system combines:
- structured patient profile extraction
- clinical term normalization
- deterministic eligibility filtering
- heuristic biomarker and treatment checks
- LLM-generated recommendation summaries
- follow-up oncology Q&A through chat

---

## Key Capabilities

- **Patient record ingestion** from PDFs up to **10MB**
- **Manual structured entry** for missing or corrected details
- **4-node LangGraph workflow** for extraction, trial search, evaluation, and recommendation
- **3-tier match ranking**: Strong, Moderate, Possible
- **Completeness auditing** with ordered missing-field suggestions
- **Oncology chat assistant** grounded in patient and trial context
- **Production deployment** across Railway and Vercel

---

## Tech Stack

### Frontend
- Next.js
- TypeScript
- Tailwind CSS

### Backend
- FastAPI
- Python
- SQLModel / SQLAlchemy
- PostgreSQL

### AI / Orchestration
- LangGraph
- LangChain
- Groq
- Llama 3.3 70B

### External Integrations
- ClinicalTrials.gov
- PubMed

---

## Architecture

### Trial Matching Flow
1. Extract text from patient PDF
2. Build structured `PatientProfile`
3. Normalize oncology terms
4. Run **4-step trial matching pipeline**
5. Return ranked matches and recommendation summary

### LangGraph Nodes
- `extract_cancer_type`
- `search_trials`
- `evaluate_trials`
- `recommend_trials`

### Matching Strategy
- hard filters for status, sex, and age
- heuristic biomarker and treatment-history checks
- deterministic scoring
- LLM-generated eligibility reasoning

---

## Project Structure

```txt
clinical-trial-matcher/
├── Backend/
│   ├── .env.example
│   ├── requirements.txt
│   └── App/
│       ├── main.py
│       ├── schemas.py
│       ├── normalizer.py
│       ├── database.py
│       ├── db_service.py
│       ├── models.py
│       ├── config/
│       ├── core/
│       ├── agents/
│       ├── tools/
│       ├── memory/
│       └── prompts/
└── frontend/
    ├── app/
    ├── components/
    ├── package.json
    └── next.config.ts
