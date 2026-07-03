# agents/research_agent.py
"""
Research agent — medical literature search and summarization.
Responsibility: Search PubMed and summarize findings for clinical context.
Depends on: agents/base.py, agents/chains.py, tools/search_tool.py
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base import BaseAgent
from agents.chains import get_llm
from tools.search_tool import SearchTool

logger = logging.getLogger("uvicorn.error")

__all__ = ["ResearchAgent"]

# -----------------------------------------------------------
# SYSTEM PROMPT
# -----------------------------------------------------------

_RESEARCH_SYSTEM_PROMPT = """You are a medical research summarizer specializing in oncology.

Your job is to summarize PubMed research findings in plain, patient-friendly language.

Rules:
- Summarize key findings clearly and concisely
- Highlight relevance to the patient's specific cancer type and biomarkers
- Mention the number of studies found
- Note the recency of the research
- NEVER overstate findings or claim certainty where none exists
- ALWAYS end with a disclaimer that findings are informational only
- Keep the summary under 300 words
- Use simple language — avoid excessive medical jargon
"""

_RESEARCH_USER_PROMPT = """
Patient Context:
{patient_context}

Research Query:
{query}

PubMed Papers Found ({paper_count} papers):
{papers_text}

Please provide a plain-language summary of what this research means
for the patient's situation. Focus on practical relevance.
"""


# -----------------------------------------------------------
# RESEARCH AGENT
# -----------------------------------------------------------

class ResearchAgent(BaseAgent):
    """
    Medical literature research agent.

    Workflow:
        1. Builds smart PubMed query from patient + trial context
        2. Searches PubMed via SearchTool
        3. Summarizes findings via LLM
        4. Returns plain-language research summary
    """

    def __init__(self):
        self._search_tool = SearchTool()

    @property
    def name(self) -> str:
        return "research_agent"

    # -----------------------------------------------------------
    # INTERNAL HELPERS
    # -----------------------------------------------------------

    def _build_pubmed_query(
        self,
        patient_profile: Optional[Dict[str, Any]],
        topic: Optional[str] = None,
    ) -> str:
        """
        Builds a targeted PubMed search query from patient context.

        Examples:
            "EGFR Exon 19 deletion NSCLC treatment 2023 2024"
            "Stage IV non-small cell lung cancer immunotherapy"
            "PD-L1 high expression pembrolizumab efficacy"
        """
        parts = []

        # Add specific topic if provided
        if topic:
            parts.append(topic)

        if patient_profile:
            # Cancer type
            cancer_type = patient_profile.get("cancer_type") or ""
            if cancer_type:
                # Simplify MeSH terms for PubMed
                if "Non-Small-Cell" in cancer_type or "NSCLC" in cancer_type:
                    parts.append("non-small cell lung cancer NSCLC")
                elif "Breast" in cancer_type:
                    parts.append("breast cancer")
                elif "Colorectal" in cancer_type:
                    parts.append("colorectal cancer")
                else:
                    parts.append(cancer_type[:50])

            # Key biomarkers — most clinically relevant
            biomarkers = patient_profile.get("biomarkers") or []
            relevant_biomarkers = [
                b for b in biomarkers
                if any(
                    marker in b.upper()
                    for marker in [
                        "EGFR", "ALK", "ROS1", "KRAS",
                        "BRAF", "PD-L1", "HER2", "BRCA",
                        "MET", "RET", "NTRK",
                    ]
                )
            ][:2]  # Max 2 biomarkers to keep query focused

            for bm in relevant_biomarkers:
                # Clean up biomarker string
                clean_bm = bm.split(":")[0].strip()
                clean_bm = clean_bm.replace("-", "").replace("+", "").strip()
                if clean_bm:
                    parts.append(clean_bm)

            # Stage
            stage = patient_profile.get("cancer_stage") or ""
            if "IV" in stage or "4" in stage:
                parts.append("metastatic advanced")
            elif "III" in stage or "3" in stage:
                parts.append("locally advanced")

        # Add recency filter
        parts.append("treatment clinical trial 2022 2023 2024")

        query = " ".join(parts)
        logger.info("ResearchAgent built query: '%s'", query)
        return query

    def _format_papers_for_prompt(
        self,
        papers: List[Dict[str, Any]],
    ) -> str:
        """Formats papers list into a readable text block for the LLM."""
        if not papers:
            return "No papers found."

        lines = []
        for i, paper in enumerate(papers, 1):
            lines.append(
                f"{i}. {paper.get('title', 'Unknown Title')} "
                f"({paper.get('year', 'N/A')}) "
                f"— {paper.get('journal', '')}"
            )
            abstract = paper.get("abstract", "")
            if abstract:
                lines.append(f"   Abstract: {abstract[:400]}...")
            lines.append("")

        return "\n".join(lines)

    def _build_patient_context_text(
        self,
        patient_profile: Optional[Dict[str, Any]],
    ) -> str:
        """Builds compact patient context for the research prompt."""
        if not patient_profile:
            return "Not available."

        parts = []
        if patient_profile.get("cancer_type"):
            parts.append(f"Cancer: {patient_profile['cancer_type']}")
        if patient_profile.get("cancer_stage"):
            parts.append(f"Stage: {patient_profile['cancer_stage']}")

        biomarkers = patient_profile.get("biomarkers") or []
        if biomarkers:
            parts.append(f"Biomarkers: {', '.join(biomarkers[:3])}")

        treatments = patient_profile.get("previous_treatments") or []
        if treatments:
            parts.append(f"Prior treatments: {', '.join(treatments[:2])}")

        return " | ".join(parts) if parts else "Not available."

    # -----------------------------------------------------------
    # PUBLIC RUN
    # -----------------------------------------------------------

    async def run(self, payload: Any) -> Any:
        """
        Execute a medical research query.

        Expected payload:
        {
            "topic": "...",               # optional specific topic
            "patient_profile": {...},     # optional patient context
            "trial_context": "...",       # optional trial context
        }

        Returns:
        {
            "query": "...",              # PubMed query used
            "paper_count": N,            # papers found
            "summary": "...",            # plain language summary
            "papers": [...],             # raw paper data
        }
        """
        if not isinstance(payload, dict):
            raise ValueError("ResearchAgent payload must be a dict.")

        topic = (payload.get("topic") or "").strip()
        patient_profile = payload.get("patient_profile")
        trial_context = payload.get("trial_context", "")

        logger.info(
            "ResearchAgent.run | topic='%s'",
            topic or "auto-generated from profile",
        )

        # --- Step 1: Build query ---
        query = self._build_pubmed_query(patient_profile, topic or None)

        # --- Step 2: Search PubMed ---
        try:
            papers = await self._search_tool.execute({"query": query})
        except Exception:
            logger.exception("ResearchAgent: SearchTool failed")
            papers = []

        if not papers:
            return {
                "query": query,
                "paper_count": 0,
                "summary": (
                    "No recent PubMed literature was found for this specific "
                    "query. This may be due to the specificity of the search "
                    "terms or network availability. Please consult an oncologist "
                    "for the latest research guidance."
                ),
                "papers": [],
            }

        # --- Step 3: Summarize via LLM ---
        patient_context_text = self._build_patient_context_text(
            patient_profile
        )
        papers_text = self._format_papers_for_prompt(papers)

        # Add trial context if available
        full_context = patient_context_text
        if trial_context:
            full_context += f"\n\nTrial Context: {trial_context[:300]}"

        user_prompt = _RESEARCH_USER_PROMPT.format(
            patient_context=full_context,
            query=query,
            paper_count=len(papers),
            papers_text=papers_text,
        )

        try:
            llm = get_llm()
            messages = [
                SystemMessage(content=_RESEARCH_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            response = await llm.ainvoke(messages)
            summary = response.content.strip()
        except Exception:
            logger.exception("ResearchAgent: LLM summarization failed")
            summary = (
                f"Found {len(papers)} relevant papers on PubMed for your query. "
                f"Key papers include: "
                + ", ".join(
                    p.get("title", "Unknown")[:60]
                    for p in papers[:3]
                )
                + ". Please consult your oncologist for interpretation."
            )

        logger.info(
            "ResearchAgent complete | query='%s' | papers=%d | summary_length=%d",
            query,
            len(papers),
            len(summary),
        )

        return {
            "query": query,
            "paper_count": len(papers),
            "summary": summary,
            "papers": [
                {
                    "pmid": p.get("pmid"),
                    "title": p.get("title"),
                    "year": p.get("year"),
                    "journal": p.get("journal"),
                }
                for p in papers
            ],
        }
