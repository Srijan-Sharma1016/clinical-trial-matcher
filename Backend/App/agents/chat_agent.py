# agents/chat_agent.py
"""
Chat agent — conversational oncology assistant.
Responsibility: Multi-turn conversation with patient profile context.
Depends on: agents/base.py, agents/chains.py, memory/in_memory.py
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agents.research_agent import ResearchAgent   
from agents.sql_agent import SQLAgent
from agents.base import BaseAgent
from agents.chains import get_llm
from memory.in_memory import InMemoryStore

logger = logging.getLogger("uvicorn.error")

__all__ = ["ChatAgent"]

# -----------------------------------------------------------
# SYSTEM PROMPT
# -----------------------------------------------------------

_CHAT_SYSTEM_PROMPT = """You are a compassionate and knowledgeable oncology assistant.
You help patients and clinicians understand clinical trial options and cancer-related questions.

Your responsibilities:
- Answer questions about clinical trials clearly and in plain language
- Explain medical terms, biomarkers, and cancer stages simply
- Help interpret trial eligibility criteria
- Provide general oncology education
- Reference the patient's profile when relevant and available

Important rules:
- NEVER provide definitive medical advice or treatment recommendations
- ALWAYS recommend consulting a qualified oncologist for medical decisions
- If you don't know something, say so clearly — never hallucinate medical facts
- Keep responses concise, warm, and supportive
- If patient profile is available, use it to personalize answers

Patient Profile Context (if available):
{patient_context}
"""

# -----------------------------------------------------------
# CHAT AGENT
# -----------------------------------------------------------

class ChatAgent(BaseAgent):
    """
    Conversational oncology assistant with session memory.

    Features:
        - Multi-turn conversation with history
        - Patient profile awareness
        - General oncology knowledge
        - Clinical trial explanation
        - In-memory session management
    """

    def __init__(self):
        self._memory = InMemoryStore()
        self._research_agent = ResearchAgent()
        self._sql_agent = SQLAgent() 

    @property
    def name(self) -> str:
        return "chat_agent"

    # -----------------------------------------------------------
    # INTERNAL HELPERS
    # -----------------------------------------------------------

    def _is_analytics_question(self, message: str) -> bool:
        """
        Detects if the user is asking an analytics/database question.
        Triggers SQLAgent if True.
        """
        analytics_triggers = [
        "how many", "how much", "count", "total",
        "show me all", "list all", "list patients",
        "how many patients", "how many trials",
        "most common", "most recent", "latest patients",
        "statistics", "stats", "summary of patients",
        "database", "records", "stored",
        "have we seen", "have we matched",
        "what patients", "which patients",
    ]
        message_lower = message.lower()
        return any(trigger in message_lower for trigger in analytics_triggers)

    def _is_research_question(self, message: str) -> bool:
        """
        Detects if the user is asking for research/literature context.
        Triggers ResearchAgent if True.
        """
        research_triggers = [
            "research", "study", "studies", "evidence", "literature",
            "paper", "papers", "published", "pubmed", "journal",
            "what does research", "what do studies", "clinical evidence",
            "data show", "data says", "proven", "efficacy", "findings",
            "latest", "recent findings", "what science says",
        ]
        message_lower = message.lower()
        return any(trigger in message_lower for trigger in research_triggers)

    def _build_patient_context(
        self,
        patient_profile: Optional[Dict[str, Any]],
    ) -> str:
        """
        Builds a plain-text patient context string for the system prompt.
        Returns 'Not available' if no profile provided.
        """
        if not patient_profile:
            return "Not available."

        lines = []

        if patient_profile.get("age"):
            lines.append(f"- Age: {patient_profile['age']}")
        if patient_profile.get("gender"):
            lines.append(f"- Gender: {patient_profile['gender']}")
        if patient_profile.get("cancer_type"):
            lines.append(f"- Cancer Type: {patient_profile['cancer_type']}")
        if patient_profile.get("cancer_stage"):
            lines.append(f"- Cancer Stage: {patient_profile['cancer_stage']}")
        if patient_profile.get("diagnosis"):
            lines.append(f"- Diagnosis: {patient_profile['diagnosis']}")

        biomarkers = patient_profile.get("biomarkers") or []
        if biomarkers:
            lines.append(f"- Biomarkers: {', '.join(biomarkers)}")

        treatments = patient_profile.get("previous_treatments") or []
        if treatments:
            lines.append(f"- Previous Treatments: {', '.join(treatments)}")

        if patient_profile.get("country"):
            lines.append(f"- Country: {patient_profile['country']}")

        return "\n".join(lines) if lines else "Not available."

    def _build_trial_context(
        self,
        trial_matches: Optional[Dict[str, Any]],
    ) -> str:
        """
        Builds a plain-text trial context string from match results.
        Summarizes top trials for the LLM context window.
        """
        if not trial_matches:
            return ""

        eligibility_results = trial_matches.get("eligibility_results") or []
        if not eligibility_results:
            return ""

        lines = ["\nTrial Match Context:"]

        # Include top 3 trials by score
        sorted_results = sorted(
            eligibility_results,
            key=lambda r: r.get("score", 0),
            reverse=True,
        )[:3]

        for result in sorted_results:
            lines.append(
                f"- {result.get('title', 'Unknown')} "
                f"({result.get('nct_id', 'N/A')}) "
                f"| Score: {result.get('score', 0)} "
                f"| Filter: {'Pass' if result.get('hard_filter_pass') else 'Fail'}"
            )

        final_recs = trial_matches.get("final_recommendations", "")
        if final_recs:
            # Truncate to avoid bloating context
            lines.append(
                f"\nFinal Recommendations Summary:\n"
                f"{final_recs[:500]}..."
                if len(final_recs) > 500
                else f"\nFinal Recommendations Summary:\n{final_recs}"
            )

        return "\n".join(lines)

    async def _load_history(
        self,
        session_id: str,
    ) -> List[Any]:
        """Loads conversation history for a session."""
        history = await self._memory.load(session_id)
        return history if isinstance(history, list) else []

    async def _save_history(
        self,
        session_id: str,
        history: List[Any],
    ) -> None:
        """Saves conversation history — keeps last 20 messages to manage tokens."""
        # Keep last 20 messages max — prevents token overflow
        trimmed = history[-20:] if len(history) > 20 else history
        await self._memory.save(session_id, trimmed)

    # -----------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------
    def _is_research_question(self, message: str) -> bool:
        """
        Detects if the user is asking for research/literature context.
        Triggers ResearchAgent if True.
        """
        research_triggers = [
            "research", "study", "studies", "evidence", "literature",
            "paper", "papers", "published", "pubmed", "journal",
            "what does research", "what do studies", "clinical evidence",
            "data show", "data says", "proven", "efficacy", "findings",
            "latest", "recent findings", "what science says",
        ]
        message_lower = message.lower()
        return any(trigger in message_lower for trigger in research_triggers)

    def _is_research_question(self, message: str) -> bool:
        """
        Detects if the user is asking for research/literature context.
        Triggers ResearchAgent if True.
        """
        research_triggers = [
            "research", "study", "studies", "evidence", "literature",
            "paper", "papers", "published", "pubmed", "journal",
            "what does research", "what do studies", "clinical evidence",
            "data show", "data says", "proven", "efficacy", "findings",
            "latest", "recent findings", "what science says",
        ]
        return any(t in message.lower() for t in research_triggers)

    def _is_analytics_question(self, message: str) -> bool:
        """
        Detects if the user is asking an analytics/database question.
        Triggers SQLAgent if True.
        """
        analytics_triggers = [
            "how many", "how much", "count", "total",
            "show me all", "list all", "list patients",
            "how many patients", "how many trials",
            "most common", "most recent", "latest patients",
            "statistics", "stats", "summary of patients",
            "database", "records", "stored",
            "have we seen", "have we matched",
            "what patients", "which patients",
        ]
        return any(t in message.lower() for t in analytics_triggers)

    async def run(self, payload: Any) -> Any:
        """
        Handle a conversational message.

        Expected payload:
        {
            "session_id": "...",          # optional — generated if missing
            "message": "...",             # required — user message
            "patient_profile": {...},     # optional — for context
            "trial_matches": {...},       # optional — for context
        }

        Returns:
        {
            "session_id": "...",
            "response": "...",
            "message_count": N,
        }
        """
        # --- Validate input ---
        if not isinstance(payload, dict):
            raise ValueError("ChatAgent payload must be a dict.")

        message = (payload.get("message") or "").strip()
        if not message:
            raise ValueError("ChatAgent payload must include a 'message' field.")

        # --- Session management ---
        session_id = payload.get("session_id") or str(uuid.uuid4())
        patient_profile = payload.get("patient_profile")
        trial_matches = payload.get("trial_matches")

        logger.info(
            "ChatAgent.run | session_id=%s | message_length=%d",
            session_id,
            len(message),
        )

        # --- Build base context ---
        patient_context = self._build_patient_context(patient_profile)
        trial_context = self._build_trial_context(trial_matches)
        full_context = patient_context + trial_context

        # --- Research enrichment ---
        # Triggered only when research keywords detected + profile available
        research_context = ""
        if self._is_research_question(message) and patient_profile:
            logger.info(
                "ChatAgent: Research question detected — triggering ResearchAgent"
            )
            try:
                research_result = await self._research_agent.run({
                    "patient_profile": patient_profile,
                    "topic": message[:200],
                    "trial_context": trial_context[:300],
                })
                if research_result.get("summary"):
                    research_context = (
                        f"\n\nRecent Research Context "
                        f"({research_result.get('paper_count', 0)} papers found):\n"
                        f"{research_result['summary']}"
                    )
                    logger.info(
                        "ChatAgent: Research context added | papers=%d",
                        research_result.get("paper_count", 0),
                    )
            except Exception:
                logger.exception(
                    "ChatAgent: ResearchAgent call failed — "
                    "continuing without research context"
                )

        # --- Analytics enrichment ---
        # Triggered only when analytics keywords detected
        # Token efficient: SQLAgent uses small model + no LLM summarization
        analytics_context = ""
        if self._is_analytics_question(message):
            logger.info(
                "ChatAgent: Analytics question detected — triggering SQLAgent"
            )
            try:
                sql_result = await self._sql_agent.run({
                    "question": message,
                })
                if sql_result.get("success") and sql_result.get("answer"):
                    analytics_context = (
                        f"\n\nDatabase Query Results:\n"
                        f"{sql_result['answer']}"
                    )
                    logger.info(
                        "ChatAgent: Analytics context added | rows=%d",
                        sql_result.get("row_count", 0),
                    )
                elif not sql_result.get("success"):
                    analytics_context = (
                        f"\n\nDatabase Query Note: "
                        f"{sql_result.get('answer', '')}"
                    )
            except Exception:
                logger.exception(
                    "ChatAgent: SQLAgent call failed — "
                    "continuing without analytics context"
                )

        # --- Merge ALL context ---
        full_context = full_context + research_context + analytics_context

        # --- Load history ---
        history = await self._load_history(session_id)

        # --- Build messages for LLM ---
        system_content = _CHAT_SYSTEM_PROMPT.format(
            patient_context=full_context
        )

        messages = [SystemMessage(content=system_content)]
        messages.extend(history)
        messages.append(HumanMessage(content=message))

        # --- Invoke LLM ---
        try:
            llm = get_llm()
            response = await llm.ainvoke(messages)
            response_text = response.content.strip()
        except Exception:
            logger.exception(
                "ChatAgent LLM call failed for session '%s'", session_id
            )
            response_text = (
                "I'm sorry, I encountered an error processing your question. "
                "Please try again or consult your oncologist directly."
            )

        # --- Update and save history ---
        history.append(HumanMessage(content=message))
        history.append(AIMessage(content=response_text))
        await self._save_history(session_id, history)

        logger.info(
            "ChatAgent response | session_id=%s | "
            "response_length=%d | history_length=%d",
            session_id,
            len(response_text),
            len(history),
        )

        return {
            "session_id": session_id,
            "response": response_text,
            "message_count": len(history) // 2,
        }

    async def clear_session(self, session_id: str) -> None:
        """Clears conversation history for a session."""
        await self._memory.delete(session_id)
        logger.info("ChatAgent session cleared | session_id=%s", session_id)

    async def get_history(
        self,
        session_id: str,
    ) -> List[Dict[str, str]]:
        """
        Returns conversation history as a list of
        {"role": "user/assistant", "content": "..."} dicts.
        Useful for frontend rendering.
        """
        history = await self._load_history(session_id)
        result = []
        for msg in history:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
        return result
