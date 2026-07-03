# agents/oncology_agent.py
"""
Single deep oncology agent.
Responsibility: Unified conversational agent with tool-calling capability.
Replaces: chat_agent.py + research_agent.py + sql_agent.py (as orchestrator)
Depends on: agents/base.py, agents/chains.py, memory/in_memory.py,
            tools/search_tool.py, database.py
"""

import asyncio
import concurrent.futures
import inspect
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from agents.base import BaseAgent
from agents.chains import get_llm, get_tool_llm
from memory.in_memory import InMemoryStore
from tools.search_tool import SearchTool
from agents.sql_agent import SQLAgent

logger = logging.getLogger("uvicorn.error")

__all__ = ["OncologyAgent"]

# -----------------------------------------------------------
# SYSTEM PROMPT
# -----------------------------------------------------------

_SYSTEM_PROMPT = """You are a compassionate and highly knowledgeable oncology assistant.
You help patients and clinicians understand clinical trial options and cancer-related questions.

You have access to two powerful tools:
1. search_pubmed       — Search PubMed for peer-reviewed medical literature
2. query_database      — Query the application database for patient and trial analytics

Your responsibilities:
- Answer questions about clinical trials clearly in plain language
- Explain medical terms, biomarkers, and cancer stages simply
- Search PubMed when asked about research, evidence, or latest findings
- Query the database when asked about patient counts, statistics, or records
- Use BOTH tools only when clearly required by the user's question
- Provide general oncology education

Decision rules for tool use:
- Use tools only when necessary.
- Questions about research/studies/evidence/latest findings → use search_pubmed.
- Questions about counts/statistics/records/system data → use query_database.
- Questions needing both research and system data → use both tools only when clearly required.
- General oncology questions or questions answerable from provided patient/trial context → respond directly without tools.
- If you use a tool, call only the provided tool names.
- Tool arguments must be valid JSON matching the tool schema.

Important rules:
- NEVER provide definitive medical advice.
- ALWAYS recommend consulting a qualified oncologist.
- Never hallucinate medical facts.
- Keep responses concise, warm, and supportive.
- Always use patient profile context when available.
- If clinical trial matches are provided, explain them as options to discuss with the care team, not as guaranteed recommendations.

Patient Profile Context:
{patient_context}
"""

# -----------------------------------------------------------
# TOOL DEFINITIONS
# -----------------------------------------------------------

_search_tool = SearchTool()
_sql_agent = SQLAgent()


def _run_backend_call(value: Any) -> Any:
    """
    Safely handles either:
    - sync backend return values
    - async coroutine/awaitable backend calls

    Tool functions are sync because LangChain tool wrappers can run them
    in a worker context. If the backend is async, this runs it inside
    a separate thread with its own event loop to avoid FastAPI loop conflicts.
    """
    if not inspect.isawaitable(value):
        return value

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, value)
        return future.result(timeout=30)


@tool
def search_pubmed(query: str) -> str:
    """
    Search PubMed for peer-reviewed medical literature.

    Use this when the user asks about research, studies, evidence,
    clinical findings, or latest medical developments.

    Args:
        query: Medical search query e.g. 'EGFR Exon 19 NSCLC treatment 2024'

    Returns:
        Summary of relevant research papers found on PubMed.
    """
    logger.info("Tool: search_pubmed called | query='%s'", query)

    try:
        papers = _run_backend_call(_search_tool.execute({"query": query}))

        if not papers:
            return "No relevant PubMed papers found for this query."

        if not isinstance(papers, list):
            return json.dumps(papers, ensure_ascii=False)

        lines = [f"Found {len(papers)} papers on PubMed:\n"]

        for i, paper in enumerate(papers[:4], 1):
            if not isinstance(paper, dict):
                lines.append(f"{i}. {str(paper)}")
                lines.append("")
                continue

            title = paper.get("title", "Unknown")
            year = paper.get("year", "N/A")
            journal = paper.get("journal", "")

            lines.append(f"{i}. {title} ({year}) — {journal}")

            abstract = paper.get("abstract", "")
            if abstract:
                lines.append(f"   Key finding: {str(abstract)[:300]}...")

            lines.append("")

        return "\n".join(lines)

    except Exception:
        logger.exception("Tool: search_pubmed failed")
        return "PubMed search encountered an error. Please try again."


@tool
def query_database(question: str) -> str:
    """
    Query the application database for patient profiles and trial match analytics.

    Use this when the user asks about counts, statistics, records,
    or any data stored in the system.

    Args:
        question: Natural language question e.g. 'How many patients have been analyzed?'

    Returns:
        Query results as plain text.
    """
    logger.info("Tool: query_database called | question='%s'", question)

    try:
        result = _run_backend_call(_sql_agent.run({"question": question}))

        if not isinstance(result, dict):
            return str(result)

        if result.get("success"):
            return result.get("answer", "No results found.")

        return (
            result.get("answer")
            or result.get("error")
            or "Could not retrieve data for that question."
        )

    except Exception:
        logger.exception("Tool: query_database failed")
        return "Database query encountered an error. Please try again."


_TOOLS = [search_pubmed, query_database]


# -----------------------------------------------------------
# ONCOLOGY AGENT
# -----------------------------------------------------------

class OncologyAgent(BaseAgent):
    """
    Single deep oncology agent with tool-calling capability.

    Architecture:
        - Tool-bound LLM decides whether tools are needed
        - Tool calls executed and results fed back to LLM
        - Fallback plain LLM handles Groq tool-call generation failures
        - Session memory maintained per conversation
    """

    def __init__(self):
        super().__init__()
        self._memory = InMemoryStore()
        self._llm = get_llm()
        self._llm_with_tools = get_tool_llm().bind_tools(_TOOLS)

    @property
    def name(self) -> str:
        return "oncology_agent"

    # -----------------------------------------------------------
    # INTERNAL HELPERS
    # -----------------------------------------------------------

    def _stringify(self, value: Any) -> str:
        """
        Converts any value into a safe string for prompts/responses.
        """
        if value is None:
            return ""

        if isinstance(value, str):
            return value

        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return str(value)

        return str(value)

    def _is_tool_use_failure(self, exc: Exception) -> bool:
        """
        Detects Groq/LangChain tool-call generation failures.

        Groq can reject malformed function/tool calls before Python receives
        them. When that happens, retry with plain non-tool LLM.
        """
        text = str(exc).lower()

        return (
            "tool_use_failed" in text
            or "failed to call a function" in text
            or "failed_generation" in text
            or "invalid_request_error" in text
        )

    def _normalize_tool_args(
        self,
        tool_name: Optional[str],
        tool_args: Any,
    ) -> Dict[str, Any]:
        """
        Makes tool arguments safe before invoking LangChain tools.
        Handles dict args, JSON string args, or plain string args.
        """
        if isinstance(tool_args, dict):
            return tool_args

        if isinstance(tool_args, str):
            try:
                parsed = json.loads(tool_args)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

            if tool_name == "search_pubmed":
                return {"query": tool_args}

            if tool_name == "query_database":
                return {"question": tool_args}

            return {"input": tool_args}

        return {}

    def _build_patient_context(
        self,
        patient_profile: Optional[Dict[str, Any]],
    ) -> str:
        if not patient_profile:
            return "No patient profile available."

        parts = []

        if patient_profile.get("age"):
            parts.append(f"Age: {self._stringify(patient_profile.get('age'))}")

        if patient_profile.get("gender"):
            parts.append(
                f"Gender: {self._stringify(patient_profile.get('gender'))}"
            )

        if patient_profile.get("cancer_type"):
            parts.append(
                f"Cancer: {self._stringify(patient_profile.get('cancer_type'))}"
            )

        if patient_profile.get("cancer_stage"):
            parts.append(
                f"Stage: {self._stringify(patient_profile.get('cancer_stage'))}"
            )

        biomarkers = patient_profile.get("biomarkers") or []
        if isinstance(biomarkers, list) and biomarkers:
            biomarker_text = [self._stringify(b) for b in biomarkers[:4]]
            parts.append(f"Biomarkers: {', '.join(biomarker_text)}")
        elif biomarkers:
            parts.append(f"Biomarkers: {self._stringify(biomarkers)}")

        treatments = patient_profile.get("previous_treatments") or []
        if isinstance(treatments, list) and treatments:
            treatment_text = [self._stringify(t) for t in treatments[:3]]
            parts.append(f"Prior treatments: {', '.join(treatment_text)}")
        elif treatments:
            parts.append(f"Prior treatments: {self._stringify(treatments)}")

        return " | ".join(parts) if parts else "Profile incomplete."

    async def _load_history(self, session_id: str) -> List[Any]:
        history = await self._memory.load(session_id)
        return history if isinstance(history, list) else []

    async def _save_history(
        self,
        session_id: str,
        history: List[Any],
    ) -> None:
        # Keep last 20 messages to manage tokens.
        trimmed = history[-20:] if len(history) > 20 else history
        await self._memory.save(session_id, trimmed)

    async def _invoke_llm_with_tool_fallback(
        self,
        messages: List[Any],
        session_id: str,
    ):
        """
        Calls the tool-bound LLM first.

        If Groq fails while generating a tool call, retry using plain LLM.
        Returns:
            (ai_response, used_fallback)
        """
        try:
            response = await self._llm_with_tools.ainvoke(messages)
            return response, False

        except Exception as exc:
            if not self._is_tool_use_failure(exc):
                raise

            logger.warning(
                "OncologyAgent: Groq tool-call generation failed; "
                "falling back to plain LLM | session_id=%s | error=%s",
                session_id,
                str(exc),
            )

            fallback_messages = messages + [
                SystemMessage(
                    content=(
                        "Internal tool calling failed. "
                        "Answer directly using the available patient profile, "
                        "trial matching context, and conversation history. "
                        "Do not mention internal tool failure to the user."
                    )
                )
            ]

            response = await self._llm.ainvoke(fallback_messages)
            return response, True

    async def _synthesize_with_tool_results(
        self,
        original_messages: List[Any],
        tool_results: List[ToolMessage],
        session_id: str,
    ) -> str:
        """
        Generates the final user-facing response after tool execution.

        First tries the normal tool-message flow. If Groq fails again,
        falls back to a plain LLM prompt with tool results serialized as text.
        """
        try:
            final_response = await self._llm_with_tools.ainvoke(
                original_messages + tool_results
            )
            return self._stringify(final_response.content).strip()

        except Exception as exc:
            logger.warning(
                "OncologyAgent: final tool-result synthesis failed; "
                "retrying with plain LLM | session_id=%s | error=%s",
                session_id,
                str(exc),
            )

            tool_result_text = "\n\n".join(
                self._stringify(msg.content) for msg in tool_results
            )

            plain_messages = []

            # Keep only system/human/assistant content messages.
            # Avoid passing ToolMessage to a plain model if provider dislikes it.
            for msg in original_messages:
                if isinstance(msg, ToolMessage):
                    continue

                plain_messages.append(msg)

            plain_messages.append(
                SystemMessage(
                    content=(
                        "Use the following tool results to answer the user's "
                        "question clearly and safely. Do not mention internal "
                        "tool mechanics.\n\n"
                        f"Tool results:\n{tool_result_text}"
                    )
                )
            )

            final_response = await self._llm.ainvoke(plain_messages)
            return self._stringify(final_response.content).strip()

    async def _execute_tool_calls(
        self,
        tool_calls: List[Any],
    ) -> List[ToolMessage]:
        """
        Executes all tool calls requested by the LLM.
        Returns ToolMessage results to feed back into the conversation.
        """
        tool_map = {t.name: t for t in _TOOLS}
        results = []

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                logger.warning(
                    "OncologyAgent: malformed tool_call ignored | %s",
                    tool_call,
                )
                continue

            tool_name = tool_call.get("name")
            raw_tool_args = tool_call.get("args", {})
            tool_args = self._normalize_tool_args(tool_name, raw_tool_args)
            tool_call_id = tool_call.get("id") or str(uuid.uuid4())

            logger.info(
                "OncologyAgent: executing tool '%s' | args=%s",
                tool_name,
                tool_args,
            )

            selected_tool = tool_map.get(tool_name)

            if not selected_tool:
                result_content = f"Tool '{tool_name}' not found."
            else:
                try:
                    result_content = await selected_tool.ainvoke(tool_args)
                    result_content = self._stringify(result_content)
                except Exception:
                    logger.exception(
                        "OncologyAgent: tool '%s' execution failed",
                        tool_name,
                    )
                    result_content = (
                        f"Tool '{tool_name}' encountered an error."
                    )

            results.append(
                ToolMessage(
                    content=result_content,
                    tool_call_id=tool_call_id,
                )
            )

        return results

    # -----------------------------------------------------------
    # PUBLIC METHODS
    # -----------------------------------------------------------

    async def get_history(
        self,
        session_id: str,
    ) -> List[Dict[str, str]]:
        """
        Returns conversation history as list of role/content dicts for frontend.
        """
        history = await self._load_history(session_id)

        result = []

        for msg in history:
            if isinstance(msg, HumanMessage):
                result.append(
                    {
                        "role": "user",
                        "content": self._stringify(msg.content),
                    }
                )

            elif isinstance(msg, AIMessage) and msg.content:
                result.append(
                    {
                        "role": "assistant",
                        "content": self._stringify(msg.content),
                    }
                )

        return result

    async def clear_session(self, session_id: str) -> None:
        """
        Clears conversation history for a session.
        """
        await self._memory.delete(session_id)

        logger.info(
            "OncologyAgent session cleared | session_id=%s",
            session_id,
        )

    async def run(self, payload: Any) -> Any:
        """
        Handle a conversational message with tool-calling.

        Expected payload:
        {
            "session_id": "...",          # optional
            "message": "...",             # required
            "patient_profile": {...},     # optional
            "trial_matches": {...},       # optional
        }

        Returns:
        {
            "session_id": "...",
            "response": "...",
            "message_count": N,
            "tools_used": [...],
        }
        """
        # --- Validate ---
        if not isinstance(payload, dict):
            raise ValueError("OncologyAgent payload must be a dict.")

        message = self._stringify(payload.get("message")).strip()

        if not message:
            raise ValueError("OncologyAgent payload must include 'message'.")

        # --- Session ---
        session_id = payload.get("session_id") or str(uuid.uuid4())
        patient_profile = payload.get("patient_profile")
        trial_matches = payload.get("trial_matches")

        logger.info(
            "OncologyAgent.run | session_id=%s | message_length=%d",
            session_id,
            len(message),
        )

        # --- Build system prompt with patient context ---
        patient_context = self._build_patient_context(patient_profile)

        # Add trial context if available.
        if trial_matches:
            if isinstance(trial_matches, dict):
                trial_count = trial_matches.get("trial_count", 0)
                cancer_type = trial_matches.get("cancer_type", "")
                final_recs = trial_matches.get("final_recommendations", "")
            else:
                trial_count = ""
                cancer_type = ""
                final_recs = trial_matches

            final_recs_text = self._stringify(final_recs)

            patient_context += (
                f"\n\nTrial Matching Context:"
                f"\n- Cancer Type: {self._stringify(cancer_type)}"
                f"\n- Trials Evaluated: {self._stringify(trial_count)}"
                f"\n- Recommendations Summary: {final_recs_text[:800]}"
            )

        system_content = _SYSTEM_PROMPT.format(
            patient_context=patient_context
        )

        # --- Load history ---
        history = await self._load_history(session_id)

        # --- Build initial messages ---
        messages = [SystemMessage(content=system_content)]
        messages.extend(history)
        messages.append(HumanMessage(content=message))

        tools_used: List[str] = []
        response_text = ""

        try:
            # --- Step 1: Tool-bound LLM decides what to do ---
            ai_response, used_fallback = await self._invoke_llm_with_tool_fallback(
                messages,
                session_id,
            )

            if used_fallback:
                response_text = self._stringify(ai_response.content).strip()
                tools_used = []

                logger.info(
                    "OncologyAgent: fallback plain LLM response generated"
                )

            else:
                # --- Step 2: Check if LLM wants to use tools ---
                tool_calls = getattr(ai_response, "tool_calls", []) or []

                if tool_calls:
                    tool_names = [
                        tc.get("name")
                        for tc in tool_calls
                        if isinstance(tc, dict) and tc.get("name")
                    ]
                    tools_used = tool_names

                    logger.info(
                        "OncologyAgent: LLM requested tools: %s",
                        tool_names,
                    )

                    # Add LLM's tool-call message to conversation.
                    messages.append(ai_response)

                    # --- Step 3: Execute tools ---
                    tool_results = await self._execute_tool_calls(tool_calls)

                    # --- Step 4: Generate final response with tool results ---
                    response_text = await self._synthesize_with_tool_results(
                        messages,
                        tool_results,
                        session_id,
                    )

                else:
                    response_text = self._stringify(ai_response.content).strip()

                    logger.info(
                        "OncologyAgent: Direct response — no tools used"
                    )

        except Exception:
            logger.exception(
                "OncologyAgent LLM call failed for session '%s'",
                session_id,
            )

            response_text = (
                "I'm sorry, I encountered an error. "
                "Please try again or consult your oncologist directly."
            )

        if not response_text:
            response_text = (
                "I’m sorry, I couldn’t generate a complete response. "
                "Please try asking again, and discuss clinical decisions with "
                "a qualified oncologist."
            )

        # --- Update and save history ---
        history.append(HumanMessage(content=message))
        history.append(AIMessage(content=response_text))

        await self._save_history(session_id, history)

        logger.info(
            "OncologyAgent response | session_id=%s | "
            "response_length=%d | tools_used=%s | history=%d",
            session_id,
            len(response_text),
            tools_used,
            len(history),
        )

        return {
            "session_id": session_id,
            "response": response_text,
            "message_count": len(history) // 2,
            "tools_used": tools_used,
        }
