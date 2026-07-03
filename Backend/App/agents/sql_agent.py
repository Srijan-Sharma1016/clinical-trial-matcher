# agents/sql_agent.py
"""
SQL agent — natural language to SQL query executor.
Responsibility: Convert user questions to safe SELECT queries.
Token efficient: minimal schema, small model, truncated results.
Depends on: agents/base.py, database.py, models.py
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from sqlmodel import Session, text

from agents.base import BaseAgent
from config.settings import GROQ_API_KEY
from database import engine

logger = logging.getLogger("uvicorn.error")

__all__ = ["SQLAgent"]

# -----------------------------------------------------------
# TOKEN EFFICIENT — Use small fast model for SQL generation
# -----------------------------------------------------------

_sql_llm = ChatGroq(
    model="llama-3.1-8b-instant",   # ← small model, fast, cheap
    api_key=GROQ_API_KEY,
    temperature=0,
    max_retries=2,
)

# -----------------------------------------------------------
# MINIMAL SCHEMA CONTEXT
# Token efficient — only what the LLM needs, nothing more
# -----------------------------------------------------------

_MINIMAL_SCHEMA = """
Tables:
1. patient_profiles
   - id, age, gender, cancer_type, cancer_stage
   - biomarkers (JSON array), previous_treatments (JSON array)
   - created_at

2. trial_matches
   - id, patient_profile_id (FK), nct_id, title
   - match_explanation (text), created_at
"""

# -----------------------------------------------------------
# SYSTEM PROMPT — token efficient, concise
# -----------------------------------------------------------

_SQL_SYSTEM_PROMPT = """You are a SQL query generator for a clinical trial database.
Generate a single safe PostgreSQL SELECT query.

Schema:
{schema}

Rules:
- SELECT only — no INSERT, UPDATE, DELETE, DROP, ALTER
- Max 5 rows — always add LIMIT 5
- Simple queries only — no complex subqueries
- Return ONLY the SQL query — no explanation, no markdown, no backticks

If the question cannot be answered with SELECT, respond with: INVALID
"""

_SQL_USER_PROMPT = "Question: {question}\nSQL:"


# -----------------------------------------------------------
# SQL AGENT
# -----------------------------------------------------------

class SQLAgent(BaseAgent):
    """
    Token-efficient natural language to SQL agent.

    Strategy:
        - Small model (llama-3.1-8b-instant) for SQL generation
        - Minimal schema in prompt (100 tokens)
        - Hard SELECT-only enforcement
        - Results truncated to 5 rows max
        - No LLM summarization — direct results returned
    """

    @property
    def name(self) -> str:
        return "sql_agent"

    # -----------------------------------------------------------
    # INTERNAL HELPERS
    # -----------------------------------------------------------

    def _is_safe_query(self, sql: str) -> bool:
        """
        Hard safety check — only allows SELECT statements.
        Blocks any destructive or mutating SQL.
        """
        if not sql:
            return False

        sql_upper = sql.strip().upper()

        # Must start with SELECT
        if not sql_upper.startswith("SELECT"):
            return False

        # Block dangerous keywords
        dangerous = [
            "INSERT", "UPDATE", "DELETE", "DROP",
            "ALTER", "TRUNCATE", "CREATE", "REPLACE",
            "EXEC", "EXECUTE", "GRANT", "REVOKE",
        ]
        return not any(keyword in sql_upper for keyword in dangerous)

    def _generate_sql(self, question: str) -> Optional[str]:
        """
        Calls small LLM to generate SQL from natural language.
        Returns None if generation fails or query is invalid.
        """
        messages = [
            SystemMessage(
                content=_SQL_SYSTEM_PROMPT.format(schema=_MINIMAL_SCHEMA)
            ),
            HumanMessage(
                content=_SQL_USER_PROMPT.format(question=question)
            ),
        ]

        try:
            response = _sql_llm.invoke(messages)
            sql = response.content.strip()

            # Clean up common LLM artifacts
            sql = sql.replace("```sql", "").replace("```", "").strip()

            if sql.upper() == "INVALID":
                logger.info("SQLAgent: LLM marked question as INVALID")
                return None

            logger.info("SQLAgent generated SQL: %s", sql)
            return sql

        except Exception:
            logger.exception("SQLAgent: SQL generation failed")
            return None

    def _execute_query(
        self,
        sql: str,
    ) -> Dict[str, Any]:
        """
        Executes the SQL query safely.
        Returns results as list of dicts, truncated to 5 rows.
        """
        try:
            with Session(engine) as session:
                result = session.execute(text(sql))
                columns = list(result.keys())
                rows = result.fetchmany(5)   # ← hard limit 5 rows

                formatted_rows = [
                    dict(zip(columns, row))
                    for row in rows
                ]

                logger.info(
                    "SQLAgent query executed | rows=%d | columns=%s",
                    len(formatted_rows),
                    columns,
                )

                return {
                    "columns": columns,
                    "rows": formatted_rows,
                    "row_count": len(formatted_rows),
                }

        except Exception as e:
            logger.exception("SQLAgent query execution failed")
            return {
                "columns": [],
                "rows": [],
                "row_count": 0,
                "error": str(e),
            }

    def _format_results_as_text(
        self,
        columns: List[str],
        rows: List[Dict[str, Any]],
    ) -> str:
        """
        Formats query results as plain readable text.
        No LLM call needed — token efficient!
        """
        if not rows:
            return "No results found."

        lines = []

        # Header
        lines.append(f"Found {len(rows)} result(s):")
        lines.append("")

        # Rows
        for i, row in enumerate(rows, 1):
            row_parts = []
            for col in columns:
                val = row.get(col)
                if val is not None:
                    # Truncate long values
                    val_str = str(val)
                    if len(val_str) > 80:
                        val_str = val_str[:80] + "..."
                    row_parts.append(f"{col}: {val_str}")
            lines.append(f"{i}. {' | '.join(row_parts)}")

        return "\n".join(lines)

    # -----------------------------------------------------------
    # PUBLIC RUN
    # -----------------------------------------------------------

    async def run(self, payload: Any) -> Any:
        """
        Execute a natural language database query.

        Expected payload:
        {
            "question": "How many patients have we matched?"
        }

        Returns:
        {
            "question": "...",
            "sql": "...",
            "answer": "...",       # plain text — no LLM summarization
            "row_count": N,
            "success": True/False,
        }
        """
        if not isinstance(payload, dict):
            raise ValueError("SQLAgent payload must be a dict.")

        question = (payload.get("question") or "").strip()
        if not question:
            raise ValueError("SQLAgent payload must include a 'question' field.")

        logger.info("SQLAgent.run | question='%s'", question)

        # --- Step 1: Generate SQL ---
        sql = self._generate_sql(question)

        if not sql:
            return {
                "question": question,
                "sql": None,
                "answer": (
                    "I couldn't generate a valid database query for that question. "
                    "Try asking something like: "
                    "'How many patients have we analyzed?' or "
                    "'What are the most recent trial matches?'"
                ),
                "row_count": 0,
                "success": False,
            }

        # --- Step 2: Safety check ---
        if not self._is_safe_query(sql):
            logger.warning(
                "SQLAgent: Unsafe query blocked | sql=%s", sql
            )
            return {
                "question": question,
                "sql": sql,
                "answer": "That query is not permitted for security reasons.",
                "row_count": 0,
                "success": False,
            }

        # --- Step 3: Execute ---
        result = self._execute_query(sql)

        if result.get("error"):
            return {
                "question": question,
                "sql": sql,
                "answer": (
                    f"The query ran into an error: {result['error']}. "
                    "Please try rephrasing your question."
                ),
                "row_count": 0,
                "success": False,
            }

        # --- Step 4: Format results as text (NO LLM call!) ---
        answer = self._format_results_as_text(
            result["columns"],
            result["rows"],
        )

        logger.info(
            "SQLAgent complete | question='%s' | rows=%d",
            question,
            result["row_count"],
        )

        return {
            "question": question,
            "sql": sql,
            "answer": answer,
            "row_count": result["row_count"],
            "success": True,
        }
