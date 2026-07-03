# tools/db_tool.py
"""
Database query tool.
Responsibility: Handle structured database queries from agent context.
Extend this for NL-to-SQL or direct DB lookup features.
"""

import logging
from typing import Any

from tools.base_tool import BaseTool

logger = logging.getLogger("uvicorn.error")

__all__ = ["DBTool"]


class DBTool(BaseTool):
    """
    Executes structured queries against the application database.

    Current status: Placeholder — not yet implemented.

    Planned capabilities:
        - Natural language → SQL via LLM chain
        - Direct structured query execution via SQLModel
        - Patient profile lookups
        - Trial match history retrieval
    """

    @property
    def name(self) -> str:
        return "db_tool"

    @property
    def description(self) -> str:
        return (
            "Queries the application database for patient profiles "
            "and trial match history. "
            "Supports structured queries and natural language lookups."
        )

    async def execute(self, payload: Any) -> Any:
        """
        Execute a database query.

        Future implementation outline:
            1. Parse payload — NL query string or structured dict
            2. If NL: run through NL-to-SQL LLM chain
            3. Execute via SQLModel session against engine
            4. Return results as list of dicts or model instances
            5. Handle empty results and access control

        Args:
            payload: Query string or structured query dict.

        Raises:
            NotImplementedError: Until implementation is complete.
        """
        logger.warning(
            "DBTool.execute() called but not yet implemented. "
            "payload_type=%s",
            type(payload).__name__,  # type only — not content (PHI risk)
        )
        raise NotImplementedError("DBTool.execute() is not yet implemented.")
