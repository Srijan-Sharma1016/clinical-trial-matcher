# tools/base_tool.py
"""
Abstract base tool class.
Responsibility: Define the contract all tools must follow.
"""

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["BaseTool"]


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Every tool must implement:
        - name       (property) — unique identifier for agent routing
        - description (property) — human-readable purpose
        - execute()  — primary action

    Optional hooks:
        - validate_input() — pre-execute validation
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique tool identifier.
        Used by agents for tool routing and selection.
        Example: 'clinicaltrials_search'
        """
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description of what this tool does.
        Used by LLM agents to decide when to invoke the tool.
        Example: 'Search ClinicalTrials.gov for recruiting trials.'
        """
        ...

    @abstractmethod
    async def execute(self, payload: Any) -> Any:
        """
        Execute the tool's primary action.

        Args:
            payload: Tool-specific input data.

        Returns:
            Tool-specific output. Subclasses should
            narrow the type in their own signatures.
        """
        ...

    def validate_input(self, payload: Any) -> bool:
        """
        Optional pre-execute input validation hook.
        Override in subclasses for input-specific checks.
        Default: always returns True.
        """
        return True

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"
