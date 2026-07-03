# agents/base.py
"""
Base agent abstraction.
Responsibility: Define the contract every agent must fulfill.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Every agent must implement run(). 
    Optional hooks: validate_input(), setup(), teardown().
    """

    @property
    def name(self) -> str:
        """Human-readable agent name derived from class name."""
        return self.__class__.__name__

    @abstractmethod
    async def run(self, payload: Any) -> Any:
        """
        Execute the agent's primary task.

        Args:
            payload: Agent-specific input data.

        Returns:
            Agent-specific output. Subclasses should
            narrow the type in their own signatures.
        """
        pass

    def validate_input(self, payload: Any) -> bool:
        """
        Optional pre-run input validation hook.
        Override in subclasses for input-specific checks.
        Default: always returns True.
        """
        return True

    async def setup(self) -> None:
        """
        Optional async initialization hook.
        Called before run() if agent needs async setup
        (e.g. DB connections, loading tools).
        """
        pass

    async def teardown(self) -> None:
        """
        Optional async cleanup hook.
        Called after run() completes or fails.
        """
        pass

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"&lt;{self.__class__.__name__}&gt;"
