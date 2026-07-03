# core/callbacks.py
"""
Logging and tracing callbacks.
Responsibility: Centralized observability hooks for the pipeline.
Extend PipelineCallback for LangSmith / OpenTelemetry / custom tracing.
"""

import logging
import time
from abc import ABC
from typing import Any, Dict, Optional

logger = logging.getLogger("uvicorn.error")

__all__ = ["BasePipelineCallback", "PipelineCallback", "default_callback"]


# -----------------------------------------------------------
# ABSTRACT BASE
# -----------------------------------------------------------

class BasePipelineCallback(ABC):
    """
    Abstract interface for pipeline observability callbacks.
    Subclass this to integrate LangSmith, OpenTelemetry, or custom tracing.
    """

    def on_node_start(self, node_name: str, state: Dict[str, Any]) -> None:
        """Called when a LangGraph node begins execution."""
        ...

    def on_node_end(self, node_name: str, state: Dict[str, Any]) -> None:
        """Called when a LangGraph node completes successfully."""
        ...

    def on_error(self, node_name: str, error: Exception) -> None:
        """Called when a LangGraph node raises an exception."""
        ...

    def on_pipeline_start(self, payload: Dict[str, Any]) -> float:
        """Called when the full pipeline begins. Returns start timestamp."""
        ...

    def on_pipeline_end(self, start_time: float, success: bool) -> None:
        """Called when the full pipeline completes or fails."""
        ...

    def on_llm_start(
        self,
        chain_name: str,
        trial_id: Optional[str] = None,
    ) -> float:
        """Called before each LLM chain invocation. Returns start timestamp."""
        ...

    def on_llm_end(
        self,
        chain_name: str,
        start_time: float,
        trial_id: Optional[str] = None,
    ) -> None:
        """Called after each LLM chain invocation."""
        ...

    def on_trials_fetched(self, cancer_type: str, count: int) -> None:
        """Called after ClinicalTrials.gov search completes."""
        ...


# -----------------------------------------------------------
# DEFAULT LOGGING IMPLEMENTATION
# -----------------------------------------------------------

class PipelineCallback(BasePipelineCallback):
    """
    Default pipeline callback — logs all events via uvicorn logger.
    Extend this class to add LangSmith, OpenTelemetry, or custom tracing.
    """

    def on_node_start(self, node_name: str, state: Dict[str, Any]) -> None:
        # NOTE: Do NOT log state — it contains patient PHI
        logger.info(">>> Node started   : %s", node_name)

    def on_node_end(self, node_name: str, state: Dict[str, Any]) -> None:
        # NOTE: Do NOT log state — it contains patient PHI
        logger.info("<<< Node completed : %s", node_name)

    def on_error(self, node_name: str, error: Exception) -> None:
        logger.exception(
            "!!! Node failed    : %s | error_type=%s | error=%s",
            node_name,
            type(error).__name__,
            str(error),
        )

    def on_pipeline_start(self, payload: Dict[str, Any]) -> float:
        # NOTE: Do NOT log payload — it contains patient PHI
        logger.info("=" * 50)
        logger.info("Pipeline started.")
        return time.time()

    def on_pipeline_end(self, start_time: float, success: bool) -> None:
        elapsed = round(time.time() - start_time, 3)
        status = "SUCCESS" if success else "FAILED"
        logger.info(
            "Pipeline finished  | status=%s | elapsed=%ss",
            status,
            elapsed,
        )
        logger.info("=" * 50)

    def on_llm_start(
        self,
        chain_name: str,
        trial_id: Optional[str] = None,
    ) -> float:
        logger.info(
            "LLM call started   | chain=%s | trial_id=%s",
            chain_name,
            trial_id or "N/A",
        )
        return time.time()

    def on_llm_end(
        self,
        chain_name: str,
        start_time: float,
        trial_id: Optional[str] = None,
    ) -> None:
        elapsed = round(time.time() - start_time, 3)
        logger.info(
            "LLM call completed | chain=%s | trial_id=%s | elapsed=%ss",
            chain_name,
            trial_id or "N/A",
            elapsed,
        )

    def on_trials_fetched(self, cancer_type: str, count: int) -> None:
        logger.info(
            "Trials fetched     | cancer_type=%s | count=%d",
            cancer_type,
            count,
        )


# -----------------------------------------------------------
# DEFAULT INSTANCE — import and use directly
# -----------------------------------------------------------

default_callback = PipelineCallback()
