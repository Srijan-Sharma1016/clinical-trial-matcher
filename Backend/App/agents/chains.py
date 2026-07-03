# agents/chains.py
"""
LLM engine and chain instances.
Responsibility: Instantiate the LLM and bind it to prompt templates.
Depends on: config/settings.py, prompts/templates.py
"""

import logging
from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from config.settings import (
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    GROQ_TEMPERATURE,
    GROQ_MAX_RETRIES,
)
from prompts.templates import (
    cancer_type_prompt,
    eligibility_prompt,
    recommendation_prompt,
)

logger = logging.getLogger("uvicorn.error")

__all__ = [
    "get_llm",
    "get_tool_llm",
    "get_cancer_type_chain",
    "get_eligibility_chain",
    "get_recommendation_chain",
]

# -----------------------------------------------------------
# LLM INSTANCE — Lazy Singleton
# -----------------------------------------------------------

@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """
    Lazy singleton LLM instance.
    Used for all standard chain invocations.
    Created on first call — not at import time.
    """
    logger.info(
        "Initializing LLM | model=%s | temperature=%s | max_retries=%s",
        GROQ_MODEL_NAME,
        GROQ_TEMPERATURE,
        GROQ_MAX_RETRIES,
    )
    return ChatGroq(
        model=GROQ_MODEL_NAME,
        api_key=GROQ_API_KEY,
        temperature=GROQ_TEMPERATURE,
        max_retries=GROQ_MAX_RETRIES,
    )


@lru_cache(maxsize=1)
def get_tool_llm() -> ChatGroq:
    """
    Dedicated LLM instance for tool calling.
    Uses Groq's tool-use optimized model.
    Separate from get_llm() to avoid contaminating
    standard chain calls with tool-use model behavior.
    """
    logger.info(
        "Initializing Tool LLM | model=llama3-groq-70b-8192-tool-use-preview"
    )
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0,
        max_retries=3,
    )


# -----------------------------------------------------------
# CHAINS — Lazy Singletons
# -----------------------------------------------------------

@lru_cache(maxsize=1)
def get_cancer_type_chain():
    """Chain for extracting cancer type from patient summary."""
    return cancer_type_prompt | get_llm() | StrOutputParser()


@lru_cache(maxsize=1)
def get_eligibility_chain():
    """Chain for assessing trial eligibility for a single trial."""
    return eligibility_prompt | get_llm() | StrOutputParser()


@lru_cache(maxsize=1)
def get_recommendation_chain():
    """Chain for generating final plain-language recommendation report."""
    return recommendation_prompt | get_llm() | StrOutputParser()
