# core/pipeline.py
"""
LangGraph workflow definition and compilation.
Responsibility: Wire all nodes into the compiled graph — nothing else.
Depends on: agents/nodes.py, core/state.py
"""

from functools import lru_cache
from langgraph.graph import StateGraph, END

from core.state import AgentState
from agents.nodes import (
    extract_cancer_type_node,
    search_trials_node,
    evaluate_trials_node,
    recommend_trials_node,
)

__all__ = ["get_trial_matching_app"]

# -----------------------------------------------------------
# NODE NAME CONSTANTS — typo-safe references
# -----------------------------------------------------------

_NODE_EXTRACT_CANCER_TYPE = "extract_cancer_type"
_NODE_SEARCH_TRIALS       = "search_trials"
_NODE_EVALUATE_TRIALS     = "evaluate_trials"
_NODE_RECOMMEND_TRIALS    = "recommend_trials"


# -----------------------------------------------------------
# CONDITIONAL EDGE ROUTER
# -----------------------------------------------------------

def _should_continue(state: AgentState) -> str:
    """
    Route to recommend_trials early if an error has been set.
    recommend_trials handles error state gracefully with a fallback message.
    """
    return "end" if state.get("error") else "continue"


# -----------------------------------------------------------
# GRAPH FACTORY — lazy singleton
# -----------------------------------------------------------

@lru_cache(maxsize=1)
def get_trial_matching_app():
    """
    Builds and compiles the LangGraph trial matching workflow.
    Lazy singleton — compiled once on first call, not at import time.

    Graph flow:
        extract_cancer_type
            ↓ (error → recommend_trials)
        search_trials
            ↓
        evaluate_trials
            ↓
        recommend_trials
            ↓
        END
    """
    graph = StateGraph(AgentState)

    # --- Register Nodes ---
    graph.add_node(_NODE_EXTRACT_CANCER_TYPE, extract_cancer_type_node)
    graph.add_node(_NODE_SEARCH_TRIALS,       search_trials_node)
    graph.add_node(_NODE_EVALUATE_TRIALS,     evaluate_trials_node)
    graph.add_node(_NODE_RECOMMEND_TRIALS,    recommend_trials_node)

    # --- Entry Point ---
    graph.set_entry_point(_NODE_EXTRACT_CANCER_TYPE)

    # --- Edges ---
    # Conditional after Node 1 — skip to recommend if error
    graph.add_conditional_edges(
        _NODE_EXTRACT_CANCER_TYPE,
        _should_continue,
        {
            "continue": _NODE_SEARCH_TRIALS,
            "end":      _NODE_RECOMMEND_TRIALS,
        },
    )

    graph.add_edge(_NODE_SEARCH_TRIALS,   _NODE_EVALUATE_TRIALS)
    graph.add_edge(_NODE_EVALUATE_TRIALS, _NODE_RECOMMEND_TRIALS)
    graph.add_edge(_NODE_RECOMMEND_TRIALS, END)

    return graph.compile()
