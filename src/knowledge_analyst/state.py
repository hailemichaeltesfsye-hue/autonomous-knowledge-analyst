"""
State definition for the Autonomous Knowledge Analyst agent.
Defines the shared state structure used across all LangGraph nodes.
"""

from typing import TypedDict, List, Dict, Any, Optional
import operator
from typing import Annotated


class EvidenceItem(TypedDict):
    """Structure for an individual evidence snippet retrieved by the tools."""
    source: str         # "vector_db" or "web_search"
    content: str        # Retrieved snippet or content summary
    query_used: str     # Query that produced this snippet
    metadata: Dict[str, Any]


class HistoryItem(TypedDict):
    """Memory layer record for keeping track of past research & refinement attempts."""
    iteration: int
    queries: List[str]
    score: float
    critique: str
    route: str


class ResearchState(TypedDict):
    """
    Shared state object for the Autonomous Knowledge Analyst LangGraph workflow.
    Tracks state across Router, Researcher, Judge, Refiner, and HITL nodes.
    """
    # Initial input
    query: str

    # Semantic Router output
    route: str                          # "vector_db", "web_search", or "hybrid"
    search_queries: List[str]          # Current list of queries to search

    # Researcher node output (Annotated with list concatenation to preserve findings across retries)
    retrieved_evidence: Annotated[List[EvidenceItem], operator.add]

    # Judge node evaluation outputs
    judge_score: float                  # Score from 0.0 to 100.0
    judge_critique: str                 # Detailed feedback on gaps/evidence
    is_sufficient: bool                # True if judge_score >= threshold (e.g., 75.0)

    # Self-Correction Loop & Retry Control
    retry_count: int                   # Current retry attempt count
    max_retries: int                   # Maximum allowed retries before forcing HITL

    # Memory Layer (Tracks iterations history across sessions)
    history: Annotated[List[HistoryItem], operator.add]

    # Report Synthesis & HITL Approval
    draft_report: str                  # Synthesized draft report presented to human
    human_approved: bool               # True if human approves the report
    human_feedback: Optional[str]      # Feedback if human rejects/requests changes
    final_report: str                  # Final approved output report
