"""
LangGraph StateGraph Builder & Compiler for Autonomous Knowledge Analyst.
Includes SqliteSaver checkpointing for cross-session execution & thread persistence.
"""

import os
from pathlib import Path
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Import state and nodes
from .state import ResearchState
from .nodes import (
    router_node,
    researcher_node,
    judge_node,
    refiner_node,
    hitl_node,
    final_answer_node,
    QUALITY_THRESHOLD
)

# Attempt to import SqliteSaver if available, fallback to MemorySaver
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    HAS_SQLITE_SAVER = True
except ImportError:
    HAS_SQLITE_SAVER = False


def route_after_judge(state: ResearchState) -> Literal["hitl", "refiner"]:
    """
    Conditional Edge router after Judge node:
    Routes to HITL if evidence score is sufficient OR max_retries exceeded.
    Otherwise routes to Refiner node for self-correction.
    """
    is_sufficient = state.get("is_sufficient", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if is_sufficient or retry_count >= max_retries:
        print(f"--> [Conditional Routing]: Route after Judge -> 'hitl' (Sufficient: {is_sufficient}, Retries: {retry_count}/{max_retries})")
        return "hitl"
    else:
        print(f"--> [Conditional Routing]: Route after Judge -> 'refiner' (Refinement needed, Attempt #{retry_count + 1})")
        return "refiner"


def route_after_hitl(state: ResearchState) -> Literal["final_answer", "refiner"]:
    """
    Conditional Edge router after HITL node:
    Routes to Final Answer if approved by human.
    Routes back to Refiner if human rejected or provided revision feedback.
    """
    approved = state.get("human_approved", True)
    if approved:
        print("--> [Conditional Routing]: Route after HITL -> 'final_answer' (Approved by Human)")
        return "final_answer"
    else:
        print("--> [Conditional Routing]: Route after HITL -> 'refiner' (Human requested revisions)")
        return "refiner"


def get_checkpointer(db_path: str = "checkpoints/checkpoints.db"):
    """
    Initialize SQLite checkpointer for persisting graph execution threads.
    Creates checkpoints directory automatically.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    if HAS_SQLITE_SAVER:
        try:
            import sqlite3
            conn = sqlite3.connect(db_path, check_same_thread=False)
            saver = SqliteSaver(conn)
            saver.setup()
            return saver
        except Exception as e:
            print(f"SqliteSaver initialization note (using MemorySaver): {e}")
            return MemorySaver()
    else:
        return MemorySaver()


def create_graph(checkpointer=None):
    """
    Builds and compiles the Autonomous Knowledge Analyst StateGraph.
    
    Args:
        checkpointer: Checkpointer instance (default: SqliteSaver/MemorySaver)
        
    Returns:
        Compiled LangGraph application
    """
    if checkpointer is None:
        checkpointer = get_checkpointer()

    builder = StateGraph(ResearchState)

    # 1. Add Workflow Nodes
    builder.add_node("router", router_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("judge", judge_node)
    builder.add_node("refiner", refiner_node)
    builder.add_node("hitl", hitl_node)
    builder.add_node("final_answer", final_answer_node)

    # 2. Add Fixed Edges
    builder.add_edge(START, "router")
    builder.add_edge("router", "researcher")
    builder.add_edge("researcher", "judge")
    builder.add_edge("refiner", "researcher")
    builder.add_edge("final_answer", END)

    # 3. Add Conditional Edges
    builder.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "hitl": "hitl",
            "refiner": "refiner"
        }
    )

    builder.add_conditional_edges(
        "hitl",
        route_after_hitl,
        {
            "final_answer": "final_answer",
            "refiner": "refiner"
        }
    )

    # 4. Compile graph with checkpointer
    app = builder.compile(checkpointer=checkpointer)
    return app


if __name__ == "__main__":
    app = create_graph()
    print("LangGraph StateGraph compiled successfully!")
    print(f"Nodes: {list(app.nodes.keys())}")
