"""
Unit tests for LangGraph state workflow, routing logic, retry limits, and checkpointer.
"""

import sys
from pathlib import Path

# Add src path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from knowledge_analyst.state import ResearchState
from knowledge_analyst.graph import (
    create_graph,
    get_checkpointer,
    route_after_judge,
    route_after_hitl
)


def test_route_after_judge_sufficient():
    state: ResearchState = {
        "query": "Test query",
        "route": "hybrid",
        "search_queries": [],
        "retrieved_evidence": [],
        "judge_score": 0.85,
        "judge_critique": "Good",
        "is_sufficient": True,
        "retry_count": 0,
        "max_retries": 3,
        "history": [],
        "draft_report": "",
        "human_approved": False,
        "human_feedback": None,
        "final_report": ""
    }
    decision = route_after_judge(state)
    assert decision == "hitl"


def test_route_after_judge_needs_refinement():
    state: ResearchState = {
        "query": "Test query",
        "route": "hybrid",
        "search_queries": [],
        "retrieved_evidence": [],
        "judge_score": 0.50,
        "judge_critique": "Missing data",
        "is_sufficient": False,
        "retry_count": 1,
        "max_retries": 3,
        "history": [],
        "draft_report": "",
        "human_approved": False,
        "human_feedback": None,
        "final_report": ""
    }
    decision = route_after_judge(state)
    assert decision == "refiner"


def test_route_after_judge_max_retries_exceeded():
    state: ResearchState = {
        "query": "Test query",
        "route": "hybrid",
        "search_queries": [],
        "retrieved_evidence": [],
        "judge_score": 0.50,
        "judge_critique": "Missing data",
        "is_sufficient": False,
        "retry_count": 3,
        "max_retries": 3,
        "history": [],
        "draft_report": "",
        "human_approved": False,
        "human_feedback": None,
        "final_report": ""
    }
    decision = route_after_judge(state)
    assert decision == "hitl"


def test_graph_compilation():
    checkpointer = get_checkpointer()
    app = create_graph(checkpointer=checkpointer)
    assert app is not None
    assert "router" in app.nodes
    assert "researcher" in app.nodes
    assert "judge" in app.nodes
    assert "refiner" in app.nodes
    assert "hitl" in app.nodes
    assert "final_answer" in app.nodes


if __name__ == "__main__":
    print("Running test_route_after_judge_sufficient...")
    test_route_after_judge_sufficient()
    print("Running test_route_after_judge_needs_refinement...")
    test_route_after_judge_needs_refinement()
    print("Running test_route_after_judge_max_retries_exceeded...")
    test_route_after_judge_max_retries_exceeded()
    print("Running test_graph_compilation...")
    test_graph_compilation()
    print("All graph tests passed successfully!")
