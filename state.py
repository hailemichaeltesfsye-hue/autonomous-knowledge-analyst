"""Root re-export for state.py"""
import sys
from pathlib import Path

# Add src to python path for easy importing
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from knowledge_analyst.state import ResearchState, EvidenceItem, HistoryItem

__all__ = ["ResearchState", "EvidenceItem", "HistoryItem"]

if __name__ == "__main__":
    # Simple validation test
    state: ResearchState = {
        "query": "Analyze whether AI and automation adoption among African businesses has increased over the last five years, and provide evidence.",
        "route": "hybrid",
        "search_queries": ["AI adoption African businesses 2020 2025"],
        "retrieved_evidence": [],
        "judge_score": 0.0,
        "judge_critique": "",
        "is_sufficient": False,
        "retry_count": 0,
        "max_retries": 3,
        "history": [],
        "draft_report": "",
        "human_approved": False,
        "human_feedback": None,
        "final_report": ""
    }
    print("State schema initialized successfully:")
    print(f"- Target Query: {state['query']}")
    print(f"- Max Retries: {state['max_retries']}")
