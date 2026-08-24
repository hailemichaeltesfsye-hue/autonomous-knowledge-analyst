
"""
CLI Interface for Autonomous Knowledge Analyst.
Supports interactive research execution, thread resumption, and HITL decision prompting.
"""
 
import sys
import uuid
import argparse
from pathlib import Path
 
# Add src path
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
 
from knowledge_analyst.graph import create_graph, get_checkpointer
from knowledge_analyst.state import ResearchState
 
# Try importing Command for LangGraph resume, fallback to a minimal shim
try:
    from langgraph.types import Command
except ImportError:
    class Command:
        def __init__(self, resume):
            self.resume = resume
 
 
def print_banner():
    print("=" * 80)
    print("      AUTONOMOUS KNOWLEDGE ANALYST — RESEARCH PIPELINE")
    print("=" * 80)
 
 
def run_cli():
    parser = argparse.ArgumentParser(description="Run Autonomous Knowledge Analyst AI Research Pipeline")
    parser.add_argument(
        "--query",
        type=str,
        default="Analyze whether AI and automation adoption among African businesses has increased over the last five years, and provide evidence.",
        help="Research topic/query to analyze"
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        help="Thread ID for resuming execution from a checkpoint"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum self-correction retries allowed"
    )
    args = parser.parse_args()
 
    print_banner()
 
    thread_id = args.thread_id or f"session-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
 
    print(f"Session Thread ID: {thread_id}")
    print(f"Target Query:      {args.query}")
    print(f"Max Retries:       {args.max_retries}\n")
 
    checkpointer = get_checkpointer()
    app = create_graph(checkpointer=checkpointer)
 
    initial_state: ResearchState = {
        "query": args.query,
        "route": "hybrid",
        "search_queries": [],
        "retrieved_evidence": [],
        "judge_score": 0.0,
        "judge_critique": "",
        "is_sufficient": False,
        "retry_count": 0,
        "max_retries": args.max_retries,
        "history": [],
        "draft_report": "",
        "human_approved": False,
        "human_feedback": None,
        "final_report": ""
    }
 
    # Check whether this thread already has a saved checkpoint (resume) or is new.
    # app.get_state(config) returns the last persisted state for this thread_id, if any.
    existing_state = app.get_state(config)
    is_resuming = bool(existing_state and existing_state.values)
 
    if is_resuming:
        print("Resuming existing session from checkpoint...\n")
        stream_input = None  # None tells LangGraph to resume from the last saved checkpoint
    else:
        print("Starting research workflow...\n")
        stream_input = initial_state
 
    # Execute graph until pause or completion
    try:
        for event in app.stream(stream_input, config=config):
            for node_name, node_output in event.items():
                print(f"\n[Completed Step: '{node_name}']")
                if "judge_score" in node_output:
                    print(f"  > Score: {node_output['judge_score']:.2f}")
                if "retry_count" in node_output:
                    print(f"  > Retry Count: {node_output['retry_count']}")
                if "final_report" in node_output:
                    print("\n" + "=" * 80)
                    print("FINAL SYNTHESIZED REPORT")
                    print("=" * 80)
                    print(node_output["final_report"])
                    return
    except Exception as e:
        print(f"\nWorkflow paused or interrupted: {e}")
 
    # Check state at checkpoint (either just paused, or resumed and now paused again)
    checkpoint_state = app.get_state(config)
 
    if checkpoint_state and checkpoint_state.next:
        next_node = checkpoint_state.next[0]
        print(f"\nWorkflow interrupted at node: '{next_node}' for Human Review.")
 
        # Inspect state values at interrupt
        values = checkpoint_state.values
        draft = values.get("draft_report", "")
        score = values.get("judge_score", 0.0)
 
        print("\n" + "-" * 60)
        print(f"HITL DRAFT REPORT SUMMARY (Judge Score: {score * 100:.0f}/100)")
        print("-" * 60)
        if draft:
            print(draft)
        else:
            print(f"Evidence Snippets Collected: {len(values.get('retrieved_evidence', []))}")
            for item in values.get("retrieved_evidence", [])[:4]:
                print(f" - [{item['source']}] {item['content'][:120]}...")
        print("\n" + "-" * 60)
 
        user_choice = input("Approve report and generate final analysis? [y/N/feedback]: ").strip()
 
        if user_choice.lower() in ["y", "yes"]:
            human_action = {"approved": True, "feedback": None}
            print("\nUser APPROVED. Resuming workflow to generate final report...")
        else:
            feedback = user_choice if user_choice.lower() not in ["n", "no", ""] else "Expand research with more statistics across East and West Africa."
            human_action = {"approved": False, "feedback": feedback}
            print(f"\nUser REJECTED with feedback: '{feedback}'. Resuming refinement loop...")
 
        # Resume graph from checkpoint with human_action payload
        for event in app.stream(Command(resume=human_action), config=config):
            for node_name, node_output in event.items():
                print(f"\n[Completed Step: '{node_name}']")
                if "final_report" in node_output:
                    print("\n" + "=" * 80)
                    print("FINAL SYNTHESIZED REPORT")
                    print("=" * 80)
                    print(node_output["final_report"])
    else:
        print("\nWorkflow completed without requiring further human review.")
 
 
if __name__ == "__main__":
    run_cli()
 