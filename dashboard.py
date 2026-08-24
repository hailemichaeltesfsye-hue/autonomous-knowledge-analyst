"""
Streamlit Interactive Dashboard for Autonomous Knowledge Analyst.
Provides visual node progress monitoring, evidence inspection, HITL approval buttons,
and thread session persistence.
"""

import sys
import uuid
import streamlit as st
from pathlib import Path

# Add src path for package imports
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from knowledge_analyst.graph import create_graph, get_checkpointer
from knowledge_analyst.state import ResearchState

try:
    from langgraph.types import Command
except ImportError:
    class Command:
        def __init__(self, resume):
            self.resume = resume

# Page Configuration
st.set_page_config(
    page_title="Autonomous Knowledge Analyst",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 Autonomous Knowledge Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Stateful AI Research Agent with Self-Correction, Persistence & Human-in-the-Loop Approval</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Agent Settings")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session-{uuid.uuid4().hex[:8]}"

thread_id = st.sidebar.text_input("Thread ID (Session Checkpoint)", value=st.session_state.thread_id)
max_retries = st.sidebar.slider("Max Self-Correction Retries", min_value=1, max_value=5, value=3)

if st.sidebar.button("🔄 New Research Session"):
    st.session_state.thread_id = f"session-{uuid.uuid4().hex[:8]}"
    st.session_state.workflow_started = False
    st.session_state.messages = []
    st.rerun()

# Initialize Checkpointer & App
checkpointer = get_checkpointer()
app = create_graph(checkpointer=checkpointer)
config = {"configurable": {"thread_id": thread_id}}

# Main Query Form
default_query = "Analyze whether AI and automation adoption among African businesses has increased over the last five years, and provide evidence."
user_query = st.text_area("Research Query / Topic:", value=default_query, height=100)

col1, col2 = st.columns([1, 4])
with col1:
    start_button = st.button("🚀 Run Autonomous Agent", type="primary", use_container_width=True)

# Graph execution handler
if start_button:
    st.session_state.workflow_started = True
    initial_state: ResearchState = {
        "query": user_query,
        "route": "hybrid",
        "search_queries": [],
        "retrieved_evidence": [],
        "judge_score": 0.0,
        "judge_critique": "",
        "is_sufficient": False,
        "retry_count": 0,
        "max_retries": max_retries,
        "history": [],
        "draft_report": "",
        "human_approved": False,
        "human_feedback": None,
        "final_report": ""
    }

    status_placeholder = st.empty()
    status_placeholder.info("⏳ Initializing workflow graph and Semantic Router...")

    # Run graph execution stream
    try:
        for event in app.stream(initial_state, config=config):
            for node_name, node_output in event.items():
                status_placeholder.info(f"⚡ Completed node: **{node_name.upper()}**")
    except Exception as e:
        status_placeholder.warning(f"Workflow reached interrupt point: {e}")

# Read graph state from checkpoint
state_snapshot = app.get_state(config)

if state_snapshot and state_snapshot.values:
    values = state_snapshot.values
    
    st.divider()

    # Metrics Dashboard Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Strategy Route", values.get("route", "N/A").upper())
    m2.metric("Evidence Count", len(values.get("retrieved_evidence", [])))
    score = values.get("judge_score", 0.0)
    m3.metric("Judge Quality Score", f"{score * 100:.0f}%", delta="Sufficient" if score >= 0.75 else "Needs Refinement")
    m4.metric("Retry Counter", f"{values.get('retry_count', 0)} / {values.get('max_retries', 3)}")

    # Judge Critique Alert
    critique = values.get("judge_critique")
    if critique:
        st.info(f"**Judge Evaluation Critique:** {critique}")

    # Tabs for detailed inspection
    t1, t2, t3 = st.tabs(["📝 Final Report / HITL Approval", "🔍 Retrieved Evidence", "📜 Research Memory Log"])

    with t1:
        # Check if final report is ready
        final_report = values.get("final_report")
        if final_report:
            st.success("✅ Business Research Analysis Report Completed!")
            st.markdown(final_report)
        else:
            # Display HITL Approval Interface
            st.warning("⚠️ **Human-in-the-Loop Review Required**")
            st.markdown("Please review the draft findings below and choose whether to approve or request refined research.")

            draft = values.get("draft_report", "")
            if draft:
                st.markdown(draft)

            st.divider()
            c_approve, c_reject = st.columns(2)

            with c_approve:
                if st.button("✅ Approve Draft & Generate Final Report", type="primary", use_container_width=True):
                    with st.spinner("Synthesizing final executive report..."):
                        human_action = {"approved": True, "feedback": None}
                        for event in app.stream(Command(resume=human_action), config=config):
                            pass
                    st.rerun()

            with c_reject:
                feedback_text = st.text_input("Revision instructions for Refiner node:", placeholder="e.g. Include more data on manufacturing automation in South Africa")
                if st.button("🔄 Request Revisions & Trigger Self-Correction", use_container_width=True):
                    with st.spinner("Refining search queries and re-evaluating..."):
                        human_action = {"approved": False, "feedback": feedback_text or "Add missing country stats"}
                        for event in app.stream(Command(resume=human_action), config=config):
                            pass
                    st.rerun()

    with t2:
        evidence = values.get("retrieved_evidence", [])
        if evidence:
            st.subheader(f"Retrieved Evidence Items ({len(evidence)})")
            for idx, item in enumerate(evidence, 1):
                with st.expander(f"[{item['source'].upper()}] Snippet #{idx} (Query: '{item.get('query_used', '')}')"):
                    st.write(item["content"])
        else:
            st.write("No evidence snippets retrieved yet.")

    with t3:
        history = values.get("history", [])
        if history:
            st.subheader("Iterative Research History & Memory Layer")
            for h in history:
                st.markdown(f"""
                - **Iteration #{h['iteration']}**: Score `{h['score']:.2f}` | Strategy: `{h['route']}`
                  - *Queries*: `{', '.join(h['queries'])}`
                  - *Critique*: {h['critique']}
                """)
        else:
            st.write("No prior refinement iterations logged.")
