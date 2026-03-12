"""
MarketMind AI — Streamlit Chat Interface.

WHAT THIS IS:
  A browser-based chat UI that lets users talk to the multi-agent pipeline.
  It calls the FastAPI backend (localhost:8000) and displays the structured
  outputs from the analyst, strategist, and content writer agents.

HOW TO RUN (requires FastAPI backend running first):
  Terminal 1: .venv/Scripts/uvicorn marketmind.api.main:app --reload --port 8000
  Terminal 2: .venv/Scripts/streamlit run ui/app.py

WHY CALL THE API RATHER THAN IMPORTING DIRECTLY?
  - Separation of concerns: UI layer doesn't know about agent internals
  - Realistic architecture: matches production where UI and backend are separate
  - Enables future mobile/web clients to reuse the same API

STREAMLIT STATE MANAGEMENT:
  st.session_state persists across reruns (each user interaction triggers a rerun).
  We use it to store:
    - messages: the chat history to display
    - thread_id: the session ID for conversation continuity
"""

import os

import httpx
import streamlit as st

# In Docker: set API_BASE=http://api:8000 via environment variable
# Locally: defaults to localhost
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MarketMind AI",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "api_status" not in st.session_state:
    st.session_state.api_status = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("MarketMind AI")
    st.caption("Multi-agent marketing intelligence")

    st.divider()

    # API health check
    if st.button("Check API Status"):
        try:
            resp = httpx.get(f"{API_BASE}/health", timeout=5)
            data = resp.json()
            st.session_state.api_status = data
        except Exception as e:
            st.session_state.api_status = {"error": str(e)}

    if st.session_state.api_status:
        status = st.session_state.api_status
        if "error" in status:
            st.error(f"API offline: {status['error']}")
            st.caption("Start with: uvicorn marketmind.api.main:app --reload --port 8000")
        else:
            st.success(f"API: {status.get('status', 'ok')}")
            mem_status = status.get("long_term_memory", "unavailable")
            if mem_status == "connected":
                st.success(f"MongoDB: connected")
            else:
                st.warning(f"MongoDB: {mem_status}")

    st.divider()

    # Session management
    st.subheader("Session")
    if st.session_state.thread_id:
        st.caption(f"Thread: `{st.session_state.thread_id[:8]}...`")
    else:
        st.caption("No active session")

    if st.button("New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = None
        st.rerun()

    st.divider()

    # Example queries
    st.subheader("Try these")
    examples = [
        "What is the churn risk of SEG003?",
        "Create a Q2 strategy for Rising Stars",
        "Write a win-back email for dormant customers",
        "Build a complete campaign for cart abandonment",
    ]
    for example in examples:
        if st.button(example, use_container_width=True, key=f"ex_{example[:20]}"):
            st.session_state._pending_query = example
            st.rerun()

    st.divider()
    st.caption("Routes: analyst · strategist · content_writer · full_pipeline")


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.header("MarketMind AI")
st.caption("Ask about customer segments, get campaign strategies, or generate marketing copy.")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "sections" in msg:
            # Render structured agent output
            route = msg.get("route", "")
            st.caption(f"Route: **{route}**")
            for section_title, section_content in msg["sections"]:
                with st.expander(section_title, expanded=True):
                    st.markdown(section_content)
        else:
            st.markdown(msg["content"])


def call_api(query: str, thread_id: str | None) -> dict:
    """Call POST /query on the FastAPI backend."""
    resp = httpx.post(
        f"{API_BASE}/query",
        json={"query": query, "thread_id": thread_id},
        timeout=120,  # LLM calls can take time
    )
    resp.raise_for_status()
    return resp.json()


def build_sections(result: dict) -> list[tuple[str, str]]:
    """Extract non-empty agent outputs as (title, content) pairs."""
    sections = []
    if result.get("analyst_output"):
        sections.append(("Data Analysis", result["analyst_output"]))
    if result.get("strategist_output"):
        sections.append(("Campaign Strategy", result["strategist_output"]))
    if result.get("content_output"):
        sections.append(("Marketing Copy", result["content_output"]))
    return sections


# Handle example button clicks
pending = st.session_state.pop("_pending_query", None)

# Chat input
user_input = st.chat_input("Ask MarketMind AI...") or pending

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Call the API
    with st.chat_message("assistant"):
        with st.spinner("Agents working..."):
            try:
                result = call_api(user_input, st.session_state.thread_id)
                st.session_state.thread_id = result["thread_id"]

                route = result.get("task_route", "unknown")
                sections = build_sections(result)

                st.caption(f"Route: **{route}**")
                for section_title, section_content in sections:
                    with st.expander(section_title, expanded=True):
                        st.markdown(section_content)

                if not sections:
                    st.warning("No output from agents. Check the API logs.")

                # Save to message history
                st.session_state.messages.append({
                    "role": "assistant",
                    "route": route,
                    "sections": sections,
                    "content": "",
                })

            except httpx.ConnectError:
                err = "Cannot connect to the API. Make sure the FastAPI server is running:\n\n```\n.venv/Scripts/uvicorn marketmind.api.main:app --reload --port 8000\n```"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

            except Exception as e:
                err = f"Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
