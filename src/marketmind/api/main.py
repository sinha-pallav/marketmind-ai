"""
FastAPI backend for MarketMind AI.

WHAT THIS DOES:
  Exposes the multi-agent graph as a REST API so any client (Streamlit UI,
  Claude Desktop, mobile app, third-party integration) can call it over HTTP.

ENDPOINTS:
  POST /query        — Run the agent pipeline, get back analysis/strategy/copy
  GET  /health       — Liveness check (also reports MongoDB connectivity)
  GET  /sessions/{thread_id} — Retrieve past session history from short-term memory

WHY FASTAPI OVER FLASK?
  - Async-first: LangGraph agents can be slow (multiple LLM calls); async keeps
    the server from blocking while waiting on the LLM API.
  - Automatic OpenAPI docs at /docs — the Streamlit UI uses these to stay in sync.
  - Pydantic validation: request/response models are defined as dataclasses,
    so bad inputs are rejected with clear error messages before hitting the LLM.

TO RUN:
  .venv/Scripts/uvicorn marketmind.api.main:app --reload --port 8000

THEN VISIT:
  http://localhost:8000/docs   — interactive Swagger UI
  http://localhost:8000/health — quick status check
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from marketmind.agents.graph import run
from marketmind.memory.short_term import get_thread_history
from marketmind.memory import long_term

app = FastAPI(
    title="MarketMind AI",
    version="0.1.0",
    description="Multi-agent marketing intelligence system — analyst, strategist, content writer.",
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """
    The body for POST /query.
    thread_id is optional — omit it to start a fresh conversation.
    Pass the same thread_id from a previous response to continue a session.
    """
    query: str
    thread_id: str | None = None


class QueryResponse(BaseModel):
    """
    Structured response from the agent pipeline.
    Only the fields that are relevant to the route taken will be populated.
    """
    thread_id: str
    task_route: str
    analyst_output: str
    strategist_output: str
    content_output: str


class HealthResponse(BaseModel):
    status: str
    version: str
    long_term_memory: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    """
    Liveness check. Returns:
      - status: always "ok" if the server is running
      - long_term_memory: "connected" or "unavailable" (MongoDB Atlas)
    """
    return HealthResponse(
        status="ok",
        version="0.1.0",
        long_term_memory="connected" if long_term.is_connected() else "unavailable",
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Run the MarketMind agent pipeline.

    The orchestrator classifies the query and routes it to:
      - analyst only
      - strategist only
      - content_writer only
      - full_pipeline (analyst → strategist → content_writer)

    Pass the returned thread_id back in subsequent requests to maintain
    conversation context within the same session.
    """
    try:
        result = run(
            user_query=request.query,
            thread_id=request.thread_id,
        )
        return QueryResponse(
            thread_id=result["thread_id"],
            task_route=result.get("task_route", ""),
            analyst_output=result.get("analyst_output", ""),
            strategist_output=result.get("strategist_output", ""),
            content_output=result.get("content_output", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{thread_id}")
def get_session(thread_id: str):
    """
    Retrieve the message history for a past conversation thread.
    Useful for displaying chat history when a user returns to a session.
    """
    messages = get_thread_history(thread_id)
    return {
        "thread_id": thread_id,
        "message_count": len(messages),
        "messages": [
            {
                "role": m.__class__.__name__.replace("Message", "").lower(),
                "content": m.content[:500] if len(m.content) > 500 else m.content,
            }
            for m in messages
        ],
    }
