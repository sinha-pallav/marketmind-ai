"""
Short-term memory via LangGraph checkpointing.

THE PROBLEM WITHOUT THIS:
  Every time you call graph.invoke(), state is reset from scratch.
  The agent has no memory of what was said 2 messages ago in the same session.

  User: "Analyse SEG002"
  Agent: [analyses SEG002]
  User: "Now write an email for them"
  Agent: "Who is 'them'?" ← no memory of the previous message

HOW LANGGRAPH CHECKPOINTING FIXES IT:
  A checkpointer saves the full AgentState after each graph run.
  When you invoke the graph again with the same thread_id, it loads the
  previous state and continues from where it left off.

  thread_id = a unique ID per conversation/session (like a chat thread)
  Every user session gets a different thread_id so conversations don't mix.

TWO CHECKPOINTER OPTIONS:
  1. MemorySaver (used here in development):
     - Stores state in Python dict in memory
     - Lost when the process stops
     - Zero setup, great for testing and local development

  2. SqliteSaver / PostgresSaver (for production):
     - Persists to disk/database — survives restarts
     - Drop-in replacement — just swap the checkpointer object
     - We'll upgrade to PostgresSaver in Week 5 when the DB is set up
"""

import uuid
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver

# A single in-memory checkpointer shared across all sessions
# In production: replace with SqliteSaver or PostgresSaver
_checkpointer = MemorySaver()


def get_checkpointer() -> MemorySaver:
    """Return the shared checkpointer instance."""
    return _checkpointer


def new_thread_id() -> str:
    """
    Generate a unique thread ID for a new conversation session.
    A thread_id is like a conversation ID — each chat window gets one.
    """
    return str(uuid.uuid4())


def make_config(thread_id: str) -> dict:
    """
    Build the LangGraph config dict for a given thread.
    Pass this as the second argument to graph.invoke(state, config).

    Example:
        config = make_config("user_pallav_session_1")
        result = graph.invoke(initial_state, config)
        # Later in same session:
        result = graph.invoke(next_state, config)  # ← remembers the first run
    """
    return {"configurable": {"thread_id": thread_id}}


def get_thread_history(thread_id: str) -> list:
    """
    Retrieve all messages from a past thread (conversation history).
    Useful for displaying chat history in the UI.
    """
    try:
        checkpoint = _checkpointer.get({"configurable": {"thread_id": thread_id}})
        if checkpoint and "channel_values" in checkpoint:
            return checkpoint["channel_values"].get("messages", [])
    except Exception:
        pass
    return []
