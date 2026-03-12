"""
The MarketMind AI agent graph — wired with LangGraph.

LANGGRAPH CONCEPTS:
  StateGraph:  The graph builder. Nodes and edges are added to it, then compiled.
  add_node:    Register an agent function as a named node in the graph.
  add_edge:    Always go from node A to node B (unconditional).
  add_conditional_edges: After node A, call a router function to decide next node.
  set_entry_point: Which node runs first when the graph is invoked.
  compile():   Validates and freezes the graph. Returns a runnable object.

GRAPH FLOW:

  START
    |
    v
  [orchestrator_node]  -- classifies the query, sets task_route
    |
    | (conditional routing based on task_route)
    |
    +--"analyst"---------> [analyst_node] -------> END
    |
    +--"strategist"------> [strategist_node] -----> END
    |
    +--"content_writer"--> [content_writer_node] -> END
    |
    +--"full_pipeline"--> [analyst_node]
                               |
                          [strategist_node]
                               |
                          [content_writer_node]
                               |
                              END

The orchestrator is a lightweight LLM call that classifies intent.
All heavy work (tool use, data retrieval, generation) happens in the
specialist agent nodes.
"""

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from marketmind.agents.analyst import analyst_node
from marketmind.agents.content_writer import content_writer_node
from marketmind.agents.state import AgentState
from marketmind.agents.strategist import strategist_node
from marketmind.memory.short_term import get_checkpointer, make_config, new_thread_id
from marketmind.memory import long_term

load_dotenv()

# ---------------------------------------------------------------------------
# Orchestrator — classifies query intent and sets routing
# ---------------------------------------------------------------------------

ORCHESTRATOR_PROMPT = """\
You are the Orchestrator for MarketMind AI. Your only job is to classify \
the user's request into one of four routing options.

Routing options:
- "analyst"        : User wants data analysis, metrics, segment info, performance data
- "strategist"     : User wants a campaign strategy, plan, or recommendation
- "content_writer" : User wants email copy, SMS text, social media posts, ad creative
- "full_pipeline"  : User wants analysis + strategy + copy (a complete campaign brief)

Reply with ONLY one of these four words. Nothing else. No explanation.

Examples:
  "What is the churn rate of SEG003?"                  -> analyst
  "Create a Q2 strategy for Rising Stars"              -> strategist
  "Write a win-back email for dormant customers"       -> content_writer
  "Build a complete campaign for cart abandonment"     -> full_pipeline
  "Which products have the highest margin?"            -> analyst
  "How should we approach the bargain hunters segment?" -> strategist
"""


def orchestrator_node(state: AgentState) -> dict:
    """
    Classifies the user query and sets task_route.
    Uses a fast, minimal LLM call — just a classification, no tools needed.
    """
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",  # Fastest/cheapest model for classification
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=10,  # Only need one word in response
    )

    messages = [
        HumanMessage(content=f"{ORCHESTRATOR_PROMPT}\n\nUser request: {state['user_query']}")
    ]
    response = llm.invoke(messages)
    route = response.content.strip().lower()

    # Validate — default to analyst if LLM returns unexpected value
    valid_routes = {"analyst", "strategist", "content_writer", "full_pipeline"}
    if route not in valid_routes:
        route = "analyst"

    print(f"  [Orchestrator] Routing to: {route}")
    return {"task_route": route}


# ---------------------------------------------------------------------------
# Conditional router — called after orchestrator to determine next node
# ---------------------------------------------------------------------------

def route_after_orchestrator(state: AgentState) -> str:
    """Returns the next node name based on task_route."""
    return state.get("task_route", "analyst")


def route_after_analyst(state: AgentState) -> str:
    """In full_pipeline, continue to strategist. Otherwise stop."""
    if state.get("task_route") == "full_pipeline":
        return "strategist"
    return END


def route_after_strategist(state: AgentState) -> str:
    """In full_pipeline, continue to content_writer. Otherwise stop."""
    if state.get("task_route") == "full_pipeline":
        return "content_writer"
    return END


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------

def build_graph():
    """
    Construct, wire, and compile the MarketMind agent graph.
    Returns a compiled LangGraph runnable.
    """
    workflow = StateGraph(AgentState)

    # Register all nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("strategist", strategist_node)
    workflow.add_node("content_writer", content_writer_node)

    # Entry point — always starts at orchestrator
    workflow.add_edge(START, "orchestrator")

    # After orchestrator: conditional routing to one of the four options
    workflow.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "analyst": "analyst",
            "strategist": "strategist",
            "content_writer": "content_writer",
            "full_pipeline": "analyst",  # full_pipeline starts at analyst
        },
    )

    # After analyst: stop OR continue to strategist (if full_pipeline)
    workflow.add_conditional_edges(
        "analyst",
        route_after_analyst,
        {"strategist": "strategist", END: END},
    )

    # After strategist: stop OR continue to content_writer (if full_pipeline)
    workflow.add_conditional_edges(
        "strategist",
        route_after_strategist,
        {"content_writer": "content_writer", END: END},
    )

    # Content writer always ends the pipeline
    workflow.add_edge("content_writer", END)

    # Attach short-term memory checkpointer
    # This enables conversation continuity across multiple graph.invoke() calls
    # when the same thread_id is used
    return workflow.compile(checkpointer=get_checkpointer())


# ---------------------------------------------------------------------------
# Helper: run the graph and return a clean final output
# ---------------------------------------------------------------------------

def run(user_query: str, thread_id: str = None) -> dict:
    """
    Run the full agent graph for a user query.

    Args:
        user_query: The user's natural language request.
        thread_id:  Optional session ID for conversation continuity.
                    If None, a new thread is created (fresh conversation).
                    Pass the same thread_id across multiple calls to maintain
                    short-term memory within a session.

    Returns a dict with:
      - task_route: which path was taken
      - analyst_output: if analyst ran
      - strategist_output: if strategist ran
      - content_output: if content writer ran
      - thread_id: the session ID (save this to continue the conversation)
    """
    if thread_id is None:
        thread_id = new_thread_id()

    graph = build_graph()
    config = make_config(thread_id)

    initial_state = {
        "messages": [],
        "user_query": user_query,
        "task_route": "",
        "analyst_output": "",
        "strategist_output": "",
        "content_output": "",
    }

    result = graph.invoke(initial_state, config)
    result["thread_id"] = thread_id

    # Persist to long-term memory (silently skipped if MongoDB not configured)
    long_term.log_session(
        thread_id=thread_id,
        user_query=user_query,
        task_route=result.get("task_route", ""),
        analyst_output=result.get("analyst_output", ""),
        strategist_output=result.get("strategist_output", ""),
        content_output=result.get("content_output", ""),
    )

    return result


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n" + "="*60)
    print("TEST 1: Single agent — analyst only")
    print("="*60)
    result = run("What is the churn risk and CLV of the Bargain Hunters segment?")
    print(f"Route taken: {result['task_route']}")
    print(f"\nAnalyst output:\n{result['analyst_output']}")

    print("\n" + "="*60)
    print("TEST 2: Full pipeline — all three agents")
    print("="*60)
    result = run(
        "Build a complete win-back campaign for our dormant customers — "
        "include the data analysis, campaign strategy, and email copy."
    )
    print(f"Route taken: {result['task_route']}")
    print(f"\nAnalyst:\n{result['analyst_output'][:400]}...")
    print(f"\nStrategist:\n{result['strategist_output'][:400]}...")
    print(f"\nContent Writer:\n{result['content_output'][:400]}...")
