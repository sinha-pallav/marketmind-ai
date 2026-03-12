"""
Shared state for the multi-agent graph.

WHAT IS STATE IN LANGGRAPH?
  State is a TypedDict (typed dictionary) that gets passed to every node
  in the graph. Each node reads from it and returns a partial update.
  LangGraph merges the updates automatically.

  Think of it like a whiteboard in a war room:
  - Every agent can read the whole board
  - Each agent writes only their section
  - The orchestrator reads everything to make decisions

WHY TypedDict?
  Python's TypedDict gives us type hints (IDE autocomplete, error catching)
  without the overhead of a full class. LangGraph requires it for state.

THE `add_messages` REDUCER:
  Normal dict fields: new value REPLACES old value
  `Annotated[list, add_messages]`: new messages are APPENDED to the list

  Without it:  state["messages"] = [new_msg]       <- history lost!
  With it:     state["messages"] = [old..., new_msg] <- history preserved

  This is critical for agents that need conversation context.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Full conversation history — add_messages ensures messages are appended
    # not replaced. Every agent and tool call adds to this list.
    messages: Annotated[list, add_messages]

    # The original user query — preserved unchanged throughout the pipeline
    user_query: str

    # Set by the orchestrator to control routing:
    # "analyst"        → only the Analyst Agent runs
    # "strategist"     → only the Strategist Agent runs
    # "content_writer" → only the Content Writer runs
    # "full_pipeline"  → Analyst -> Strategist -> Content Writer (all three)
    task_route: str

    # Outputs from each specialist agent — populated as agents complete
    analyst_output: str
    strategist_output: str
    content_output: str
