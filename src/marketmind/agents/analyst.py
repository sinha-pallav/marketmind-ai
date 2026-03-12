"""
Analyst Agent — powered by Claude.

ROLE: Data-grounded marketing analyst. Uses RAG tools to retrieve context
before answering. Produces structured analysis with specific numbers.

AGENTIC LOOP (ReAct pattern):
  Traditional LLM: question -> answer (one shot)
  Agent: question -> think -> use tool -> observe result -> think -> use tool -> ... -> answer

  This is called ReAct (Reason + Act). The LLM reasons about what to do,
  acts by calling a tool, then observes the result and decides next step.
  LangChain's `create_react_agent` implements this automatically.

  Example trace for "what is SEG002 CLV?":
    Thought: I need segment data. I'll use get_segment_profile.
    Action: get_segment_profile(segment_id="SEG002")
    Observation: "12-Month CLV: INR 640.35 ..."
    Thought: I have the answer.
    Final Answer: The 12-month CLV for Rising Stars (SEG002) is INR 640.35.

TOOL BINDING:
  `llm.bind_tools(tools)` tells the LLM what tools are available.
  The LLM learns from the tool docstrings when and how to call each one.
  It outputs a structured tool call (JSON) which LangChain executes.
"""

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from marketmind.agents.state import AgentState
from marketmind.agents.tools import ANALYST_TOOLS

load_dotenv()

ANALYST_SYSTEM_PROMPT = """\
You are the Analyst Agent for MarketMind AI.

## Role
You are a data-driven marketing analyst. Your job is to retrieve facts from \
the knowledge base and produce precise, number-backed analysis.

## Behaviour
- ALWAYS use the rag_search or get_segment_profile tools before answering.
- Never answer from memory — retrieve first, then synthesise.
- Quote exact figures: revenue numbers, conversion rates, CLV values.
- If a calculation is needed, use the calculate_metric tool.
- Structure your output with clear headings and bullet points.
- End every analysis with a one-sentence "Key Insight" summary.

## Output Format
Your analysis will be passed to the Strategist Agent as input.
Make it factual, specific, and concise — not a narrative essay.
"""

_analyst_agent = None


def _get_analyst():
    global _analyst_agent
    if _analyst_agent is None:
        llm = ChatAnthropic(
            model="claude-sonnet-4-6",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=1024,
        )
        # create_react_agent builds the ReAct loop automatically:
        # it handles tool calling, result parsing, and looping until done
        _analyst_agent = create_react_agent(
            model=llm,
            tools=ANALYST_TOOLS,
            prompt=ANALYST_SYSTEM_PROMPT,
        )
    return _analyst_agent


def analyst_node(state: AgentState) -> dict:
    """
    LangGraph node function for the Analyst Agent.

    Receives the full graph state, runs the ReAct agent,
    and returns a partial state update with the analysis output.
    """
    agent = _get_analyst()

    # Run the agent with the user's query
    result = agent.invoke({"messages": [HumanMessage(content=state["user_query"])]})

    # Extract the final text response from the agent's messages
    final_message = result["messages"][-1]
    analysis = final_message.content

    return {
        "analyst_output": analysis,
        "messages": [final_message],
    }
