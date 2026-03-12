"""
Strategist Agent — designed for GPT-4o, using Claude as placeholder.

ROLE: Takes the Analyst's data output and produces a concrete, actionable
campaign strategy. Does not do data retrieval — it synthesises and plans.

MULTI-LLM NOTE:
  In the final architecture this agent uses GPT-4o:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

  Why GPT-4o for strategy?
  - GPT-4o is strong at structured reasoning and multi-step planning
  - Using a different LLM for each agent demonstrates multi-LLM orchestration
    (a specific skill listed in AI Agent Architect job specs)
  - In production, you'd benchmark both and pick the best for each role

  TO SWAP: Change the 3 lines under "# LLM" below once OpenAI key is in .env
"""

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from marketmind.agents.state import AgentState
from marketmind.agents.tools import STRATEGIST_TOOLS

load_dotenv()

STRATEGIST_SYSTEM_PROMPT = """\
You are the Strategist Agent for MarketMind AI.

## Role
You receive a data analysis from the Analyst Agent and produce a clear,
actionable marketing campaign strategy.

## Behaviour
- Read the analyst output carefully. Build your strategy directly on the data.
- You may use rag_search to look up additional context if needed.
- Produce a structured campaign strategy with:
    1. Campaign Objective (one sentence, measurable)
    2. Target Segment (with rationale from the data)
    3. Key Message (what problem does this campaign solve for the customer?)
    4. Channel Mix (which channels, why, and what budget split)
    5. Success Metrics (KPIs with specific targets)
    6. Timeline (key milestones)
- Be specific. "Increase email revenue" is not a strategy. \
"Send a 3-email win-back sequence to SEG004 targeting 13% reactivation" is.

## Constraints
- Do not invent data that was not in the analyst output.
- Flag any assumptions you make explicitly.
"""

_strategist_agent = None


def _get_strategist():
    global _strategist_agent
    if _strategist_agent is None:
        # LLM — swap to GPT-4o when OpenAI key is available:
        # from langchain_openai import ChatOpenAI
        # llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
        llm = ChatAnthropic(
            model="claude-sonnet-4-6",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=1024,
        )
        _strategist_agent = create_react_agent(
            model=llm,
            tools=STRATEGIST_TOOLS,
            prompt=STRATEGIST_SYSTEM_PROMPT,
        )
    return _strategist_agent


def strategist_node(state: AgentState) -> dict:
    """
    LangGraph node for the Strategist Agent.
    Receives analyst output as context, produces campaign strategy.
    """
    agent = _get_strategist()

    # Build the prompt: include both user query and analyst output
    prompt = (
        f"User Request: {state['user_query']}\n\n"
        f"Analyst Data Output:\n{state.get('analyst_output', 'No analysis available.')}\n\n"
        f"Based on the above, create a campaign strategy."
    )

    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    final_message = result["messages"][-1]
    strategy = final_message.content

    return {
        "strategist_output": strategy,
        "messages": [final_message],
    }
