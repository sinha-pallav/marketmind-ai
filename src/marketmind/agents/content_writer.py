"""
Content Writer Agent — designed for Mistral, using Claude as placeholder.

ROLE: Takes the campaign strategy and writes the actual marketing copy —
email subject lines, body text, SMS messages, social captions.

MULTI-LLM NOTE:
  In the final architecture this agent uses a fine-tuned Mistral 7B (Week 7).
  Fine-tuning on marketing copy makes it better at brand voice than a generic model.

  To swap to Mistral via HuggingFace Inference API once available:
    from langchain_huggingface import HuggingFaceEndpoint
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    )

  Or via Ollama (local, free):
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="mistral")

  TO SWAP: Replace the LLM block below when Mistral is available.
"""

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from marketmind.agents.state import AgentState
from marketmind.agents.tools import CONTENT_TOOLS

load_dotenv()

CONTENT_WRITER_SYSTEM_PROMPT = """\
You are the Content Writer Agent for MarketMind AI.

## Role
You write compelling, conversion-focused marketing copy based on a campaign \
strategy and supporting customer data.

## Behaviour
- Use rag_search or get_segment_profile to understand the target audience \
  before writing. Know who you're writing to.
- Write in a warm, direct, human voice — not corporate jargon.
- Every piece of copy must have a clear call-to-action (CTA).
- Personalisation tokens should use [FIRST_NAME], [LAST_PURCHASE], [SEGMENT_OFFER].

## Output Format
Always produce ALL of the following for email campaigns:
  1. Subject Line (max 50 characters)
  2. Preview Text (max 90 characters, appears in inbox next to subject)
  3. Email Body (3 short paragraphs: hook, value, CTA)
  4. CTA Button Text (max 5 words)
  5. SMS Version (max 160 characters including link placeholder [LINK])

For social media campaigns, produce:
  1. Instagram Caption (with 5 hashtags)
  2. Short-form video hook (first 3 seconds, max 15 words)

## Constraints
- Do not use exclamation marks more than once per email.
- Do not make claims not supported by the strategy or data.
- Keep subject lines curiosity-driven, not click-bait.
"""

_content_agent = None


def _get_content_writer():
    global _content_agent
    if _content_agent is None:
        # LLM — swap to fine-tuned Mistral in Week 7:
        # from langchain_huggingface import HuggingFaceEndpoint
        # llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3", ...)
        llm = ChatAnthropic(
            model="claude-sonnet-4-6",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=1500,
        )
        _content_agent = create_react_agent(
            model=llm,
            tools=CONTENT_TOOLS,
            prompt=CONTENT_WRITER_SYSTEM_PROMPT,
        )
    return _content_agent


def content_writer_node(state: AgentState) -> dict:
    """
    LangGraph node for the Content Writer Agent.
    Receives the campaign strategy and writes marketing copy.
    """
    agent = _get_content_writer()

    prompt = (
        f"User Request: {state['user_query']}\n\n"
        f"Campaign Strategy:\n{state.get('strategist_output', 'No strategy provided.')}\n\n"
        f"Write the marketing copy for this campaign."
    )

    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    final_message = result["messages"][-1]
    content = final_message.content

    return {
        "content_output": content,
        "messages": [final_message],
    }
