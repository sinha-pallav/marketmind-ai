"""
RAG chain: retrieval + prompt + LLM = a complete answer.

This is the full loop:
  User question
    -> RAGPipeline retrieves relevant chunks
    -> Chunks injected into system prompt
    -> Claude reads the grounded prompt + user question
    -> Claude returns a data-grounded answer

LANGCHAIN CHAIN PATTERN:
  We use LCEL (LangChain Expression Language) which uses the pipe operator |
  to chain steps together. Each step receives the output of the previous step.

  prompt | llm | output_parser

  This is equivalent to:
    1. Format the prompt
    2. Send to LLM
    3. Parse the response

  LCEL makes it easy to swap components — e.g. swap Claude for GPT-4o
  by just changing the llm variable.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from marketmind.rag.pipeline import RAGPipeline
from marketmind.rag.prompt import build_prompt

load_dotenv()


def build_rag_chain(data_dir: Path):
    """
    Build the full RAG chain.

    Returns a callable: chain(question) -> answer string
    """
    # Build the retrieval pipeline (loads BM25 corpus + connects to Pinecone)
    pipeline = RAGPipeline.build(data_dir)

    # Claude Sonnet — good balance of quality and cost for analyst tasks
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=1024,
    )

    output_parser = StrOutputParser()

    def chain(question: str) -> str:
        # Step 1: retrieve grounded context
        context = pipeline.query(question, top_k=5)

        # Step 2: build system prompt with context injected
        system_prompt = build_prompt(context)

        # Step 3: call Claude
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]
        response = llm.invoke(messages)

        # Step 4: extract text from response
        return output_parser.invoke(response)

    return chain


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    data_dir = project_root / "data"

    print("Building RAG chain...")
    ask = build_rag_chain(data_dir)

    test_questions = [
        "Which customer segment has the highest churn risk and what should we do about it?",
        "What were the top 3 priorities for Q2 2026 marketing?",
        "Which contact method had the best campaign conversion rate?",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print(f"{'='*60}")
        answer = ask(question)
        print(f"A: {answer}")
