"""
Tools available to agents.

WHAT ARE TOOLS IN AGENTIC AI?
  Tools are functions that an LLM can choose to call during its reasoning.
  Instead of the LLM "knowing" everything, it can look things up, calculate,
  or trigger actions — just like a human analyst using a calculator or database.

  LangChain's @tool decorator:
  1. Wraps a Python function
  2. Extracts the docstring as the tool description (what the LLM reads to decide when to use it)
  3. Extracts the function signature as the input schema
  4. Makes it callable by any LangChain/LangGraph agent

  IMPORTANT: Write clear docstrings — the LLM reads them to decide which tool
  to call. A vague docstring = tool used incorrectly or not at all.

TOOLS BUILT HERE:
  1. rag_search          — search the marketing knowledge base (uses our RAG pipeline)
  2. calculate_metric    — compute common marketing KPIs from raw numbers
  3. get_segment_profile — quick structured lookup for a specific customer segment
"""

from pathlib import Path

from langchain_core.tools import tool

# Lazy-load the RAG pipeline — avoids loading 200 chunks + Pinecone connection
# every time this module is imported. Loads only when a tool is first called.
_rag_pipeline = None


def _get_pipeline():
    global _rag_pipeline
    if _rag_pipeline is None:
        from marketmind.rag.pipeline import RAGPipeline
        project_root = Path(__file__).parent.parent.parent.parent
        _rag_pipeline = RAGPipeline.build(project_root / "data")
    return _rag_pipeline


# ---------------------------------------------------------------------------
# Tool 1: RAG Search
# ---------------------------------------------------------------------------

@tool
def rag_search(query: str) -> str:
    """
    Search the marketing knowledge base for information about customer segments,
    campaign performance, product catalog, market data, or marketing strategy.

    Use this tool whenever you need facts, statistics, or context from the
    company's marketing data. Do not answer from memory — always retrieve first.

    Args:
        query: A natural language question or topic to search for.

    Returns:
        Relevant context from the knowledge base as formatted text.
    """
    pipeline = _get_pipeline()
    return pipeline.query(query, top_k=5)


# ---------------------------------------------------------------------------
# Tool 2: Marketing Metric Calculator
# ---------------------------------------------------------------------------

@tool
def calculate_metric(metric: str, values: dict) -> str:
    """
    Calculate a common marketing metric from provided input values.

    Supported metrics:
    - "roas"     : Return on Ad Spend. values = {"revenue": float, "ad_spend": float}
    - "clv"      : Customer Lifetime Value. values = {"avg_order_value": float, "purchase_frequency": float, "avg_customer_lifespan_years": float}
    - "cac"      : Customer Acquisition Cost. values = {"total_marketing_spend": float, "new_customers_acquired": int}
    - "churn_rate": Monthly churn rate. values = {"customers_lost": int, "customers_start_of_period": int}
    - "conversion_rate": values = {"conversions": int, "total_contacts": int}
    - "email_roi": values = {"revenue_attributed": float, "email_spend": float}

    Args:
        metric: The metric name (one of the options above).
        values: A dict of input values required for that metric.

    Returns:
        A string with the calculated result and interpretation.
    """
    metric = metric.lower().strip()

    try:
        if metric == "roas":
            roas = values["revenue"] / values["ad_spend"]
            return (
                f"ROAS = {roas:.2f}x\n"
                f"For every INR 1 spent on ads, INR {roas:.2f} in revenue was generated.\n"
                f"Benchmark: >3x is good, >5x is excellent for most retail categories."
            )

        elif metric == "clv":
            clv = (
                values["avg_order_value"]
                * values["purchase_frequency"]
                * values["avg_customer_lifespan_years"]
            )
            return (
                f"Customer Lifetime Value = INR {clv:,.2f}\n"
                f"Based on: AOV={values['avg_order_value']}, "
                f"frequency={values['purchase_frequency']}/year, "
                f"lifespan={values['avg_customer_lifespan_years']} years."
            )

        elif metric == "cac":
            cac = values["total_marketing_spend"] / values["new_customers_acquired"]
            return (
                f"Customer Acquisition Cost = INR {cac:,.2f}\n"
                f"Each new customer cost INR {cac:,.2f} to acquire.\n"
                f"Healthy ratio: CLV should be at least 3x CAC."
            )

        elif metric == "churn_rate":
            rate = (values["customers_lost"] / values["customers_start_of_period"]) * 100
            return (
                f"Monthly Churn Rate = {rate:.2f}%\n"
                f"Annual equivalent = ~{min(rate * 12, 100):.1f}%\n"
                f"Benchmark: <2% monthly is good for retail subscriptions."
            )

        elif metric == "conversion_rate":
            rate = (values["conversions"] / values["total_contacts"]) * 100
            return f"Conversion Rate = {rate:.2f}% ({values['conversions']} / {values['total_contacts']})"

        elif metric == "email_roi":
            roi = ((values["revenue_attributed"] - values["email_spend"]) / values["email_spend"]) * 100
            return (
                f"Email ROI = {roi:.0f}%\n"
                f"For every INR 1 spent on email, INR {values['revenue_attributed']/values['email_spend']:.2f} returned.\n"
                f"Industry benchmark: email typically delivers 3600-4200% ROI."
            )

        else:
            return (
                f"Unknown metric '{metric}'. "
                f"Supported: roas, clv, cac, churn_rate, conversion_rate, email_roi"
            )

    except KeyError as e:
        return f"Missing required value: {e}. Check the 'values' dict for this metric."
    except ZeroDivisionError:
        return "Cannot calculate — a denominator value is zero."


# ---------------------------------------------------------------------------
# Tool 3: Segment Profile Lookup
# ---------------------------------------------------------------------------

@tool
def get_segment_profile(segment_id: str) -> str:
    """
    Get the full profile for a specific customer segment by its ID.

    Use this when you need precise details about a segment (size, CLV,
    preferred channel, churn risk) and you already know the segment ID.
    Faster than a full RAG search when you know exactly which segment you need.

    Args:
        segment_id: The segment identifier, e.g. "SEG001", "SEG002", etc.

    Returns:
        Full segment profile as formatted text, or an error if not found.
    """
    # Segment data is small — load directly rather than via RAG
    import json
    from pathlib import Path

    import pandas as pd

    project_root = Path(__file__).parent.parent.parent.parent
    segments_path = project_root / "data" / "raw" / "customers" / "customer_segments.csv"

    try:
        df = pd.read_csv(segments_path)
        row = df[df["segment_id"].str.upper() == segment_id.upper()]

        if row.empty:
            available = ", ".join(df["segment_id"].tolist())
            return f"Segment '{segment_id}' not found. Available IDs: {available}"

        row = row.iloc[0]
        return (
            f"Segment Profile: {row['segment_name']} ({row['segment_id']})\n"
            f"Size: {row['size']:,} customers\n"
            f"Average Order Value: INR {row['avg_order_value']}\n"
            f"Purchase Frequency: {row['purchase_frequency_per_year']} times/year\n"
            f"12-Month CLV: INR {row['avg_clv_12m']}\n"
            f"Top Categories: {row['top_categories']}\n"
            f"Preferred Channel: {row['preferred_channel']}\n"
            f"Churn Risk: {row['churn_risk']}\n"
            f"Description: {row['description']}"
        )
    except Exception as e:
        return f"Error loading segment data: {e}"


# All tools in one list — passed to agents that need them
ALL_TOOLS = [rag_search, calculate_metric, get_segment_profile]
ANALYST_TOOLS = [rag_search, calculate_metric, get_segment_profile]
STRATEGIST_TOOLS = [rag_search, calculate_metric]
CONTENT_TOOLS = [rag_search, get_segment_profile]
