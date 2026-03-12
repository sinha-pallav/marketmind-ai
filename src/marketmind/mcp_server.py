"""
MCP (Model Context Protocol) Server for MarketMind AI.

WHAT IS MCP?
  MCP is an open standard by Anthropic that lets AI models connect to external
  tools and data sources in a standardised way — like a USB-C port for AI tools.

  Without MCP: each AI app builds its own custom tool integration
  With MCP:    one server exposes tools → any MCP client can use them

  In this project, our MCP server exposes:
    - rag_search          → search the marketing knowledge base
    - calculate_metric    → compute marketing KPIs
    - get_segment_profile → look up customer segment data
    - run_agent           → run the full multi-agent pipeline

  These tools can then be called by:
    - Claude Desktop (via mcp settings)
    - Any LangGraph agent with MCP tool binding
    - External services via HTTP

HOW IT WORKS:
  The MCP server runs as a separate process using stdio transport.
  The client (e.g. Claude Desktop) launches the server and communicates
  with it via stdin/stdout using JSON-RPC messages.

  To connect Claude Desktop to this server, add to claude_desktop_config.json:
  {
    "mcpServers": {
      "marketmind": {
        "command": "python",
        "args": ["-m", "marketmind.mcp_server"],
        "cwd": "C:/Users/PALLA/OneDrive/Desktop/MarketMind AI"
      }
    }
  }

TO RUN STANDALONE:
  .venv/Scripts/python -m marketmind.mcp_server
"""

import asyncio
import json
from pathlib import Path

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Build the MCP server instance
server = Server("marketmind-ai")

# Lazy-load heavy dependencies (RAG pipeline etc.) only when tools are called
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from marketmind.rag.pipeline import RAGPipeline
        project_root = Path(__file__).parent.parent.parent
        _pipeline = RAGPipeline.build(project_root / "data")
    return _pipeline


# ---------------------------------------------------------------------------
# Tool definitions — what the MCP client sees
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Declare all available tools with their schemas."""
    return [
        types.Tool(
            name="rag_search",
            description=(
                "Search the MarketMind marketing knowledge base. Use this to find "
                "information about customer segments, campaign performance, product "
                "catalog, market data, and marketing strategy documents."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language question or topic to search for",
                    }
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="calculate_metric",
            description=(
                "Calculate a marketing KPI. Supported: roas, clv, cac, "
                "churn_rate, conversion_rate, email_roi."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "description": "The metric to calculate (e.g. 'roas', 'clv')",
                    },
                    "values": {
                        "type": "object",
                        "description": "Key-value pairs of input numbers for the metric",
                    },
                },
                "required": ["metric", "values"],
            },
        ),
        types.Tool(
            name="get_segment_profile",
            description=(
                "Get the full profile for a specific customer segment. "
                "Available segments: SEG001, SEG002, SEG003, SEG004, SEG005."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "segment_id": {
                        "type": "string",
                        "description": "Segment ID, e.g. 'SEG001'",
                    }
                },
                "required": ["segment_id"],
            },
        ),
        types.Tool(
            name="run_agent",
            description=(
                "Run the full MarketMind multi-agent pipeline for a marketing request. "
                "Automatically routes to the right agents (analyst, strategist, content writer). "
                "Use for complex requests that need data analysis + strategy + copy."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The marketing request to process",
                    },
                    "thread_id": {
                        "type": "string",
                        "description": "Optional session ID for conversation continuity",
                    },
                },
                "required": ["query"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tool execution — called when the client invokes a tool
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[types.TextContent]:
    """Execute the requested tool and return results."""

    if name == "rag_search":
        pipeline = _get_pipeline()
        result = pipeline.query(arguments["query"], top_k=5)
        return [types.TextContent(type="text", text=result)]

    elif name == "calculate_metric":
        from marketmind.agents.tools import calculate_metric
        result = calculate_metric.invoke({
            "metric": arguments["metric"],
            "values": arguments["values"],
        })
        return [types.TextContent(type="text", text=result)]

    elif name == "get_segment_profile":
        from marketmind.agents.tools import get_segment_profile
        result = get_segment_profile.invoke({"segment_id": arguments["segment_id"]})
        return [types.TextContent(type="text", text=result)]

    elif name == "run_agent":
        from marketmind.agents.graph import run
        result = run(
            user_query=arguments["query"],
            thread_id=arguments.get("thread_id"),
        )
        # Return the most complete output available
        output_parts = []
        if result.get("analyst_output"):
            output_parts.append(f"## Analysis\n{result['analyst_output']}")
        if result.get("strategist_output"):
            output_parts.append(f"## Strategy\n{result['strategist_output']}")
        if result.get("content_output"):
            output_parts.append(f"## Campaign Copy\n{result['content_output']}")
        output_parts.append(f"\n_Session ID: {result.get('thread_id', 'N/A')}_")
        return [types.TextContent(type="text", text="\n\n---\n\n".join(output_parts))]

    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
