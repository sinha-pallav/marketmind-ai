# MarketMind AI — Build Progress Report
**Last Updated:** March 2026
**Current Status:** Week 1-5 Complete (RAG + Multi-Agent System + Memory + API + UI + Docker + CI/CD)

---

## 1. What We Built — Overview

| Phase | What | Files Created | Status |
|---|---|---|---|
| Project Setup | Python project structure, venv, git | `pyproject.toml`, `.gitignore`, `.env.example`, all `__init__.py` | Done |
| Data Preparation | 5 data sources across 3 formats | `data/prepare_data.py` | Done |
| RAG Ingestion | Load -> Chunk -> Embed -> Pinecone | `loaders.py`, `chunker.py`, `embedder.py`, `ingestion.py` | Done |
| RAG Retrieval | Hybrid search + compression | `retriever.py`, `compression.py`, `pipeline.py` | Done |
| LLM Chain | Prompt architecture + Claude call | `prompt.py`, `chain.py` | Done |
| Agent Tools | @tool-decorated LangChain tools | `agents/tools.py` | Done |
| Multi-Agent Graph | LangGraph orchestrator + 3 specialist agents | `agents/state.py`, `agents/analyst.py`, `agents/strategist.py`, `agents/content_writer.py`, `agents/graph.py` | Done |
| Short-Term Memory | LangGraph MemorySaver checkpointer | `memory/short_term.py` | Done |
| Long-Term Memory | MongoDB Atlas (sessions, campaigns, insights) | `memory/long_term.py` | Done |
| MCP Server | 4 tools exposed via Model Context Protocol | `mcp_server.py` | Done |
| FastAPI Backend | REST API with /query, /health, /sessions | `api/main.py` | Done |
| Streamlit UI | Chat interface calling FastAPI | `ui/app.py` | Done |
| Docker | Multi-container build for API + UI | `Dockerfile`, `Dockerfile.streamlit`, `docker-compose.yml`, `.dockerignore` | Done |
| CI/CD | Auto-deploy on push to main | `.github/workflows/deploy.yml` | Done |
| Fly.io Config | Production deployment config | `fly.toml` | Done |

---

## 2. Phase-by-Phase Detail

### Phase 1 — Project Structure

| What We Did | Why |
|---|---|
| Created `src/marketmind/` layout (not just `marketmind/`) | Industry-standard `src/` layout prevents accidentally importing local code instead of installed package — a subtle bug that wastes hours |
| Used `pyproject.toml` instead of `requirements.txt` | Modern Python packaging standard. Declares dependencies, build system, and tool config in one file. What real engineering teams use |
| Created `.env` + `.env.example` separation | `.env` holds real secrets, never committed. `.env.example` is a template committed to git so teammates know what keys are needed |
| Added `.gitignore` | Prevents committing `.env`, `.venv/`, `data/`, `__pycache__/` — any of which could expose secrets or bloat the repo |
| Created `__init__.py` in every subfolder | Tells Python "this folder is an importable package." Without this, `from marketmind.rag.loaders import load_all` would fail with ModuleNotFoundError |
| Virtual environment with Python 3.11 (not 3.14) | Python 3.14 is too new — LangChain, Pinecone, sentence-transformers don't have binary wheels for it yet. 3.11 is the stable target for AI/ML packages |

**Issues & Fixes:**

| Issue | Cause | Fix |
|---|---|---|
| `pip` broken in venv | Corrupted pip installation from initial venv creation | Ran `python -m ensurepip --upgrade` then `pip install --upgrade pip` |
| `setuptools.backends.legacy:build` error | Wrong build backend name in `pyproject.toml` | Changed to `setuptools.build_meta` (the correct name) |
| `pinecone-client` import error | Pinecone renamed their package from `pinecone-client` to `pinecone` | Updated `pyproject.toml` to `pinecone>=5.0`, reinstalled |

---

### Phase 2 — Data Preparation

**Data sources chosen:**

| File | Source | Format | Size | Why This Dataset |
|---|---|---|---|---|
| `online_retail.csv` | UCI Machine Learning Repository | CSV | 35 MB / 500K+ rows | Real e-commerce transaction data. Lets us build product performance summaries that mirror actual retail analytics work |
| `bank_marketing.csv` | UCI Machine Learning Repository | CSV | 5 MB / 41K rows | Real campaign response data (phone campaigns). Gives us conversion rates by age, job type, contact method — realistic campaign analytics |
| `customer_segments.csv` | Synthetic (generated) | CSV | 2 KB / 5 rows | Rich text descriptions of 5 segments — High-Value Loyalists, Rising Stars, Bargain Hunters, Dormant Customers, Corporate Buyers. Designed to produce excellent RAG retrieval |
| `product_catalog.json` | Synthetic (generated) | JSON | 4 KB / 5 products | Products with campaign messaging, target segments, margins. Tests JSON ingestion in RAG pipeline |
| `q1_marketing_report.pdf` | Synthetic (generated with fpdf2) | PDF | 4 KB / 2 pages | Strategy document with Q1 performance, Q2 priorities, budget allocation. Tests unstructured text extraction |

**Why multiple formats?**
Clients have data in all formats. A system that only handles CSV is not production-ready. Handling CSV + JSON + PDF is what separates a demo from a deployable product — and it's what interviewers ask about.

**Issues & Fixes:**

| Issue | Cause | Fix |
|---|---|---|
| `→` character encoding error on Windows | Windows terminal uses cp1252 encoding which doesn't support Unicode arrow character | Replaced all `→` with `->` using sed |
| `\u2014` (em dash) PDF error | fpdf2's built-in Helvetica font only supports latin-1 charset | Created `_clean()` helper that replaces em dashes, curly quotes etc. with ASCII equivalents before passing to fpdf2 |
| `ln=True` deprecation warning in fpdf2 | API changed in fpdf2 v2.5.2 | Replaced with `new_x=XPos.LMARGIN, new_y=YPos.NEXT` |
| Nested zip file in bank marketing download | UCI changed zip structure — the CSV was inside `bank-additional.zip` which was inside the outer zip | Added nested `zipfile.ZipFile` extraction: open outer zip → open inner zip → read CSV |
| `KeyError: 'bank-additional/...'` | Needed to inspect actual zip contents before hardcoding the path | Ran a quick script to print `z.namelist()` to discover actual structure |

---

### Phase 3 — RAG Ingestion Pipeline

**Key design decisions:**

| Decision | Why |
|---|---|
| Did NOT create 1 Document per CSV row for large files | `online_retail.csv` has 500K rows. Creating 500K vectors in Pinecone costs money and retrieval would return useless individual transaction lines. Instead aggregated into 170 meaningful summaries (product performance + market summaries) |
| Aggregated bank marketing into 4 statistical summary docs | "44-year-old who said no" is not useful for RAG. "Cellular contacts convert at 14.7% vs telephone at 5.2%" is useful. Aggregation makes retrieved context actionable |
| Used `sentence-transformers/all-MiniLM-L6-v2` (local) | 384-dimension, ~90 MB model, runs on CPU. No API key needed, no rate limits, no cost per embedding. Good quality for semantic retrieval |
| Chunk size 512 characters with 50 char overlap | Embedding models cap around 256-512 tokens. 512 chars ≈ 100-130 tokens — safe buffer. Overlap of 50 chars ensures sentences split at a chunk boundary still appear complete in the next chunk |
| `RecursiveCharacterTextSplitter` with custom separators | Tries to split at `\n\n` (paragraph) first, then `\n`, then `. `, then ` `. Preserves readability — avoids splitting mid-sentence unless absolutely necessary |
| Used MD5 hash of source + type + chunk_index as vector ID | Deterministic IDs mean re-running ingestion produces same IDs → Pinecone upsert safely overwrites without creating duplicates |
| Metadata stored with each vector | Enables metadata filtering in retrieval (e.g. "only return customer_segment type docs") and source attribution in answers |

**What the ingestion produced:**

| Step | Input | Output |
|---|---|---|
| Load | 5 raw files | 186 Documents |
| Chunk | 186 Documents | 200 chunks (14 documents were large enough to split) |
| Embed | 200 text strings | 200 × 384-dimensional vectors |
| Upsert | 200 vectors | Pinecone index `marketmind`, namespace `marketing-knowledge-base`, 200 total vectors |

**Issues & Fixes:**

| Issue | Cause | Fix |
|---|---|---|
| `pypdf` not installed | Not included in initial `pyproject.toml` | Added `pypdf>=4.0` to dependencies, ran pip install |
| `pdf.pages` printed as dict object, not page count | `fpdf2` changed `pages` from an int to a dict in newer versions | Changed to `len(pdf.pages)` |

---

### Phase 4 — RAG Retrieval Pipeline

**Hybrid search — why both vector + BM25:**

| Retriever | Strength | Weakness |
|---|---|---|
| Vector (Pinecone) | Finds semantically similar content even with different wording. "At-risk customers" matches "churn risk" | Can return irrelevant results that are "semantically close" but wrong |
| BM25 (keyword) | Exact matches on specific terms like "SEG003", product codes, specific numbers | Misses synonyms and paraphrasing entirely |
| Hybrid (RRF) | Gets both semantic and keyword matches. A chunk ranked high in both gets a significantly higher combined score | Slightly more complex to implement |

**Reciprocal Rank Fusion (RRF) formula:**

```
score(chunk) = Σ  1 / (60 + rank_i)
```

k=60 is a standard constant. It prevents the top-ranked result from completely dominating. A chunk ranked 1st gets 1/61 = 0.0164. Rank 10 gets 1/70 = 0.0143. Small gaps reward consistency across both retrievers.

**Context compression:**

Retrieved chunks pass through an embeddings-based filter. The query is embedded and cosine similarity is computed against each chunk. Chunks below 0.25 similarity are dropped. This removes noise before sending to the LLM — better answers, lower token cost.

**Issues & Fixes:**

| Issue | Cause | Fix |
|---|---|---|
| No issues — retrieval worked first time | — | — |

---

### Phase 5 — LLM Chain (Prompt Architecture + Claude)

**Prompt structure (4 layers):**

| Layer | Content | Purpose |
|---|---|---|
| Role | "You are the Analyst Agent for MarketMind AI..." | Sets persona, ensures consistent tone across all questions |
| Instructions | "Answer ONLY using context... cite sources... be specific with numbers..." | Reduces hallucination, enforces grounded responses |
| Guardrails | "Do not invent statistics... if outside scope say..." | Makes the system safe to demo — won't fabricate data or go off-topic |
| Context | Retrieved chunks injected at runtime via `{context}` placeholder | The actual knowledge — different for every question |

**Why this prompt architecture matters for interviews:**
This is exactly what "Context Engineer" means in job descriptions. It's not just "write a prompt" — it's designing a layered system where each component serves a specific function and failure modes are handled explicitly.

**LLM chain results (tested with 3 questions):**

| Question | Quality | Notable |
|---|---|---|
| "Which segment has highest churn risk?" | Good | Claude correctly admitted it didn't have complete segment data rather than hallucinating — guardrail worked |
| "Q2 2026 marketing priorities?" | Excellent | Pulled exact figures from PDF (25% QoQ email target, 40% TikTok increase, INR 55 lakhs recovery) |
| "Best campaign contact method?" | Excellent | Computed 3x comparison (14.7% cellular vs 5.2% telephone) from bank marketing data, made recommendation |

---

### Phase 6 — Multi-Agent System (LangGraph)

**Architecture overview:**

The system has 4 agents, each with a distinct role:

| Agent | Model | Role | Tools |
|---|---|---|---|
| Orchestrator | claude-haiku-4-5 | Classifies query intent, sets routing | None (classification only) |
| Analyst | claude-sonnet-4-6 | Retrieves data, computes KPIs, summarises findings | `rag_search`, `calculate_metric`, `get_segment_profile` |
| Strategist | claude-sonnet-4-6 (GPT-4o placeholder) | Builds campaign strategy from analyst findings | `rag_search`, `get_segment_profile` |
| Content Writer | claude-sonnet-4-6 (Mistral placeholder) | Writes email, SMS, social media copy | `rag_search`, `get_segment_profile` |

**Routing logic:**

| Route | When | What Runs |
|---|---|---|
| `analyst` | User asks for data, metrics, segment info | Analyst -> END |
| `strategist` | User asks for a campaign plan or recommendation | Strategist -> END |
| `content_writer` | User asks for copy (email, SMS, ads) | Content Writer -> END |
| `full_pipeline` | User wants analysis + strategy + copy | Analyst -> Strategist -> Content Writer -> END |

**Key design decisions:**

| Decision | Why |
|---|---|
| Orchestrator uses Haiku with `max_tokens=10` | Classification is a single word. Using Sonnet/Opus would waste money and add latency for a task that needs zero reasoning |
| ReAct pattern (`create_react_agent`) for all specialist agents | Agents reason about which tool to call, call it, observe the result, then reason again. This is more robust than hardcoding tool order — agents adapt to what they find |
| `AgentState` uses `Annotated[list, add_messages]` | LangGraph appends messages rather than replacing the list on each node update. Without this, each node would wipe the previous node's messages |
| Lazy-loaded agent singletons (`_analyst_agent = None`) | Agents hold loaded models. Creating them at import time would slow every module import. Lazy init means the first call pays the cost, subsequent calls reuse the instance |
| `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` | Windows cmd/PowerShell uses cp1252 encoding. Claude's output contains emojis and special chars. Without this, printing crashes with UnicodeEncodeError |

**Issues & Fixes:**

| Issue | Cause | Fix |
|---|---|---|
| Claude output crashes Windows terminal | cp1252 codec can't encode emoji/special chars | Added `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` in `graph.py __main__` |

---

### Phase 7 — Memory Layer

**Short-term memory (LangGraph MemorySaver):**

| What | How | Why |
|---|---|---|
| Stores full `AgentState` after each `graph.invoke()` call | LangGraph checkpointer serialises state to an in-memory dict, keyed by `thread_id` | Without this, calling the graph twice loses all context — the agent can't answer "do the same for SEG002" because it forgot what "the same" refers to |
| `thread_id` is a UUID generated per session | `str(uuid.uuid4())` | Each user conversation gets an isolated memory slot — sessions don't contaminate each other |
| `MemorySaver` is a singleton shared across all sessions | Module-level `_checkpointer = MemorySaver()` | All threads share one store. In production this would be `PostgresSaver` (Week 5 upgrade path documented in comments) |

**Long-term memory (MongoDB Atlas):**

| Collection | What It Stores | When Written |
|---|---|---|
| `sessions` | Full conversation turn: query + all agent outputs + route + timestamp | After every `graph.run()` call, automatically |
| `campaigns` | Complete generated campaigns with title, segment, strategy, copy, tags | Explicit `save_campaign()` call (e.g. from strategist/content writer) |
| `insights` | Key facts extracted during analysis (churn rate, CLV, KPIs by segment) | Explicit `save_insight()` call from analyst |

**Issues & Fixes:**

| Issue | Cause | Fix |
|---|---|---|
| `pymongo` SSL `TLSV1_ALERT_INTERNAL_ERROR` | pymongo 3.11 had TLS compatibility issues with newer Atlas clusters | Upgraded to `pymongo>=4.8` in `pyproject.toml` |
| SSL handshake still failing after upgrade | Missing CA bundle — Python's `ssl` module needs root certificates | Added `tlsCAFile=certifi.where()` to `MongoClient()` call |
| Atlas still rejecting connection | MongoDB Atlas M0 cluster had no IP whitelist entry | Added `0.0.0.0/0` to Network Access in Atlas dashboard |
| Connection now works | All three fixes combined | Long-term memory fully operational. Sessions, insights writing/reading confirmed |

---

### Phase 8 — MCP Server

**What is MCP?**
Model Context Protocol is an open standard by Anthropic. It lets any MCP-compatible client (Claude Desktop, LangGraph agents, external services) call tools via a standardised JSON-RPC interface over stdio.

**Tools exposed:**

| Tool | What It Does |
|---|---|
| `rag_search` | Queries the full RAG pipeline — hybrid search + compression + formatted results |
| `calculate_metric` | Computes marketing KPIs: ROAS, CLV, CAC, churn rate, conversion rate, email ROI |
| `get_segment_profile` | Returns the full profile JSON for SEG001-SEG005 |
| `run_agent` | Runs the complete multi-agent pipeline and returns structured output |

**To connect Claude Desktop:**
Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "marketmind": {
      "command": "python",
      "args": ["-m", "marketmind.mcp_server"],
      "cwd": "C:/Users/PALLA/OneDrive/Desktop/MarketMind AI"
    }
  }
}
```

---

### Phase 9 — FastAPI Backend + Streamlit UI

**FastAPI endpoints:**

| Endpoint | Method | What It Does |
|---|---|---|
| `/health` | GET | Returns API status + MongoDB connectivity status |
| `/query` | POST | Runs agent pipeline, returns `{thread_id, task_route, analyst_output, strategist_output, content_output}` |
| `/sessions/{thread_id}` | GET | Returns message history for a given conversation thread |
| `/docs` | GET | Auto-generated Swagger UI (free from FastAPI) |

**Why FastAPI over Flask:**

| FastAPI | Flask |
|---|---|
| Async-first — doesn't block while waiting for LLM API responses | Synchronous by default |
| Auto-generates `/docs` Swagger UI from type annotations | No built-in API docs |
| Pydantic request/response models — bad inputs rejected before hitting LLM | Manual validation needed |
| Faster (ASGI vs WSGI) | Slower for concurrent requests |

**Streamlit UI features:**

| Feature | How It Works |
|---|---|
| Chat history | `st.session_state.messages` persists across reruns — Streamlit re-runs the whole script on every interaction |
| Session continuity | `thread_id` stored in `st.session_state` — passed to every `/query` call so the agent remembers earlier messages |
| Structured output | Analyst / Strategy / Copy outputs rendered in separate `st.expander` sections, not one flat wall of text |
| API health check | Button calls `/health` and displays MongoDB connectivity status in real time |
| Example queries | Sidebar buttons pre-fill the chat input with representative queries for each route type |
| Graceful error handling | If FastAPI is not running, shows the exact command to start it |

**To run the full stack:**
```bash
# Terminal 1
.venv/Scripts/uvicorn marketmind.api.main:app --reload --port 8000

# Terminal 2
.venv/Scripts/streamlit run ui/app.py
```

---

### Phase 10 — Docker + CI/CD

**Docker architecture:**

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the FastAPI container. python:3.11-slim base, installs all deps, pre-downloads the sentence-transformers model so startup is instant |
| `Dockerfile.streamlit` | Lighter container for Streamlit UI — same base, no ML model download |
| `docker-compose.yml` | Runs both containers locally with one command (`docker compose up`). UI talks to API via Docker's internal network at `http://api:8000` |
| `.dockerignore` | Excludes `.env`, `data/`, `.venv/` from the image — secrets and 40MB of CSV files stay off the container |
| `fly.toml` | Fly.io deployment config: Singapore region (closest to India), 512MB RAM, scale-to-zero when idle |

**Why bake the sentence-transformers model into the image:**
If you don't, the first request triggers a 90MB download inside the container, causing a 60-second timeout. Baking it in at build time means any request starts instantly, every time.

**Why Docker layer ordering matters (for interviews):**
```
COPY pyproject.toml .          # Layer 1 — only changes when you add a dependency
COPY src/ src/                 # Layer 2
RUN pip install -e .           # Layer 3 — cached if Layer 1 didn't change
COPY ui/ ui/                   # Layer 4 — code changes only rebuild from here
```
If you copy all files first then install, every code change triggers a full `pip install`. The order above means `pip install` is only re-run when `pyproject.toml` changes.

**GitHub Actions CI/CD pipeline:**

| Job | Trigger | Steps |
|---|---|---|
| `test` | Every push to any branch | Install package, run import smoke tests on all 4 main modules |
| `deploy` | Push to `main` only, only if test passes | Install flyctl, run `flyctl deploy --remote-only` using `FLY_API_TOKEN` secret |

**"Remote only" deploy** means the Docker image is built on Fly.io's servers, not your laptop. This makes CI fast and avoids pushing a 2GB image over your home internet.

**Issues & Fixes:**

| Issue | Cause | Fix |
|---|---|---|
| flyctl not in PATH after winget install | Terminal needs restart to pick up new PATH entries | Close and reopen terminal after `winget install` |

---

### Phase 10 — Current File Structure

```
MarketMind AI/
├── src/marketmind/
│   ├── config.py                  Central settings (reads .env)
│   ├── mcp_server.py              MCP server (4 tools for Claude Desktop / agents)
│   ├── agents/
│   │   ├── state.py               AgentState TypedDict (shared across all nodes)
│   │   ├── tools.py               @tool functions: rag_search, calculate_metric, get_segment_profile
│   │   ├── analyst.py             Analyst agent node (ReAct, claude-sonnet-4-6)
│   │   ├── strategist.py          Strategist agent node (claude-sonnet-4-6 / GPT-4o placeholder)
│   │   ├── content_writer.py      Content Writer node (claude-sonnet-4-6 / Mistral placeholder)
│   │   └── graph.py               LangGraph StateGraph, orchestrator, routing, run()
│   ├── api/
│   │   └── main.py                FastAPI: /query, /health, /sessions/{thread_id}
│   ├── memory/
│   │   ├── short_term.py          MemorySaver checkpointer (per-session, in-memory)
│   │   └── long_term.py           MongoDB Atlas (sessions, campaigns, insights)
│   └── rag/
│       ├── loaders.py             CSV / JSON / PDF -> Documents
│       ├── chunker.py             Split large docs with overlap
│       ├── embedder.py            Text -> 384-dim vectors (local)
│       ├── ingestion.py           Full ingest pipeline (run once)
│       ├── retriever.py           Vector + BM25 + RRF fusion
│       ├── compression.py         Filter noise by cosine similarity
│       ├── pipeline.py            Clean query() interface
│       ├── prompt.py              4-layer system prompt architecture
│       └── chain.py               retrieve + prompt + Claude = answer
├── data/
│   ├── prepare_data.py            Download + generate all raw data
│   └── raw/
│       ├── transactions/online_retail.csv
│       ├── campaigns/bank_marketing.csv
│       ├── customers/customer_segments.csv
│       ├── products/product_catalog.json
│       └── reports/q1_marketing_report.pdf
├── ui/app.py                      Streamlit chat UI (calls FastAPI)
├── tests/                         Empty — Week 6
├── Dockerfile                     FastAPI container image
├── Dockerfile.streamlit           Streamlit container image
├── docker-compose.yml             Run full stack locally with one command
├── .dockerignore                  Exclude .env, data/, .venv/ from images
├── fly.toml                       Fly.io deployment config (Singapore, 512MB)
├── .github/workflows/deploy.yml   CI/CD: test + deploy on push to main
├── .env                           Real secrets (never in git)
├── .env.example                   Template (committed to git)
├── .gitignore
└── pyproject.toml                 Dependencies + build config
```

---

## 4. Skills Demonstrated So Far

| Skill | Where | Job Role It Targets |
|---|---|---|
| Python engineering-grade project structure | `pyproject.toml`, `src/` layout, `__init__.py` | AI Engineer |
| Vector databases (Pinecone) | `ingestion.py`, `retriever.py` | AI Engineer, Context Engineer |
| Embedding models (sentence-transformers) | `embedder.py` | AI Engineer, Context Engineer |
| RAG pipeline design | All of `rag/` | Context Engineer |
| Hybrid search (BM25 + vector) | `retriever.py` | Context Engineer |
| Context compression | `compression.py` | Context Engineer |
| Prompt architecture (4-layer system prompt) | `prompt.py` | Context Engineer |
| Multi-format data ingestion (CSV, JSON, PDF) | `loaders.py` | AI Engineer |
| LangChain / LCEL | `chain.py` | AI Engineer, Agent Architect |
| Claude API via `langchain-anthropic` | `chain.py` | AI Engineer |
| LangGraph StateGraph (nodes, edges, conditional routing) | `agents/graph.py` | Agent Architect |
| ReAct agent pattern (`create_react_agent`) | `analyst.py`, `strategist.py`, `content_writer.py` | Agent Architect |
| Multi-agent orchestration with intent routing | `agents/graph.py` (orchestrator node) | Agent Architect |
| @tool decorated LangChain tools | `agents/tools.py` | Agent Architect, AI Engineer |
| Short-term memory (MemorySaver checkpointer, thread_id) | `memory/short_term.py` | Agent Architect, Context Engineer |
| Long-term memory (MongoDB Atlas, multi-collection design) | `memory/long_term.py` | AI Engineer, Context Engineer |
| MCP server (Model Context Protocol) | `mcp_server.py` | AI Engineer, Agent Architect |
| FastAPI (Pydantic models, async, OpenAPI docs) | `api/main.py` | AI Engineer |
| Streamlit chat UI (session state management) | `ui/app.py` | AI Engineer |
| Multi-LLM architecture (different models per agent role) | `graph.py`, `content_writer.py`, `strategist.py` | Agent Architect |
| Docker containerisation (multi-stage, layer caching, slim images) | `Dockerfile`, `Dockerfile.streamlit` | AI Engineer |
| Docker Compose (multi-service networking, health checks, depends_on) | `docker-compose.yml` | AI Engineer |
| CI/CD with GitHub Actions (test job + deploy job, secrets) | `.github/workflows/deploy.yml` | AI Engineer |
| Cloud deployment (Fly.io, scale-to-zero, environment secrets) | `fly.toml` | AI Engineer |

---

## 5. What's Next

| Week | Focus | Key Deliverables | Status |
|---|---|---|---|
| Week 1-2 | RAG Foundation | Data prep, ingestion, hybrid retrieval, LLM chain | Done |
| Week 3-4 | Multi-Agent System | LangGraph graph, 4 agents, tools, short-term + long-term memory, MCP server | Done |
| Week 5 | Engineering Layer | FastAPI backend, Streamlit UI, Docker, CI/CD, Fly.io config | Done |
| Week 5 (deploy) | Live Deployment | Push repo to GitHub, deploy to Fly.io, get public URL | In Progress |
| Week 6 | Observability | LangSmith tracing, RAGAs evaluation, cost tracking | Upcoming |
| Week 7 | Fine-Tuning | QLoRA on Mistral 7B, swap content writer agent | Upcoming |
| Week 8 | Portfolio Polish | README, Loom demo, live deployment | Upcoming |

---

## 6. Key Concepts to Remember

| Concept | One-Line Explanation |
|---|---|
| `src/` layout | Prevents Python from importing local uninstalled code by accident |
| `pyproject.toml` | Modern single-file project config — replaces `setup.py` + `requirements.txt` |
| Virtual environment | Isolated Python sandbox per project — packages installed here don't affect system Python |
| Document (LangChain) | Standard unit: `page_content` (text) + `metadata` (dict of source info) |
| Chunking with overlap | Split long text into pieces with shared edges so sentences at boundaries aren't cut off |
| Embedding | Text converted to a list of numbers (vector). Similar text = nearby vectors |
| Cosine similarity | Distance metric between two vectors. 1.0 = identical, 0 = unrelated, -1 = opposite |
| BM25 | Keyword-based ranking algorithm. Scores how well a document matches query terms |
| RRF | Reciprocal Rank Fusion. Combines ranked lists from different retrievers without needing common score scales |
| Upsert | Insert + Update. If vector ID exists, overwrite. If not, insert. Safe to run multiple times |
| Namespace (Pinecone) | Partition within an index. Like a folder — can query or clear one namespace without touching others |
| Guardrails (prompt) | Explicit rules in the system prompt that prevent the LLM from hallucinating or going off-topic |
| LangGraph StateGraph | A directed graph where each node is a function that reads and updates shared state. Edges define control flow |
| AgentState | The shared data structure passed between all nodes. Each node reads from it and writes updates back |
| add_messages | A LangGraph reducer that appends to the messages list instead of overwriting it on each node update |
| Conditional edges | After a node runs, a router function inspects state and returns the name of the next node to run |
| ReAct pattern | Reason -> Act -> Observe loop. Agent decides which tool to call, calls it, reads the result, then decides what to do next |
| @tool decorator | Wraps a Python function so LangGraph/LangChain agents can discover and call it. Docstring becomes the tool's description |
| thread_id | A UUID that identifies a conversation session. LangGraph uses it as the key to load/save state from the checkpointer |
| MemorySaver | In-memory LangGraph checkpointer. Lost on restart. For development only — replaced with PostgresSaver in production |
| MongoDB document model | Schema-free storage. Each document is a dict — fields can vary between documents. No migrations needed for new fields |
| MCP (Model Context Protocol) | Anthropic open standard for exposing AI tools via JSON-RPC over stdio. Any MCP client can call any MCP server's tools |
| FastAPI Pydantic models | Request/response classes that validate input types before any code runs. Bad data is rejected with 422 errors automatically |
| Streamlit session_state | A dict that persists across reruns. Streamlit re-runs the entire script on every user interaction — session_state is how you keep data |
| Multi-LLM architecture | Different agents use different models for their role. Orchestrator uses cheap/fast Haiku; specialists use capable Sonnet |
| Docker layer caching | Each `RUN`/`COPY` line is a cached layer. Order deps before code so pip install is only re-run when pyproject.toml changes, not on every code edit |
| docker-compose service networking | Services talk to each other by service name (e.g. `http://api:8000`), not localhost. Docker creates a shared virtual network automatically |
| `depends_on` with `condition: service_healthy` | Streamlit waits for FastAPI's health check to pass before starting — prevents startup race conditions |
| `scale-to-zero` (Fly.io) | Container shuts down when idle and wakes on first request. Saves free tier quota, costs nothing when nobody is using it |
| CI/CD secrets | API keys are stored as encrypted repository secrets in GitHub Settings, injected as env vars at deploy time — never in code or config files |
| `--remote-only` deploy | Docker image is built on Fly.io's servers, not locally. Avoids uploading a 2GB image over home internet |
