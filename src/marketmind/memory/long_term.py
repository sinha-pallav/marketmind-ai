"""
Long-term memory via MongoDB.

WHAT GETS STORED HERE:
  Unlike short-term memory (which is per-session), long-term memory persists
  across days and weeks. It stores:

  1. session_log    — every query + response + route taken + timestamp
  2. campaign_store — campaigns created (analysis + strategy + copy), searchable
  3. insights_store — key facts the analyst extracted (segment stats, KPIs)

WHY MONGODB (not SQL)?
  - Marketing outputs are semi-structured: a campaign brief has flexible fields
    (sometimes there's email copy, sometimes just strategy, etc.)
  - MongoDB's document model handles variable structure naturally
  - No schema migrations needed when we add new fields later
  - `pymongo` is simple — no ORM needed for this use case

COLLECTION DESIGN:
  database: marketmind
    ├── sessions      — one document per conversation turn
    ├── campaigns     — one document per generated campaign
    └── insights      — one document per extracted data insight

MONGODB SETUP (MongoDB Atlas free tier — no installation needed):
  1. Go to cloud.mongodb.com → create free account
  2. Create a free M0 cluster (shared, always free)
  3. Database Access → Add user (username + password)
  4. Network Access → Add IP Address → Allow access from anywhere (0.0.0.0/0)
  5. Clusters → Connect → Drivers → copy the connection string
  6. Replace <password> in string with your password
  7. Add to .env: MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/marketmind
"""

import os
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

_client = None
_db = None


def _get_db():
    """
    Lazy connection to MongoDB.
    Connects only when first needed, not at import time.
    Returns None gracefully if MONGODB_URL is not set yet.
    """
    global _client, _db
    if _db is not None:
        return _db

    mongo_url = os.getenv("MONGODB_URL", "")
    if not mongo_url or "user:password" in mongo_url:
        # URL not configured yet — return None so callers can handle gracefully
        return None

    try:
        import certifi
        from pymongo import MongoClient
        _client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
        # Verify connection
        _client.admin.command("ping")
        _db = _client["marketmind"]
        print("  [MongoDB] Connected to long-term memory store.")
        return _db
    except Exception as e:
        print(f"  [MongoDB] Could not connect: {e}. Long-term memory disabled.")
        return None


# ---------------------------------------------------------------------------
# Session logging — every query + response is recorded
# ---------------------------------------------------------------------------

def log_session(
    thread_id: str,
    user_query: str,
    task_route: str,
    analyst_output: str = "",
    strategist_output: str = "",
    content_output: str = "",
) -> Optional[str]:
    """
    Record a complete conversation turn to MongoDB.
    Returns the inserted document ID, or None if MongoDB is not available.
    """
    db = _get_db()
    if db is None:
        return None

    doc = {
        "thread_id": thread_id,
        "timestamp": datetime.now(timezone.utc),
        "user_query": user_query,
        "task_route": task_route,
        "analyst_output": analyst_output,
        "strategist_output": strategist_output,
        "content_output": content_output,
    }

    result = db["sessions"].insert_one(doc)
    return str(result.inserted_id)


# ---------------------------------------------------------------------------
# Campaign store — save and search generated campaigns
# ---------------------------------------------------------------------------

def save_campaign(
    thread_id: str,
    title: str,
    target_segment: str,
    analyst_summary: str,
    strategy: str,
    copy: str,
    tags: list[str] = None,
) -> Optional[str]:
    """
    Save a complete campaign to the campaign store.
    Campaigns can be searched later by segment, title, or tags.
    """
    db = _get_db()
    if db is None:
        return None

    doc = {
        "thread_id": thread_id,
        "created_at": datetime.now(timezone.utc),
        "title": title,
        "target_segment": target_segment,
        "analyst_summary": analyst_summary,
        "strategy": strategy,
        "copy": copy,
        "tags": tags or [],
    }

    result = db["campaigns"].insert_one(doc)
    return str(result.inserted_id)


def get_campaigns(target_segment: str = None, limit: int = 5) -> list:
    """
    Retrieve past campaigns, optionally filtered by segment.
    Ordered by most recent first.
    """
    db = _get_db()
    if db is None:
        return []

    query = {}
    if target_segment:
        query["target_segment"] = {"$regex": target_segment, "$options": "i"}

    campaigns = (
        db["campaigns"]
        .find(query, {"_id": 0})
        .sort("created_at", -1)
        .limit(limit)
    )
    return list(campaigns)


# ---------------------------------------------------------------------------
# Insights store — key facts extracted by the analyst
# ---------------------------------------------------------------------------

def save_insight(
    segment_id: str,
    insight_type: str,
    content: str,
    source_query: str,
) -> Optional[str]:
    """
    Store a key insight extracted during analysis.
    Agents can retrieve past insights rather than re-running full analysis.

    insight_type examples: "churn_risk", "clv", "campaign_performance", "product_revenue"
    """
    db = _get_db()
    if db is None:
        return None

    doc = {
        "segment_id": segment_id,
        "insight_type": insight_type,
        "content": content,
        "source_query": source_query,
        "created_at": datetime.now(timezone.utc),
    }

    result = db["insights"].insert_one(doc)
    return str(result.inserted_id)


def get_recent_insights(segment_id: str = None, limit: int = 10) -> list:
    """Retrieve recent insights, optionally filtered by segment."""
    db = _get_db()
    if db is None:
        return []

    query = {}
    if segment_id:
        query["segment_id"] = segment_id.upper()

    insights = (
        db["insights"]
        .find(query, {"_id": 0})
        .sort("created_at", -1)
        .limit(limit)
    )
    return list(insights)


def is_connected() -> bool:
    """Check if long-term memory is available."""
    return _get_db() is not None
