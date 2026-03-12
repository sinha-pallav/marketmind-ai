"""
Ingestion pipeline: orchestrates load -> chunk -> embed -> store in Pinecone.

PINECONE CONCEPTS:
  - Index:     A named vector database. Think of it like a table in SQL.
  - Vector:    A stored item = {id, values (the embedding), metadata}.
  - Upsert:    Insert or update vectors. Idempotent — safe to run multiple times.
  - Namespace: A partition within an index (like a folder). We use one per
               data source so we can filter or clear them independently.

FLOW:
  1. Load all documents from raw data files
  2. Chunk large documents into smaller pieces
  3. Embed each chunk (text -> 384-dim vector)
  4. Upsert vectors to Pinecone in batches of 100
"""

import hashlib
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

from marketmind.rag.chunker import chunk_documents, preview_chunks
from marketmind.rag.embedder import EMBEDDING_DIMENSION, embed_texts
from marketmind.rag.loaders import load_all

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "marketmind")
NAMESPACE = "marketing-knowledge-base"


# ---------------------------------------------------------------------------
# Pinecone helpers
# ---------------------------------------------------------------------------

def get_pinecone_index():
    """Connect to Pinecone and return the index object."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"  Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            # Serverless is Pinecone's free tier — no dedicated pods needed
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("  Index created.")
    else:
        print(f"  Index '{INDEX_NAME}' already exists.")

    return pc.Index(INDEX_NAME)


def _doc_id(doc: Document, chunk_index: int) -> str:
    """
    Generate a stable, unique ID for each chunk.
    Using a hash of source + chunk_index means re-running ingestion
    produces the same IDs -> Pinecone upsert safely overwrites old vectors.
    """
    raw = f"{doc.metadata.get('source', '')}::{doc.metadata.get('type', '')}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def upsert_to_pinecone(
    index,
    chunks: List[Document],
    batch_size: int = 100,
) -> None:
    """
    Embed all chunks and upsert to Pinecone in batches.

    Args:
        index:      Pinecone index object.
        chunks:     List of chunked Documents.
        batch_size: Pinecone recommends batches of 100 for best throughput.
    """
    total = len(chunks)
    print(f"  Upserting {total} chunks to Pinecone in batches of {batch_size}...")

    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start: batch_start + batch_size]
        texts = [doc.page_content for doc in batch]

        # Embed this batch
        vectors_values = embed_texts(texts)

        # Build the list of (id, values, metadata) tuples Pinecone expects
        vectors = []
        for doc, values in zip(batch, vectors_values):
            chunk_idx = doc.metadata.get("chunk_index", 0)
            vec_id = _doc_id(doc, chunk_idx)

            # Pinecone metadata must be flat key-value (no nested dicts)
            # We store the original text so we can retrieve it later
            metadata = {k: str(v) for k, v in doc.metadata.items()}
            metadata["text"] = doc.page_content[:1000]  # Pinecone metadata limit

            vectors.append({
                "id": vec_id,
                "values": values,
                "metadata": metadata,
            })

        index.upsert(vectors=vectors, namespace=NAMESPACE)
        done = min(batch_start + batch_size, total)
        print(f"    Upserted {done}/{total}")

    print("  Done upserting.")


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def run_ingestion(data_dir: Path) -> None:
    """
    Full ingestion pipeline: load -> chunk -> embed -> store.

    Args:
        data_dir: Path to the project's data/ directory.
    """
    print("\n=== Step 1: Loading documents ===")
    docs = load_all(data_dir)
    print(f"  Total raw documents loaded: {len(docs)}")

    print("\n=== Step 2: Chunking ===")
    chunks = chunk_documents(docs, chunk_size=512, chunk_overlap=50)
    print(f"  Total chunks after splitting: {len(chunks)}")
    preview_chunks(chunks, n=2)

    print("\n=== Step 3: Connecting to Pinecone ===")
    index = get_pinecone_index()

    print("\n=== Step 4: Embedding & Upserting ===")
    upsert_to_pinecone(index, chunks)

    # Print index stats so you can verify
    stats = index.describe_index_stats()
    print(f"\n=== Done! Pinecone index stats ===")
    print(f"  Total vectors: {stats.total_vector_count}")
    print(f"  Namespace '{NAMESPACE}': "
          f"{stats.namespaces.get(NAMESPACE, {})}")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    data_dir = project_root / "data"
    run_ingestion(data_dir)
