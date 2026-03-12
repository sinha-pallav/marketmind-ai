"""
Context compression: filters retrieved chunks to remove noise.

THE PROBLEM it solves:
  After hybrid retrieval you have 5-10 chunks. But some chunks are only
  partially relevant — they matched the query but contain a lot of extra text
  the LLM doesn't need. Feeding noisy context to the LLM leads to:
    - Worse answers (distracted by irrelevant text)
    - Higher token costs (paying for context that doesn't help)

TWO APPROACHES:
  1. Embeddings-based filter (used here for Week 1):
     - Compute cosine similarity between query embedding and each chunk
     - Drop chunks below a similarity threshold (default: 0.3)
     - Fast, free, no LLM API call needed
     - Limitation: operates at chunk level, can't trim within a chunk

  2. LLM-based compression (Week 6, after LangSmith observability is set up):
     - Send each chunk to an LLM with prompt: "Extract only the parts relevant to: {query}"
     - More precise but costs tokens and adds latency
     - Worth it for production; too expensive for early development

We use approach 1 now and will swap to approach 2 in Week 6.
"""

from typing import List

import numpy as np
from langchain_core.documents import Document

from marketmind.rag.embedder import embed_texts


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors. Range: -1 to 1, higher = more similar."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def compress(
    query: str,
    docs: List[Document],
    similarity_threshold: float = 0.25,
) -> List[Document]:
    """
    Filter chunks that are not similar enough to the query.

    Args:
        query:                The user's question.
        docs:                 Retrieved chunks from the hybrid retriever.
        similarity_threshold: Chunks with cosine similarity below this are removed.
                              0.25 is conservative — keeps most chunks.
                              Raise to 0.4+ for stricter filtering.

    Returns:
        Filtered list of documents, each with 'compression_score' in metadata.
    """
    if not docs:
        return docs

    # Embed the query and all chunk texts in one batch for efficiency
    all_texts = [query] + [doc.page_content for doc in docs]
    all_embeddings = embed_texts(all_texts, batch_size=64)

    query_embedding = all_embeddings[0]
    chunk_embeddings = all_embeddings[1:]

    filtered = []
    for doc, chunk_emb in zip(docs, chunk_embeddings):
        score = cosine_similarity(query_embedding, chunk_emb)
        doc.metadata["compression_score"] = round(score, 4)
        if score >= similarity_threshold:
            filtered.append(doc)

    # Sort by compression score so best chunks come first
    filtered.sort(key=lambda d: d.metadata["compression_score"], reverse=True)
    return filtered


def format_context(docs: List[Document]) -> str:
    """
    Format retrieved documents into a single context string for the LLM prompt.
    Each chunk is separated by a divider and labelled with its source type.
    """
    if not docs:
        return "No relevant context found."

    parts = []
    for i, doc in enumerate(docs, start=1):
        source_type = doc.metadata.get("type", "unknown").replace("_", " ").title()
        score = doc.metadata.get("compression_score", "")
        score_str = f" (relevance: {score})" if score else ""
        parts.append(
            f"[Context {i} — {source_type}{score_str}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(parts)
