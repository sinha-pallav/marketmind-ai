"""
Hybrid retriever: combines vector search (Pinecone) + keyword search (BM25).

WHY HYBRID?
  Vector search alone has a weakness: it finds semantically "similar" chunks
  but can miss exact matches. For example:
    Query: "SEG003 churn risk"
    Vector search might return "High-Value Loyalists" (semantically about segments)
    BM25 will correctly prioritise the chunk that literally contains "SEG003"

  BM25 alone misses synonyms and paraphrasing:
    Query: "which segment is most at-risk of leaving?"
    BM25 won't match "churn" if the user said "leaving"
    Vector search handles this naturally

  Hybrid = best of both worlds.

RECIPROCAL RANK FUSION (RRF):
  Instead of combining raw scores (which have different scales), we combine ranks.
  Formula: score(chunk) = sum(1 / (k + rank_i)) for each retriever i
  k=60 is a standard constant that prevents high ranks from dominating.

  Example with k=60:
    Rank 1 -> 1/(60+1) = 0.0164
    Rank 2 -> 1/(60+2) = 0.0161
    Rank 10 -> 1/(60+10) = 0.0143
  Differences are small — a chunk ranked 1st in both retrievers wins clearly.
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from pinecone import Pinecone
from rank_bm25 import BM25Okapi

from marketmind.rag.chunker import chunk_documents
from marketmind.rag.embedder import embed_texts, get_model
from marketmind.rag.loaders import load_all

load_dotenv()

# RRF constant — standard value, no need to tune
_RRF_K = 60


class HybridRetriever:
    """
    Retrieves relevant chunks using vector search + BM25, fused with RRF.

    Usage:
        retriever = HybridRetriever.build(data_dir)
        results = retriever.retrieve("which segments are high churn risk?")
        for doc in results:
            print(doc.page_content)
            print(doc.metadata)
    """

    def __init__(
        self,
        chunks: List[Document],
        pinecone_index,
        namespace: str,
    ):
        self._chunks = chunks
        self._index = pinecone_index
        self._namespace = namespace

        # Build BM25 index from the same corpus stored in Pinecone
        # Tokenise by splitting on whitespace and lowercasing
        tokenised = [doc.page_content.lower().split() for doc in chunks]
        self._bm25 = BM25Okapi(tokenised)

    @classmethod
    def build(cls, data_dir: Path) -> "HybridRetriever":
        """
        Factory method: loads data, builds BM25 corpus, connects to Pinecone.
        Call this once at startup — not on every query.
        """
        print("Building HybridRetriever...")

        # Load and chunk corpus (same process as ingestion — must be identical)
        print("  Loading corpus for BM25...")
        docs = load_all(data_dir)
        chunks = chunk_documents(docs, chunk_size=512, chunk_overlap=50)
        print(f"  BM25 corpus: {len(chunks)} chunks")

        # Connect to Pinecone
        print("  Connecting to Pinecone...")
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME", "marketmind")
        index = pc.Index(index_name)

        namespace = "marketing-knowledge-base"
        return cls(chunks, index, namespace)

    # ------------------------------------------------------------------
    # Core retrieval methods
    # ------------------------------------------------------------------

    def _vector_search(self, query: str, top_k: int) -> List[Document]:
        """Embed the query and find nearest neighbours in Pinecone."""
        query_vector = embed_texts([query])[0]

        results = self._index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=self._namespace,
            include_metadata=True,
        )

        docs = []
        for match in results.matches:
            text = match.metadata.get("text", "")
            metadata = dict(match.metadata)
            metadata["vector_score"] = match.score
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

    def _bm25_search(self, query: str, top_k: int) -> List[Document]:
        """Find chunks with highest BM25 keyword overlap with the query."""
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        # Get indices of top_k highest scores
        top_indices = np.argsort(scores)[::-1][:top_k]

        docs = []
        for idx in top_indices:
            doc = self._chunks[idx]
            metadata = dict(doc.metadata)
            metadata["bm25_score"] = float(scores[idx])
            docs.append(Document(
                page_content=doc.page_content,
                metadata=metadata,
            ))
        return docs

    def _rrf_fusion(
        self,
        vector_docs: List[Document],
        bm25_docs: List[Document],
        top_k: int,
    ) -> List[Document]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.
        Returns top_k documents sorted by combined RRF score.
        """
        # Use page_content as the unique key for deduplication
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(vector_docs):
            key = doc.page_content[:200]  # first 200 chars as fingerprint
            scores[key] = scores.get(key, 0) + 1 / (_RRF_K + rank + 1)
            doc_map[key] = doc

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content[:200]
            scores[key] = scores.get(key, 0) + 1 / (_RRF_K + rank + 1)
            # Merge metadata if doc already seen from vector search
            if key in doc_map:
                doc_map[key].metadata.update({
                    k: v for k, v in doc.metadata.items()
                    if k not in doc_map[key].metadata
                })
            else:
                doc_map[key] = doc

        # Sort by RRF score descending, return top_k
        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        results = []
        for key in sorted_keys[:top_k]:
            doc = doc_map[key]
            doc.metadata["rrf_score"] = round(scores[key], 6)
            results.append(doc)
        return results

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_type: Optional[str] = None,
    ) -> List[Document]:
        """
        Main retrieval method. Returns top_k most relevant chunks.

        Args:
            query:       Natural language question.
            top_k:       Number of chunks to return after fusion.
            filter_type: Optional metadata filter, e.g. "customer_segment"
                         to restrict results to a specific document type.
        """
        # Fetch more candidates than needed, then fuse down to top_k
        candidates = top_k * 3

        vector_results = self._vector_search(query, candidates)
        bm25_results = self._bm25_search(query, candidates)
        fused = self._rrf_fusion(vector_results, bm25_results, top_k * 2)

        # Optional: filter by document type
        if filter_type:
            fused = [d for d in fused if d.metadata.get("type") == filter_type]

        return fused[:top_k]

    def retrieve_with_scores(self, query: str, top_k: int = 5) -> List[tuple]:
        """Same as retrieve() but also returns RRF scores. Useful for debugging."""
        docs = self.retrieve(query, top_k)
        return [(doc, doc.metadata.get("rrf_score", 0)) for doc in docs]
