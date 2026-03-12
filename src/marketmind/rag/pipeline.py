"""
RAG pipeline: the single entry point agents use to answer questions.

This combines retriever + compression into one clean interface:

    pipeline = RAGPipeline.build(data_dir)
    answer_context = pipeline.query("which segments have highest churn risk?")

The returned context string is ready to be injected into an LLM prompt.
"""

from pathlib import Path
from typing import Optional

from marketmind.rag.compression import compress, format_context
from marketmind.rag.retriever import HybridRetriever


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever):
        self._retriever = retriever

    @classmethod
    def build(cls, data_dir: Path) -> "RAGPipeline":
        retriever = HybridRetriever.build(data_dir)
        return cls(retriever)

    def query(
        self,
        question: str,
        top_k: int = 5,
        filter_type: Optional[str] = None,
        similarity_threshold: float = 0.25,
    ) -> str:
        """
        Retrieve relevant context for a question.

        Args:
            question:             The user's natural language question.
            top_k:                Max chunks to return.
            filter_type:          Optional: restrict to a document type
                                  e.g. "customer_segment", "product_catalog",
                                  "product_performance", "marketing_report".
            similarity_threshold: Drop chunks below this relevance score.

        Returns:
            Formatted context string ready to inject into an LLM prompt.
        """
        docs = self._retriever.retrieve(question, top_k=top_k, filter_type=filter_type)
        compressed = compress(question, docs, similarity_threshold=similarity_threshold)
        return format_context(compressed)

    def query_with_docs(self, question: str, top_k: int = 5):
        """Same as query() but returns raw Document objects instead of a string."""
        docs = self._retriever.retrieve(question, top_k=top_k)
        return compress(question, docs)
