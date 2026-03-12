"""
Text chunker: splits large Documents into smaller overlapping chunks.

WHY chunking is necessary:
  - LLMs have a context window limit (e.g. 8K tokens)
  - Embedding models have their own limit (usually 512 tokens)
  - Retrieving a 10-page PDF as one block means most of it is irrelevant noise
  - Small, focused chunks = better retrieval precision

HOW overlap works:
  Given chunk_size=500 and chunk_overlap=50:
  [-------- chunk 1 (500 chars) --------]
                              [-------- chunk 2 (500 chars) --------]
  The 50-char overlap ensures a sentence split at the boundary of chunk 1
  still appears in full at the start of chunk 2. No context is lost.

STRATEGY:
  - Structured documents (segments, products, campaign summaries) are already
    small — they pass through unchanged if under chunk_size.
  - Large documents (PDF pages, long retail summaries) get split.
  - We use RecursiveCharacterTextSplitter which tries to split at paragraph
    boundaries first, then sentences, then words — preserving readability.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Split documents into chunks, preserving all original metadata.
    Documents already smaller than chunk_size pass through untouched.

    Args:
        docs:          List of Documents from the loaders.
        chunk_size:    Maximum characters per chunk (not tokens — we keep it
                       conservative so it stays under most embedding model limits).
        chunk_overlap: Characters shared between consecutive chunks.

    Returns:
        List of chunked Documents, each with original metadata preserved plus
        a 'chunk_index' field added.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try splitting at these boundaries in order (most to least preferred)
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunked = []
    for doc in docs:
        splits = splitter.split_documents([doc])
        for i, split in enumerate(splits):
            split.metadata["chunk_index"] = i
            split.metadata["total_chunks"] = len(splits)
            chunked.append(split)

    return chunked


def preview_chunks(chunks: List[Document], n: int = 3) -> None:
    """Print the first n chunks for inspection during development."""
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Showing first {n}:\n")
    for chunk in chunks[:n]:
        print(f"  [type={chunk.metadata.get('type')} | "
              f"chunk {chunk.metadata.get('chunk_index', 0)+1}/"
              f"{chunk.metadata.get('total_chunks', 1)}]")
        print(f"  {chunk.page_content[:120].replace(chr(10), ' ')}...")
        print()
