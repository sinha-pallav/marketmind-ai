"""
Embedding model wrapper.

WHAT embeddings are:
  An embedding converts text into a list of numbers (a vector). Texts with
  similar meaning end up as vectors that are "close" to each other in vector
  space. This is what makes semantic search possible — you embed the user's
  query and find the chunks whose vectors are closest.

  Example:
    "best performing product"   -> [0.12, -0.45, 0.87, ...]  (384 numbers)
    "top revenue items"         -> [0.11, -0.43, 0.85, ...]  (very close!)
    "quarterly budget forecast" -> [-0.3,  0.21, 0.02, ...]  (far away)

MODEL CHOICE:
  We use "all-MiniLM-L6-v2" from sentence-transformers:
    - Dimension: 384 (compact, fast)
    - Runs locally — no API key needed for embeddings
    - Good balance of speed vs. quality for retrieval tasks
    - Free, no rate limits

  In Week 2 we'll add OpenAI's text-embedding-3-small for comparison.
"""

from typing import List

from sentence_transformers import SentenceTransformer

# Load once at module level — avoids reloading the model on every call
# The model is ~90 MB and is downloaded to ~/.cache/huggingface on first run
_model: SentenceTransformer | None = None
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"  Loading embedding model '{MODEL_NAME}'...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """
    Embed a list of strings into vectors.

    Args:
        texts:      List of text strings to embed.
        batch_size: How many texts to process at once. Larger = faster but
                    uses more RAM. 64 is safe for most machines.

    Returns:
        List of vectors, one per input text. Each vector has 384 floats.
    """
    model = get_model()
    # encode() returns a numpy array — we convert to plain Python lists
    # so they can be serialised to JSON and sent to Pinecone
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.tolist()
