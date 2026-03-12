# ============================================================
# MarketMind AI — Dockerfile (FastAPI backend)
#
# WHAT THIS DOES:
#   Builds a container image for the FastAPI API server.
#   The image is used by docker-compose locally and by
#   Fly.io in production.
#
# BASE IMAGE CHOICE — python:3.11-slim:
#   "slim" is a minimal Debian image with Python installed.
#   It's ~50MB vs ~900MB for the full image.
#   We only install what we explicitly need — keeps image small
#   and reduces the attack surface (fewer packages = fewer CVEs).
#
# LAYER ORDERING — why dependencies are installed before code:
#   Docker caches each layer. If we copy all files first, then
#   install deps, ANY code change invalidates the deps layer and
#   triggers a full reinstall. By copying pyproject.toml first
#   and installing deps before copying src/, code changes only
#   rebuild the last two layers. Much faster on repeated builds.
#
# SENTENCE-TRANSFORMERS MODEL (pre-baked):
#   The all-MiniLM-L6-v2 model is ~90MB. If we don't pre-download
#   it, the first API request triggers a download inside the container,
#   causing a 30-60 second timeout. Baking it in means the image is
#   larger but startup is instant and deterministic.
# ============================================================

FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# System dependencies needed by some Python packages
# (sentence-transformers needs these for its C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the dependency manifest first (for layer caching)
COPY pyproject.toml .

# Create a minimal src structure so pip can find the package
COPY src/ src/

# Install the package and all dependencies
# --no-cache-dir: don't store pip download cache (saves image space)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

# Pre-download the sentence-transformers model into the image.
# This runs at build time, not at container startup.
# The model is cached in /root/.cache/huggingface/
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Expose the API port
EXPOSE 8000

# Health check — Docker will restart the container if this fails
# --interval: check every 30s
# --timeout: give the app 10s to respond
# --start-period: wait 60s before first check (app needs time to start)
# --retries: fail after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=5)"

# Run the FastAPI app with uvicorn
# --host 0.0.0.0: listen on all interfaces (required in containers)
# --port 8000: match the EXPOSE above
# NO --reload: reload is for development only, wastes resources in production
CMD ["uvicorn", "marketmind.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
