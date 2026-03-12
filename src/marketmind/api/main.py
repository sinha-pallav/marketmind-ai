"""FastAPI application entry point."""

from fastapi import FastAPI

app = FastAPI(title="MarketMind AI", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}
