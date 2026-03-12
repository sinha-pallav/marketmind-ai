"""Central configuration — reads from .env file."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Vector DB
    pinecone_api_key: str = ""
    pinecone_index_name: str = "marketmind"

    # Databases
    postgres_url: str = ""
    mongodb_url: str = ""

    # Observability
    langsmith_api_key: str = ""
    langsmith_project: str = "marketmind"

    # App
    environment: str = "development"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
