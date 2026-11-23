"""Environment configuration shared by setup, ingestion, and retrieval."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


@dataclass(frozen=True)
class Settings:
    pinecone_api_key: str
    openai_api_key: str
    index_name: str
    namespace: str
    cloud: str
    region: str


def load_settings() -> Settings:
    load_dotenv()
    required = ("PINECONE_API_KEY", "OPENAI_API_KEY", "INDEX_NAME")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise ValueError(f"Set these variables in .env: {', '.join(missing)}")
    namespace = os.getenv("PINECONE_NAMESPACE", "coderhouse-demo").strip()
    if not namespace:
        raise ValueError("PINECONE_NAMESPACE must not be empty.")
    return Settings(
        pinecone_api_key=os.environ["PINECONE_API_KEY"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        index_name=os.environ["INDEX_NAME"],
        namespace=namespace,
        cloud=os.getenv("PINECONE_CLOUD", "aws"),
        region=os.getenv("PINECONE_REGION", "us-east-1"),
    )
