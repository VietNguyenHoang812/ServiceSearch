from functools import lru_cache
from qdrant_client import QdrantClient


@lru_cache()
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        host=...,
        port=...,
        api_key=...
    )

