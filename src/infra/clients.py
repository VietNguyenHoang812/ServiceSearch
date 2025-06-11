from functools import lru_cache
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from xinference.client import Client as XinferenceClient

from src.core.config import XINFERENCE_CONFIG, QDRANT_CONFIG, NEO4J_CONFIG


@lru_cache()
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=QDRANT_CONFIG["URL"],
    )

@lru_cache()
def get_xinference_client() -> XinferenceClient:
    return XinferenceClient(
        base_url=XINFERENCE_CONFIG["URL"],
    )

@lru_cache()
def get_neo4j_driver() -> GraphDatabase:
    return GraphDatabase.driver(
        uri=NEO4J_CONFIG["URI"],
        auth=(NEO4J_CONFIG["USER"], NEO4J_CONFIG["PASSWORD"]),
    )
