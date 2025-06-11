from typing import List

from src.infra import model_rerank


def embedding_text(input_text: str, model) -> list:
    embedding_result = model.create_embedding(input_text)
    vector_embedding = embedding_result["data"][0]["embedding"]
    
    return vector_embedding

def rerank(query: str, corpus: List[str]):
    rerank_metadata = model_rerank.rerank(corpus, query)
    rerank_results = rerank_metadata["results"]
    
    return rerank_results