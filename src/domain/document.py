from qdrant_client import models

from src.infra import qclient, model_bm42, model_dense
from utils.vector_search import embedding_text


def dense_search_vt(query_str: str, collection_name: str, threshold: float=0.4, limit: int=5):
    topk = qclient.query_points(
        collection_name = collection_name, 
        query = embedding_text(query_str, model_dense),
        limit = limit
    )
        
    list_point = []
    for index, point in enumerate(topk.points):
        if point.score < threshold:
            continue
        try:
            new_point = {
                "chunk": point.payload["chunk"],
                "id": point.payload["id"],
                "citation": point.payload["name"],
                "score": point.score
            }
        except:
            new_point = {
                "chunk": point.payload["chunk"],
                # "id": point.payload["id"],
                "citation": point.payload["name"],
                "score": point.score
            }
        
        list_point.append(new_point)
    
    return list_point

def hybrid_search(query_str: str, collection_name: str, threshold: float=0.45, limit: int=5):
    sparse_embedding = list(model_bm42.query_embed(query_str))[0]
    
    topk = qclient.query_points(
        collection_name = collection_name, 
        prefetch = [
            models.Prefetch(query = sparse_embedding.as_object(), using = "text-sparse", limit = 5),
            models.Prefetch(query = embedding_text(query_str, model_dense), using = "text-dense", limit = 5)
        ],
        query = models.FusionQuery(fusion=models.Fusion.RRF), # combine score 
        limit = limit
    )
        
    list_point = []
    for index, point in enumerate(topk.points):
        if point.score >= threshold:
            try:
                new_point = {
                    "chunk": point.payload["chunk"],
                    "id": point.payload["id"],
                    "citation": point.payload["name"],
                    "score": point.score
                }
            except:
                new_point = {
                    "chunk": point.payload["chunk"],
                    # "id": point.payload["id"],
                    "citation": point.payload["name"],
                    "score": point.score
                }
            list_point.append(new_point)
    
    return list_point