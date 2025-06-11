from qdrant_client import models

from src.infra import qclient, model_bm42, old_model_dense, model_dense
from utils.vector_search import embedding_text


def dense_search_netbi(query_str: str, collection_name: str, threshold: float=0.35, limit: int=5):
    topk = qclient.query_points(
        collection_name = collection_name, 
        query = embedding_text(query_str, old_model_dense),
        limit = limit
    )
        
    list_point = []
    for index, point in enumerate(topk.points):
        if point.score < threshold:
            continue
        new_point = {
            "chunk": point.payload["chunk"],
            "tt": point.payload["tt"],
            "name": f'{point.payload["doc_code"]} - {point.payload["name"]}',
            "score": point.score
        }
        
        list_point.append(new_point)
    
    return list_point

# Currently, this function is not used
def dense_search_trick(query_str: str, collection_name: str, threshold: float=0.9, limit: int=1):
    topk = qclient.query_points(
        collection_name = collection_name, 
        query = embedding_text(query_str, model_dense), # combine score 
        limit = limit
    )
        
    list_point = []
    for index, point in enumerate(topk.points):
        if point.score < threshold:
            continue
        new_point = {
            "chunk": point.payload["chunk"],
            "name": f'{point.payload["name"]}',
            "score": point.score
        }
        
        list_point.append(new_point)
    
    return list_point

# Currently, this function is not used
def hybrid_search_netbi(query_str: str, collection_name: str, filter: list, threshold: float=0.45, limit: int=5):
    sparse_embedding = list(model_bm42.query_embed(query_str))[0]
    
    topk = qclient.query_points(
        collection_name = collection_name, 
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="field",
                    match=models.MatchAny(any=filter)
                )
            ]
        ),
        prefetch = [
            models.Prefetch(query = sparse_embedding.as_object(), using = "text-sparse", limit = 5),
            models.Prefetch(query = embedding_text(query_str, old_model_dense), using = "text-dense", limit = 5)
        ],
        query = models.FusionQuery(fusion=models.Fusion.RRF), # combine score 
        limit = limit
    )
        
    list_point = []
    for index, point in enumerate(topk.points):
        if point.score >= threshold:
            new_point = {
                "chunk": point.payload["chunk"],
                "name": f'{point.payload["doc_code"]} - {point.payload["name"]}' if point.payload["doc_code"] is not None else point.payload["name"],
                "score": point.score
            }
            list_point.append(new_point)
    
    return list_point