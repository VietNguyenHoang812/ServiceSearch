
from fastapi import APIRouter, Request

from src.domain.netbi import (
    dense_search_netbi,
    dense_search_trick,
)
from src.utils.vector_search import rerank


router = APIRouter()

# Search for definition of KPI
@router.post("/search")
async def query(request: Request):
    request = await request.json()
    query = request['query']
    
    # Search in NetBI PL02 for definition
    netbi_hit = dense_search_netbi(query_str=query, collection_name="netbi_dense_short_v5", threshold=0.35, limit=5)
        
    # Rerank
    rerank_limit = 5
    rerank_threshold = 0.2
    concat_points = netbi_hit
    
    corpus = []
    for hit in concat_points:
        chunk = hit["name"]
        corpus.append(chunk)

    if len(corpus) == 0:
        return []
    
    print("=========RERANK IN NETBI===============")
    rerank_results = rerank(query, corpus)
    topk = rerank_results[:rerank_limit]
    
    final_points = []
    for rerank_result in topk:
        idx = rerank_result["index"]
        rerank_score = rerank_result["relevance_score"]
        if rerank_score < rerank_threshold:
            continue
        concat_points[idx]["score"] = rerank_score
        final_points.append(concat_points[idx])
    
    return final_points

# Currently, this function is not used: Search for list of serving KPI
@router.post("/netbi/search/trick")
async def query(request: Request):
    request = await request.json()
    query = request['query']

    # Dense search
    netbi_hit = dense_search_trick(query_str=query, collection_name="trick_netbi_name_dense", threshold=0, limit=5)
    
    # Rerank
    rerank_limit = 1
    rerank_threshold = 0.9
    
    corpus = []
    for hit in netbi_hit:
        chunk = hit["name"]
        corpus.append(chunk)

    print("=========RERANK IN NETBI TRICK=========")
    rerank_results = rerank(query, corpus)
    topk = rerank_results[:rerank_limit]
    
    final_points = []
    for rerank_result in topk:
        idx = rerank_result["index"]
        rerank_score = rerank_result["relevance_score"]
        if rerank_score < rerank_threshold:
            continue
        netbi_hit[idx]["score"] = rerank_score
        final_points.append(netbi_hit[idx])
    
    return final_points