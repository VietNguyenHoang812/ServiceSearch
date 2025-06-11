
from fastapi import APIRouter, Request

from src.domain.document import (
    dense_search_vt,
    hybrid_search,
)
from src.utils.graph_search import get_doc_node, get_father_node, get_children_nodes, get_node, concat_contents, create_citation
from src.utils.vector_search import rerank


router = APIRouter()

dense_collection_name = 'duongbt6_dense_bge_prod'
sparse_collection_name = 'duongbt6_hybrid_bge_prod'

@router.post("/vo_tuyen/search")
async def query(request: Request):
    request = await request.json()
    query = request['query']
    print(query)
    
    # Dense search
    dense_points = dense_search_vt(query_str=query, collection_name=dense_collection_name, threshold=0.4, limit=7)
        
    # Hybrid search
    hybrid_points = hybrid_search(query_str=query, collection_name=sparse_collection_name, threshold=0.45, limit=5)
    
    # Rerank    
    rerank_limit = 5
    rerank_threshold = 0.1
    temp_points = hybrid_points + dense_points

    concat_points, corpus = [], []
    for hit in temp_points:
        try:
            point = {
                "chunk": hit["chunk"],
                "id": hit["id"],
                "citation": hit["citation"],
                "score": 0,
            }
        except:
            point = {
                "chunk": hit["chunk"],
                # "citation": f'{hit["doc_code"]} - {hit["name"]}' if hit["doc_code"] is not None else hit["name"],
                "citation": hit["citation"],
                "score": 0,
            }
        if point in concat_points:
            continue
        concat_points.append(point)
        corpus.append(hit["chunk"])    

    ## Calculate rerank score
    print("=========RERANK =========")
    rerank_results = rerank(query, corpus)
    topk = rerank_results[:rerank_limit]
    
    vector_points = []
    for rerank_result in topk:
        idx = rerank_result["index"]
        rerank_score = rerank_result["relevance_score"]
        if rerank_score < rerank_threshold:
            continue
        concat_points[idx]["score"] = rerank_score
        vector_points.append(concat_points[idx])
    
    print("=========== VECTOR POINTS ===========")
    print(vector_points)
    final_points = []
    for point in vector_points:
        try:
            node_id = point["id"]
            print("Node ID: ", node_id)
            print("Chunk  : ", point["chunk"])
            if "C0" in node_id and len(node_id.split("_")) == 3:
                print("C0")
                father_node = get_father_node(node_id)
                children_node = get_node(node_id)
            elif "C" in node_id:
                print("C")
                father_node = get_father_node(node_id)
                # print(father_node)
                father_node_id = father_node[0]["id"]
                children_node = get_children_nodes(father_node_id)
            elif "H" in node_id:
                print("H")
                father_node = get_node(node_id)
                children_node = get_children_nodes(node_id)

            list_node =  father_node + children_node
            print("List nodes: ", list_node)
            content = concat_contents(list_node, node_id)
            print("Content: ", content)
            try:
                citation = create_citation(node_id)
            except Exception as e:
                print("ERROR!!!! ", e)
                print(point)
                citation = ""
        except Exception as e:
            print("BIGG ERRRROR: ", e)
            content = point["chunk"]
            citation = point["citation"]
        
        final_point = {
            "chunk": content,
            "citation": citation,
        }
        if final_point in final_points:
            continue
        final_points.append(final_point)
    
    print("=========== FINAL POINTS ===========")
    print(final_points)
    
    return final_points