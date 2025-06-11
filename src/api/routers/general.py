import datetime
import requests

from fastapi import APIRouter, Request
from qdrant_client import models

from src.infra import qdrant_client, model_bm42, model_dense
from src.utils.vector_search import embedding_text


router = APIRouter()

#######################################
########### Example ###################
#######################################

@router.post("/search_top_nearestQ")
async def query(request: Request):
    request = await request.json()
    query_str = request['query']
    collection_name = "ads_nearestQ_dr_bm42"  
    threshold = 0
    limit = 20
    
    # Hybrid search
    sparse_embedding = list(model_bm42.query_embed(query_str))[0]
    
    topk = qdrant_client.query_points(
        collection_name = collection_name, 
        prefetch = [
            models.Prefetch(query = sparse_embedding.as_object(), using = "text-sparse", limit = 20),
            models.Prefetch(query = embedding_text(query_str, model_dense), using = "text-dense", limit = 20)
        ],
        query = models.FusionQuery(fusion=models.Fusion.RRF), # combine score 
        limit = limit
    )
        
    list_nearest_question = []
    for index, point in enumerate(topk.points):
        if point.score >= threshold:
            near_question = point.payload["question"]
            list_nearest_question.append(near_question)

    return list_nearest_question

#######################################
############## VAI ####################
#######################################

@router.post("/search_vai")
async def query(request: Request):
    request = await request.json()
    query_str = request['query']
    
    # Input call VAI
    url = "http://10.207.163.17:8082/v1/chat-messages"
    api_key = "Bearer app-cvCggurQPKrv0HGWuu8QD78t"
    headers = {
        "authorization": api_key,
    }
    body = {
        "inputs": {},
        "query": query_str,
        "response_mode": "blocking",
        "conversation_id": "",
        "user": "ADS"
    }
    
    start_time = datetime.datetime.now()
    try:
        request = requests.post(url=url, headers=headers, json=body)
        sys_ans = request.json()["answer"]
        sys_status = 200
        sys_eval = "NA"
    except Exception as e:
        print(e)
        sys_ans = ""
        sys_status = 500
        sys_eval = "NA"
    end_time = datetime.datetime.now()
    response_time = end_time - start_time
    response_time = round(response_time.total_seconds()*1000, 3)
    
    agent_response = {
        "sys_ans": sys_ans,
        "sys_name": "VAI",
        "sys_response_time": response_time,
        "sys_eval": sys_eval,
        "sys_status": sys_status,
    }
    
    return agent_response