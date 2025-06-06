import datetime
import json
import numpy as np
import os
import requests

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastembed import SparseTextEmbedding
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from typing import List
from xinference.client import Client


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CLIENT
qclient = QdrantClient(url="http://10.207.163.17:9005") # Qdrant
xclient = Client("http://10.207.163.17:9004")           # Xinference
# URI = "neo4j://10.207.163.17:8200"
URI = "neo4j://10.207.163.17:8120"
AUTH = ("neo4j", "password_123")

driver = GraphDatabase.driver(URI, auth=AUTH)           # Neo4j
driver.verify_connectivity()
print("Connection established!")

# LOAD MODEL
## Dense model
dense_model_name = "embedding-dr-bge-v2"
model_dense = xclient.get_model(dense_model_name)

old_dense_model_name = "result_embedding_dr"
old_model_dense = xclient.get_model(old_dense_model_name)

## Sparse model
sparse_model_name = "Qdrant/bm42-all-minilm-l6-v2-attentions"
model_bm42 = SparseTextEmbedding(sparse_model_name, cache_dir="/weights", local_files_only=True)

## Rerank model
rerank_model_name = "bge-rerank-v2-m3-0"
model_rerank = xclient.get_model(rerank_model_name)

# sparse_collection_name = "vietnh41_hybrid_bge_test"
# dense_collection_name = "vietnh41_dense_bge_test"

dense_collection_name = 'duongbt6_dense_bge_prod'
sparse_collection_name = 'duongbt6_hybrid_bge_prod'

# UTILS FUNCTIONS
def embedding_text(input_text: str, model) -> list:
    embedding_result = model.create_embedding(input_text)
    vector_embedding = embedding_result["data"][0]["embedding"]
    
    return vector_embedding

def rerank(query: str, corpus: List[str]):
    rerank_metadata = model_rerank.rerank(corpus, query)
    rerank_results = rerank_metadata["results"]
    
    return rerank_results

# Search
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

def create_citation(node_id: str) -> str:
    metadata_node = get_metadata_node(node_id)[0]
    metadata = metadata_node["content"]
    doc_name, code, day_publish, time_publish, author, approver = metadata.split("\n")
    
    code = code.split("Mã hiệu: ")[-1]
    doc_name = doc_name.split(": ")[-1]
    
    citation = f"{code} - {doc_name}"
    
    return citation

###################################
######## Search in GraphDB ########
###################################
def get_node(node_id: str):
    query = """
        MATCH (n{id: $node_id})
        RETURN n
    """

    records, summary, keys = driver.execute_query(
        query,
        node_id=node_id,
        database="neo4j"
    )
    
    record_list = []
    for record in records:
        # print(record.data(), type(record.data()))
        record_list.append(record.data()["n"])
        
    return record_list

def get_father_node(node_id: str):
    query = """
        MATCH (c_o{id: $node_id})<-[r:HAS]-(n)
        RETURN n
    """

    records, summary, keys = driver.execute_query(
        query,
        node_id=node_id,
        database="neo4j"
    )
    
    record_list = []
    for record in records:
        # print(record.data(), type(record.data()))
        record_list.append(record.data()["n"])
        
    return record_list

def get_children_nodes(node_id: str):
    # Node Content
    if "C" in node_id:
        return []
    # Note Heading
    elif "H" in node_id:
        query = """
            MATCH (c_o:Heading {id: $node_id})-[r:HAS]->(n)
            RETURN n
        """
    # Note Document
    else:
        query = """
            MATCH (c_o:Document {id: $node_id})-[r:HAS]->(n:Heading)
            RETURN c_o, n
        """
        
    records, summary, keys = driver.execute_query(
        query,
        node_id=node_id,
        database="neo4j"
    )
    
    record_list = []
    for record in records:
        # print(record.data(), type(record.data()))
        record_list.append(record.data()["n"])
        
    return record_list

def get_sibling_nodes(node_id: str):
    # Node Content
    if "C" in node_id:
        query = """
            MATCH (c_o:Content {id: $node_id})<-[r:HAS]-(h:Heading)
            MATCH (h)-[r:HAS]->(n:Content)
            RETURN n
        """
    # Note Heading
    elif "H" in node_id:
        query = """
            MATCH (c_o:Heading {id: $node_id})-[r:IS_NEXT_TO]-(n:Heading)
            RETURN c_o, n
        """
    else:
        query = """
            MATCH (c_o:Document {id: $node_id})<-[]-(n)
            RETURN n
        """

    records, summary, keys = driver.execute_query(
        query,
        node_id=node_id,
        database="neo4j"
    )
    
    record_list = []
    for record in records:
        # print(record.data(), type(record.data()))
        record_list.append(record.data()["n"])
        
    return record_list

def get_metadata_node(node_id: str):
    id_doc, id_file_type = node_id.split("_")[0], node_id.split("_")[1]
    if id_file_type.startswith("VB") and len(id_file_type) > 2:
        meta_data_node_id = f"{id_doc}_{id_file_type}_C0"
    else:
        meta_data_node_id = f"{id_doc}_VB_C0"
    
    query = """
        MATCH (n{id: $node_id})
        RETURN n
    """

    records, summary, keys = driver.execute_query(
        query,
        node_id=meta_data_node_id,
        database="neo4j"
    )
    
    record_list = []
    for record in records:
        # print(record.data(), type(record.data()))
        record_list.append(record.data()["n"])
        
    return record_list

def get_doc_node(metadata_node_id: str):    
    query = """
        MATCH (c_o{id: $node_id})<-[r:HAS]-(n:Document)
        RETURN n
    """

    records, summary, keys = driver.execute_query(
        query,
        node_id=metadata_node_id,
        database="neo4j"
    )
    
    record_list = []
    for record in records:
        # print(record.data(), type(record.data()))
        record_list.append(record.data()["n"])
        
    return record_list

def concat_contents(record_list: list, node_id: str):
    content = ""
    record_list.sort(key=lambda x: x["id"])
    for record in record_list:
        try:
            content += record["title"] + "\n"
        except:
            try:
                content += record["content"] + "\n"
            except:
                content += record["document_name"] + "\n"
    content = content[:-2]
    
    if len(content) > 80000:
        content = ""
        node_idx = [i for i, r in enumerate(record_list) if r['id'] == node_id][0]
        window = 10
        
        record_list = record_list[node_idx - window:node_idx + window]
        
        for record in record_list:
            try:
                content += record["title"] + "\n"
            except:
                try:
                    content += record["content"] + "\n"
                except:
                    content += record["document_name"] + "\n"
        content = content[:-2]
        print('CONTENT')
        print(content)
    
    return content

###################################################
################### API ###########################
###################################################

#######################################
############## NetBI ##################
#######################################

# Search for list of serving KPI
@app.post("/netbi/search/trick")
async def query(request: Request):
    ori_start_time = datetime.datetime.now()
    request = await request.json()
    query = request['query']

    # Dense search
    start_time = datetime.datetime.now()
    netbi_hit = dense_search_trick(query_str=query, collection_name="trick_netbi_name_dense", threshold=0, limit=5)
    end_time_search = datetime.datetime.now()
    search_time = end_time_search - start_time
    search_time = round(search_time.total_seconds(), 3)
    
    # Rerank
    rerank_limit = 1
    rerank_threshold = 0.9
    
    corpus = []
    for hit in netbi_hit:
        chunk = hit["name"]
        corpus.append(chunk)

    print("=========RERANK IN NETBI TRICK=========")
    start_time = datetime.datetime.now()
    rerank_results = rerank(query, corpus)
    end_time_rerank = datetime.datetime.now()
    rerank_time = end_time_rerank - start_time
    rerank_time = round(rerank_time.total_seconds(), 3)
    topk = rerank_results[:rerank_limit]
    
    final_points = []
    for rerank_result in topk:
        idx = rerank_result["index"]
        rerank_score = rerank_result["relevance_score"]
        if rerank_score < rerank_threshold:
            continue
        netbi_hit[idx]["score"] = rerank_score
        final_points.append(netbi_hit[idx])
        
    end_final_time = datetime.datetime.now()
    final_time = end_final_time - ori_start_time
    final_time = round(final_time.total_seconds(), 3)
    
    print(f"Search time: {search_time}s")
    print(f"Rerank time: {rerank_time}s")
    print(f"Final time: {final_time}s")
    
    return final_points

# Search for definition of KPI
@app.post("/netbi/search")
async def query(request: Request):
    request = await request.json()
    query = request['query']
    
    # Search in NetBI PL02 for definition
    start_time = datetime.datetime.now()
    netbi_hit = dense_search_netbi(query_str=query, collection_name="netbi_dense_short_v5", threshold=0.35, limit=5)
    end_time_search_netbi = datetime.datetime.now()
    search_time_netbi = end_time_search_netbi - start_time
    search_time_netbi = round(search_time_netbi.total_seconds(), 3)
    
    # Search in STKT
    start_time = datetime.datetime.now()
    stkt_hit = hybrid_search_netbi(query_str=query, collection_name="hybrid_dr_bm42", filter=["netbi"], threshold=0.45, limit=5)
    end_time_search_stkt = datetime.datetime.now()
    search_time_stkt = end_time_search_stkt - start_time
    search_time_stkt = round(search_time_stkt.total_seconds(), 3)
    
    # Rerank
    rerank_limit = 5
    rerank_threshold = 0.05
    concat_points = netbi_hit + stkt_hit
    
    corpus = []
    for hit in concat_points:
        chunk = hit["chunk"]
        corpus.append(chunk)
    print("=========RERANK IN NETBI===============")
    start_time = datetime.datetime.now()
    rerank_results = rerank(query, corpus)
    end_time_rerank = datetime.datetime.now()
    rerank_time = end_time_rerank - start_time
    rerank_time = round(rerank_time.total_seconds(), 3)
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

#######################################
############## Vo Tuyen ###############
#######################################

@app.post("/vo_tuyen/search")
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

#######################################
############## Example ################
#######################################

@app.post("/search_top_nearestQ")
async def query(request: Request):
    request = await request.json()
    query_str = request['query']
    collection_name = "ads_nearestQ_dr_bm42"  
    threshold = 0
    limit = 20
    
    # Hybrid search
    sparse_embedding = list(model_bm42.query_embed(query_str))[0]
    
    topk = qclient.query_points(
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

@app.post("/search_vai")
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

#####################
### Upload points ###
#####################

def upload_hybrid(chunks, collection_name, is_delete: bool=False):
    if is_delete:
        qclient.delete_collection(collection_name = collection_name)
    if not qclient.collection_exists(collection_name):    
        qclient.create_collection(
            collection_name=collection_name,
            vectors_config={
                "text-dense": models.VectorParams(
                    size=model_encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "text-sparse": models.SparseVectorParams(
                    modifier = models.Modifier.IDF
                )
            }
        ) 
    print("Upload points")
    qclient.upload_points(
        collection_name = collection_name,
        points = [
            models.PointStruct(
                id = i,
                vector = {
                    "text-dense": model_encoder.encode(row['content']).tolist(),
                    "text-sparse": models.SparseVector(
                        indices = list(model_sparse.embed(row['content']))[0].indices.tolist(),
                        values = list(model_sparse.embed(row['content']))[0].values.tolist()
                    )
                },
                payload = {
                    "chunk": row["content"],
                    "content_id": row["content_id"]
                }
            )
            for i, row in enumerate(chunks)
        ]
    )

def upload_dense(chunks, collection_name: str, is_delete: bool=False):
    if is_delete:
        qclient.delete_collection(collection_name = collection_name)
    if not qclient.collection_exists(collection_name):    
        qclient.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
                size=model_encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
                distance=models.Distance.COSINE,
            )
    )
    print("Upload points")
    qclient.upload_points(
        collection_name = collection_name,
        points = [
            models.PointStruct(
                id = i,
                vector = model_encoder.encode(row['content']).tolist(),
                payload = {
                    "chunk": row["content"],
                    "content_id": row["content_id"]
                }
            )
            for i, row in enumerate(chunks)
        ]
    )
        
@app.post("/upload_hybrid")
async def upload_hybrid_points(request: Request):
    request = await request.json()
    print(request)
    
    upload_hybrid(chunks=request['data'], collection_name=request["collection_name"])
    print('Done')
    return 'Done'

@app.post("/upload_dense")
async def upload_dense_points(request: Request):
    request = await request.json()
    print(request)
    
    upload_dense(chunks=request['data'], collection_name=request["collection_name"])
    print('Done')
    return 'Done'


#####################################
########## GET IMAGE ################
#####################################

# import os
# from datetime import datetime
# from fastapi import Response
# from fastapi.responses import FileResponse
# from minio import Minio
# from minio.error import S3Error
# from io import BytesIO

# from src.core.config import MINIO_CONFIG


# minio_qclient = Minio(
#     MINIO_CONFIG['ENDPOINT'],
#     access_key=MINIO_CONFIG['ACCESS_KEY'],
#     secret_key=MINIO_CONFIG['SECRET_KEY'],
#     secure=MINIO_CONFIG['SECURE']
# )

# 1
# @router.get("/netmind-image/{image_name}")
# async def get_image(image_name: str):
#     return get_image_minio(image_name)

# def get_image_minio(image_name):
#     print(image_name)
#     bucket_name = "netmind"
    
#     folder_1, folder_2 = image_name.split("_")[0], image_name.split("_")[1]
#     full_path = f"{folder_1}/{folder_2}/{image_name}"
#     try:
#         # Lấy đối tượng từ 
#         try:
#             response = minio_qclient.get_object(bucket_name, full_path)
#         except:
#             # response = backup_minio_qclient.get_object(bucket_name, image_name)
#             response = minio_qclient.get_object(bucket_name, full_path)

#         # Đọc nội dung của hình ảnh vào một BytesIO object
#         image_data = BytesIO(response.read())
#         image_data.seek(0)

#         # Trả về hình ảnh dưới dạng HTTP response
#         return Response(content=image_data.getvalue(), media_type='image/png')

#     except S3Error as e:
#         # Xử lý lỗi nếu có
#         return Response(f"File không tồn tại!", status_code=404)