





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