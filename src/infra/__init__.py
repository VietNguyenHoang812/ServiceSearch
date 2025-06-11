from .clients import get_qdrant_client, get_xinference_client, get_neo4j_driver
from .models import ModelBM42Singleton


# Clients
qdrant_client = get_qdrant_client()
xinference_client = get_xinference_client()
neo4j_driver = get_neo4j_driver()

# LOAD MODEL
dense_model_name = "embedding-dr-bge-v2"
model_dense = xinference_client.get_model(dense_model_name)

old_dense_model_name = "result_embedding_dr"
old_model_dense = xinference_client.get_model(old_dense_model_name)

rerank_model_name = "bge-rerank-v2-m3-0"
model_rerank = xinference_client.get_model(rerank_model_name)

model_bm42 = ModelBM42Singleton.get_model()