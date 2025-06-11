from fastembed import SparseTextEmbedding

class ModelBM42Singleton:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelBM42Singleton, cls).__new__(cls)
            cls._model = SparseTextEmbedding(
                "Qdrant/bm42-all-minilm-l6-v2-attentions", 
                cache_dir="/weights", 
                local_files_only=True
            )
        return cls._instance

    @classmethod
    def get_model(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._model