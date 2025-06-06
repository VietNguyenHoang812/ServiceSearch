from .app import AppSettings
from src.utils.loader import load_yaml


class ProdAppSettings(AppSettings):
    debug: bool = False
    docs_url: str = None
    openapi_prefix: str = None
    openapi_url: str = None
    redoc_url: str = None
    title: str = "VTNet Prod"
    version: str = "0.0.1"

    env_configs = load_yaml('env/prod.yaml')