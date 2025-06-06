from .app import AppSettings
from src.utils.loader import load_yaml


class DevAppSettings(AppSettings):
    version: str = "1.0"
    env_configs = load_yaml('env/dev.yaml')

