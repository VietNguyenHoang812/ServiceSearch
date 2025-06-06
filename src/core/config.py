from functools import lru_cache
from typing import Dict, Type

from .settings.base import BaseAppSettings, AppEnvTypes
from .settings.app import AppSettings
from .settings.dev import DevAppSettings
from .settings.prod import ProdAppSettings


environments: Dict[AppEnvTypes, Type[AppSettings]] = {
    AppEnvTypes.prod: ProdAppSettings, 
    AppEnvTypes.dev: DevAppSettings
}


@lru_cache() 
def get_app_settings() -> AppSettings:
    app_env = BaseAppSettings().app_env
    config = environments[app_env]
    return config()


env_config = get_app_settings().env_configs

APPLICATION_CODE = env_config['APP']['APPLICATION_CODE']
APP_IP = env_config['APP']['IP']
MINIO_CONFIG = env_config['STORAGE']['MINIO']
NETBI_CONFIG = env_config['REQUEST']['NETBI']
URL_BACKEND_API = env_config['REQUEST']['BACKEND_API']['URL']
INPUT_DIR = env_config['STORAGE']['LOCAL_INPUT_DIR']
OUTPUT_DIR = env_config['STORAGE']['LOCAL_OUTPUT_DIR']
EMAIL_CONFIG = env_config['REQUEST']['EMAIL']
CHATBOT_VAI_CONFIG = env_config['REQUEST']['CHATBOT_VAI']
CHATBOT_VOFFICE_CONFIG = env_config['REQUEST']['CHATBOT_VOFFICE']
MYSQL_CONFIG = env_config['STORAGE']['MYSQL']
GOOGLE_CONFIG = env_config['REQUEST']['GOOGLE']
TAVILY_CONFIG = env_config['REQUEST']['TAVILY']
WHITELIST = env_config['APP']['WHITELIST']
VAI_TTS = env_config['REQUEST']['VAI_TTS']
VAI_ASR = env_config['REQUEST']['VAI_ASR']