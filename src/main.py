import logging
logging.getLogger('urllib3').setLevel(logging.WARNING)

import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI

from src.api.main import api_router
from src.core.config import get_app_settings


def get_application() -> FastAPI:
    settings = get_app_settings()
    application = FastAPI(**settings.fastapi_kwargs)
    application.include_router(api_router, prefix=settings.api_prefix)
    return application


app = get_application()