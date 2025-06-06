from fastapi import APIRouter, Depends

from src.api.dependencies.security import validate_whitelist
from src.api.routers import (
    netbi, 
    votuyen,
    general
)


api_router = APIRouter(
    dependencies=[Depends(validate_whitelist)]
)

api_router.include_router(netbi.router, tags=['netbi'], prefix='/netbi')
api_router.include_router(votuyen.router, tags=['votuyen'], prefix='/votuyen')
api_router.include_router(general.router, tags=['general'])
