from fastapi import APIRouter

from src.api.routers import (
    netbi, 
)


api_router = APIRouter()

api_router.include_router(netbi.router, tags=['netbi'], prefix='/netbi')
api_router.include_router(netbi.router, tags=['document'], prefix='/document')

