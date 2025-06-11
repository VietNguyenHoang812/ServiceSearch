
from fastapi import Response
from fastapi.responses import FileResponse
from io import BytesIO
from minio import Minio
from minio.error import S3Error

from src.core.config import MINIO_CONFIG


minio_client = Minio(
    MINIO_CONFIG['ENDPOINT'],
    access_key=MINIO_CONFIG['ACCESS_KEY'],
    secret_key=MINIO_CONFIG['SECRET_KEY'],
    secure=MINIO_CONFIG['SECURE'],
    cert_check=False
)