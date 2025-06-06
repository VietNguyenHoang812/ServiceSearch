import os

from datetime import datetime
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
    secure=MINIO_CONFIG['SECURE']
)
    
TEMP_DIR = "/tmp"    
    
    
def get_image_minio(image_name):
    print(image_name)
    bucket_name = "netbi-chart"
    
    if "png" in image_name:
        try:
            # Lấy đối tượng từ 
            try:
                response = minio_client.get_object(bucket_name, image_name)
            except:
                # response = backup_minio_client.get_object(bucket_name, image_name)
                response = minio_client.get_object(bucket_name, image_name)
                
            # Đọc nội dung của hình ảnh vào một BytesIO object
            image_data = BytesIO(response.read())
            image_data.seek(0)

            # Trả về hình ảnh dưới dạng HTTP response
            return Response(content=image_data.getvalue(), media_type='image/png')

        except S3Error as e:
            # Xử lý lỗi nếu có
            return Response(f"File không tồn tại!", status_code=404)
    
    elif "xlsx" in image_name:
        return "File xlsx không cho phép"
    
    else:
        temp_file_path = os.path.join(TEMP_DIR, image_name)
        try:
            # Lấy đối tượng từ MinIO
            try:
                response = minio_client.fget_object(bucket_name, image_name, temp_file_path)
            except:
                # response = backup_minio_client.fget_object(bucket_name, image_name, temp_file_path)
                response = minio_client.fget_object(bucket_name, image_name, temp_file_path)
                
            # Trả về hình ảnh dưới dạng HTTP response
            return FileResponse(path=temp_file_path, filename=image_name, media_type='application/zip')

        except S3Error as e:
            # Xử lý lỗi nếu có
            return Response(f"File không tồn tại!", status_code=404)