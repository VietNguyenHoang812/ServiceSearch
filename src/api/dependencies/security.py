from fastapi import Request
from fastapi.exceptions import HTTPException

from src.core.config import WHITELIST
from src.resources.const import HOST_DOES_NOT_ALLOW


wildcard_ips = [ip[:-1] for ip in WHITELIST if ip.endswith('.*')]


def validate_whitelist(request: Request):
    client_host = request.client.host
    
    if client_host in WHITELIST:
        return
    
    for ip in wildcard_ips:
        if client_host.startswith(ip):
            return
       
    raise HTTPException(
        status_code=403,
        detail=HOST_DOES_NOT_ALLOW
    )