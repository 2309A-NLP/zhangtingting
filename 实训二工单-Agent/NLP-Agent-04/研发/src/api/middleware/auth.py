"""API Key 认证中间件"""

from __future__ import annotations

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from config import settings

security = HTTPBearer(auto_error=False)

# 无需认证的路径
_WHITELIST = {"/health", "/", "/docs", "/redoc", "/openapi.json"}


class AuthMiddleware(BaseHTTPMiddleware):
    """API Key 认证中间件"""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        if request.url.path in _WHITELIST:
            return await call_next(request)

        auth_header = request.headers.get("X-API-Key") or request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]
        else:
            api_key = auth_header

        valid_keys = settings.API_KEYS_LIST
        if valid_keys and api_key not in valid_keys:
            raise HTTPException(status_code=401, detail="无效的 API Key")

        return await call_next(request)


async def get_api_key(request: Request) -> str:
    """依赖注入方式获取 API Key（从 X-API-Key 或 Authorization 头提取）"""
    if not settings.API_KEY_ENABLED:
        return "public"
    auth_header = request.headers.get("X-API-Key") or request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:]
    else:
        api_key = auth_header
    if not api_key or api_key not in settings.API_KEYS_LIST:
        raise HTTPException(status_code=401, detail="无效的 API Key")
    return api_key
