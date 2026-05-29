from __future__ import annotations
'''提供用户注册、登录、获取个人信息的 API 接口。'''
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_auth_service, get_current_user_id, get_db_session
from app.api.schemas import Envelope, LoginRequest, RegisterRequest, TokenResponse, UserProfileResponse
from app.core.config import get_settings
from app.core.response import success_response
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=Envelope[UserProfileResponse])
async def register(
    payload: RegisterRequest,
    db_session: AsyncSession = Depends(get_db_session),
    auth_service: AuthService = Depends(get_auth_service),
):
    try:
        user = await auth_service.register_user(
            db_session,
            username=payload.username,
            password=payload.password,
            email=payload.email,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return success_response(UserProfileResponse(user_id=user.user_id, username=user.username, email=user.email))


@router.post("/login", response_model=Envelope[TokenResponse])
async def login(
    payload: LoginRequest,
    db_session: AsyncSession = Depends(get_db_session),
    auth_service: AuthService = Depends(get_auth_service),
):
    try:
        user, token = await auth_service.authenticate_user(
            db_session,
            username=payload.username,
            password=payload.password,
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    settings = get_settings()
    return success_response(
        TokenResponse(
            access_token=token,
            expires_in_seconds=settings.app_access_token_expire_minutes * 60,
            user_id=user.user_id,
            username=user.username,
        )
    )


@router.get("/me", response_model=Envelope[UserProfileResponse])
async def me(
    db_session: AsyncSession = Depends(get_db_session),
    current_user_id: str | None = Depends(get_current_user_id),
    auth_service: AuthService = Depends(get_auth_service),
):
    if not current_user_id:
        # 需要认证
        raise HTTPException(status_code=401, detail="Authentication required.")

    user = await auth_service.get_user_by_id(db_session, user_id=current_user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")

    return success_response(UserProfileResponse(user_id=user.user_id, username=user.username, email=user.email))

'''
总结：三个接口的职责
接口	            方法	       作用	             需要认证
/auth/register	POST	创建新账号	         ❌ 不需要
/auth/login	    POST	登录，获取 token	     ❌ 不需要
/auth/me	    GET	    获取当前用户信息	     ✅ 需要（JWT token）

注册
用户 → POST /auth/register → auth_service.register_user → MySQL 保存用户
                                                              │
                                                              ▼
登录                                        返回 {user_id, username, email}
用户 → POST /auth/login → auth_service.authenticate_user
                              │                    │
                              │ 验证密码            │ 签发 JWT token
                              ▼                    ▼
                          MySQL 查询用户         jwt.encode()
                                                    │
                                                    ▼
                              返回 {access_token, user_id, username}

获取个人信息
用户 → GET /auth/me (带 token) → dependencies.get_current_user_id 解析 token
                                              │
                                              ▼
                              auth_service.get_user_by_id → MySQL 查询
                                              │
                                              ▼
                              返回 {user_id, username, email}
'''