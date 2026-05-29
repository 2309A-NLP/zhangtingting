from __future__ import annotations
'''
管理 AI 角色的 CRUD 操作，包括系统预置角色和用户自定义角色。
(CRUD 就是管理数据的四种基本操作：增、删、改、查)
'''

# Query 是用来处理 URL 查询参数的工具，可以给查询参数添加验证、默认值、描述等信息。
'''
	        Query	            Path	          Body
数据位置	   URL 问号后面	       URL 路径中	    请求体
示例	       /api/roles?page=1   /api/roles/123	{"name": "律师"}
用途	       过滤、分页、排序	   资源ID	        创建/更新数据

Query 提供什么？
1. 自动生成 API 文档:访问 http://localhost:8000/docs 会自动显示 page 参数的说明
2.自动请求验证
请求：GET /roles?page=0
page 要求 ge=1（≥1），但传了 0 → FastAPI 自动返回 422 错误
3.类型转换
请求：GET /roles?page=1
page 在 URL 里是字符串 "1"，自动转成 int 类型
'''
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user_id, get_db_session, get_role_service, require_user_match
from app.api.schemas import (
    CreateCustomRoleRequest,
    DeleteCustomRoleResponse,
    Envelope,
    RoleDetectRequest,
    RoleDetectResponse,
    RoleSchema,
    RolesListResponse,
    UpdateCustomRoleRequest,
)
from app.core.response import success_response
from app.services.role_service import RoleService

router = APIRouter(prefix="/roles", tags=["roles"])
'''
路由概览
方法	     路径	                    功能	                        需要认证
GET	    /roles	                    获取角色列表	                可选（有 user_id 时需要）
POST	/roles/detect	            检测用户意图，匹配最适合的角色	✅ 需要
POST	/roles/custom	            创建自定义角色	            ✅ 需要
PUT	    /roles/custom/{role_id}	    更新自定义角色	            ✅ 需要
DELETE  /roles/custom/{role_id}	    删除自定义角色	            ✅ 需要
'''

@router.get("", response_model=Envelope[RolesListResponse])
async def list_roles(
    user_id: str | None = Query(default=None),  # 查询参数 ?user_id=xxx
    db_session: AsyncSession = Depends(get_db_session),
    current_user_id: str | None = Depends(get_current_user_id),
    role_service: RoleService = Depends(get_role_service),
):
    # 如果指定了 user_id，只能查自己的角色。
    if user_id:
        require_user_match(user_id, current_user_id)
    roles = await role_service.list_roles(db_session, user_id=user_id)
    return success_response(
        RolesListResponse(
            total=len(roles),
            items=[
                RoleSchema(
                    role_id=item.role_id,
                    name=item.name,
                    category=item.category,
                    role_type=item.role_type,  # type: ignore[arg-type]
                    system_prompt=item.system_prompt,
                    knowledge_base_id=item.knowledge_base_id,
                    created_at=item.created_at,
                )
                for item in roles
            ],
        )
    )


@router.post("/detect", response_model=Envelope[RoleDetectResponse])
async def detect_role(
    payload: RoleDetectRequest,
    db_session: AsyncSession = Depends(get_db_session),
    current_user_id: str | None = Depends(get_current_user_id),
    role_service: RoleService = Depends(get_role_service),
):
    require_user_match(payload.user_id, current_user_id)
    role, action, confidence, reason = await role_service.detect_role(
        db_session,
        user_id=payload.user_id,
        query=payload.query,
    )
    return success_response(
        RoleDetectResponse(
            role_id=role.role_id,
            role_name=role.name,
            category=role.category,
            action=action,  # type: ignore[arg-type]
            confidence=confidence,
            reason=reason,
        )
    )


@router.post("/custom", response_model=Envelope[RoleSchema])
async def create_custom_role(
    payload: CreateCustomRoleRequest,
    db_session: AsyncSession = Depends(get_db_session),
    current_user_id: str | None = Depends(get_current_user_id),
    role_service: RoleService = Depends(get_role_service),
):
    require_user_match(payload.user_id, current_user_id)
    try:
        role = await role_service.create_custom_role(
            db_session,
            user_id=payload.user_id,
            name=payload.name,
            system_prompt=payload.system_prompt,
            category=payload.category,
            knowledge_base_id=payload.knowledge_base_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return success_response(
        RoleSchema(
            role_id=role.role_id,
            name=role.name,
            category=role.category,
            role_type=role.role_type,  # type: ignore[arg-type]
            system_prompt=role.system_prompt,
            knowledge_base_id=role.knowledge_base_id,
            created_at=role.created_at,
        )
    )


@router.delete("/custom/{role_id}", response_model=Envelope[DeleteCustomRoleResponse])
async def delete_custom_role(
    role_id: str,
    user_id: str = Query(...),
    db_session: AsyncSession = Depends(get_db_session),
    current_user_id: str | None = Depends(get_current_user_id),
    role_service: RoleService = Depends(get_role_service),
):
    require_user_match(user_id, current_user_id)
    try:
        await role_service.delete_custom_role(
            db_session,
            user_id=user_id,
            role_id=role_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return success_response(
        DeleteCustomRoleResponse(
            success=True,
            role_id=role_id,
        )
    )


@router.put("/custom/{role_id}", response_model=Envelope[RoleSchema])
async def update_custom_role(
    role_id: str,
    payload: UpdateCustomRoleRequest,
    db_session: AsyncSession = Depends(get_db_session),
    current_user_id: str | None = Depends(get_current_user_id),
    role_service: RoleService = Depends(get_role_service),
):
    require_user_match(payload.user_id, current_user_id)
    try:
        role = await role_service.update_custom_role(
            db_session,
            user_id=payload.user_id,
            role_id=role_id,
            name=payload.name,
            system_prompt=payload.system_prompt,
            category=payload.category,
            knowledge_base_id=payload.knowledge_base_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return success_response(
        RoleSchema(
            role_id=role.role_id,
            name=role.name,
            category=role.category,
            role_type=role.role_type,  # type: ignore[arg-type]
            system_prompt=role.system_prompt,
            knowledge_base_id=role.knowledge_base_id,
            created_at=role.created_at,
        )
    )
