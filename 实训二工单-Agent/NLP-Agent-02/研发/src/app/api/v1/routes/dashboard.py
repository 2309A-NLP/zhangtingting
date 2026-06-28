from typing import Annotated
# 用户端  仪表盘（Dashboard）路由文件
from fastapi import APIRouter, Depends, status

from app.schemas.common import ApiResponse
from app.schemas.dashboard import DashboardOverview
from app.services.dashboard_service import DashboardService, get_dashboard_service

router = APIRouter()


@router.get(
    "/overview",  # 获取仪表盘总览数据
    response_model=ApiResponse[DashboardOverview],
    status_code=status.HTTP_200_OK)
async def get_dashboard_overview(
    service: Annotated[DashboardService, Depends(get_dashboard_service)],
) -> ApiResponse[DashboardOverview]:
    result = await service.get_overview()
    return ApiResponse(code=200, message="success", data=result)
