from typing import Annotated
# 健康检查（Health Check）路由文件
from fastapi import APIRouter, Depends

from app.core.config import settings
from app.schemas.common import ApiResponse
from app.schemas.health import HealthBasicStatus, HealthReadinessStatus
from app.services.health_service import HealthService, get_health_service

router = APIRouter()


@router.get(
    "/health",  # 存活检查
    response_model=ApiResponse[HealthBasicStatus])
async def health_check() -> ApiResponse[HealthBasicStatus]:
    return ApiResponse(
        code=200,
        message="success",
        data=HealthBasicStatus(status="ok", role=settings.app_role),
    )
'''
作用： 检查服务是否活着（正在运行）
元素	    说明
路径	    GET /health
逻辑	    直接返回 "ok"，不做任何依赖检查
返回	    {"status": "ok", "role": "api"}
用途  	Kubernetes livenessProbe、负载均衡器健康检查
'''

@router.get(
    "/health/ready",  # 就绪检查
    response_model=ApiResponse[HealthReadinessStatus])
async def readiness_check(
    service: Annotated[HealthService, Depends(get_health_service)],
) -> ApiResponse[HealthReadinessStatus]:
    result = await service.get_readiness()
    return ApiResponse(code=200, message="success", data=result)
'''
作用： 检查服务是否准备好接收流量
元素	   说明
路径	   GET /health/ready
逻辑	   检查所有依赖服务是否可用（数据库、Redis、消息队列等）
返回	   每个依赖的健康状态
用途	   Kubernetes readinessProbe、服务发现
'''

'''
核心区别：/health vs /health/ready
维度	           /health（存活）	/health/ready（就绪）
检查什么	      进程是否还在运行	    所有依赖是否都可用
逻辑	          立即返回 ok	    检查 DB、Redis、LLM 等
开销	          极低（几乎没有）	    中等（需要查询依赖）
失败后果	      K8s 重启容器	    K8s 暂停转发流量
什么时候用	  每秒检查一次	    每 10-30 秒检查一次

容器启动
    ↓
/health/ready 返回 not_ready（数据库还没连上）
    ↓
K8s 不转发流量到这个 Pod
    ↓
5 秒后再次检查 /health/ready
    ↓
数据库连接成功 → 返回 ready
    ↓
K8s 开始转发流量 ✅
    ↓
... 如果 /health 失败（进程挂了）→ K8s 重启容器

轻量级存活检查
    不查数据库、不做任何 I/O
    即使数据库挂了，/health 依然返回 200
    保证 K8s 不会因为数据库临时问题而重启容器
重量级就绪检查
    检查数据库、Redis、LLM 服务等
    如果任何依赖不可用，返回 not_ready
    K8s 会停止转发流量，但不会重启容器
'''