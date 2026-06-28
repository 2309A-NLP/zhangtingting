from typing import Annotated
# Prometheus 监控指标导出路由
# 专门用于暴露服务的监控数据，供 Prometheus 定期抓取
from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse  # FastAPI 的纯文本响应类（不是 JSON）

from app.services.metrics_service import MetricsService, get_metrics_service

router = APIRouter()


@router.get(
    "/metrics",  # 返回 Prometheus 格式的监控指标
    response_class=PlainTextResponse)
async def get_metrics(
    service: Annotated[MetricsService, Depends(get_metrics_service)],
) -> PlainTextResponse:
    content = await service.render_prometheus_metrics()
    return PlainTextResponse(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")
'''
元素	               说明
路径	            GET /metrics
响应类型	        PlainTextResponse（不是 JSON）
Content-Type	text/plain; version=0.0.4; charset=utf-8（Prometheus 标准格式）
逻辑	            调用 MetricsService.render_prometheus_metrics() 生成指标文本
用途	            供 Prometheus 定期抓取
'''