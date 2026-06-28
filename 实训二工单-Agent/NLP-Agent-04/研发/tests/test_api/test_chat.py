"""API 测试"""

from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport

from src.api.app import app


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health_check(client):
    """健康检查接口"""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_root(client):
    """根路径"""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "基金数据问答智能体系统" in data["service"]
