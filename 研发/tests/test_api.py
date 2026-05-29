from __future__ import annotations

import json

import pytest

from app.api.routers import chat as chat_router_module
from app.api.routers import knowledge as knowledge_router_module
from app.api.schemas import ChatResponse, ContextSourceSchema
from app.chat.models import BuiltContext, ChatCompletionResult, ContextSource


def unwrap(payload: dict):
    return payload["data"] if "data" in payload else payload


@pytest.mark.asyncio
async def test_list_roles(api_client):
    response = await api_client.get("/api/v1/roles", params={"user_id": "test-user-001"})
    assert response.status_code == 200
    assert "X-Request-Id" in response.headers
    data = unwrap(response.json())
    assert data["total"] >= 2
    assert any(item["role_id"] == "lawyer_01" for item in data["items"])


@pytest.mark.asyncio
async def test_detect_role(api_client):
    response = await api_client.post(
        "/api/v1/roles/detect",
        json={"user_id": "test-user-001", "query": "How should I handle a contract dispute?"},
    )
    assert response.status_code == 200
    data = unwrap(response.json())
    assert data["role_id"] == "lawyer_01"
    assert data["action"] == "matched"


@pytest.mark.asyncio
async def test_create_custom_role(api_client):
    response = await api_client.post(
        "/api/v1/roles/custom",
        json={
            "user_id": "test-user-001",
            "name": "My advisor",
            "system_prompt": "You are my professional advisor.",
            "category": "general",
        },
    )
    assert response.status_code == 200
    data = unwrap(response.json())
    assert data["role_id"] == "custom_00002"
    assert data["name"] == "My advisor"


@pytest.mark.asyncio
async def test_chat_non_stream(api_client, monkeypatch):
    class FakeLimiter:
        async def check(self, *, user_id: str, role_id: str) -> None:
            return None

    class FakeContextBuilder:
        async def build(self, **kwargs):
            return BuiltContext(
                messages=[
                    {"role": "system", "content": "test system prompt"},
                    {"role": "user", "content": kwargs["query"]},
                ],
                context_sources=[
                    ContextSource(
                        doc_id="doc_001",
                        chunk_id="chunk_001",
                        source="sample.txt",
                        score=0.95,
                        text="Contract disputes should start with evidence collection.",
                    )
                ],
                rewritten_query="contract dispute handling",
            )

    class FakeLLMClient:
        async def complete(self, *, messages, temperature=0.3, max_tokens=1024):
            return ChatCompletionResult(
                content="Start by preserving evidence and reviewing the contract clauses.",
                model="fake-vllm",
                tokens_used=123,
                degraded_to_online_api=False,
            )

    class FakeCacheService:
        async def get_cached_response(self, *, user_id: str, role_id: str, query: str):
            return None

        async def set_cached_response(self, *, user_id: str, role_id: str, query: str, response: ChatResponse):
            return None

    class FakeMemoryService:
        async def update_summary(self, *, user_id: str, role_id: str, query: str, response: str):
            return "memory updated"

    async def fake_persist_chat(**kwargs):
        return None

    monkeypatch.setattr(chat_router_module, "RedisLeakyBucketRateLimiter", FakeLimiter)
    monkeypatch.setattr(chat_router_module, "ContextBuilder", FakeContextBuilder)
    monkeypatch.setattr(chat_router_module, "LLMClient", FakeLLMClient)
    monkeypatch.setattr(chat_router_module, "ChatCacheService", FakeCacheService)
    monkeypatch.setattr(chat_router_module, "MemoryService", FakeMemoryService)
    monkeypatch.setattr(chat_router_module, "_persist_chat", fake_persist_chat)

    response = await api_client.post(
        "/api/v1/chat",
        json={
            "user_id": "test-user-001",
            "role_id": "lawyer_01",
            "query": "How should I handle a contract dispute?",
            "stream": False,
        },
    )
    assert response.status_code == 200
    data = unwrap(response.json())
    assert data["role_id"] == "lawyer_01"
    assert data["response"].startswith("Start by preserving evidence")
    assert data["context_sources"][0]["doc_id"] == "doc_001"
    assert data["tokens_used"] == 123


@pytest.mark.asyncio
async def test_chat_cache_hit(api_client, monkeypatch):
    class FakeLimiter:
        async def check(self, *, user_id: str, role_id: str) -> None:
            return None

    class FakeCacheService:
        async def get_cached_response(self, *, user_id: str, role_id: str, query: str):
            return ChatResponse(
                request_id="old-request-id",
                role_id=role_id,
                role_name="Civil Lawyer",
                session_id="old-session-id",
                response="Cached answer",
                context_sources=[
                    ContextSourceSchema(
                        doc_id="doc_cache",
                        chunk_id="chunk_cache",
                        source="cache.txt",
                        score=0.88,
                    )
                ],
                tokens_used=77,
                latency_ms=999,
                model="cache-model",
                degraded_to_online_api=False,
                rewritten_query=query,
            )

        async def set_cached_response(self, *, user_id: str, role_id: str, query: str, response: ChatResponse):
            return None

    class FailContextBuilder:
        async def build(self, **kwargs):
            raise AssertionError("ContextBuilder should not be called on cache hit")

    monkeypatch.setattr(chat_router_module, "RedisLeakyBucketRateLimiter", FakeLimiter)
    monkeypatch.setattr(chat_router_module, "ChatCacheService", FakeCacheService)
    monkeypatch.setattr(chat_router_module, "ContextBuilder", FailContextBuilder)

    response = await api_client.post(
        "/api/v1/chat",
        json={
            "user_id": "test-user-001",
            "role_id": "lawyer_01",
            "query": "repeat question",
            "stream": False,
        },
    )
    assert response.status_code == 200
    data = unwrap(response.json())
    assert data["response"] == "Cached answer"
    assert data["session_id"] != "old-session-id"
    assert data["request_id"] == response.headers["X-Request-Id"]


@pytest.mark.asyncio
async def test_chat_stream(api_client, monkeypatch):
    class FakeLimiter:
        async def check(self, *, user_id: str, role_id: str) -> None:
            return None

    class FakeContextBuilder:
        async def build(self, **kwargs):
            return BuiltContext(
                messages=[{"role": "user", "content": kwargs["query"]}],
                context_sources=[],
                rewritten_query=kwargs["query"],
            )

    class FakeLLMClient:
        async def stream(self, *, messages, temperature=0.3, max_tokens=1024):
            yield {"event": "delta", "data": json.dumps({"content": "Part one ", "model": "fake", "degraded": False})}
            yield {"event": "delta", "data": json.dumps({"content": "part two", "model": "fake", "degraded": False})}
            yield {"event": "end", "data": json.dumps({"model": "fake", "degraded": False})}

    class FakeCacheService:
        async def get_cached_response(self, *, user_id: str, role_id: str, query: str):
            return None

        async def set_cached_response(self, *, user_id: str, role_id: str, query: str, response: ChatResponse):
            return None

    class FakeMemoryService:
        async def update_summary(self, *, user_id: str, role_id: str, query: str, response: str):
            return "memory updated"

    async def fake_persist_chat(**kwargs):
        return None

    monkeypatch.setattr(chat_router_module, "RedisLeakyBucketRateLimiter", FakeLimiter)
    monkeypatch.setattr(chat_router_module, "ContextBuilder", FakeContextBuilder)
    monkeypatch.setattr(chat_router_module, "LLMClient", FakeLLMClient)
    monkeypatch.setattr(chat_router_module, "ChatCacheService", FakeCacheService)
    monkeypatch.setattr(chat_router_module, "MemoryService", FakeMemoryService)
    monkeypatch.setattr(chat_router_module, "_persist_chat", fake_persist_chat)

    async with api_client.stream(
        "POST",
        "/api/v1/chat",
        json={
            "user_id": "test-user-001",
            "role_id": "lawyer_01",
            "query": "Please stream the answer",
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        body = ""
        async for line in response.aiter_lines():
            body += line

    assert "Part one" in body
    assert "part two" in body

# 标记这是一个异步测试
@pytest.mark.asyncio
async def test_clear_chat(api_client, monkeypatch):
    class FakeRedis:
        async def scan_iter(self, match=None, count=None):
            for key in ("cache:test-user-001:lawyer_01:a", "cache:test-user-001:lawyer_01:b"):
                yield key

        async def delete(self, *keys):
            return len(keys)

    monkeypatch.setattr(chat_router_module, "get_redis", lambda: FakeRedis())

    response = await api_client.post(
        "/api/v1/chat/clear",
        json={"user_id": "test-user-001", "role_id": "lawyer_01"},
    )
    assert response.status_code == 200
    data = unwrap(response.json())
    assert data["success"] is True
    assert len(data["cleared_keys"]) == 5


@pytest.mark.asyncio
async def test_knowledge_upload(api_client, monkeypatch, tmp_path):
    enqueued: dict[str, object] = {}

    class FakeStorageService:
        def build_raw_object_name(self, *, user_id: str, role_id: str, task_id: str, file_name: str) -> str:
            return f"{user_id}/{role_id}/raw/{task_id}/{file_name}"

        async def upload_file(self, *, bucket: str, object_name: str, local_path: str, content_type: str | None = None, metadata=None):
            return f"minio://{bucket}/{object_name}"

    class FakeQueue:
        async def enqueue(self, task):
            enqueued["task"] = task

    settings = knowledge_router_module.get_settings()
    settings.upload_dir = str(tmp_path)

    monkeypatch.setattr(knowledge_router_module, "MinioStorageService", FakeStorageService)
    monkeypatch.setattr(knowledge_router_module, "get_knowledge_task_queue", lambda: FakeQueue())

    files = {"file": ("test.txt", "Contract evidence includes the signed agreement and payment records.", "text/plain")}
    data = {
        "user_id": "test-user-001",
        "role_id": "lawyer_01",
        "mode": "incremental",
    }
    response = await api_client.post("/api/v1/knowledge/upload", data=data, files=files)
    assert response.status_code == 200
    payload = unwrap(response.json())
    assert payload["status"] == "queued"
    assert payload["role_id"] == "lawyer_01"
    assert enqueued["task"].role_id == "lawyer_01"


@pytest.mark.asyncio
async def test_knowledge_task_status(api_client, monkeypatch):
    class FakeRedis:
        async def hgetall(self, key):
            return {
                "task_id": "task_001",
                "user_id": "test-user-001",
                "role_id": "lawyer_01",
                "mode": "incremental",
                "status": "success",
                "doc_id": "task_001",
                "source_uri": "minio://rag-raw/test.txt",
                "parsed_artifact_uri": "minio://rag-parsed/manifest.json",
                "chunk_count": "3",
                "started_at": "100",
                "finished_at": "120",
            }

    monkeypatch.setattr(knowledge_router_module, "get_redis", lambda: FakeRedis())

    response = await api_client.get(
        "/api/v1/knowledge/tasks/task_001",
        params={"user_id": "test-user-001", "role_id": "lawyer_01"},
    )
    assert response.status_code == 200
    payload = unwrap(response.json())
    assert payload["task_id"] == "task_001"
    assert payload["status"] == "success"
    assert payload["chunk_count"] == 3
