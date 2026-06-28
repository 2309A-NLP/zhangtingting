from fastapi.testclient import TestClient

from app.agents.llm_response_writer import AgentLLMResponseWriter
from app.main import app


def test_agent_reply_uses_llm_rewriter_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr("app.core.config.settings.llm_reply_enabled", True)

    async def fake_rewrite_message(
        self: AgentLLMResponseWriter,
        *,
        intent: str,
        agent_state: str,
        tool_name: str | None,
        tool_arguments: dict[str, object],
        execution_result: object,
        fallback_message: str,
    ) -> str:
        if agent_state == "confirm":
            assert intent == "create"
            assert tool_name == "schedule_create"
            return "我已经整理好提醒内容了，要不要现在创建这条提醒？"
        assert intent == "create"
        assert agent_state == "reply"
        assert tool_name == "schedule_create"
        assert fallback_message == "操作已完成。"
        return "已为您安排好提醒，明天下午五点记得学习。"

    monkeypatch.setattr(AgentLLMResponseWriter, "rewrite_message", fake_rewrite_message)

    with TestClient(app) as client:
        first_response = client.post(
            "/api/v1/agent/chat",
            json={
                "session_id": "llm-reply-demo-1",
                "user_input": "明天下午5点提醒我学习",
                "confirmed": False,
                "context": {},
            },
        )
        assert first_response.status_code == 200
        first_body = first_response.json()
        assert first_body["data"]["agent_state"] == "confirm"
        assert first_body["data"]["user_message"] == "我已经整理好提醒内容了，要不要现在创建这条提醒？"

        second_response = client.post(
            "/api/v1/agent/chat",
            json={
                "session_id": "llm-reply-demo-1",
                "user_input": "确认",
                "confirmed": True,
                "context": {},
            },
        )
        assert second_response.status_code == 200
        body = second_response.json()
        assert body["data"]["agent_state"] == "reply"
        assert body["data"]["user_message"] == "已为您安排好提醒，明天下午五点记得学习。"
