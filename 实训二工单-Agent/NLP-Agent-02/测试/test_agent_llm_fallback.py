from fastapi.testclient import TestClient

from app.agents.llm_parser import AgentLLMParser
from app.agents.llm_plan_schemas import LLMPlanResponse
from app.agents.llm_planner import AgentLLMPlanner
from app.main import app
from app.schemas.agent import AgentStateResponse


def test_agent_chat_uses_llm_fallback_when_rule_parser_cannot_handle_input(
    monkeypatch,
) -> None:
    async def fake_plan(
        self: AgentLLMPlanner,
        *,
        session_id: str,
        user_input: str,
        context: dict[str, object],
    ) -> LLMPlanResponse | None:
        assert session_id == "llm-fallback-demo-1"
        assert "tomorrow" in user_input
        return None

    async def fake_parse(
        self: AgentLLMParser,
        *,
        session_id: str,
        user_input: str,
        context: dict[str, object],
    ) -> AgentStateResponse | None:
        assert session_id == "llm-fallback-demo-1"
        assert "tomorrow" in user_input
        return AgentStateResponse(
            agent_state="confirm",
            intent="create",
            user_message="我准备为您新增一条日程，请确认是否创建。",
            parser_source="llm",
            tool_name="schedule_create",
            tool_arguments={
                "content": "study",
                "schedule_date": "2026-06-17",
                "schedule_time": "17:00:00",
                "cycle_rule": "once",
                "cycle_value": None,
                "source_text": user_input,
            },
        )

    monkeypatch.setattr(AgentLLMPlanner, "plan", fake_plan)
    monkeypatch.setattr(AgentLLMParser, "parse", fake_parse)

    payload = {
        "session_id": "llm-fallback-demo-1",
        "user_input": "tomorrow at 5pm remind me to study",
        "confirmed": False,
        "context": {},
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/agent/chat", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["agent_state"] == "confirm"
        assert body["data"]["parser_source"] == "llm"
        assert body["data"]["tool_name"] == "schedule_create"

        session_response = client.get("/api/v1/agent/sessions/llm-fallback-demo-1")
        assert session_response.status_code == 200
        session_body = session_response.json()
        assert session_body["data"]["session_id"] == "llm-fallback-demo-1"
        assert session_body["data"]["has_pending_confirmation"] is True


def test_agent_chat_can_use_llm_plan_before_fallback(monkeypatch) -> None:
    async def fake_plan(
        self: AgentLLMPlanner,
        *,
        session_id: str,
        user_input: str,
        context: dict[str, object],
    ) -> LLMPlanResponse | None:
        assert session_id == "llm-plan-demo-1"
        return LLMPlanResponse(
            complexity="complex",
            intent="create",
            action="confirm",
            extracted={
                "content": "deep work",
                "schedule_date": "2026-06-18",
                "schedule_time": "21:00:00",
                "cycle_rule": "once",
                "cycle_value": None,
                "source_text": user_input,
            },
            missing_fields=[],
            reply_style="concise",
            reasoning_summary="create reminder",
        )

    async def fake_parse(
        self: AgentLLMParser,
        *,
        session_id: str,
        user_input: str,
        context: dict[str, object],
    ) -> AgentStateResponse | None:
        return None

    monkeypatch.setattr(AgentLLMPlanner, "plan", fake_plan)
    monkeypatch.setattr(AgentLLMParser, "parse", fake_parse)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/agent/chat",
            json={
                "session_id": "llm-plan-demo-1",
                "user_input": "make me a complex study reminder for tomorrow night",
                "confirmed": False,
                "context": {},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["agent_state"] == "confirm"
        assert body["data"]["parser_source"] == "llm_plan"
        assert body["data"]["tool_name"] == "schedule_create"
