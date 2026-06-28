from fastapi.testclient import TestClient

from app.agents.llm_parser import AgentLLMParser
from app.main import app
from app.schemas.agent import AgentStateResponse


def test_agent_llm_audit_api_returns_session_logs(monkeypatch) -> None:
    async def fake_parse(
        self: AgentLLMParser,
        *,
        session_id: str,
        user_input: str,
        context: dict[str, object],
    ) -> AgentStateResponse | None:
        await self._record_audit(
            session_id=session_id,
            user_input=user_input,
            parser_stage="parse",
            success=True,
            request_payload={"user_input": user_input},
            raw_response_text='{"agent_state":"confirm"}',
            parsed_response={
                "agent_state": "confirm",
                "intent": "create",
                "user_message": "请确认是否创建。",
                "tool_name": "schedule_create",
            },
            error_message=None,
        )
        return AgentStateResponse(
            agent_state="confirm",
            intent="create",
            user_message="请确认是否创建。",
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

    monkeypatch.setattr(AgentLLMParser, "parse", fake_parse)

    payload = {
        "session_id": "llm-audit-api-demo-1",
        "user_input": "please set up a study reminder in a natural way",
        "confirmed": False,
        "context": {},
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/agent/chat", json=payload)
        assert response.status_code == 200

        audit_response = client.get("/api/v1/agent/llm-audit/llm-audit-api-demo-1")
        assert audit_response.status_code == 200
        audit_body = audit_response.json()
        assert len(audit_body["data"]) >= 1
        assert all(item["session_id"] == "llm-audit-api-demo-1" for item in audit_body["data"])
        assert any(item["parser_stage"] == "parse" and item["success"] is True for item in audit_body["data"])
        parse_item = next(item for item in audit_body["data"] if item["parser_stage"] == "parse")
        assert parse_item["request_payload"]["user_input"] == "please set up a study reminder in a natural way"

        recent_response = client.get("/api/v1/agent/llm-audit?limit=10")
        assert recent_response.status_code == 200
        recent_body = recent_response.json()
        assert recent_body["data"]["total"] >= 1
        assert any(item["session_id"] == "llm-audit-api-demo-1" for item in recent_body["data"]["items"])

        filtered_response = client.get("/api/v1/agent/llm-audit?session_id=llm-audit-api-demo-1&success=true")
        assert filtered_response.status_code == 200
        filtered_body = filtered_response.json()
        assert filtered_body["data"]["total"] >= 1
        assert any(item["session_id"] == "llm-audit-api-demo-1" for item in filtered_body["data"]["items"])
