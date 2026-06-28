from app.agents.llm_parser import AgentLLMParser
from app.agents.llm_schemas import LLMParseResponse


class FakeLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses

    async def chat_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        use_json_mode: bool = False,
    ) -> str:
        assert system_prompt
        assert user_prompt
        return self._responses.pop(0)


async def test_llm_parser_normalizes_schedule_create_arguments() -> None:
    client = FakeLLMClient(
        [
            '{"agent_state":"confirm","intent":"create","user_message":"请确认","tool_name":"schedule_create","tool_arguments":{"content":"学习","schedule_date":"2026-06-17","schedule_time":"17:00:00"},"missing_fields":[],"target_id":null}'
        ]
    )
    parser = AgentLLMParser(client=client)

    result = await parser.parse(
        session_id="llm-parser-demo-1",
        user_input="tomorrow at 5pm remind me to study",
        context={},
    )

    assert result is not None
    assert result.parser_source == "llm"
    assert result.tool_arguments["cycle_rule"] == "once"
    assert result.tool_arguments["cycle_value"] is None
    assert result.tool_arguments["source_text"] == "tomorrow at 5pm remind me to study"


async def test_llm_parser_normalizes_schedule_delete_id_field() -> None:
    client = FakeLLMClient(
        [
            '{"agent_state":"confirm","intent":"delete","user_message":"请确认删除","tool_name":"schedule_delete","tool_arguments":{"id":12},"missing_fields":[],"target_id":12}'
        ]
    )
    parser = AgentLLMParser(client=client)

    result = await parser.parse(
        session_id="llm-parser-demo-2",
        user_input="delete schedule 12",
        context={},
    )

    assert result is not None
    assert result.tool_arguments["schedule_id"] == 12
    assert "id" not in result.tool_arguments


async def test_llm_parser_uses_repair_flow_for_invalid_json() -> None:
    client = FakeLLMClient(
        [
            '{"agent_state"',
            '{"agent_state":"confirm","intent":"create","user_message":"请确认","tool_name":"schedule_create","tool_arguments":{"content":"学习","schedule_date":"2026-06-17","schedule_time":"17:00:00","cycle_rule":"once","cycle_value":null,"source_text":"tomorrow at 5pm remind me to study"},"missing_fields":[],"target_id":null}',
        ]
    )
    parser = AgentLLMParser(client=client)

    result = await parser.parse(
        session_id="llm-parser-demo-3",
        user_input="tomorrow at 5pm remind me to study",
        context={},
    )

    assert result is not None
    assert result.parser_source == "llm"
    assert result.agent_state == "confirm"


def test_llm_schema_rejects_unknown_tool_name() -> None:
    try:
        LLMParseResponse.model_validate(
            {
                "agent_state": "confirm",
                "intent": "create",
                "user_message": "请确认",
                "tool_name": "schedule_remove",
                "tool_arguments": {},
                "missing_fields": [],
                "target_id": None,
            }
        )
    except ValueError:
        return

    raise AssertionError("Expected LLMParseResponse to reject unknown tool_name")


async def test_llm_parser_supports_pending_confirmation_cancel_tool() -> None:
    client = FakeLLMClient(
        [
            '{"agent_state":"reply","intent":"unknown","user_message":"已取消","tool_name":"conversation_cancel","tool_arguments":{},"missing_fields":[],"target_id":null}'
        ]
    )
    parser = AgentLLMParser(client=client)

    result = await parser.parse_pending_confirmation(
        session_id="llm-parser-pending-1",
        user_input="cancel it",
        pending_tool_name="schedule_create",
        pending_tool_arguments={
            "content": "study",
            "schedule_date": "2026-06-17",
            "schedule_time": "17:00:00",
        },
        pending_intent="create",
        context={},
    )

    assert result is not None
    assert result.tool_name == "conversation_cancel"
    assert result.parser_source == "llm_pending_confirmation"
