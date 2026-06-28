from app.agents.llm_plan_schemas import LLMPlanResponse
from app.agents.llm_planner import AgentLLMPlanner


class FakePlannerClient:
    def __init__(self, response: str) -> None:
        self._response = response

    async def chat_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        use_json_mode: bool = False,
    ) -> str:
        assert system_prompt
        assert user_prompt
        assert use_json_mode is True
        return self._response


async def test_llm_planner_returns_structured_plan() -> None:
    planner = AgentLLMPlanner(
        client=FakePlannerClient(
            '{"complexity":"complex","intent":"create","action":"confirm","extracted":{"content":"study","schedule_date":"2026-06-18","schedule_time":"19:00:00","cycle_rule":"once"},"missing_fields":[],"reply_style":"concise","reasoning_summary":"create reminder"}'
        )
    )

    result = await planner.plan(
        session_id="planner-demo-1",
        user_input="tomorrow 7pm remind me to study and then ask me again",
        context={},
    )

    assert result is not None
    assert isinstance(result, LLMPlanResponse)
    assert result.complexity == "complex"
    assert result.intent == "create"
    assert result.action == "confirm"
