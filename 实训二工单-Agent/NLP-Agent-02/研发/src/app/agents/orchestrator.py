import json
from datetime import date, datetime, time
# Agent 编排器，是对话式 AI Agent 的核心中枢。
# 它负责处理用户的自然语言请求，通过规则匹配和 LLM 理解用户意图，执行相应操作（如创建日程、查询、删除等），并管理对话状态和确认流程。
from sqlalchemy.ext.asyncio import AsyncSession
from structlog import get_logger

from app.agents.executor import AgentToolExecutor
from app.agents.llm_parser import AgentLLMParser
from app.agents.llm_planner import AgentLLMPlanner
from app.agents.llm_response_writer import AgentLLMResponseWriter
from app.agents.llm_suggestion_writer import AgentLLMSuggestionWriter
from app.agents.parser import AgentRuleParser
from app.agents.prompts import SYSTEM_PROMPT
from app.core.config import settings
from app.schemas.agent import (
    AgentExecuteResponse,
    AgentNaturalLanguageRequest,
    AgentStateResponse,
    AgentToolRequest,
    AgentToolResult,
)
from app.services.conversation_history_service import ConversationHistoryService
from app.services.conversation_service import ConversationService

logger = get_logger()
'''
AgentOrchestrator
├── 初始化
│   ├── AgentToolExecutor（工具执行器）
│   ├── AgentRuleParser（规则解析器）
│   ├── AgentLLMPlanner（LLM 规划器）
│   ├── AgentLLMParser（LLM 解析器）
│   ├── AgentLLMResponseWriter（响应生成器）
│   ├── AgentLLMSuggestionWriter（建议生成器）
│   ├── ConversationHistoryService（对话历史服务）
│   └── ConversationService（会话状态服务）
├── 核心入口
│   └── execute_natural_language() → 处理自然语言请求
├── 工具执行
│   └── execute() → 直接执行工具调用
├── 私有方法
│   ├── _handle_confirmed_request() → 处理已确认请求
│   ├── _handle_pending_confirmation_follow_up() → 处理待确认跟进
│   ├── _run_llm_planner() → 运行 LLM 规划
│   ├── _record_conversation_history() → 记录对话历史
│   ├── _resolve_tool_name() → 解析工具名称
│   ├── _build_fallback_suggestions() → 构建备选建议
│   └── _merge_pending_tool_arguments() → 合并工具参数
└── 配置依赖
    └── settings.llm_debug_logging（LLM 调试日志开关）
    
用户输入
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ execute_natural_language()                                      │
└─────────────────────────┬───────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ 1. 清理过期确认状态                                             │
└─────────────────────────┬───────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. _handle_confirmed_request()                                  │
│    如果 request.confirmed == True                               │
│    → 执行待确认操作 → 返回                                      │
└─────────────────────────┬───────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. _handle_pending_confirmation_follow_up()                     │
│    如果存在待确认状态                                            │
│    ├── 用户确认 → 执行 → 返回                                   │
│    ├── 用户取消 → 取消 → 返回                                   │
│    └── 用户修改 → 更新参数 → 继续流程                            │
└─────────────────────────┬───────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. _parser.parse() - 规则解析                                   │
│    ├── 成功 → 使用规则结果                                       │
│    └── 失败 (unknown/clarify) → 继续                            │
└─────────────────────────┬───────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. _run_llm_planner() - LLM 规划                               │
│    ├── 成功 → 使用规划结果                                       │
│    └── 失败 → 继续                                              │
└─────────────────────────┬───────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. _llm_parser.parse() - LLM 解析（最终兜底）                   │
│    └── 成功 → 使用解析结果                                       │
└─────────────────────────┬───────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. 判断 agent_state                                             │
│    ├── clarify → 生成建议 → 返回（不执行）                      │
│    ├── confirm → 保存确认状态 → 返回（待用户确认）               │
│    └── execute → 执行工具                                       │
└─────────────────────────┬───────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ 8. _executor.execute() - 执行工具                               │
│    ├── schedule_create → 创建日程                               │
│    ├── schedule_list → 查询日程                                 │
│    ├── schedule_update → 更新日程                               │
│    └── schedule_delete → 删除日程                               │
└─────────────────────────┬───────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ 9. _llm_response_writer.rewrite_reply() - 生成回复              │
│    └── 生成友好的自然语言回复                                    │
└─────────────────────────┬───────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ 10. _record_conversation_history() - 记录对话历史                │
└─────────────────────────┬───────────────────────────────────────┘
                          ↓
                    返回 AgentStateResponse
'''

class AgentOrchestrator:
    def __init__(self, session: AsyncSession) -> None:
        self._executor = AgentToolExecutor(session)
        self._parser = AgentRuleParser()
        self._llm_planner = AgentLLMPlanner.from_session(session)
        self._llm_parser = AgentLLMParser.from_session(session)
        self._llm_response_writer = AgentLLMResponseWriter()
        self._llm_suggestion_writer = AgentLLMSuggestionWriter()
        self._conversation_history_service = ConversationHistoryService.from_session(session)
        self._conversation_service = ConversationService.from_session(session)

    async def execute(self, request: AgentToolRequest) -> AgentExecuteResponse:
        try:
            # 直接执行工具调用（不需要自然语言解析）
            result = await self._executor.execute(request.tool_name, **request.arguments)
            tool_result = AgentToolResult(
                success=True,
                tool_name=request.tool_name,
                data=result,
                error=None,
            )
            return AgentExecuteResponse(success=True, system_prompt=SYSTEM_PROMPT, tool_result=tool_result)
        except Exception as exc:  # pragma: no cover
            tool_result = AgentToolResult(
                success=False,
                tool_name=request.tool_name,
                data=None,
                error=str(exc),
            )
            return AgentExecuteResponse(success=False, system_prompt=SYSTEM_PROMPT, tool_result=tool_result)

    # 处理用户的自然语言请求
    async def execute_natural_language(
        self,
        request: AgentNaturalLanguageRequest,
    ) -> AgentStateResponse:
        # 清理过期确认：每次处理前清理过期的确认状态
        await self._conversation_service.clear_expired_confirmations()

        # 1.处理已确认请求
        confirmed_result = await self._handle_confirmed_request(request)
        if confirmed_result is not None:
            await self._record_conversation_history(request, confirmed_result)
            return confirmed_result

        # 2.处理待确认跟进
        pending_confirmation_result = await self._handle_pending_confirmation_follow_up(request)
        if pending_confirmation_result is not None:
            await self._record_conversation_history(request, pending_confirmation_result)
            return pending_confirmation_result

        # 规则解析
        state = self._parser.parse(request.user_input, confirmed=False)
        state.parser_source = "rule"
        if settings.llm_debug_logging:
            logger.info(
                "agent_rule_parse_completed",
                session_id=request.session_id,
                agent_state=state.agent_state,
                intent=state.intent,
                parser_source=state.parser_source,
            )

        # LLM 规划（规则失败时）
        # 触发条件：规则解析失败（unknown）或需要澄清
        if state.intent == "unknown" or state.agent_state == "clarify":
            planned_state = await self._run_llm_planner(request)
            if planned_state is not None:
                state = planned_state
            elif settings.llm_debug_logging:
                logger.info(
                    "agent_llm_planner_returned_none",
                    session_id=request.session_id,
                )

        # LLM 解析（LLM 规划也失败时）
        if state.intent == "unknown" or state.agent_state == "clarify":
            if settings.llm_debug_logging:
                logger.info(
                    "agent_llm_fallback_attempted",
                    session_id=request.session_id,
                    user_input=request.user_input,
                    rule_agent_state=state.agent_state,
                    rule_intent=state.intent,
                )
            llm_state = await self._llm_parser.parse(
                session_id=request.session_id,
                user_input=request.user_input,
                context=request.context,
            )
            if llm_state is not None:
                if settings.llm_debug_logging:
                    logger.info(
                        "agent_llm_fallback_succeeded",
                        session_id=request.session_id,
                        agent_state=llm_state.agent_state,
                        intent=llm_state.intent,
                        tool_name=llm_state.tool_name,
                    )
                state = llm_state
            elif settings.llm_debug_logging:
                logger.info(
                    "agent_llm_fallback_returned_none",
                    session_id=request.session_id,
                )
        # 多层降级：规则 → LLM 规划 → LLM 解析

        # 处理非执行状态
        if state.agent_state in {"confirm", "clarify"}:
            # 生成友好回复
            state.user_message = await self._llm_response_writer.rewrite_message(
                intent=state.intent,
                agent_state=state.agent_state,
                tool_name=state.tool_name,
                tool_arguments=state.tool_arguments,
                execution_result=None,
                fallback_message=state.user_message,
            )

        # 生成建议
        if state.agent_state == "clarify":
            state.suggested_inputs = await self._llm_suggestion_writer.generate_suggestions(
                intent=state.intent,
                agent_state=state.agent_state,
                user_input=request.user_input,
                missing_fields=state.missing_fields,
                fallback_suggestions=self._build_fallback_suggestions(state.missing_fields),
            )

        # 保存确认状态：需要用户确认的操作存入数据库
        if state.agent_state == "confirm" and state.tool_name is not None:
            await self._conversation_service.save_confirmation(
                session_id=request.session_id,
                intent=state.intent,
                agent_state=state.agent_state,
                tool_name=state.tool_name,
                tool_arguments=state.tool_arguments,
                user_message=state.user_message,
            )

        # 非执行状态返回：不需要执行操作，直接返回
        if state.agent_state != "execute" or state.tool_name is None:
            await self._record_conversation_history(request, state)
            return state

        # 执行工具
        result = await self._executor.execute(state.tool_name, **state.tool_arguments)
        # 生成回复：根据执行结果生成回复消息
        user_message = await self._llm_response_writer.rewrite_reply(
            intent=state.intent,
            agent_state="reply",
            tool_name=state.tool_name,
            tool_arguments=state.tool_arguments,
            execution_result=result,
            fallback_message="操作已完成。",
        )
        response = AgentStateResponse(
            agent_state="reply",
            intent=state.intent,
            user_message=user_message,
            parser_source=state.parser_source,
            tool_name=state.tool_name,
            tool_arguments=state.tool_arguments,
            execution_result=result,
            target_id=state.target_id,
        )
        # 返回响应：记录历史并返回
        await self._record_conversation_history(request, response)
        return response


    # 处理已确认请求
    async def _handle_confirmed_request(
        self,
        request: AgentNaturalLanguageRequest,
    ) -> AgentStateResponse | None:
        if not request.confirmed:
            return None

        stored = await self._conversation_service.get_confirmation(request.session_id)
        if stored is None or stored.tool_name is None or stored.tool_arguments_json is None:
            return None

        tool_arguments = json.loads(stored.tool_arguments_json)
        result = await self._executor.execute(stored.tool_name, **tool_arguments)
        # 执行完成后，清除存储中的确认记录，防止重复执行
        await self._conversation_service.clear_confirmation(request.session_id)
        # 调用 LLM 改写器，把工具执行结果转换成自然语言回复
        user_message = await self._llm_response_writer.rewrite_reply(
            intent=stored.intent,
            agent_state="reply",
            tool_name=stored.tool_name,
            tool_arguments=tool_arguments,
            execution_result=result,
            fallback_message="操作已完成。",
        )
        return AgentStateResponse(
            agent_state="reply",
            intent=stored.intent,
            user_message=user_message,
            parser_source="conversation",
            tool_name=stored.tool_name,
            tool_arguments=tool_arguments,
            execution_result=result,
        )

    # 处理待确认跟进
    async def _handle_pending_confirmation_follow_up(
        self,
        request: AgentNaturalLanguageRequest,
    ) -> AgentStateResponse | None:
        # 查询会话存储中暂存的"待确认操作"
        existing = await self._conversation_service.get_confirmation(request.session_id)
        if existing is None or existing.tool_name is None or existing.tool_arguments_json is None:
            return None

        tool_arguments = json.loads(existing.tool_arguments_json)
        # 把用户输入标准化（去空格、转小写），方便关键词匹配
        normalized_input = request.user_input.strip().lower()

        # 关键词匹配 肯定确认
        if normalized_input in {"确认", "好的", "确定", "是的", "yes", "ok", "okay"}:
            # 执行 → 清除 → 改写 → 返回
            result = await self._executor.execute(existing.tool_name, **tool_arguments)
            await self._conversation_service.clear_confirmation(request.session_id)
            user_message = await self._llm_response_writer.rewrite_reply(
                intent=existing.intent,
                agent_state="reply",
                tool_name=existing.tool_name,
                tool_arguments=tool_arguments,
                execution_result=result,
                fallback_message="操作已完成。",
            )
            return AgentStateResponse(
                agent_state="reply",
                intent=existing.intent,
                user_message=user_message,
                parser_source="conversation",
                tool_name=existing.tool_name,
                tool_arguments=tool_arguments,
                execution_result=result,
            )

        # 关键词匹配 否定取消
        if normalized_input in {"取消", "不用了", "算了", "cancel", "no"}:
            await self._conversation_service.clear_confirmation(request.session_id)
            return AgentStateResponse(
                agent_state="reply",
                intent="unknown",
                user_message="好的，已取消这次待确认的日程操作。",
                parser_source="conversation",
                tool_name="conversation_cancel",
                tool_arguments={},
            )

        # 复杂情况：交给 LLM 解析
        llm_state = await self._llm_parser.parse_pending_confirmation(
            session_id=request.session_id,
            user_input=request.user_input,
            pending_tool_name=existing.tool_name,
            pending_tool_arguments=tool_arguments,
            pending_intent=existing.intent,
            context=request.context,
        )
        if llm_state is None:
            return None
        # 用户想取消  LLM 判断用户想取消 → 清除记录 → 返回取消状态
        if llm_state.tool_name == "conversation_cancel":
            await self._conversation_service.clear_confirmation(request.session_id)
            return llm_state
        # 用户想执行操作
        if llm_state.agent_state == "execute" and llm_state.tool_name is not None:
            merged_arguments = self._merge_pending_tool_arguments(
                original=tool_arguments,
                updates=llm_state.tool_arguments,
            )
            result = await self._executor.execute(llm_state.tool_name, **merged_arguments)
            await self._conversation_service.clear_confirmation(request.session_id)
            user_message = await self._llm_response_writer.rewrite_reply(
                intent=existing.intent,
                agent_state="reply",
                tool_name=llm_state.tool_name,
                tool_arguments=merged_arguments,
                execution_result=result,
                fallback_message="操作已完成。",
            )
            return AgentStateResponse(
                agent_state="reply",
                intent=existing.intent,
                user_message=user_message,
                parser_source=llm_state.parser_source,
                tool_name=llm_state.tool_name,
                tool_arguments=merged_arguments,
                execution_result=result,
                target_id=llm_state.target_id,
            )

        # 用户需要进一步确认或澄清
        if llm_state.agent_state in {"confirm", "clarify"} and llm_state.tool_name is not None:
            merged_arguments = self._merge_pending_tool_arguments(
                original=tool_arguments,
                updates=llm_state.tool_arguments,
            )
            response = AgentStateResponse(
                agent_state=llm_state.agent_state,
                intent=existing.intent,
                user_message=llm_state.user_message,
                parser_source=llm_state.parser_source,
                tool_name=llm_state.tool_name,
                tool_arguments=merged_arguments,
                missing_fields=llm_state.missing_fields,
                target_id=llm_state.target_id,
            )
            if response.agent_state == "confirm":
                await self._conversation_service.save_confirmation(
                    session_id=request.session_id,
                    intent=existing.intent,
                    agent_state=response.agent_state,
                    tool_name=response.tool_name,
                    tool_arguments=response.tool_arguments,
                    user_message=response.user_message,
                )
            return response

        # 其他情况
        return llm_state

    # LLM 规划
    async def _run_llm_planner(
        self,
        request: AgentNaturalLanguageRequest,
    ) -> AgentStateResponse | None:
        # 调用 LLM 生成计划
        # 用户说了什么（user_input）
        # 会话上下文（context，比如历史对话）
        plan = await self._llm_planner.plan(
            session_id=request.session_id,
            user_input=request.user_input,
            context=request.context,
        )
        if plan is None:
            return None

        if settings.llm_debug_logging:
            logger.info(
                "agent_llm_plan_succeeded",
                session_id=request.session_id,
                complexity=plan.complexity,
                intent=plan.intent,
                action=plan.action,
            )

        if plan.intent == "unknown":
            return None

        # 准备响应数据
        extracted = dict(plan.extracted)
        tool_name = self._resolve_tool_name(plan.intent, extracted)
        missing_fields = list(plan.missing_fields)
        # 需要澄清（action == "clarify"）
        if plan.action == "clarify":
            return AgentStateResponse(
                agent_state="clarify",
                intent=plan.intent,
                user_message="请补充一下缺少的信息。",
                parser_source="llm_plan",
                tool_name=tool_name,
                tool_arguments=extracted,
                missing_fields=missing_fields,
            )
        # 需要确认（action == "confirm"）
        if plan.action == "confirm":
            return AgentStateResponse(
                agent_state="confirm",
                intent=plan.intent,
                user_message="我准备帮您处理这条日程，请确认。",
                parser_source="llm_plan",
                tool_name=tool_name,
                tool_arguments=extracted,
                missing_fields=missing_fields,
            )
        # 直接执行（action == "execute"）
        if plan.action == "execute" and tool_name is not None:
            return AgentStateResponse(
                agent_state="execute",
                intent=plan.intent,
                user_message="正在处理。",
                parser_source="llm_plan",
                tool_name=tool_name,
                tool_arguments=extracted,
                missing_fields=missing_fields,
            )
        # 默认回复（action == "reply" 或其他）
        return AgentStateResponse(
            agent_state="reply",
            intent=plan.intent,
            user_message="好的。",
            parser_source="llm_plan",
            tool_name=tool_name,
            tool_arguments=extracted,
            missing_fields=missing_fields,
        )

    # 解析工具名称
    @staticmethod
    def _resolve_tool_name(intent: str, extracted: dict[str, object]) -> str | None:
        if intent == "create":
            return "schedule_create"
        if intent == "query":
            return "schedule_list"
        if intent == "update":
            return "schedule_update"
        if intent == "delete":
            return "schedule_delete"
        tool_name = extracted.get("tool_name")
        # 兜底逻辑：从 extracted 中取
        return tool_name if isinstance(tool_name, str) else None

    # 把每一次用户请求和系统响应的完整信息，保存到会话历史服务中，用于后续的上下文理解、调试、分析和优化。
    async def _record_conversation_history(
        self,
        request: AgentNaturalLanguageRequest,
        response: AgentStateResponse,
    ) -> None:
        await self._conversation_history_service.record(
            session_id=request.session_id,
            user_input=request.user_input,
            confirmed=request.confirmed,
            context=request.context,
            parser_source=response.parser_source,
            intent=response.intent,
            agent_state=response.agent_state,
            tool_name=response.tool_name,
            tool_arguments=response.tool_arguments,
            missing_fields=response.missing_fields,
            suggested_inputs=response.suggested_inputs,
            target_id=response.target_id,
            execution_result=response.execution_result,
            user_message=response.user_message,
        )

    # 当系统需要用户补充信息时，根据缺少的字段类型，返回一组示例输入供用户参考。
    @staticmethod
    def _build_fallback_suggestions(missing_fields: list[str]) -> list[str]:
        if "schedule_date" in missing_fields and "schedule_time" in missing_fields:
            return [
                "明天下午5点提醒我开会",
                "后天早上9点提醒我学习",
                "2026-06-20 18:00 提醒我运动",
            ]
        if "schedule_time" in missing_fields:
            return [
                "明天上午9点提醒我开会",
                "后天下午3点提醒我学习",
            ]
        if "target_id" in missing_fields:
            return [
                "取消日程 12",
                "删除日程 5",
            ]
        return [
            "明天下午5点提醒我开会",
            "后天早上9点提醒我学习",
        ]

    # 合并工具参数
    # 把用户新提供的参数（updates）合并到原始参数（original）中，同时把日期时间对象转成字符串格式。
    @staticmethod
    def _merge_pending_tool_arguments(
        *,
        original: dict[str, object],
        updates: dict[str, object],
    ) -> dict[str, object]:
        merged = dict(original)
        for key, value in updates.items():
            if value is not None:
                merged[key] = value

        for key in ("schedule_date", "schedule_time"):
            value = merged.get(key)
            if isinstance(value, (date, datetime, time)):
                merged[key] = value.isoformat()
        return merged
