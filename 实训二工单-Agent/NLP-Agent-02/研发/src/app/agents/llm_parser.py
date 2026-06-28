import json
import re
from datetime import date
from typing import Any
# LLM 解析器，是 Agent 系统的 AI 大脑。
# 当规则解析器无法理解用户输入时，它调用大语言模型来解析用户的自然语言请求，提取意图、参数和操作。
# 它还具备自动修复能力，当 LLM 返回格式错误的 JSON 时，会再次调用 LLM 来修复。
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from structlog import get_logger

from app.agents.llm_prompts import (
    LLM_FALLBACK_SYSTEM_PROMPT,
    LLM_JSON_REPAIR_SYSTEM_PROMPT,
    LLM_PENDING_CONFIRMATION_SYSTEM_PROMPT,
)
from app.agents.llm_schemas import LLMParseResponse
from app.core.config import settings
from app.llm import LLMClient, get_llm_client
from app.schemas.agent import AgentStateResponse
from app.services.llm_audit_service import LLMAuditLogService

logger = get_logger()

'''
AgentLLMParser
├── 初始化（依赖注入）
│   ├── LLMClient（LLM 客户端）
│   └── LLMAuditLogService（审计日志服务）
├── 工厂方法
│   └── from_session() → 从数据库会话创建 Parser
├── 核心方法
│   ├── parse() → 解析用户输入（最终兜底）
│   └── parse_pending_confirmation() → 解析待确认场景的跟进输入
├── JSON 处理
│   ├── _extract_json_from_text() → 从 LLM 响应中提取 JSON
│   └── _repair_invalid_response() → 修复无效的 JSON 响应
├── 参数规范化
│   └── _normalize_tool_arguments() → 统一工具参数格式
├── 审计记录
│   └── _record_audit() → 记录所有 LLM 调用
└── 配置依赖
    └── settings.llm_debug_logging（LLM 调试日志开关）
'''

class AgentLLMParser:
    def __init__(
        self,
        client: LLMClient | None = None,
        audit_service: LLMAuditLogService | None = None,
    ) -> None:
        self._client = client or get_llm_client()
        self._audit_service = audit_service

    @classmethod
    def from_session(cls, session: AsyncSession, client: LLMClient | None = None) -> "AgentLLMParser":
        return cls(
            client=client,
            audit_service=LLMAuditLogService.from_session(session),
        )

    # 核心解析方法
    async def parse(
        self,
        *,
        session_id: str,
        user_input: str,
        context: dict[str, Any],
    ) -> AgentStateResponse | None:
        if self._client is None:
            if settings.llm_debug_logging:
                logger.info("llm_parser_skipped_no_client")
            return None

        user_prompt = json.dumps(
            {
                "today": date.today().isoformat(),
                "user_input": user_input,
                "context": context,
            },
            ensure_ascii=False,
        )

        # LLM 调用与响应处理
        raw_content: str | None = None
        payload: dict[str, Any] | None = None
        try:
            if settings.llm_debug_logging:
                logger.info(
                    "llm_parse_started",
                    user_input=user_input,
                    context_keys=sorted(context.keys()),
                )
            raw_content = await self._client.chat_text(
                system_prompt=LLM_FALLBACK_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            if settings.llm_debug_logging:
                logger.info(
                    "llm_parse_raw_content_received",
                    content_length=len(raw_content),
                    preview=raw_content[:300],
                )
            payload = self._extract_json_from_text(raw_content)
            parsed = LLMParseResponse.model_validate(payload)
        except (ValidationError, ValueError, KeyError, TypeError) as exc:
            logger.warning("llm_parse_invalid_response", error=str(exc))
            await self._record_audit(
                session_id=session_id,
                user_input=user_input,
                parser_stage="parse",
                success=False,
                request_payload=json.loads(user_prompt),
                raw_response_text=raw_content,
                parsed_response=payload,
                error_message=str(exc),
            )
            # 尝试调用 LLM 修复（_repair_invalid_response）
            repaired = await self._repair_invalid_response(
                session_id=session_id,
                user_input=user_input,
                raw_content=raw_content,
            )
            if repaired is None:
                return None
            parsed = repaired
        except Exception as exc:  # pragma: no cover
            logger.warning("llm_parse_failed", error=str(exc))
            await self._record_audit(
                session_id=session_id,
                user_input=user_input,
                parser_stage="parse",
                success=False,
                request_payload=json.loads(user_prompt),
                raw_response_text=raw_content,
                parsed_response=payload,
                error_message=str(exc),
            )
            return None

        if settings.llm_debug_logging:
            logger.info(
                "llm_parse_succeeded",
                agent_state=parsed.agent_state,
                intent=parsed.intent,
                tool_name=parsed.tool_name,
            )
        await self._record_audit(
            session_id=session_id,
            user_input=user_input,
            parser_stage="parse",
            success=True,
            request_payload=json.loads(user_prompt),
            raw_response_text=raw_content,
            parsed_response=parsed.model_dump(),
            error_message=None,
        )

        # 规范化参数：统一不同工具的参数格式
        tool_arguments = self._normalize_tool_arguments(
            tool_name=parsed.tool_name,
            tool_arguments=parsed.tool_arguments,
            source_text=user_input,
        )

        return AgentStateResponse(
            agent_state=parsed.agent_state,
            intent=parsed.intent,
            user_message=parsed.user_message,
            parser_source="llm",
            tool_name=parsed.tool_name,
            tool_arguments=tool_arguments,
            missing_fields=parsed.missing_fields,
            target_id=parsed.target_id,
        )

    # 处理待确认场景
    async def parse_pending_confirmation(
        self,
        *,
        session_id: str,
        user_input: str,
        pending_tool_name: str,
        pending_tool_arguments: dict[str, Any],
        pending_intent: str,
        context: dict[str, Any],
    ) -> AgentStateResponse | None:
        if self._client is None:
            return None

        user_prompt = json.dumps(
            {
                "session_id": session_id,
                "user_input": user_input,
                "pending_intent": pending_intent,
                "pending_tool_name": pending_tool_name,
                "pending_tool_arguments": pending_tool_arguments,
                "context": context,
            },
            ensure_ascii=False,
        )

        raw_content: str | None = None
        payload: dict[str, Any] | None = None
        try:
            raw_content = await self._client.chat_text(
                system_prompt=LLM_PENDING_CONFIRMATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            payload = self._extract_json_from_text(raw_content)
            parsed = LLMParseResponse.model_validate(payload)
        except (ValidationError, ValueError, KeyError, TypeError) as exc:
            logger.warning("llm_pending_confirmation_invalid_response", error=str(exc))
            await self._record_audit(
                session_id=session_id,
                user_input=user_input,
                parser_stage="pending_confirmation",
                success=False,
                request_payload=json.loads(user_prompt),
                raw_response_text=raw_content,
                parsed_response=payload,
                error_message=str(exc),
            )
            return None
        except Exception as exc:  # pragma: no cover
            logger.warning("llm_pending_confirmation_failed", error=str(exc))
            await self._record_audit(
                session_id=session_id,
                user_input=user_input,
                parser_stage="pending_confirmation",
                success=False,
                request_payload=json.loads(user_prompt),
                raw_response_text=raw_content,
                parsed_response=payload,
                error_message=str(exc),
            )
            return None

        await self._record_audit(
            session_id=session_id,
            user_input=user_input,
            parser_stage="pending_confirmation",
            success=True,
            request_payload=json.loads(user_prompt),
            raw_response_text=raw_content,
            parsed_response=parsed.model_dump(),
            error_message=None,
        )

        tool_arguments = self._normalize_tool_arguments(
            tool_name=parsed.tool_name,
            tool_arguments=parsed.tool_arguments,
            source_text=user_input,
        )

        return AgentStateResponse(
            agent_state=parsed.agent_state,
            intent=parsed.intent,
            user_message=parsed.user_message,
            parser_source="llm_pending_confirmation",
            tool_name=parsed.tool_name,
            tool_arguments=tool_arguments,
            missing_fields=parsed.missing_fields,
            target_id=parsed.target_id,
        )

    @staticmethod
    def _normalize_tool_arguments(
        *,
        tool_name: str | None,
        tool_arguments: dict[str, Any],
        source_text: str,
    ) -> dict[str, Any]:
        normalized = dict(tool_arguments)

        if tool_name == "schedule_list" and "date" in normalized and "date_value" not in normalized:
            normalized["date_value"] = normalized.pop("date")

        if tool_name in {"schedule_get", "schedule_update", "schedule_delete"}:
            if "id" in normalized and "schedule_id" not in normalized:
                normalized["schedule_id"] = normalized.pop("id")

        if tool_name == "schedule_create":
            normalized.setdefault("cycle_rule", "once")
            normalized.setdefault("cycle_value", None)
            normalized.setdefault("source_text", source_text)

        if tool_name == "schedule_update":
            normalized.setdefault("source_text", source_text)

        return normalized

    # 提取 JSON
    @staticmethod
    def _extract_json_from_text(content: str) -> dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            # 移除开头的代码块标记
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            # 移除结尾的代码块标记
            cleaned = re.sub(r"\s*```$", "", cleaned)

        # 提取 JSON 内容  从第一个 { 开始，到最后一个 } 结束（贪婪匹配） 能处理多行 JSON
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match is None:
            raise ValueError("LLM response does not contain JSON")
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("LLM response JSON must be an object")
        return parsed

    # 当 LLM 返回的原始内容解析失败时，用另一个 LLM 来"修复"或"重新生成"合法的 JSON 响应。
    async def _repair_invalid_response(
        self,
        *,
        session_id: str,
        user_input: str,
        raw_content: str | None,
    ) -> LLMParseResponse | None:
        if self._client is None or raw_content is None:
            if settings.llm_debug_logging:
                logger.info(
                    "llm_repair_skipped",
                    has_client=self._client is not None,
                    has_raw_content=raw_content is not None,
                )
            return None

        repair_prompt = json.dumps(
            {
                "user_input": user_input,
                "raw_model_output": raw_content,
            },
            ensure_ascii=False,
        )

        try:
            if settings.llm_debug_logging:
                logger.info(
                    "llm_repair_started",
                    raw_content_length=len(raw_content),
                    raw_preview=raw_content[:300],
                )
            repaired_content = await self._client.chat_text(
                system_prompt=LLM_JSON_REPAIR_SYSTEM_PROMPT,
                user_prompt=repair_prompt,
                use_json_mode=True,
            )
            if settings.llm_debug_logging:
                logger.info(
                    "llm_repair_content_received",
                    content_length=len(repaired_content),
                    preview=repaired_content[:300],
                )
            repaired_payload = self._extract_json_from_text(repaired_content)
            parsed = LLMParseResponse.model_validate(repaired_payload)
            if settings.llm_debug_logging:
                logger.info(
                    "llm_repair_succeeded",
                    agent_state=parsed.agent_state,
                    intent=parsed.intent,
                    tool_name=parsed.tool_name,
                )
            await self._record_audit(
                session_id=session_id,
                user_input=user_input,
                parser_stage="repair",
                success=True,
                request_payload=json.loads(repair_prompt),
                raw_response_text=repaired_content,
                parsed_response=parsed.model_dump(),
                error_message=None,
            )
            return parsed
        except (ValidationError, ValueError, KeyError, TypeError) as exc:
            logger.warning("llm_repair_invalid_response", error=str(exc))
            await self._record_audit(
                session_id=session_id,
                user_input=user_input,
                parser_stage="repair",
                success=False,
                request_payload=json.loads(repair_prompt),
                raw_response_text=locals().get("repaired_content"),
                parsed_response=None,
                error_message=str(exc),
            )
            return None
        except Exception as exc:  # pragma: no cover
            logger.warning("llm_repair_failed", error=str(exc))
            await self._record_audit(
                session_id=session_id,
                user_input=user_input,
                parser_stage="repair",
                success=False,
                request_payload=json.loads(repair_prompt),
                raw_response_text=locals().get("repaired_content"),
                parsed_response=None,
                error_message=str(exc),
            )
            return None

    # 记录 LLM 解析过程的审计日志
    async def _record_audit(
        self,
        *,
        session_id: str,
        user_input: str,
        parser_stage: str,
        success: bool,
        request_payload: dict[str, Any] | None,
        raw_response_text: str | None,
        parsed_response: dict[str, Any] | None,
        error_message: str | None,
    ) -> None:
        if self._audit_service is None:
            return
        provider = self._client.provider_name if self._client is not None else None
        model_name = self._client.model_name if self._client is not None else None
        await self._audit_service.record(
            session_id=session_id,
            user_input=user_input,
            parser_stage=parser_stage,
            success=success,
            provider=provider,
            model_name=model_name,
            request_payload=request_payload,
            raw_response_text=raw_response_text,
            parsed_response=parsed_response,
            error_message=error_message,
        )


