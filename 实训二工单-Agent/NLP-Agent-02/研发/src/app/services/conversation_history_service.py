import json
from datetime import date, datetime, time
from typing import Annotated, Any

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.repositories.conversation_history_repository import ConversationHistoryRepository
from app.schemas.conversation_history import (
    ConversationHistoryCreate,
    ConversationHistoryList,
    ConversationHistoryQuery,
    ConversationHistoryRead,
)

'''
ConversationHistoryService
├── 初始化（依赖注入）
│   └── ConversationHistoryRepository（数据访问层）
├── 工厂方法
│   └── from_session() → 从数据库会话创建 Service
├── 日志记录
│   └── record() → 记录一条对话历史（19个参数，自动 JSON 序列化）
├── 日志查询
│   ├── list_by_session_id() → 按会话 ID 查询所有日志
│   ├── list_logs() → 多条件筛选列表（分页 + 过滤）
│   └── export_logs() → 多条件筛选导出（无分页限制）
├── 私有工具方法
│   ├── _dump_json() → 安全 JSON 序列化（处理 None 和特殊类型）
│   └── _json_default() → 自定义序列化器（处理 Pydantic 模型、枚举、日期等）
└── FastAPI 依赖注入
    └── get_conversation_history_service() → 从会话创建 Service 实例
'''

class ConversationHistoryService:
    def __init__(self, repository: ConversationHistoryRepository) -> None:
        self._repository = repository

    @classmethod
    def from_session(cls, session: AsyncSession) -> "ConversationHistoryService":
        return cls(ConversationHistoryRepository(session))

    # 记录对话历史
    async def record(
        self,
        *,
        session_id: str,
        user_input: str,
        confirmed: bool,
        context: dict[str, Any],
        parser_source: str | None,
        intent: str,
        agent_state: str,
        tool_name: str | None,
        tool_arguments: dict[str, Any],
        missing_fields: list[str],
        suggested_inputs: list[str],
        target_id: int | None,
        execution_result: Any,
        user_message: str | None,
    ) -> ConversationHistoryRead:
        payload = ConversationHistoryCreate(
            session_id=session_id,
            user_input=user_input,
            confirmed=confirmed,
            context_json=self._dump_json(context),
            parser_source=parser_source,
            intent=intent,
            agent_state=agent_state,
            tool_name=tool_name,
            tool_arguments_json=self._dump_json(tool_arguments),
            missing_fields_json=self._dump_json(missing_fields),
            suggested_inputs_json=self._dump_json(suggested_inputs),
            target_id=target_id,
            execution_result_json=self._dump_json(execution_result),
            user_message=user_message,
        )
        '''
        session_id：会话 ID
        user_input：用户原始输入
        confirmed：是否已确认
        context：上下文信息（如当前页面、用户信息等）
        parser_source：解析器来源（如 "llm", "rule"）
        intent：识别出的意图
        agent_state：Agent 状态
        tool_name：调用的工具名称
        tool_arguments：工具参数
        missing_fields：缺失的字段列表
        suggested_inputs：建议的输入值
        target_id：目标 ID（如订单 ID）
        execution_result：执行结果（任意类型）
        user_message：用户消息
        '''
        record = await self._repository.create(payload)
        return ConversationHistoryRead.model_validate(record)

    # 按会话id查询
    async def list_by_session_id(self, session_id: str) -> list[ConversationHistoryRead]:
        records = await self._repository.list_by_session_id(session_id)
        return [ConversationHistoryRead.model_validate(item) for item in records]

    # 多条件筛选列表
    async def list_logs(
        self,
        *,
        session_id: str | None = None,
        parser_source: str | None = None,
        agent_state: str | None = None,
        intent: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> ConversationHistoryList:
        query = ConversationHistoryQuery(
            session_id=session_id,
            parser_source=parser_source,
            agent_state=agent_state,
            intent=intent,
            limit=limit,
            offset=offset,
        )
        records, total = await self._repository.list_logs(query)
        items = [ConversationHistoryRead.model_validate(item) for item in records]
        return ConversationHistoryList(total=total, limit=limit, offset=offset, items=items)

    # 多条件筛选导出
    async def export_logs(
        self,
        *,
        session_id: str | None = None,
        parser_source: str | None = None,
        agent_state: str | None = None,
        intent: str | None = None,
    ) -> list[ConversationHistoryRead]:
        query = ConversationHistoryQuery(
            session_id=session_id,
            parser_source=parser_source,
            agent_state=agent_state,
            intent=intent,
        )
        records = await self._repository.export_logs(query)
        return [ConversationHistoryRead.model_validate(item) for item in records]

    # 安全 JSON 序列化
    @staticmethod
    def _dump_json(payload: Any) -> str | None:
        if payload is None:
            return None
        # default=ConversationHistoryService._json_default
        # 当 json.dumps() 遇到无法序列化的类型时，会调用这个函数
        return json.dumps(payload, ensure_ascii=False, default=ConversationHistoryService._json_default)

    # 自定义序列化器
    @staticmethod
    def _json_default(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if hasattr(value, "value"):
            return value.value
        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        return str(value)


async def get_conversation_history_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ConversationHistoryService:
    return ConversationHistoryService.from_session(session)

'''
对比	              ConversationService	         ConversationHistoryService
职责	              管理会话确认状态（临时）	            记录对话历史（持久化）
数据特点	          临时数据，有 TTL（10分钟）	        永久数据，只增不删
操作	               upsert（覆盖更新）	                create（只追加）
典型字段	       intent, agent_state, tool_name	 包含上述 + 执行结果、缺失字段、建议输入
过期处理	           有 expires_at + 定时清理	           无过期概念
JSON 序列化	       仅 tool_arguments	         多个字段 JSON 化 + 自定义序列化器
'''