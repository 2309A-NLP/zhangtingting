import json
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.exceptions import NotFoundError
from app.repositories.conversation_repository import ConversationRepository
from app.schemas.conversation import (
    ConversationSessionList,
    ConversationSessionView,
    ConversationStateCreate,
    ConversationStateRead,
)

'''
ConversationService
├── 初始化（依赖注入）
│   └── ConversationRepository（数据访问层）
├── 工厂方法
│   └── from_session() → 从数据库会话创建 Service
├── 会话状态管理
│   ├── save_confirmation() → 保存/更新会话确认状态（带10分钟过期）
│   ├── get_confirmation() → 获取单个确认状态（可能为空）
│   └── clear_confirmation() → 清除确认状态（软删除/硬删除）
├── 会话查询
│   ├── get_session() → 获取单个会话详情（不存在则抛 NotFoundError）
│   ├── list_sessions() → 会话列表（分页 + 过期过滤）
│   └── export_sessions() → 导出全部会话
├── 清理维护
│   └── clear_expired_confirmations() → 批量清理过期确认状态
├── 私有方法
│   └── _build_session_view() → 构建会话视图对象（计算 is_expired + has_pending_confirmation）
├── FastAPI 依赖注入
│   └── get_conversation_service() → 从会话创建 Service 实例
└── 数据模型
    ├── ConversationStateCreate（创建/更新载荷）
    ├── ConversationStateRead（读取模型）
    ├── ConversationSessionView（视图模型）
    │   ├── is_expired（是否过期）
    │   └── has_pending_confirmation（是否有待确认操作）
    └── ConversationSessionList（列表响应）
'''

class ConversationService:
    def __init__(self, repository: ConversationRepository) -> None:
        self._repository = repository

    @classmethod
    def from_session(cls, session: AsyncSession) -> "ConversationService":
        return cls(ConversationRepository(session))

    # 保存确认状态  agent 解析出来的“待确认动作”存起来
    async def save_confirmation(
        self,
        *,
        session_id: str,
        intent: str,
        agent_state: str,
        tool_name: str | None,
        tool_arguments: dict[str, object],
        user_message: str,
    ) -> ConversationStateRead:
        payload = ConversationStateCreate(
            session_id=session_id,
            intent=intent,
            agent_state=agent_state,
            tool_name=tool_name,
            tool_arguments_json=json.dumps(tool_arguments, ensure_ascii=False),
            user_message=user_message,
            expires_at=datetime.now() + timedelta(minutes=10),  # 10 分钟内用户需要确认操作
        )
        # 幂等性 如果 session_id 存在则更新，否则插入
        record = await self._repository.upsert(payload)
        return ConversationStateRead.model_validate(record)

    # 获取确认状态
    async def get_confirmation(self, session_id: str) -> ConversationStateRead | None:
        record = await self._repository.get_by_session_id(session_id)
        if record is None:
            return None
        return ConversationStateRead.model_validate(record)

    # 获取会话详情
    async def get_session(self, session_id: str) -> ConversationSessionView:
        record = await self._repository.get_by_session_id(session_id)
        if record is None:
            raise NotFoundError(f"Conversation session {session_id} not found")
        # 调用私有方法 _build_session_view，计算 is_expired 和 has_pending_confirmation
        return self._build_session_view(ConversationStateRead.model_validate(record))

    # 会话列表
    async def list_sessions(
        self,
        *,
        include_expired: bool = False,
        limit: int = 20,
        offset: int = 0,
    ) -> ConversationSessionList:
        now = datetime.now()
        records, total = await self._repository.list_sessions(
            now=now,
            include_expired=include_expired,
            limit=limit,
            offset=offset,
        )
        items = [
            self._build_session_view(ConversationStateRead.model_validate(record), now=now)
            for record in records
        ]
        return ConversationSessionList(total=total, limit=limit, offset=offset, items=items)

    # 导出全部会话
    async def export_sessions(self, *, include_expired: bool = False) -> list[ConversationSessionView]:
        now = datetime.now()
        records = await self._repository.export_sessions(now=now, include_expired=include_expired)
        return [
            self._build_session_view(ConversationStateRead.model_validate(record), now=now)
            for record in records
        ]

    # 清除确认状态
    async def clear_confirmation(self, session_id: str) -> None:
        deleted = await self._repository.clear(session_id)
        if not deleted:
            raise NotFoundError(f"Conversation session {session_id} not found")

    # 批量清理过期
    async def clear_expired_confirmations(self) -> int:
        return await self._repository.clear_expired(datetime.now())

    # 构建会话视图
    @staticmethod
    def _build_session_view(
        state: ConversationStateRead,
        *,
        now: datetime | None = None,
    ) -> ConversationSessionView:
        reference_time = now or datetime.now()
        tool_arguments: dict[str, object] = {}
        if state.tool_arguments_json:
            tool_arguments = json.loads(state.tool_arguments_json)

        is_expired = state.expires_at is not None and state.expires_at <= reference_time
        # 计算是否有待确认操作：同时满足三个条件
        # agent_state 是 "confirm"（等待用户确认）
        # tool_name 不为 None（有工具需要执行）
        # 未过期（is_expired 为 False）
        has_pending_confirmation = (
            state.agent_state == "confirm"
            and state.tool_name is not None
            and not is_expired
        )

        return ConversationSessionView(
            id=state.id,
            session_id=state.session_id,
            intent=state.intent,
            agent_state=state.agent_state,
            tool_name=state.tool_name,
            tool_arguments=tool_arguments,
            user_message=state.user_message,
            expires_at=state.expires_at,
            created_at=state.created_at,
            updated_at=state.updated_at,
            is_expired=is_expired,
            has_pending_confirmation=has_pending_confirmation,
        )


async def get_conversation_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ConversationService:
    return ConversationService.from_session(session)
