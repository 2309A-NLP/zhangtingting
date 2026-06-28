from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any
# 工具执行器，是 Agent 系统的"手"。
# 它负责接收解析后的工具调用请求，查找对应的工具函数，规范化参数，然后执行实际的业务操作。
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.tools import ScheduleAgentTools
'''
AgentToolExecutor
├── 初始化
│   ├── ScheduleAgentTools（日程工具集）
│   └── _registry（工具注册表）
│       ├── schedule_create → 创建日程
│       ├── schedule_list → 查询日程
│       ├── schedule_get → 获取单个日程
│       ├── schedule_update → 更新日程
│       └── schedule_delete → 删除日程
├── 核心方法
│   └── execute() → 执行工具调用
├── 参数规范化
│   ├── _normalize_tool_kwargs() → 统一参数格式
│   └── _parse_legacy_datetime() → 解析旧版日期时间格式
└── 工具依赖
    └── ScheduleAgentTools（实际的工具实现）
'''

class AgentToolExecutor:
    def __init__(self, session: AsyncSession) -> None:
        self._tools = ScheduleAgentTools(session)
        # Callable[..., Awaitable[object]]：接受任意参数，返回 Awaitable 的可调用对象
        # 即：所有工具函数都是 async def 返回 object
        self._registry: dict[str, Callable[..., Awaitable[object]]] = {
            "schedule_create": self._tools.schedule_create,
            "schedule_list": self._tools.schedule_list,
            "schedule_get": self._tools.schedule_get,
            "schedule_update": self._tools.schedule_update,
            "schedule_delete": self._tools.schedule_delete,
        }

    # execute（执行工具）
    async def execute(self, tool_name: str, **kwargs: Any) -> object:
        tool = self._registry.get(tool_name)
        if tool is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        normalized_kwargs = self._normalize_tool_kwargs(tool_name, kwargs)
        # 调用实际的工具函数并返回结果
        return await tool(**normalized_kwargs)

    @staticmethod
    def _normalize_tool_kwargs(tool_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(kwargs)

        if tool_name == "schedule_create":
            if "date" in normalized and "schedule_date" not in normalized:
                normalized["schedule_date"] = normalized.pop("date")
            legacy_datetime = normalized.pop("datetime", None)
            if legacy_datetime is not None:
                parsed_datetime = AgentToolExecutor._parse_legacy_datetime(legacy_datetime)
                if parsed_datetime is not None:
                    # datetime → schedule_date + schedule_time
                    if "schedule_date" not in normalized:
                        normalized["schedule_date"] = parsed_datetime.date().isoformat()
                    if "schedule_time" not in normalized:
                        normalized["schedule_time"] = parsed_datetime.time().isoformat()
            normalized.pop("date_value", None)

        if tool_name == "schedule_list":
            if "date" in normalized and "date_value" not in normalized:
                normalized["date_value"] = normalized.pop("date")

        if tool_name in {"schedule_get", "schedule_update", "schedule_delete"}:
            if "id" in normalized and "schedule_id" not in normalized:
                normalized["schedule_id"] = normalized.pop("id")

        return normalized

    # 解析旧版日期时间
    @staticmethod
    def _parse_legacy_datetime(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value
        if not isinstance(value, str):
            return None

        normalized_value = value.strip()
        if not normalized_value:
            return None

        if normalized_value.endswith("Z"):
            normalized_value = normalized_value[:-1] + "+00:00"
        '''
        规则： 将 Z（UTC 时区标记）替换为 +00:00（ISO 8601 标准格式）
        为什么需要这个？
        Z 是 ISO 8601 的 UTC 时区简写，但 Python 的 fromisoformat() 不支持 Z
        必须转换成 +00:00 才能解析
        '''

        try:
            return datetime.fromisoformat(normalized_value)
        except ValueError:
            return None
