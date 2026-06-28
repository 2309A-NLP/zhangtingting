from datetime import date, time
# 日程工具集，是 Agent 系统直接调用的业务操作层。
# 它接收来自 AgentToolExecutor 的规范化参数，调用 ScheduleService 执行实际的 CRUD 操作，是 Agent 系统与业务服务的桥梁。
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import CycleRule, ScheduleStatus
from app.schemas.schedule import (
    ScheduleCreate,
    ScheduleList,
    ScheduleQuery,
    ScheduleRead,
    ScheduleUpdate,
)
from app.services.schedule_service import ScheduleService
'''
ScheduleAgentTools
├── 初始化
│   └── ScheduleService（日程服务）
├── 工具方法
│   ├── schedule_create() → 创建日程
│   ├── schedule_list() → 查询日程列表
│   ├── schedule_get() → 获取单个日程
│   ├── schedule_update() → 更新日程
│   └── schedule_delete() → 删除日程
└── 数据转换
    ├── CycleRule(cycle_rule) → 字符串转枚举
    └── ScheduleStatus(status) → 字符串转枚举
'''

class ScheduleAgentTools:
    def __init__(self, session: AsyncSession) -> None:
        self._service = ScheduleService.from_session(session)

    async def schedule_create(
        self,
        *,
        content: str,
        schedule_time: time,
        schedule_date: date | None = None,
        cycle_rule: str = "once",
        cycle_value: str | None = None,
        source_text: str | None = None,
    ) -> ScheduleRead:
        payload = ScheduleCreate(
            content=content,
            schedule_date=schedule_date,
            schedule_time=schedule_time,
            cycle_rule=CycleRule(cycle_rule),
            cycle_value=cycle_value,
            source_text=source_text,
        )
        return await self._service.create(payload)

    async def schedule_list(
        self,
        *,
        date_value: date | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        status: str | None = None,
    ) -> ScheduleList:
        query = ScheduleQuery(
            date=date_value,
            start_date=start_date,
            end_date=end_date,
            status=ScheduleStatus(status) if status else None,
        )
        return await self._service.list_all(query)

    async def schedule_get(self, *, schedule_id: int) -> ScheduleRead:
        return await self._service.get_by_id(schedule_id)

    async def schedule_update(
        self,
        *,
        schedule_id: int,
        content: str | None = None,
        schedule_date: date | None = None,
        schedule_time: time | None = None,
        cycle_rule: str | None = None,
        cycle_value: str | None = None,
        source_text: str | None = None,
        status: str | None = None,
    ) -> ScheduleRead:
        payload = ScheduleUpdate(
            content=content,
            schedule_date=schedule_date,
            schedule_time=schedule_time,
            cycle_rule=CycleRule(cycle_rule) if cycle_rule else None,
            cycle_value=cycle_value,
            source_text=source_text,
            status=ScheduleStatus(status) if status else None,
        )
        return await self._service.update(schedule_id, payload)

    async def schedule_delete(self, *, schedule_id: int) -> dict[str, int]:
        await self._service.delete(schedule_id)
        return {"id": schedule_id}
