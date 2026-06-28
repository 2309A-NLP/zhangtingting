import re
from datetime import date
from typing import Any
# 规则解析器，是 Agent 系统的第一道防线。
# 它使用正则表达式和关键词匹配快速解析用户输入，提取意图、参数和操作。相比 LLM 解析，它速度快、成本低、可预测。
from app.core.enums import CycleRule
# CycleRule：循环规则枚举
# ONCE：一次性
# INTERVAL_DAYS：间隔天数
# WEEKDAY：工作日
from app.schemas.agent import AgentStateResponse
# 中文日期时间解析器
from app.utils.nlp_datetime import ChineseDateTimeParser

'''
AgentRuleParser
├── 初始化
│   └── ChineseDateTimeParser（中文日期时间解析器）
├── 核心方法
│   └── parse() → 解析用户输入，返回 AgentStateResponse
├── 私有方法
│   └── _parse_create() → 解析"创建日程"相关参数
└── 支持的意图
    ├── query → 查询日程
    │   └── "今天的日程" / "今天有什么安排"
    ├── create → 创建日程
    │   ├── "提醒我..."（普通创建）
    │   ├── "每隔N天..."（间隔循环）
    │   └── "工作日..." / "周一到周五..."（工作日循环）
    ├── delete → 删除日程
    │   └── "取消日程 N" / "删除日程 N"
    └── unknown → 无法识别
        └── 返回友好提示
'''

class AgentRuleParser:
    def __init__(self) -> None:
        self._datetime_parser = ChineseDateTimeParser()

    def parse(self, user_input: str, confirmed: bool = False) -> AgentStateResponse:
        normalized = user_input.strip()
        # 意图1：查询日程（query）
        if any(keyword in normalized for keyword in ["今天的日程", "今天有什么安排"]):
            return AgentStateResponse(
                agent_state="execute",
                intent="query",
                user_message="正在为您查询今天的日程安排。",
                tool_name="schedule_list",
                tool_arguments={
                    "date_value": date.today().isoformat(),
                    "status": "active",
                },
            )

        # 意图2a：间隔循环创建（每隔N天）
        if any(keyword in normalized for keyword in ["每隔", "隔天", "每两天", "每三天"]):
            # 调用 _parse_create() 提取参数
            parsed = self._parse_create(normalized)
            if parsed is None:
                return AgentStateResponse(
                    agent_state="clarify",
                    intent="create",
                    user_message="请补充具体日期和时间，例如：后天下午5点，每隔三天提醒我开会。",
                    missing_fields=["schedule_date", "schedule_time"],
                )

            if not confirmed:
                cycle_label = (
                    f"每隔 {parsed['cycle_value']} 天"
                    if parsed["cycle_rule"] == CycleRule.INTERVAL_DAYS.value and parsed["cycle_value"]
                    else "循环"
                )
                return AgentStateResponse(
                    agent_state="confirm",
                    intent="create",
                    user_message=(
                        f"我准备为您新增一条循环日程：{parsed['schedule_date']} "
                        f"{parsed['schedule_time']} {parsed['content']}，循环规则为{cycle_label}。请确认是否创建？"
                    ),
                    tool_name="schedule_create",
                    tool_arguments=parsed,
                )

            return AgentStateResponse(
                agent_state="execute",
                intent="create",
                user_message="正在为您创建循环日程。",
                tool_name="schedule_create",
                tool_arguments=parsed,
            )

        # 意图2b：工作日循环创建
        if any(keyword in normalized for keyword in ["工作日", "周一到周五", "周一至周五"]):
            parsed = self._parse_create(normalized)
            if parsed is None:
                return AgentStateResponse(
                    agent_state="clarify",
                    intent="create",
                    user_message="请补充具体时间，例如：每周一到周五早上9点提醒我打卡。",
                    missing_fields=["schedule_time"],
                )

            parsed["cycle_rule"] = CycleRule.WEEKDAY.value
            if not confirmed:
                return AgentStateResponse(
                    agent_state="confirm",
                    intent="create",
                    user_message=(
                        f"我准备为您新增一条循环日程：{parsed['schedule_date']} "
                        f"{parsed['schedule_time']} {parsed['content']}，循环规则为工作日。请确认是否创建？"
                    ),
                    tool_name="schedule_create",
                    tool_arguments=parsed,
                )

            return AgentStateResponse(
                agent_state="execute",
                intent="create",
                user_message="正在为您创建工作日日程。",
                tool_name="schedule_create",
                tool_arguments=parsed,
            )

        # 意图3：删除日程（delete）
        if "取消日程" in normalized or "删除日程" in normalized:
            match = re.search(r"(取消日程|删除日程)\s*(\d+)", normalized)
            if not match:
                return AgentStateResponse(
                    agent_state="clarify",
                    intent="delete",
                    user_message="请告诉我您要取消哪一条日程，例如：取消日程1。",
                    missing_fields=["target_id"],
                )

            schedule_id = int(match.group(2))
            if not confirmed:
                return AgentStateResponse(
                    agent_state="confirm",
                    intent="delete",
                    user_message=f"我准备删除日程 {schedule_id}。删除后将无法恢复，请确认是否删除？",
                    tool_name="schedule_delete",
                    tool_arguments={"schedule_id": schedule_id},
                    target_id=schedule_id,
                )

            return AgentStateResponse(
                agent_state="execute",
                intent="delete",
                user_message=f"正在为您删除日程 {schedule_id}。",
                tool_name="schedule_delete",
                tool_arguments={"schedule_id": schedule_id},
                target_id=schedule_id,
            )

        # 意图4：创建日程（create）
        if "提醒我" in normalized or "添加日程" in normalized:
            parsed = self._parse_create(normalized)
            if parsed is None:
                return AgentStateResponse(
                    agent_state="clarify",
                    intent="create",
                    user_message="请告诉我具体日期和时间，例如：明天下午5点提醒我开会。",
                    missing_fields=["schedule_date", "schedule_time"],
                )

            if not confirmed:
                return AgentStateResponse(
                    agent_state="confirm",
                    intent="create",
                    user_message=(
                        f"我准备为您新增一条日程：{parsed['schedule_date']} "
                        f"{parsed['schedule_time']} {parsed['content']}，请确认是否创建？"
                    ),
                    tool_name="schedule_create",
                    tool_arguments=parsed,
                )

            return AgentStateResponse(
                agent_state="execute",
                intent="create",
                user_message="正在为您创建日程。",
                tool_name="schedule_create",
                tool_arguments=parsed,
            )

        return AgentStateResponse(
            agent_state="reply",
            intent="unknown",
            user_message="抱歉，我暂时还不能稳定处理这类请求，您可以换一种更明确的说法试试。",
        )

    # 解析创建参数
    def _parse_create(self, user_input: str) -> dict[str, Any] | None:
        content_match = re.search(r"提醒我(.+)", user_input)
        if content_match is None:
            content_match = re.search(r"添加日程[:：]?\s*(.+)", user_input)
        if content_match is None:
            return None

        content = content_match.group(1).strip()
        if not content:
            return None

        parsed_datetime = self._datetime_parser.parse(user_input)
        if parsed_datetime.schedule_date is None or parsed_datetime.schedule_time is None:
            return None

        clean_content = content
        for token in [
            "明天",
            "后天",
            "今天",
            "每隔一天",
            "每隔两天",
            "每隔三天",
            "每隔四天",
            "每隔五天",
            "每隔六天",
            "每隔七天",
            "每隔八天",
            "每隔九天",
            "下午5点",
            "早上9点",
            "17:00",
            "09:00",
            "周一",
            "周二",
            "周三",
            "周四",
            "周五",
            "周六",
            "周日",
            "星期一",
            "星期二",
            "星期三",
            "星期四",
            "星期五",
            "星期六",
            "星期日",
            "星期天",
            "工作日",
            "周一到周五",
            "周一至周五",
        ]:
            clean_content = clean_content.replace(token, "")
        clean_content = clean_content.strip()

        # 作用： 检测用户是否指定了重复规则
        cycle_rule = CycleRule.ONCE.value
        cycle_value: str | None = None
        interval_days = self._datetime_parser.parse_interval_days(user_input)
        if interval_days is not None:
            cycle_rule = CycleRule.INTERVAL_DAYS.value
            cycle_value = str(interval_days)

        return {
            "content": clean_content or content,
            "schedule_date": parsed_datetime.schedule_date,
            "schedule_time": parsed_datetime.schedule_time,
            "cycle_rule": cycle_rule,
            "cycle_value": cycle_value,
            "source_text": user_input,
        }
