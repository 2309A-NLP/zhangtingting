from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import httpx

from config import OPENAI_API_KEY, OPENAI_BASE_URL, MAX_TOOL_CALL_ROUNDS, MODEL_NAME
from tools.db_tools import delete_record, insert_record, query_records, verify_record
from agent.memory import SessionMemory
from agent.prompts import load_system_prompt

logger = logging.getLogger(__name__)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "insert_record",
            "description": "往 money_notes 表写入一条收支记录，写入前需先向用户确认所有字段。",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "日期，格式 YYYY-MM-DD，如 2025-07-05"},
                    "member": {
                        "type": "string",
                        "description": "成员名称（爸爸/妈妈/女儿）",
                        "enum": ["爸爸", "妈妈", "女儿"],
                    },
                    "type": {"type": "string", "description": "收支类型", "enum": ["支出", "收入"]},
                    "category": {"type": "string", "description": "消费/收入类别，如 餐饮、服装、工资、报销"},
                    "amount": {"type": "number", "description": "金额（元），必须为正数"},
                    "note": {"type": "string", "description": "备注说明（可选）"},
                },
                "required": ["date", "member", "type", "category", "amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_records",
            "description": "从 money_notes 表中查询收支记录，支持多条件组合筛选。",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_from": {"type": "string", "description": "开始日期 YYYY-MM-DD，None 表示不限"},
                    "date_to": {"type": "string", "description": "结束日期 YYYY-MM-DD，None 表示不限"},
                    "member": {"type": "string", "description": "成员名称（爸爸/妈妈/女儿）"},
                    "type": {"type": "string", "description": "收支类型（支出/收入）"},
                    "category": {"type": "string", "description": "类别关键词（模糊匹配）"},
                    "keyword": {"type": "string", "description": "备注模糊搜索关键词"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_record",
            "description": "根据记录 ID 删除一条记录。删除前必须展示记录内容并征得用户明确确认。",
            "parameters": {
                "type": "object",
                "properties": {
                    "record_id": {"type": "integer", "description": "要删除的记录 ID"},
                },
                "required": ["record_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_record",
            "description": "验证指定 ID 的记录是否已成功写入数据库。",
            "parameters": {
                "type": "object",
                "properties": {
                    "record_id": {"type": "integer", "description": "记录 ID"},
                },
                "required": ["record_id"],
            },
        },
    },
]


TOOL_MAP = {
    "insert_record": insert_record,
    "query_records": query_records,
    "delete_record": delete_record,
    "verify_record": verify_record,
}

# 假langchain 遗留代码
class LLMAgent:
    def __init__(self) -> None:
        self.memory = SessionMemory()
        self.system_prompt = load_system_prompt()

    def reply(self, user_input: str) -> str:
        self.memory.add_user(user_input)

        reply_text = self._run_agent()
        self.memory.add_assistant(reply_text)
        return reply_text

    def _run_agent(self) -> str:
        for _round in range(MAX_TOOL_CALL_ROUNDS):
            # 调用 LLM 获取响应
            response = self._call_llm(self.memory.snapshot())
            # 如果有工具调用，执行工具调用
            if response.tool_calls:
                # 遍历工具调用
                for tc in response.tool_calls:
                    # 获取工具函数
                    fn = TOOL_MAP.get(tc.function.name)
                    if not fn:   # 如果工具函数不存在，添加工具结果
                        self.memory.add_tool_result(tc.id, f"未知工具：{tc.function.name}")
                        continue   # 跳过
                    try:
                        # 解析工具调用参数
                        args = json.loads(tc.function.arguments)
                        # 执行工具函数
                        result = fn(**args)
                        # 添加工具调用
                        self.memory.add_tool_call(tc.id, tc.function.name, tc.function.arguments)
                        # 添加工具结果
                        self.memory.add_tool_result(tc.id, json.dumps(result, ensure_ascii=False))
                    except Exception as exc:   # 如果工具函数执行失败，添加工具结果
                        self.memory.add_tool_result(tc.id, json.dumps({"success": False, "message": str(exc)}))
                continue
            # 如果没有工具调用，返回响应内容
            if response.content:   # 如果响应有内容，返回内容
                return response.content
            # 如果响应没有内容，返回提示
            return "抱歉，服务端未返回有效内容，请稍后重试。" 
        return "处理轮次超出限制，请简化您的请求。"   # 如果处理轮次超出限制，返回提示

    def _call_llm(self, messages: list[dict[str, Any]]) -> Any:
        system_msg = {"role": "system", "content": self.system_prompt}
        payload = {
            "model": MODEL_NAME,
            "messages": [system_msg] + messages,  # 系统提示词 + 对话历史
            "tools": TOOLS,  # 工具列表
            "tool_choice": "auto",  # 自动选择工具
            "temperature": 0,  # 温度
        }
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",  # 授权
            "Content-Type": "application/json",  # 内容类型 
        }

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]    # 取第一个响应（通常只有一个）
        msg = choice["message"]         # 获取消息内容

        if msg.get("tool_calls"):
            # 如果有工具调用，返回工具调用响应
            return _ToolCallResponse(
                tool_calls=[
                    _ToolCall(
                        id=tc["id"],
                        function=_FunctionCall(name=tc["function"]["name"], arguments=tc["function"]["arguments"]),
                    )
                    for tc in msg["tool_calls"]
                ],
                content=msg.get("content") or "",
            )
        return _ToolCallResponse(tool_calls=[], content=msg.get("content") or "")

# 工具调用
class _ToolCall:
    def __init__(self, id: str, function: "_FunctionCall") -> None:
        self.id = id
        self.function = function

# 函数调用
class _FunctionCall:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments

# 工具调用响应
class _ToolCallResponse:
    def __init__(self, tool_calls: list[_ToolCall], content: str) -> None:
        self.tool_calls = tool_calls   # 工具调用列表
        self.content = content         # 消息内容
