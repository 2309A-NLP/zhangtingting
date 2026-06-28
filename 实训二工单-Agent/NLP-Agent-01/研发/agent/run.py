# 工单编号：人工智能NLP-Agent数字人项目-记账本任务
# 来源：北京八维信息集团 · 八维文化与产业研究院

from __future__ import annotations

import logging

from langchain.agents import create_agent      # 创建 Agent 的核心函数
from langchain_core.messages import HumanMessage, SystemMessage  # 消息类型
from langchain_openai import ChatOpenAI        # OpenAI 聊天模型

from agent.guardrails import GuardrailAgent    # 安全护栏（内容过滤）
from agent.prompts import SYSTEM_PROMPT        # 系统提示词
from config import DEFAULT_WELCOME, MODEL_NAME, OPENAI_API_KEY, OPENAI_BASE_URL
from tools.db_tools import delete_record, insert_record, query_records, verify_record
# 设置日志格式
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# 获取日志记录器
logger = logging.getLogger(__name__)


TOOLS = [  # 工具列表
    insert_record,
    query_records,
    delete_record,
    verify_record,
]

_agent_graph = None  # 全局缓存 ReAct Agent 图
_guardrail_agent = GuardrailAgent()  # 安全护栏实例


def _build_graph():
    """构建并缓存 ReAct Agent 图。"""
    global _agent_graph
    # 如果缓存存在，直接返回
    if _agent_graph is not None:
        return _agent_graph

    # 如果 API 密钥未设置，抛出错误
    if not OPENAI_API_KEY:
        raise ValueError("未设置 OPENAI_API_KEY 环境变量，请检查 .env 文件")

    # 创建 OpenAI 聊天模型
    llm = ChatOpenAI(
        model=MODEL_NAME or "gpt-4o",
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0,  # 温度=0 让输出更确定 每次回答几乎一样
    )

    # 创建 ReAct Agent 图
    _agent_graph = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )
    return _agent_graph


def run(user_input: str) -> str:
    """
    单轮对话入口。接收用户输入，返回 Agent 回复。
    :param user_input: 用户的自然语言输入
    :return: Agent 的自然语言回复
    """
    # GuardrailAgent 的作用：
    # 过滤不当内容（如政治敏感、色情等）
    # 处理特定命令（如 /help、/start）
    # 返回欢迎语或错误提示
    guardrail_reply = _guardrail_agent.reply(user_input)
    # 如果安全护栏回复不为空且不等于默认欢迎语，返回安全护栏回复
    if guardrail_reply and guardrail_reply != DEFAULT_WELCOME:
        return guardrail_reply
    if guardrail_reply == DEFAULT_WELCOME:
        return guardrail_reply
    
    graph = _build_graph()
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]})
    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        return last.content
    return "处理完成，无返回内容。"
