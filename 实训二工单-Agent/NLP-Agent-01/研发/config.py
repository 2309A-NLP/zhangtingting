# 工单编号：人工智能NLP-Agent数字人项目-记账本任务
# 来源：北京八维信息集团 · 八维文化与产业研究院

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
# 从 .env 文件中读取键值对，并将其加载到环境变量中（即 os.environ）。
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
# 数据库模式文件路径
SCHEMA_PATH = BASE_DIR / "db" / "schema.sql"
# 提示词文件路径
PROMPT_PATH = BASE_DIR / "prompt.md"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
# 数据库路径
DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "money_notes.db"))    
# 日志级别
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
# 最大数据库重试次数
MAX_DB_RETRIES = int(os.getenv("MAX_DB_RETRIES", "3"))
# 最大工具调用回合数
MAX_TOOL_CALL_ROUNDS = int(os.getenv("MAX_TOOL_CALL_ROUNDS", "8"))
# 代理模式
AGENT_MODE = os.getenv("AGENT_MODE", "langchain").strip().lower()
# 支持的代理模式
SUPPORTED_AGENT_MODES = {"rule", "llm", "langchain"}

DEFAULT_WELCOME = (
    "您好，欢迎使用咱们小家专属记账本！\n"
    "请按照\"x年x月x日，谁做什么事收入/支出多少钱\"的格式来输入。\n"
    "请告诉我你的账目需求吧~"
)
