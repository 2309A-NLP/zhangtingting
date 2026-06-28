# 记账本智能体项目

本项目按照 `prompt.md` 的要求构建，目标是实现一个面向家庭场景的“记账本智能体”。

## 已按提示词落地的部分

- 项目目录结构按 `db / tools / agent / config.py / main.py` 组织
- 数据表名为 `money_notes`
- 数据库默认使用 SQLite
- 所有数据库操作均通过参数化查询完成
- 提供了四个核心工具函数：
  - `insert_record`
  - `query_records`
  - `delete_record`
  - `verify_record`
- `agent/prompts.py` 读取 `prompt.md` 并生成 `SYSTEM_PROMPT`
- `agent/run.py` 提供 `build_agent()` 与 `run()` 入口
- `main.py` 提供命令行交互循环
- 每次工具调用记录结构化日志
- 已加入显式护栏状态机：补槽追问、写入前确认、删除前确认、多笔账目初步拆分

## 项目结构

```text
NLP-Agent-01/
├── agent/
│   ├── __init__.py
│   ├── langchain_agent.py
│   ├── llm_agent.py
│   ├── memory.py
│   ├── prompts.py
│   └── run.py
├── db/
│   ├── __init__.py
│   ├── init.py
│   ├── operations.py
│   └── schema.sql
├── tools/
│   ├── __init__.py
│   ├── db_tools.py
│   └── logging_utils.py
├── config.py
├── main.py
├── prompt.md
├── requirements.txt
└── README.md
```

## 环境变量

请复制 `.env.example` 为 `.env` 后填写：

```env
OPENAI_API_KEY=sk-xxxxx
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o
DB_PATH=money_notes.db
LOG_LEVEL=INFO
MAX_DB_RETRIES=3
MAX_TOOL_CALL_ROUNDS=8
AGENT_MODE=langchain
```

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 运行

```bash
python main.py
```

## 说明

当前默认按 `prompt.md` 的方向对齐为 `langchain` 模式。
如果本地尚未配置可用的 `OPENAI_API_KEY`，程序将无法真正启动 LLM Agent。

## 后续仍可继续增强

- 更严格的意图分类路由
- 更稳定的“新增/删除前确认”状态控制
- 多笔账目拆分
- 更复杂的时间解析（上周五、上个月等）
- Web API / 数字人对接层
