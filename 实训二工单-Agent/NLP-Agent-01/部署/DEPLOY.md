# NLP-Agent-01 家庭记账助手 — 部署文档

> 工单编号：人工智能NLP-Agent数字人项目-记账本任务
> 来源：北京八维信息集团 · 八维文化与产业研究院
> 版本：v1.0
> 日期：2026-06-28

---

## 一、系统概述

**NLP-Agent-01 家庭记账助手**是一款基于大语言模型的智能对话式记账工具，家庭成员（爸爸、妈妈、女儿）可通过自然语言完成收支记录、查询、统计和删除操作。

### 1.1 核心能力

| 功能 | 说明 |
|------|------|
| 语音记账 | 输入"今天中午吃饭花了13"，自动解析日期/成员/事项/金额 |
| 多轮确认 | 记录前弹出确认卡片，确认后写入数据库 |
| 重复检测 | 自动检测疑似重复记录，防止重复记账 |
| 灵活查询 | 支持按日期/成员/类型/类别/关键词多维查询 |
| 删除操作 | 支持删除指定记录，带二次确认 |
| 引用查询 | 支持"刚才那笔"、"上一条"等上下文引用 |

### 1.2 技术架构

```
┌─────────────┐
│  main.py    │  ← 主入口，交互循环
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│  agent/run.py           │  ← LangChain ReAct Agent
│  agent/guardrails.py    │  ← 规则引擎 + 状态机
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  tools/db_tools.py     │  ← Function Calling 工具
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  db/operations.py       │  ← SQLite CRUD
│  money_notes.db         │  ← 数据持久化
└─────────────────────────┘
```

**技术栈：**

- Python 3.10+
- LangChain + LangGraph（ReAct Agent）
- ChatOpenAI 兼容接口（支持 SiliconFlow / OpenAI / 任意兼容 API）
- SQLite（无外部依赖）
- python-dotenv（环境变量）

---

## 二、环境准备

### 2.1 硬件要求

| 项目 | 最低要求 | 推荐 |
|------|---------|------|
| CPU | 1 核 | 2 核以上 |
| 内存 | 512 MB | 1 GB 以上 |
| 磁盘 | 100 MB | 500 MB 以上 |
| 网络 | 能访问 LLM API | 稳定宽带 |

### 2.2 软件要求

- **操作系统**：Windows 10/11、macOS、Linux
- **Python**：3.10 及以上
- **pip**：最新版本

### 2.3 获取 API Key

本项目使用 SiliconFlow（DeepSeek 模型）作为 LLM 提供商，也可替换为 OpenAI 或其他兼容 API。

**SiliconFlow 申请步骤：**

1. 访问 [https://siliconflow.cn](https://siliconflow.cn) 注册账号
2. 登录后在「API 密钥」页面创建新密钥
3. 复制密钥备用（格式：`sk-xxx`）

> 若使用 OpenAI，替换 `OPENAI_BASE_URL` 和 `MODEL_NAME` 即可。

---

## 三、部署步骤

### 3.1 下载源码

```bash
# 方法一：Git 克隆（如果有仓库）
git clone <仓库地址>
cd NLP-Agent-01

# 方法二：直接解压 ZIP 包
unzip NLP-Agent-01.zip
cd NLP-Agent-01
```

### 3.2 创建虚拟环境（推荐）

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3.3 安装依赖

```bash
pip install -r requirements.txt
```

**依赖说明：**

| 包 | 版本 | 用途 |
|----|------|------|
| langchain | ≥1.0.0 | Agent 框架 |
| langchain-openai | ≥0.2.0 | OpenAI 兼容接口 |
| langgraph | ≥0.2.0 | 图计算引擎 |
| python-dotenv | ≥1.0.0 | 环境变量读取 |
| httpx | ≥0.28.1 | HTTP 客户端 |

### 3.4 配置环境变量

```bash
# 复制模板文件
copy .env.example .env    # Windows
# cp .env.example .env   # macOS / Linux
```

编辑 `.env` 文件，填入真实密钥：

```bash
# 必填项
OPENAI_API_KEY=sk-your-real-api-key-here
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
MODEL_NAME=deepseek-ai/DeepSeek-V4-Flash

# 可选项（有默认值）
DB_PATH=money_notes.db
LOG_LEVEL=INFO
MAX_DB_RETRIES=3
MAX_TOOL_CALL_ROUNDS=8
AGENT_MODE=langchain
```

> **注意**：`.env` 文件包含敏感信息，请勿提交到代码仓库。若使用 Git，请确认 `.gitignore` 中包含 `.env`。

### 3.5 初始化数据库

首次运行时会自动创建 SQLite 数据库文件 `money_notes.db`，无需手动操作。

如需手动初始化：

```bash
python -c "from db.init import init_db; init_db()"
```

数据库文件位于项目根目录（如需修改路径，修改 `.env` 中的 `DB_PATH`）。

---

## 四、启动与使用

### 4.1 启动应用

```bash
python main.py
```

启动成功后会看到欢迎界面：

```
==================================================
您好，欢迎使用咱们小家专属记账本！
请按照"x年x月x日，谁做什么事收入/支出多少钱"的格式来输入。
请告诉我你的账目需求吧~
输入 '退出' 或 'quit' 结束对话
==================================================
```

### 4.2 使用示例

| 操作 | 示例输入 | 说明 |
|------|---------|------|
| 记支出 | `今天中午吃饭花了13` | 自动识别日期、成员、金额、事项 |
| 记收入 | `这个月工资发了8000` | 自动识别收入类型 |
| 查询支出 | `看看这个月花了多少钱` | 返回本月支出汇总 |
| 查询明细 | `爸爸这周支出明细` | 返回指定成员的记录列表 |
| 删除记录 | `删除刚才那条` | 先查后删，带确认 |
| 退出 | `退出` 或 `quit` | 结束对话 |

### 4.3 交互流程示例

```
您：今天中午吃饭花了13
小账：好的，我来帮您记录：
  日期：2026年6月28日
  成员：爸爸
  类型：支出
  事项：中午吃饭
  金额：13.0元
  确认无误吗？确认后我立即写入数据库~

您：确认
小账：记录已保存 ✅
  日期：2026年6月28日
  成员：爸爸
  类型：支出
  事项：中午吃饭
  金额：13.0元
  备注：
```

---

## 五、配置参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_API_KEY` | （必填） | LLM API 密钥 |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API 端点地址 |
| `MODEL_NAME` | `gpt-4o` | 模型名称 |
| `DB_PATH` | `money_notes.db` | SQLite 数据库路径（支持绝对路径）|
| `LOG_LEVEL` | `INFO` | 日志级别：`DEBUG`/`INFO`/`WARNING`/`ERROR` |
| `MAX_DB_RETRIES` | `3` | 数据库写入失败重试次数 |
| `MAX_TOOL_CALL_ROUNDS` | `8` | Agent 单次对话中工具调用最大轮数 |
| `AGENT_MODE` | `langchain` | 运行模式：`langchain`（当前）/ `llm` / `rule` |

---

## 六、数据库说明

### 6.1 表结构

```sql
CREATE TABLE money_notes (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    date     TEXT    NOT NULL,
    member   TEXT    NOT NULL,
    type     TEXT    NOT NULL CHECK (type IN ('支出', '收入')),
    category TEXT    NOT NULL,
    amount   REAL    NOT NULL CHECK (amount > 0),
    note     TEXT    DEFAULT ''
);

CREATE INDEX idx_member_date ON money_notes(member, date);
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INTEGER | 主键，自增 |
| `date` | TEXT | 日期，格式 `YYYY-MM-DD` |
| `member` | TEXT | 成员：`爸爸`/`妈妈`/`女儿` |
| `type` | TEXT | 类型：`支出` 或 `收入` |
| `category` | TEXT | 事项类别 |
| `amount` | REAL | 金额（正数）|
| `note` | TEXT | 备注 |

### 6.2 备份与恢复

**备份：**
```bash
# 复制数据库文件
copy money_notes.db money_notes_backup_20260628.db
```

**恢复：**
```bash
# 替换文件后重启即可
copy money_notes_backup_20260628.db money_notes.db
```

---

## 七、日志说明

### 7.1 日志级别

修改 `.env` 中 `LOG_LEVEL` 可控制日志详细程度：

| 级别 | 输出内容 |
|------|---------|
| `DEBUG` | 工具调用参数、完整请求/响应详情 |
| `INFO` | 用户输入、Agent 回复、错误信息（推荐）|
| `WARNING` | 仅警告和错误 |
| `ERROR` | 仅错误 |

### 7.2 日志格式

```
2026-06-28 14:30:01 [INFO] [USER] 今天中午吃饭花了13
2026-06-28 14:30:01 [INFO] [AGENT] 好的，我来帮您记录：...
```

---

## 八、常见问题

### Q1：启动报错 `OPENAI_API_KEY not set`
**原因**：`.env` 中未填写或未正确加载 API Key。
**解决**：确认 `.env` 文件存在且 `OPENAI_API_KEY=sk-xxx` 格式正确，`.venv` 环境已激活。

### Q2：启动报错 `No module named 'langchain'`
**原因**：未安装依赖或未激活虚拟环境。
**解决**：确认 `.venv` 已激活，执行 `pip install -r requirements.txt`。

### Q3：API 请求超时或失败
**原因**：网络无法访问 LLM API 端点，或 API Key 无效/余额不足。
**解决**：检查网络、确认 `OPENAI_BASE_URL` 正确、登录 SiliconFlow 检查账户余额。

### Q4：数据库报错 `UNIQUE constraint failed`
**原因**：数据库文件被多进程同时写入。
**解决**：确认同一时间只有一个 `python main.py` 进程在运行。

### Q5：中文输入乱码（Windows）
**解决**：确保终端编码为 UTF-8（`main.py` 中已自动处理）。

---

## 九、多环境部署示例

### 9.1 开发环境（本地）

`.env` 配置：
```bash
OPENAI_API_KEY=sk-dev-key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
MODEL_NAME=deepseek-ai/DeepSeek-V4-Flash
LOG_LEVEL=DEBUG
```

### 9.2 生产环境

- 使用 Nginx / Apache 做反向代理（非必需，纯命令行应用）
- 使用 `screen` / `tmux` 保持后台运行：
  ```bash
  screen -S记账本 -dm python main.py
  ```
- 日志输出重定向：
  ```bash
  python main.py >> app.log 2>&1
  ```
- 配置进程管理（systemd 示例）：
  ```ini
  [Unit]
  Description=NLP-Agent-01 家庭记账助手
  After=network.target

  [Service]
  Type=simple
  User=your-user
  WorkingDirectory=/path/to/NLP-Agent-01
  ExecStart=/path/to/NLP-Agent-01/.venv/bin/python main.py
  Restart=on-failure

  [Install]
  WantedBy=multi-user.target
  ```

---

## 十、项目结构总览

```
NLP-Agent-01/
├── config.py              # 全局配置
├── requirements.txt       # 依赖声明
├── main.py               # 主入口（交互循环）
├── prompt.md             # Agent 系统提示词
├── .env                  # 环境变量（勿提交）
├── .env.example          # 环境变量模板
│
├── agent/
│   ├── guardrails.py     # 核心：规则引擎 + 状态机
│   ├── run.py            # LangChain Agent 入口
│   ├── llm_agent.py      # LLM 模式 Agent（备用）
│   ├── langchain_agent.py # 兼容层包装器
│   ├── prompts.py        # 提示词加载
│   └── memory.py         # 会话记忆
│
├── tools/
│   ├── db_tools.py       # Function Calling 工具
│   └── logging_utils.py  # 工具调用日志
│
└── db/
    ├── schema.sql        # 建表语句
    ├── init.py           # 初始化脚本
    └── operations.py     # 底层 CRUD
```
