# 基金数据问答智能体系统

基于大语言模型（LLM）的 **NL2SQL** 基金数据智能问答系统。

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
cd fund-qa-agent

# 安装依赖
pip install -e ".[dev]"

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY 等配置
```

### 2. 数据准备

```bash
# 方式一：下载完整数据集（推荐）
bash scripts/download_data.sh
# 方式二：从本地复制
cp /path/to/博金杯比赛数据.db data/raw/
cp /path/to/question.jsonl data/raw/

# 初始化元数据
python scripts/init_db.py
```

### 3. 运行服务

```bash
# 开发模式（热重载）
make run

# 生产模式
make run-prod

# 访问 API 文档
# http://localhost:8000/docs
```

### 4. 测试与评测

```bash
# 运行单元测试
make test

# 运行评测（100 题抽样）
python scripts/run_eval.py
```

## 项目结构

```
fund-qa-agent/
├── config/                # 配置
├── src/
│   ├── api/               # FastAPI 接口
│   │   ├── routes/        # 路由
│   │   ├── middleware/    # 中间件
│   │   └── schemas/       # 请求/响应模型
│   ├── core/              # 核心业务
│   │   ├── engine/        # NL2SQL 引擎
│   │   ├── retriever/     # 检索模块
│   │   └── models/        # 领域模型
│   ├── services/          # 服务层
│   ├── batch/             # 批处理
│   └── utils/             # 工具
├── scripts/               # 运维脚本
├── tests/                 # 测试
├── eval/                  # 评测
├── data/                  # 数据（gitignored）
├── docs/                  # 文档
├── Dockerfile
└── docker-compose.yml
```

## API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/chat` | 单轮问答 |
| POST | `/api/v1/chat/stream` | 流式问答 |
| POST | `/api/v1/batch` | 批量问答 |
| GET | `/api/v1/batch/{id}` | 批量状态 |
| GET | `/api/v1/tables` | 数据表信息 |
| GET | `/health` | 健康检查 |

## 技术栈

- **框架**: FastAPI + LangChain
- **模型**: Qwen / DeepSeek / GPT-4o-mini
- **向量库**: ChromaDB
- **缓存**: Memory / Redis
- **部署**: Docker

## 前置条件

- Python 3.10+
- LLM API Key（OpenAI / DeepSeek / DashScope）
- 磁盘空间 ≥ 10GB（含数据库 1.46G）

详见 `docs/DEPLOY.md`
