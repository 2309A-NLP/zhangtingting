# 部署文档

## 环境要求

| 资源 | 最低配置 | 推荐配置 |
|---|---|---|
| CPU | 4 核 | 8 核 |
| 内存 | 8 GB | 16 GB |
| 磁盘 | 10 GB | 20 GB |
| GPU（本地模型）| 8 GB VRAM | 16 GB VRAM |

## 部署方式

### 方式一：Docker Compose（推荐）

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env

# 2. 启动服务
docker-compose up -d

# 3. 查看日志
docker-compose logs -f

# 4. 停止服务
docker-compose down
```

### 方式二：裸机部署

```bash
# 1. 安装依赖
pip install -e ".[all]"

# 2. 配置环境变量
cp .env.example .env

# 3. 启动
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 2

# 或使用 PM2
pm2 start "uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 2" --name fund-qa
```

## 配置说明

主要环境变量（详见 `.env.example`）：

| 变量 | 说明 | 示例 |
|---|---|---|
| `LLM_PROVIDER` | LLM 提供商 | openai / deepseek / dashscope |
| `LLM_API_KEY` | API 密钥 | sk-xxx |
| `LLM_MODEL_NAME` | 模型名称 | gpt-4o-mini |
| `DB_PATH` | 数据库路径 | data/raw/博金杯比赛数据.db |
| `CACHE_TYPE` | 缓存类型 | memory / redis |

## 健康检查

```bash
curl http://localhost:8000/health
# {"status":"ok","version":"1.0.0","timestamp":...}
```

## 监控

Prometheus 指标（需启用）：`GET /api/v1/metrics`
