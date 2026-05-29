# Multi-Role RAG Backend

基于 FastAPI 的多角色 RAG 后端，支持：

- 多角色对话：律师 / 医生 / 股票专家 / 历史人物 / 自定义角色
- 多用户与多角色隔离：Redis / MySQL / Milvus 全链路按 `user_id + role_id` 隔离
- 知识库处理链路：上传 -> 解析 -> 清洗 -> 分块 -> 向量化 -> Milvus 入库
- 本地推理优先：vLLM
- 在线降级兜底：SiliconFlow
- 流式输出：SSE

## 目录结构

```text
app/
  api/           FastAPI 路由和请求响应 schema
  chat/          上下文构建、角色守卫、限流、LLM 客户端
  core/          配置和日志
  db/            MySQL / Redis / Milvus 客户端
  knowledge/     知识库 pipeline
  retriever/     Query 改写、混合检索、重排
  services/      角色服务等业务服务
scripts/
  mysql-init/    数据库初始化 SQL
  sample_data/   示例知识库文本
tests/           API 和 RAGAS 测试
```

## 环境要求

- Docker Desktop
- Linux containers 模式
- Python 3.11+（本地运行时）
- 可访问的 vLLM 服务，或可用的 SiliconFlow API Key

## 快速启动

1. 复制环境变量模板：

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

2. 按需修改 `.env`

至少建议检查：

- `APP_SECRET_KEY`
- `SILICONFLOW_API_KEY`
- `VLLM_BASE_URL`
- `EMBEDDING_DEVICE`
- `RERANK_DEVICE`

3. 拉起依赖和 API：

```bash
docker compose up -d --build
```

4. 初始化数据库：

Linux/macOS:

```bash
bash scripts/init_db.sh
```

Windows PowerShell:

```powershell
.\scripts\init_db.ps1
```

5. 健康检查：

```bash
curl http://localhost:8000/api/v1/health
```

## 本地开发运行

如果你不想把 API 放进容器，也可以只启动依赖：

```bash
docker compose up -d mysql redis etcd minio milvus attu
```

然后本地运行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 核心接口

- `POST /api/v1/chat`
- `POST /api/v1/chat/clear`
- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `GET /api/v1/auth/me`
- `GET /api/v1/roles`
- `POST /api/v1/roles/detect`
- `POST /api/v1/roles/custom`
- `POST /api/v1/knowledge/upload`
- `GET /api/v1/health`

## 示例请求

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test-user-001",
    "role_id": "lawyer_01",
    "query": "我的合同纠纷怎么处理？",
    "stream": false
  }'
```

建议先注册再登录：

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "demo_user",
    "password": "demo123456",
    "email": "demo@example.com"
  }'
```

然后获取 Bearer Token：

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "demo_user",
    "password": "demo123456"
  }'
```

数据库迁移：

```bash
alembic upgrade head
```

## 示例知识库

仓库内已提供示例文档：

- `scripts/sample_data/lawyer_01`
- `scripts/sample_data/doctor_01`
- `scripts/sample_data/stock_01`
- `scripts/sample_data/history_01`

可用这些文档直接测试 `/api/v1/knowledge/upload`

## 已知边界

- 当前知识库上传为同步执行，不是后台任务队列
- BM25 采用租户内候选集本地构建，适合 Phase 1，不适合超大规模语料
- PaddleOCR、BGE、Reranker 依赖较重，首次构建镜像较慢
- 若容器内调用本机 vLLM，请确认 `VLLM_BASE_URL` 指向 `host.docker.internal`

## 测试

安装测试依赖后执行：

```bash
python -m pytest tests/test_api.py -q
```

RAGAS 集成评测默认跳过，启用方式：

```bash
RUN_RAGAS_TESTS=true python -m pytest tests/test_ragas.py -q
```

## 下一步建议

- 把知识库上传改成异步任务队列
- 加 Alembic 迁移
- 为 API 增加 JWT 鉴权
- 为 Milvus / MinIO / vLLM 增加更严格的健康检查
