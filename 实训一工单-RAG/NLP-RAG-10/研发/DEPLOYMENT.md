# 工单编号：人工智能 NLP-RAG-金融问答系统部署
# Docker 部署文档

## 1. 系统概述

### 1.1 项目简介
本项目是基于 RAG（检索增强生成）技术的金融问答系统，支持对 PDF 金融文档（如招股说明书）进行智能问答，提供文本问答、语音问答、证据引用等功能。

### 1.2 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        金融问答系统架构                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐  │
│   │   用户界面   │     │   FastAPI 后端  │     │   vLLM 模型服务  │  │
│   │  (Gradio)   │◄───►│   (端口 8000)   │◄───►│  (端口 8002)    │  │
│   │  (端口 7860) │     │                 │     │                 │  │
│   └─────────────┘     └────────┬────────┘     └─────────────────┘  │
│                                 │                                    │
│         ┌───────────────────────┼───────────────────────┐            │
│         │                       │                       │            │
│         ▼                       ▼                       ▼            │
│   ┌───────────┐         ┌───────────┐           ┌───────────┐       │
│   │  Milvus   │         │  MongoDB  │           │   MinIO   │       │
│   │ (向量存储) │         │ (文档存储) │           │ (对象存储) │       │
│   └───────────┘         └───────────┘           └───────────┘       │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                    基础设施层 (etcd)                         │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 服务组件

| 服务名称 | 镜像/构建 | 端口 | 功能描述 |
|---------|----------|------|---------|
| backend | Dockerfile | 8000 | FastAPI 后端 API 服务 |
| frontend | Dockerfile.frontend | 7860 | Gradio 前端界面 |
| milvus | milvusdb/milvus:v2.4.15 | 19530, 9091 | 向量数据库 |
| mongodb | mongo:7.0 | 27017 | 文档数据库 |
| minio | minio/minio | 9100, 9101 | S3 兼容对象存储 |
| etcd | quay.io/coreos/etcd | 2379 | Milvus 元数据存储 |
| vllm | vllm/vllm-openai | 8000 | LLM 推理服务 (需 GPU) |
| attu | zilliz/attu:v2.4 | 3000 | Milvus Web 管理界面 |
| mongo-express | mongo-express:1.0.2 | 8081 | MongoDB Web 管理界面 |

## 2. 前置要求

### 2.1 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|-----|---------|---------|
| CPU | 4 核 | 8 核或以上 |
| 内存 | 16 GB | 32 GB 或以上 |
| GPU | 可选 | NVIDIA GPU + CUDA 12.0+ |
| 磁盘 | 50 GB | 100 GB SSD |

### 2.2 软件要求

- **Docker**: 24.0+
- **Docker Compose**: 2.20+
- **NVIDIA Docker** (可选): 用于 GPU 加速

### 2.3 检查 Docker 安装

```powershell
# 检查 Docker 版本
docker --version

# 检查 Docker Compose 版本
docker compose version

# 验证 Docker 运行状态
docker ps
```

## 3. 快速开始

### 3.1 目录结构准备

```powershell
# 确保项目目录存在
cd D:\Desktop\NLP-RAG-04

# 创建必要的目录结构
mkdir -p volumes/etcd,volumes/minio,volumes/mongodb,volumes/milvus,data,model,artifacts,reports,config
```

### 3.2 模型文件准备

如果使用本地 vLLM 模型服务，需要准备模型文件：

```powershell
# 创建模型目录
mkdir -p model/llm

# 下载 Qwen2.5-0.5B-Instruct 模型（或其他兼容模型）
# 模型应放在 model/llm/Qwen2.5-0.5B-Instruct 目录下
```

### 3.3 启动服务

#### 方式一：使用部署脚本（推荐）

```powershell
# 完整构建并启动
.\deploy.ps1 -Build

# 仅启动（跳过构建）
.\deploy.ps1 -NoBuild

# 查看状态
.\deploy.ps1 -Status

# 查看日志
.\deploy.ps1 -Logs

# 停止服务
.\deploy.ps1 -Stop

# 清理所有数据
.\deploy.ps1 -Clean
```

#### 方式二：直接使用 Docker Compose

```powershell
# 构建镜像
docker compose build backend frontend

# 启动基础设施服务
docker compose up -d etcd minio mongodb milvus

# 等待 Milvus 就绪（约 30-60 秒）
docker compose ps

# 启动应用服务
docker compose up -d backend frontend

# 查看状态
docker compose ps
```

### 3.4 访问服务

启动成功后，可通过以下地址访问：

| 服务 | 地址 | 说明 |
|-----|------|-----|
| 前端界面 | http://localhost:7860 | Gradio 问答界面 |
| 后端 API | http://localhost:8000 | FastAPI 文档 |
| API 文档 | http://localhost:8000/docs | Swagger UI |
| Milvus Console | http://localhost:9091 | Milvus 管理 |
| Attu | http://localhost:3011 | 向量管理 |
| MinIO Console | http://localhost:9101 | 对象存储管理 |
| Mongo Express | http://localhost:8082 | MongoDB 管理 |

## 4. 配置说明

### 4.1 环境变量配置

主要配置文件为 `.env.docker`，关键配置项：

```env
# Milvus 向量数据库
MILVUS_URI=http://milvus:19530

# MongoDB 文档数据库
MONGODB_URI=mongodb://mongodb:27017
MONGODB_DB_NAME=nlp_rag

# MinIO 对象存储
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# LLM 模型配置（vLLM）
LLM_FALLBACK_API_URL=http://vllm:8000/v1/chat/completions
LLM_FALLBACK_MODEL_NAME=Qwen2.5-0.5B-Instruct
```

### 4.2 端口配置

如需修改端口，编辑 `docker-compose.yml` 中的端口映射：

```yaml
services:
  backend:
    ports:
      - "8000:8000"  # 修改为 "新端口:8000"
  frontend:
    ports:
      - "7860:7860"  # 修改为 "新端口:7860"
```

## 5. 健康检查

### 5.1 使用健康检查脚本

```powershell
# 单次检查
.\health-check.ps1

# 持续监控
.\health-check.ps1 -Watch
```

### 5.2 手动检查

```powershell
# 检查容器状态
docker compose ps

# 检查特定服务日志
docker compose logs backend --tail=50

# 检查健康状态
docker inspect nlp-rag-backend --format='{{.State.Health.Status}}'
```

## 6. 数据管理

### 6.1 数据持久化

所有数据通过 Docker 卷持久化存储：

| 数据类型 | 卷路径 | 说明 |
|---------|-------|-----|
| Milvus 数据 | `./volumes/milvus` | 向量索引数据 |
| MongoDB 数据 | `./volumes/mongodb` | 文档和元数据 |
| MinIO 数据 | `./volumes/minio` | 文件对象存储 |
| etcd 数据 | `./volumes/etcd` | Milvus 配置数据 |
| 应用数据 | `./data` | PDF 文件 |
| 模型文件 | `./model` | 本地模型 |
| 产物数据 | `./artifacts` | 处理结果 |

### 6.2 备份数据

```powershell
# 备份整个数据目录
Copy-Item -Recurse ./volumes ./volumes-backup-$(Get-Date -Format "yyyyMMdd")

# 备份特定服务数据
docker compose stop mongodb
Copy-Item -Recurse ./volumes/mongodb ./mongodb-backup
docker compose start mongodb
```

### 6.3 恢复数据

```powershell
# 停止服务
docker compose down

# 恢复数据
Copy-Item -Recurse ./mongodb-backup ./volumes/mongodb

# 重启服务
docker compose up -d
```

## 7. 常见问题

### 7.1 Milvus 启动失败

**问题**: Milvus 容器无法启动

**解决方案**:
1. 检查 etcd 是否正常运行
2. 检查 MinIO 是否正常运行
3. 清理并重新创建卷：
```powershell
docker compose down -v
docker compose up -d etcd minio
Start-Sleep -Seconds 30
docker compose up -d milvus
```

### 7.2 vLLM 启动失败

**问题**: vLLM 容器需要 NVIDIA GPU

**解决方案**:
1. 确认已安装 NVIDIA Docker 支持：
```powershell
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```
2. 如无 GPU，可使用云端 API 替代（修改 LLM 配置）

### 7.3 端口冲突

**问题**: 端口已被占用

**解决方案**: 修改 `docker-compose.yml` 中的端口映射

### 7.4 内存不足

**问题**: 容器因 OOM 被终止

**解决方案**:
1. 增加 Docker 内存限制
2. 减小模型大小
3. 调整 `gpu-memory-utilization` 参数

## 8. 验收标准检查

### 8.1 容器启动与运行

- [ ] `docker run` 命令能够成功启动所有容器
- [ ] 所有容器运行过程中无异常日志
- [ ] 健康检查全部通过

### 8.2 容器数据管理

- [ ] Docker 卷正确挂载，数据持久化有效
- [ ] 容器重启后数据不丢失
- [ ] 服务间数据共享正常

### 8.3 网络配置

- [ ] 容器间网络通信正常
- [ ] 服务端口映射正确
- [ ] 外部可访问所有服务

### 8.4 功能验证

- [ ] 前端界面可正常访问
- [ ] 后端 API 响应正常
- [ ] 问答功能可用

## 9. 安全建议

1. **修改默认密码**: MinIO 和 Mongo Express 使用默认密码，建议修改
2. **限制网络访问**: 生产环境建议配置防火墙
3. **定期备份**: 建立数据备份机制
4. **更新镜像**: 定期更新基础镜像修复安全漏洞

## 10. 技术支持

如遇到问题，请检查：
1. Docker 和 Docker Compose 版本
2. 日志输出：`docker compose logs <service>`
3. 资源使用：`docker stats`
4. 网络连通性：`docker compose exec backend ping mongodb`

---

**文档版本**: V1.0  
**创建日期**: 2025-02-06  
**工单编号**: 人工智能 NLP-RAG-金融问答系统部署
