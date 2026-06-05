#!/bin/bash
#==============================================================================
# NLP-RAG-04 一键部署脚本
# 用法：bash deploy.sh [选项]
#
# 选项:
#   --full        完整部署（含 vLLM + Attu）
#   --minimal     仅部署基础设施（etcd + minio + milvus）
#   --skip-deps   跳过 Python 依赖安装
#   --skip-ingest 跳过 PDF 数据接入
#   --skip-vllm   跳过 vLLM 容器启动（GPU 不可用时使用）
#   --help        显示帮助
#
# 默认行为：完整部署（full）
#==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# ---- 颜色定义 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ---- 参数解析 ----
DEPLOY_MODE="full"
SKIP_DEPS=false
SKIP_INGEST=false
SKIP_VLLM=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --full)       DEPLOY_MODE="full"; shift ;;
    --minimal)    DEPLOY_MODE="minimal"; shift ;;
    --skip-deps)  SKIP_DEPS=true; shift ;;
    --skip-ingest) SKIP_INGEST=true; shift ;;
    --skip-vllm)  SKIP_VLLM=true; shift ;;
    --help)       head -30 "$0"; exit 0 ;;
    *)            log_error "未知选项: $1"; exit 1 ;;
  esac
done

echo ""
echo "============================================"
echo " NLP-RAG-04 部署脚本"
echo " 模式: $DEPLOY_MODE"
echo "============================================"
echo ""

# ---- 1. 环境检查 ----
log_info "检查环境依赖..."

command -v docker >/dev/null 2>&1 || {
  log_error "docker 未安装，请先安装 Docker Desktop"
  exit 1
}

command -v conda >/dev/null 2>&1 && {
  log_info "检测到 conda，使用 conda 环境 nlp-rag"
  CONDA_ENV="nlp-rag"
  # 检查环境是否存在
  conda env list | grep -q "$CONDA_ENV" || {
    log_warn "conda 环境 $CONDA_ENV 不存在，使用系统 Python"
    CONDA_ENV=""
  }
} || {
  log_info "未检测到 conda，使用系统 Python"
  CONDA_ENV=""
}

# ---- 2. Python 依赖安装 ----
if [ "$SKIP_DEPS" = false ]; then
  log_info "安装 Python 依赖..."
  if [ -n "$CONDA_ENV" ]; then
    conda run -n "$CONDA_ENV" pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  else
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  fi
  log_info "Python 依赖安装完成"
fi

# ---- 3. Docker 服务启动 ----
log_info "启动 Docker 基础设施..."

# 检查 Docker Desktop 是否运行
docker ps >/dev/null 2>&1 || {
  log_error "Docker Desktop 未运行，请先启动 Docker Desktop"
  exit 1
}

# 启动 etcd（Milvus 依赖）
log_info "启动 etcd..."
docker compose up -d etcd
sleep 3

# 启动 minio
log_info "启动 MinIO..."
docker compose up -d minio
sleep 2

# 启动 Milvus
log_info "启动 Milvus..."
docker compose up -d milvus

# 等待 Milvus 就绪
log_info "等待 Milvus 就绪..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:9092/health >/dev/null 2>&1; then
    log_info "Milvus 就绪!"
    break
  fi
  if [ "$i" -eq 30 ]; then
    log_warn "Milvus 未在预期时间内就绪，继续启动其他服务..."
  fi
  sleep 2
done

# 启动 Attu（仅 full 模式）
if [ "$DEPLOY_MODE" = "full" ]; then
  log_info "启动 Attu（Milvus 管理界面）..."
  docker compose up -d attu
fi

# 启动 vLLM（full 模式且不跳过）
if [ "$DEPLOY_MODE" = "full" ] && [ "$SKIP_VLLM" = false ]; then
  log_info "启动 vLLM（LLM 推理服务）..."
  docker compose up -d vllm
  log_warn "vLLM 模型加载可能需要 2-5 分钟，请稍候..."
fi

log_info "所有 Docker 服务已启动"
docker compose ps

# ---- 4. FastAPI 后端启动 ----
log_info "启动 FastAPI 后端..."
if [ -n "$CONDA_ENV" ]; then
  conda run -n "$CONDA_ENV" uvicorn app.main:app --host 0.0.0.0 --port 8000 &
else
  uvicorn app.main:app --host 0.0.0.0 --port 8000 &
fi
FASTAPI_PID=$!
sleep 3

# 检查 FastAPI 是否启动
if curl -sf http://localhost:8000/api/health >/dev/null 2>&1; then
  log_info "FastAPI 已就绪!"
else
  log_warn "FastAPI 尚未就绪，请稍后检查"
fi

# ---- 5. Gradio 前端启动 ----
log_info "启动 Gradio 前端..."
if [ -n "$CONDA_ENV" ]; then
  conda run -n "$CONDA_ENV" python frontend/gradio_app.py &
else
  python frontend/gradio_app.py &
fi
GRADIO_PID=$!
sleep 2

# ---- 6. PDF 数据接入（可选） ----
if [ "$SKIP_INGEST" = false ]; then
  log_info "开始 PDF 数据接入..."
  log_warn "请确保 PDF 文件已放置在 data/ 目录下"
  
  if [ -n "$CONDA_ENV" ]; then
    conda run -n "$CONDA_ENV" python test/batch_fill_rag_answers.py --ingest || {
      log_warn "数据接入未执行（可能是 batch_fill_rag_answers.py 不支持 --ingest 参数）"
      log_info "尝试直接调用 FastAPI 接入接口..."
      curl -X POST http://localhost:8000/api/ingest || log_warn "API 接入失败，请手动调用"
    }
  else
    python test/batch_fill_rag_answers.py --ingest || {
      curl -X POST http://localhost:8000/api/ingest || log_warn "API 接入失败，请手动调用"
    }
  fi
fi

# ---- 7. 部署完成 ----
echo ""
echo "============================================"
echo " 部署完成!"
echo "============================================"
echo ""
echo " FastAPI 后端: http://localhost:8000"
echo " FastAPI API 文档: http://localhost:8000/docs"
echo " Gradio 前端: http://localhost:7860"
echo " Attu（Milvus 管理）: http://localhost:3001"
echo " MinIO Console: http://localhost:9101"
echo ""
echo " 后台进程 PID: FastAPI=$FASTAPI_PID, Gradio=$GRADIO_PID"
echo " 停止命令: kill $FASTAPI_PID $GRADIO_PID"
echo ""

# ---- 8. 清理函数 ----
cleanup() {
  log_info "正在停止服务..."
  kill $FASTAPI_PID 2>/dev/null || true
  kill $GRADIO_PID 2>/dev/null || true
  log_info "服务已停止"
}
trap cleanup EXIT

# 保持后台进程运行
wait $FASTAPI_PID $GRADIO_PID