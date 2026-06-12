# 基于 PDF 文档的 RAG 问答系统

本项目面向 `data/招股说明书1.pdf`，采用模式 A 本地部署：

- Embedding：本地 `bge-m3`
- LLM：本地 `Qwen2.5-7B-Instruct-GPTQ-Int4`
- 语音识别：本地 `faster-whisper`
- 向量库：Docker 中的 Milvus
- 推理服务：Docker 中的 vLLM
- 后端：FastAPI
- 前端：Gradio

## 当前本地模型

已适配以下目录：

- Embedding：`D:\Desktop\NLP-RAG-01\model\embedding\bge-m3\bge-m3\BAAI\bge-m3`
- LLM：`D:\Desktop\NLP-RAG-01\model\llm\Qwen2.5-7B-Instruct-GPTQ-Int4`

## Docker 服务

模式 A 需要以下镜像：

- `quay.io/coreos/etcd`
- `minio/minio`
- `milvusdb/milvus`
- `zilliz/attu`
- `vllm/vllm-openai`

旧项目中已经下载过的镜像可以直接复用，不需要重复下载。只要版本兼容、端口不冲突即可。

## 推荐环境安装

建议使用 Conda 新建独立环境：

```bash
conda create -n nlp-rag python=3.10 -y
conda activate nlp-rag
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

如果旧环境里存在 `torch` DLL 问题，不建议继续复用旧环境。

## 启动顺序

### 1. 启动 Docker 服务

```bash
docker compose up -d
```

启动后主要端口：

- Milvus：`19531`
- Milvus Health：`9092`
- MinIO API：`9100`
- MinIO Console：`9101`
- Attu：`3001`
- vLLM OpenAI API：`8002`

### 2. 启动后端

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. 构建索引

```bash
curl -X POST "http://127.0.0.1:8000/api/ingest"
```

### 4. 启动前端

```bash
python frontend/gradio_app.py
```

### 5. 语音问答依赖

语音转文字功能需要：

- `faster-whisper`
- `ffmpeg`

如果你在 Windows 上还没装 `ffmpeg`，需要先安装并加入 `PATH`。

## 环境变量

参考 `.env.example`：

```bash
MILVUS_URI=http://127.0.0.1:19530
LLM_PROVIDER=openai_compatible
LLM_API_URL=http://127.0.0.1:8002/v1/chat/completions
LLM_MODEL_NAME=Qwen2.5-7B-Instruct-GPTQ-Int4
LLM_LOCAL_MODEL_PATH=D:\Desktop\NLP-RAG-01\model\llm\Qwen2.5-7B-Instruct-GPTQ-Int4
EMBEDDING_MODEL_PATH=D:\Desktop\NLP-RAG-01\model\embedding\bge-m3\bge-m3\BAAI\bge-m3
OCR_LANG=ch
```

## 功能说明

- PDF 清洗：移除页眉、页脚、逻辑页码
- 表格提取：保留表格结构并转 Markdown
- OCR 兜底：页面文本极少时调用 OCR
- 语音前置处理：上传或录制语音后，先转文字再进入 RAG 问答
- 脱敏：姓名、身份证号、金额、涉密单位标识
- Query 理解：意图识别、消歧、问题改写、复杂问题拆分
- 检索：Milvus Top-K=5
- 生成：通过本地 vLLM 返回基于证据的答案

## 评估

```bash
python -m app.evaluate
```

产物：

- `reports/evaluation.csv`
- `reports/metrics.md`

## 说明

- 测试题中的页码可能是招股书逻辑页，不一定等于 PDF 物理页
- 如果 vLLM 启动失败，优先检查 GPU、CUDA、显存与 GPTQ 支持情况
- 如果只想先验证检索链路，可以把 `LLM_PROVIDER=extractive`
