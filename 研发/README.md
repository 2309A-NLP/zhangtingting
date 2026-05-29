# 多角色 RAG 智能问答系统

## 项目简介

这是一个面向多用户、多角色、多轮对话场景的 RAG 智能问答系统。

项目目标不是只做一个“能调用大模型聊天”的简单 Demo，而是构建一套较完整的知识增强问答链路，包括：

- 多角色人设管理
- 多轮对话记忆
- 本地知识库导入与检索
- 混合召回与重排
- 本地大模型推理与降级策略
- 复杂 PDF 文档解析

该系统适合用于课程设计、毕业设计、RAG 学习实践以及本地私有知识助手原型开发。

## 项目定位

本项目的核心定位是：

- 支持多用户、多角色隔离
- 支持本地知识库上传、解析、向量化、检索
- 支持本地模型与在线模型降级切换
- 支持复杂文档场景下的 RAG 问答

相较于基础版 RAG，本项目已经实现了较完整的优化链路：

`query rewrite -> dense retrieval + BM25 retrieval -> RRF fusion -> rerank -> context filtering -> grounded answer`

## 主要功能

### 1. 用户与角色体系

- 支持用户注册、登录、鉴权
- 支持预设角色与自定义角色
- 支持同一用户在不同角色之间切换
- 支持角色级知识隔离

典型角色示例：

- 法律顾问
- 医疗顾问
- 投资顾问
- 历史讲解角色

### 2. 多轮对话

- 支持多轮对话上下文维护
- 支持 Redis 最近对话缓存
- 支持长期记忆摘要
- 支持 MySQL 历史对话回补

### 3. 知识库导入

- 支持 PDF / TXT / JSON / HTML 文件上传
- 支持增量导入和全量覆盖
- 支持同内容文件去重
- 支持覆盖旧版本知识

### 4. 文档解析

- 支持复杂 PDF 本地多策略解析
- 支持 OCR 回退
- 支持表格抽取
- 支持解析质量评估
- 支持解析缓存
- 支持在本地解析质量不足时回退到 MinerU API

### 5. RAG 检索能力

- 向量检索
- BM25 检索
- RRF 融合
- BGE reranker 重排
- Query Rewrite
- 基于租户和角色的过滤检索

### 6. 模型推理

- 支持本地模型推理
- 支持 vLLM OpenAI 兼容接口
- 支持本地 CPU fallback
- 支持在线 API 降级

## 系统架构

整体架构可以概括为以下几层：

### 1. 接口层

- FastAPI 提供 HTTP API
- 提供认证、聊天、角色、知识库等接口

### 2. 数据存储层

- MySQL：用户、角色、对话、知识文件记录
- Redis：会话缓存、记忆摘要、任务状态
- Milvus：向量存储与检索
- MinIO：原始文件与解析产物存储

### 3. 知识处理层

- 文档加载器
- 文本清洗器
- 语义切块器
- 向量化器
- Milvus 入库服务

### 4. 检索增强层

- Query Rewrite
- Dense Retrieval
- BM25 Retrieval
- RRF Fusion
- Rerank
- Context Builder

### 5. 模型推理层

- 本地 Transformers 模型
- vLLM 服务
- 在线模型备用接口

## 关键技术栈

### 后端

- Python
- FastAPI
- SQLAlchemy
- Pydantic

### 数据与中间件

- MySQL
- Redis
- Milvus
- MinIO

### RAG 与模型

- BGE / M3E Embedding
- BGE Reranker
- Qwen2.5-0.5B-Instruct
- vLLM

### 文档处理

- PyMuPDF
- PyPDF
- pdfplumber
- pdfminer
- PaddleOCR

### 部署

- Docker
- Docker Compose

## 项目亮点

### 1. 多角色隔离

项目不是单一聊天机器人，而是围绕“一个用户可以拥有多个角色”来设计。

每个角色可以拥有：

- 独立的系统提示词
- 独立的知识空间
- 独立的对话上下文

### 2. 混合检索优化

项目已经实现较完整的检索增强链路：

- Query Rewrite
- Dense + BM25 多路召回
- RRF 融合
- Rerank
- Context Filtering

这使得它在结构化知识问答、长文档问答、专业领域问答中比基础向量检索更稳定。

### 3. 复杂 PDF 处理能力

很多 RAG 项目只支持简单文本导入，而本项目已经对复杂 PDF 做了专门设计：

- 多策略本地解析
- OCR 补救
- 质量评估
- MinerU API 回退
- 解析缓存

### 4. 本地模型友好

项目已经考虑低资源机器场景，适合 8GB 显存左右设备实验：

- 可用小模型 Qwen2.5-0.5B-Instruct
- 支持本地 CPU fallback
- 支持 vLLM 独立部署

## 当前已实现的 RAG 优化点

项目中已经实现的 RAG 优化包括：

- 混合检索
- BM25 + 向量召回
- RRF 融合
- Cross-encoder 重排
- Query Rewrite
- 文本清洗
- 结构化语义切块
- 上下文过滤
- 多轮记忆注入
- 基于证据的回答约束
- PDF 解析质量回退
- PDF 解析缓存

更完整的技术说明见：

- [RAG_optimization_notes.md](C:/Users/25921/Desktop/rag-app/RAG_optimization_notes.md)

## 目录说明

当前项目主要目录含义如下：

- `app/`：后端业务代码
- `data/`：模型文件、缓存、上传文件、解析结果等数据目录
- `scripts/`：部署、初始化、打包等脚本

在完整项目版本中，通常还会包含：

- `frontend/`：前端页面代码
- `.env`：本地运行配置
- `docker-compose.yml`：容器编排配置

## 运行方式

### 1. 本地开发运行

完整项目通常通过 Docker Compose 启动依赖服务：

- MySQL
- Redis
- Milvus
- MinIO
- vLLM（可选）
- API

常见流程：

1. 准备 `.env`
2. 准备本地模型目录
3. 启动 Docker 服务
4. 启动后端 API
5. 启动前端页面
6. 测试登录、聊天、知识上传

### 2. 服务器部署

服务器部署建议采用：

- Ubuntu 22.04
- Docker
- Docker Compose
- NVIDIA Driver + NVIDIA Container Toolkit（如需 GPU）

本项目已经额外提供了部署脚本：

- [scripts/deploy_project.sh](C:/Users/25921/Desktop/rag-app/scripts/deploy_project.sh)

## 适用场景

本项目适合以下场景：

- 多角色知识问答系统
- 本地私有知识助手
- 法律 / 医疗 / 投资等垂直领域问答原型
- 毕业设计 / 课程设计
- RAG 技术学习与实验

## 当前状态说明

根据当前项目演进情况，这个系统已经完成了以下关键部分：

- 基础后端链路
- 用户登录与鉴权
- 多角色聊天
- 本地知识入库
- 混合检索增强
- 本地模型接入
- 前后端联调基础能力

后续仍可以继续增强的方向包括：

- Multi-query retrieval
- HyDE
- Parent-child retrieval
- Graph retrieval
- Answer verification
- 自适应 top-k
- 在线检索缓存

## 一句话总结

这是一个面向多角色、多轮对话和私有知识增强场景的 RAG 系统原型，已经具备比较完整的文档处理、混合检索、上下文构建和本地模型推理能力，适合继续扩展为课程设计项目、毕业设计项目或中小型私有知识助手系统。
