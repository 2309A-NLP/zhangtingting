# RAG 优化说明文档

## 概述

本项目已经实现了一条相对完整的优化版 RAG 流水线，而不是最基础的“全部向量化后做一次向量检索”的简单方案。

当前主链路可以概括为：

`query rewrite -> dense retrieval + BM25 retrieval -> RRF fusion -> rerank -> context filtering -> evidence-grounded answer`

下面按照模块和链路，对项目中已经实现的 RAG 优化点进行结构化总结。

## 1. 检索侧优化

### 1.1 混合检索：向量召回 + BM25 召回

项目没有只依赖单一路检索方式。

当前同时使用：

- 基于 embedding 的向量检索
- 基于关键词重叠的 BM25 词法检索

这两条召回分支是在同一个检索器中完成的。

代码位置：

- `HybridRetriever` 入口：[app/retriever/hybrid.py](C:/Users/25921/Desktop/rag-app/app/retriever/hybrid.py:23)
- 主检索流程：[app/retriever/hybrid.py](C:/Users/25921/Desktop/rag-app/app/retriever/hybrid.py:39)
- 向量召回：[app/retriever/hybrid.py](C:/Users/25921/Desktop/rag-app/app/retriever/hybrid.py:81)
- BM25 召回：[app/retriever/hybrid.py](C:/Users/25921/Desktop/rag-app/app/retriever/hybrid.py:95)

为什么这属于 RAG 优化：

- 向量检索擅长语义相似匹配
- BM25 擅长精确词项匹配、法条名、专业术语和低频关键词命中
- 两者结合后，召回更稳健，漏召风险更低

### 1.2 RRF 融合

在完成向量召回和 BM25 召回之后，项目并不是简单把结果拼接起来。

当前采用的是 Reciprocal Rank Fusion，也就是 RRF 融合。

代码位置：

- 融合调用：[app/retriever/hybrid.py](C:/Users/25921/Desktop/rag-app/app/retriever/hybrid.py:109)
- 融合实现：[app/retriever/hybrid.py](C:/Users/25921/Desktop/rag-app/app/retriever/hybrid.py:271)

为什么这属于 RAG 优化：

- 不同召回通道的分数尺度通常不一致
- RRF 不依赖严格的分数归一化
- 这是一种稳定、实用、适合多路召回场景的融合方法

### 1.3 融合后的重排 rerank

在多路召回和 RRF 融合之后，项目还会用 `bge-reranker` 做交叉编码器重排。

代码位置：

- Reranker 类：[app/retriever/reranker.py](C:/Users/25921/Desktop/rag-app/app/retriever/reranker.py:20)
- Rerank 入口：[app/retriever/reranker.py](C:/Users/25921/Desktop/rag-app/app/retriever/reranker.py:32)
- 实际打分计算：[app/retriever/reranker.py](C:/Users/25921/Desktop/rag-app/app/retriever/reranker.py:110)

为什么这属于 RAG 优化：

- 向量召回偏向提升召回率
- Rerank 偏向提升最终排序精度
- 它可以显著提升最终送入大模型的上下文质量

### 1.4 检索前的 Query Rewrite

在进入检索之前，项目会先对用户问题做查询改写，使其更适合检索。

改写时会结合会话历史，处理：

- 指代消解
- 主语省略
- 模糊表达
- 依赖上下文的追问

代码位置：

- Query Rewriter 类：[app/retriever/query_rewrite.py](C:/Users/25921/Desktop/rag-app/app/retriever/query_rewrite.py:30)
- 改写入口：[app/retriever/query_rewrite.py](C:/Users/25921/Desktop/rag-app/app/retriever/query_rewrite.py:36)
- 改写 provider 选择：[app/retriever/query_rewrite.py](C:/Users/25921/Desktop/rag-app/app/retriever/query_rewrite.py:90)

为什么这属于 RAG 优化：

- 检索质量很大程度取决于 query 是否清晰
- 多轮对话中常常会出现信息不完整的追问
- Query Rewrite 可以提升多轮场景下的召回质量

### 1.5 按租户和角色范围做检索过滤

项目检索时会基于 `tenant_key`，并在必要时结合 `role_category` 做过滤。

代码位置：

- 过滤条件构建：[app/retriever/hybrid.py](C:/Users/25921/Desktop/rag-app/app/retriever/hybrid.py:298)

为什么这属于 RAG 优化：

- 它能减少无关知识带来的噪声
- 能防止不同用户、不同角色之间的知识串扰
- 在多用户、多角色系统里能明显提升检索相关性

## 2. 知识入库侧优化

### 2.1 结构化语义切块与重叠切块

项目没有采用最简单的按固定行数或固定字符粗暴切块。

当前使用 `RecursiveCharacterTextSplitter`，并且支持：

- 可配置 `chunk_size`
- 可配置 `chunk_overlap`
- 面向中英文的分隔符设计
- 基于 token 长度的切分

代码位置：

- Chunker 类：[app/knowledge/chunker.py](C:/Users/25921/Desktop/rag-app/app/knowledge/chunker.py:13)
- Splitter 初始化：[app/knowledge/chunker.py](C:/Users/25921/Desktop/rag-app/app/knowledge/chunker.py:21)
- 切块入口：[app/knowledge/chunker.py](C:/Users/25921/Desktop/rag-app/app/knowledge/chunker.py:31)

为什么这属于 RAG 优化：

- 重叠切块可以减少边界信息丢失
- 更合理的块形态会同时提升召回质量和回答的可解释性
- 基于 token 的切分比单纯按字符数更稳定

### 2.2 标题路径保留

在切块过程中，项目会尽量推断每个 chunk 的 `heading_path`。

代码位置：

- 标题路径推断：[app/knowledge/chunker.py](C:/Users/25921/Desktop/rag-app/app/knowledge/chunker.py:81)

为什么这属于 RAG 优化：

- 保留文档结构有助于提高可解释性
- 每个 chunk 的上下文身份更强
- 对法律、政策、规章、结构化手册类文档尤其有价值

### 2.3 向量化之前先做文本清洗

在切块和向量化之前，项目会先清洗解析后的文本。

当前处理内容包括：

- 去除重复页眉页脚
- 去除广告性文本
- 去除 URL
- 去除异常特殊字符
- 最小有效长度校验
- 敏感词拦截

代码位置：

- Cleaner 类：[app/knowledge/cleaner.py](C:/Users/25921/Desktop/rag-app/app/knowledge/cleaner.py:25)
- 去页眉页脚：[app/knowledge/cleaner.py](C:/Users/25921/Desktop/rag-app/app/knowledge/cleaner.py:170)
- 去广告内容：[app/knowledge/cleaner.py](C:/Users/25921/Desktop/rag-app/app/knowledge/cleaner.py:192)
- 敏感词检测：[app/knowledge/cleaner.py](C:/Users/25921/Desktop/rag-app/app/knowledge/cleaner.py:205)
- 最小长度校验：[app/knowledge/cleaner.py](C:/Users/25921/Desktop/rag-app/app/knowledge/cleaner.py:216)

为什么这属于 RAG 优化：

- 脏文本会同时降低 embedding 质量和检索质量
- 去除重复噪声能够提升向量表达的纯度
- 文本清洗可以明显减少知识库污染

### 2.4 复杂 PDF 的质量评估与降级处理

项目并没有只依赖单一 PDF 解析器。

当前实现了：

- 本地多策略解析
- OCR 回退
- 表格提取
- 本地解析质量评分
- 当质量不达标时回退到 MinerU API

代码位置：

- Parser 类：[app/knowledge/pdf_parser.py](C:/Users/25921/Desktop/rag-app/app/knowledge/pdf_parser.py:77)
- 缓存目录初始化：[app/knowledge/pdf_parser.py](C:/Users/25921/Desktop/rag-app/app/knowledge/pdf_parser.py:82)
- 本地质量评估：[app/knowledge/pdf_parser.py](C:/Users/25921/Desktop/rag-app/app/knowledge/pdf_parser.py:178)
- MinerU 回退调用：[app/knowledge/pdf_parser.py](C:/Users/25921/Desktop/rag-app/app/knowledge/pdf_parser.py:194)
- 质量评分函数：[app/knowledge/pdf_parser.py](C:/Users/25921/Desktop/rag-app/app/knowledge/pdf_parser.py:333)

为什么这属于 RAG 优化：

- 文档解析质量会直接影响后续切块和召回质量
- 复杂 PDF 是真实 RAG 系统中最常见的失败源之一
- 质量感知式降级能在不总是依赖远程解析的前提下提高系统鲁棒性

### 2.5 PDF 解析缓存

项目会缓存已经解析过的 PDF 结果，避免重复做昂贵的解析。

代码位置：

- 缓存命中路径：[app/knowledge/pdf_parser.py](C:/Users/25921/Desktop/rag-app/app/knowledge/pdf_parser.py:103)
- 缓存读取：[app/knowledge/pdf_parser.py](C:/Users/25921/Desktop/rag-app/app/knowledge/pdf_parser.py:508)
- 缓存写入：[app/knowledge/pdf_parser.py](C:/Users/25921/Desktop/rag-app/app/knowledge/pdf_parser.py:593)

为什么这属于 RAG 优化：

- 能减少重复解析成本
- 提升知识入库效率
- 对重复上传、反复测试、评测场景都很有帮助

## 3. 上下文构建侧优化

### 3.1 短期记忆 + 长期记忆

Context Builder 并不只依赖知识库检索结果。

它还会同时整合：

- Redis 中的最近对话记忆
- 长期记忆摘要
- 必要时从 MySQL 读取历史对话

代码位置：

- Context Builder 类：[app/chat/context_builder.py](C:/Users/25921/Desktop/rag-app/app/chat/context_builder.py:28)
- 最近记忆加载：[app/chat/context_builder.py](C:/Users/25921/Desktop/rag-app/app/chat/context_builder.py:61)
- 长期记忆加载：[app/chat/context_builder.py](C:/Users/25921/Desktop/rag-app/app/chat/context_builder.py:62)
- 记忆拼接渲染：[app/chat/context_builder.py](C:/Users/25921/Desktop/rag-app/app/chat/context_builder.py:261)

为什么这属于 RAG 优化：

- 真实问答质量不仅依赖知识检索，也依赖对话连续性
- 多轮 RAG 场景中，显式记忆注入非常重要

### 3.2 基于证据的回答约束

项目在提示词中明确要求模型优先依据检索证据来回答。

代码位置：

- 证据回答规则常量：[app/chat/context_builder.py](C:/Users/25921/Desktop/rag-app/app/chat/context_builder.py:19)
- 注入到消息列表：[app/chat/context_builder.py](C:/Users/25921/Desktop/rag-app/app/chat/context_builder.py:134)

为什么这属于 RAG 优化：

- 可以降低模型幻觉风险
- 能强化 grounded answer 的回答方式
- 让系统更贴合检索增强生成的目标

### 3.3 Prompt 注入前的上下文过滤

项目不会把召回结果原样全部塞进模型。

而是会在注入 prompt 之前做过滤，主要包括：

- 空文本过滤
- 部分分数过滤
- 限制最终引用证据条数

代码位置：

- 结果过滤逻辑：[app/chat/context_builder.py](C:/Users/25921/Desktop/rag-app/app/chat/context_builder.py:171)
- 最大证据条数限制：[app/chat/context_builder.py](C:/Users/25921/Desktop/rag-app/app/chat/context_builder.py:18)

为什么这属于 RAG 优化：

- 可以防止 prompt 过载
- 保持上下文简洁
- 有助于提高提示词质量和 token 利用效率

## 4. 可配置优化参数

这些优化并不是全部写死的，很多关键环节都已经做成了配置项。

主要可调参数包括：

- `RETRIEVAL_BM25_TOP_K`
- `RETRIEVAL_VECTOR_TOP_K`
- `RETRIEVAL_RRF_K`
- `RETRIEVAL_ENABLE_QUERY_REWRITE`
- `RETRIEVAL_ENABLE_RERANK`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `MINERU_FALLBACK_QUALITY_THRESHOLD`
- `PDF_PARSE_CACHE_DIR`

代码位置：

- 配置项定义：[app/core/config.py](C:/Users/25921/Desktop/rag-app/app/core/config.py:244)
- 检索相关参数：[app/core/config.py](C:/Users/25921/Desktop/rag-app/app/core/config.py:259)

为什么这属于 RAG 优化：

- 便于针对不同任务、不同模型、不同硬件环境进行调优
- 也让系统更适合做评测、对比实验和后续优化

## 5. 当前项目中已经具备的 RAG 优化能力总结

当前项目已经具备以下较为实用的 RAG 优化能力：

- 混合检索
- BM25 + 向量多路召回
- RRF 融合
- Cross-encoder 重排
- Query Rewrite
- 基于租户和角色的检索过滤
- 带重叠的语义切块
- 文档结构标题路径保留
- 向量化前的文本清洗
- 复杂 PDF 的质量感知降级
- PDF 解析缓存
- 对话记忆融合
- 基于证据的 Prompt 约束
- 生成前的上下文过滤

## 6. 当前尚未明确实现的高级 RAG 优化

根据当前代码，以下更高级的 RAG 策略暂时还没有明确落地：

- Multi-query expansion
- HyDE
- Parent-child retrieval
- Graph retrieval
- Self-reflection / answer verification
- Adaptive top-k
- 在线查询结果缓存

## 7. 一句话定位

这个项目已经不再是一个基础 RAG Demo，而是实现了一条围绕混合检索、重排、鲁棒文档解析与基于证据的上下文构建所展开的、具备实用性的多阶段优化版 RAG 流水线。
