from __future__ import annotations
'''定义了检索模块的核心数据结构'''
from dataclasses import dataclass, field
from typing import Any

# 检索到的单个结果
@dataclass(slots=True)
class RetrievedChunk:
    id: str                      # chunk 的唯一 ID
    doc_id: str                  # 所属文档 ID
    chunk_id: str                # chunk ID（如 "doc_123_0"）
    text: str                    # chunk 的文本内容
    source: str                  # 来源（MinIO 路径或 URL）
    role_category: str           # 角色分类（用于过滤）
    score: float                 # 相似度分数（越高越相关）
    # score 的含义：
    # 向量检索：余弦相似度，0-1 之间，越接近 1 越相似
    # BM25：相关度分数，可以是任意正数
    retrieval_type: str          # 检索类型（"dense" 或 "bm25"）
    heading_path: str = ""       # 章节路径（如 "第1章 > 第2节"）
    metadata: dict[str, Any] = field(default_factory=dict)  # 扩展元数据

# 检索结果包
@dataclass(slots=True)
class RetrievalBundle:
    query: str                      # 原始用户查询
    rewritten_query: str            # 改写后的查询
    dense_results: list[RetrievedChunk]   # 向量检索结果列表
    bm25_results: list[RetrievedChunk]    # BM25 检索结果列表
    # 用途：在 context_builder.py 中，最终使用的是 fused_results（融合结果）。
    fused_results: list[RetrievedChunk]   # 融合后的最终结果

# 查询改写结果  用途：query_rewrite.py 的输出。
@dataclass(slots=True)
class RewriteResult:
    original_query: str      # 原始查询
    rewritten_query: str     # 改写后的查询
    reason: str              # 为什么要这样改写（可解释性）
