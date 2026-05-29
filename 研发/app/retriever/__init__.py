"""Retrieval services."""
'''
用户输入: "它多少钱？"
    │
    ▼
query_rewrite.py
    │
    ├── 输出 RewriteResult
    │       ├── rewritten_query: "产品X的价格是多少？"
    │       └── reason: "补充了上下文"
    │
    ▼
hybrid.py（混合检索）
    │
    ├── 向量检索 → list[RetrievedChunk] (dense_results)
    ├── BM25检索 → list[RetrievedChunk] (bm25_results)
    ├── 融合排序 → list[RetrievedChunk] (fused_results)
    │
    ▼
RetrievalBundle（打包返回）
    │
    ├── query: "它多少钱？"
    ├── rewritten_query: "产品X的价格是多少？"
    ├── dense_results: [...]
    ├── bm25_results: [...]
    └── fused_results: [...]
    │
    ▼
context_builder.py → 从 fused_results 取前 K 条 → 拼进 prompt
'''