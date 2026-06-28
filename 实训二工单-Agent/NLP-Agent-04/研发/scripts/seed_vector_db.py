#!/usr/bin/env python3
"""
向量数据库初始化脚本
将few-shot案例索引到ChromaDB向量数据库中
"""

from __future__ import annotations

import json
from pathlib import Path

from config import settings
from src.core.retriever.vector_store import VectorStore
from src.services.embedding_service import EmbeddingService


def main():
    print("[INFO] 开始初始化向量数据库...")

    # 检查few-shot文件
    few_shot_path = Path(settings.FEW_SHOT_PATH)
    if not few_shot_path.exists():
        print(f"[ERROR] Few-shot文件不存在: {few_shot_path}")
        print("请先运行 scripts/build_few_shot.py 生成示例")
        return

    # 加载示例
    print(f"[INFO] 加载Few-shot案例: {few_shot_path}")
    with open(few_shot_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    print(f"   共加载 {len(examples)} 条示例")

    # 初始化向量存储
    print("[INFO] 初始化Embedding模型...")
    embedding_service = EmbeddingService()
    vector_store = VectorStore(embedding_service)

    # 清空旧数据
    if vector_store.count() > 0:
        print(f"[INFO] 清空旧数据 ({vector_store.count()} 条)")
        vector_store.delete_collection()
        vector_store = VectorStore(embedding_service)

    # 添加示例到向量库
    print("[INFO] 索引示例到向量数据库...")
    ids = [f"example_{i}" for i in range(len(examples))]
    texts = [ex["question"] for ex in examples]
    metadatas = [{
        "sql": ex["sql"],
        "table_names": ",".join(ex.get("table_names", [])),
        "category": ex.get("category", "data_query"),
    } for ex in examples]

    vector_store.add_texts(ids, texts, metadatas)
    print(f"[OK] 向量数据库初始化完成，共索引 {vector_store.count()} 条示例")

    # 验证
    print("\n[验证] 测试向量检索...")
    test_questions = [
        "2021年哪些基金收益率最高？",
        "股票涨跌幅如何计算？",
        "基金的规模是多少？",
    ]
    for q in test_questions:
        results = vector_store.similarity_search(q, top_k=1)
        if results:
            print(f"   Q: {q[:30]}...")
            print(f"   A: {results[0]['document'][:30]}...")
            print()


if __name__ == "__main__":
    main()
