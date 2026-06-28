"""Few-shot 检索模块 — 从向量库检索相似案例"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from config import settings
from src.core.models import FewShotExample, QuestionCategory


class FewShotRetriever:
    """基于关键词 + 向量的 Few-shot 案例检索"""

    def __init__(
        self,
        examples_path: Optional[str] = None,
        use_vector: bool = True,
    ):
        self._examples: list[FewShotExample] = []
        self._use_vector = use_vector
        self._vector_store = None
        path = Path(examples_path or settings.FEW_SHOT_PATH)
        if path.exists():
            self._load(path)

    def _load(self, path: Path) -> None:
        """从 JSON 文件加载案例"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            self._examples.append(FewShotExample(
                question=item["question"],
                sql=item["sql"],
                table_names=item.get("table_names", []),
                category=QuestionCategory(item.get("category", "data_query")),
            ))

        # 如果启用向量检索且有示例，加载向量存储
        if self._use_vector and self._examples:
            self._init_vector_store()

    def _init_vector_store(self) -> None:
        """初始化向量存储"""
        try:
            from src.core.retriever.vector_store import VectorStore
            from src.services.embedding_service import EmbeddingService

            # 创建向量存储
            self._vector_store = VectorStore(EmbeddingService())

            # 如果向量库为空，从few-shot示例初始化
            if self._vector_store.count() == 0:
                self._index_examples()
        except Exception as e:
            # 向量服务不可用，降级到关键词匹配
            self._use_vector = False
            self._vector_store = None

    def _index_examples(self) -> None:
        """将示例索引到向量库"""
        if not self._vector_store or not self._examples:
            return

        ids = [f"example_{i}" for i in range(len(self._examples))]
        texts = [ex.question for ex in self._examples]
        metadatas = [{
            "sql": ex.sql,
            "table_names": ",".join(ex.table_names),
            "category": ex.category.value,
        } for ex in self._examples]

        self._vector_store.add_texts(ids, texts, metadatas)

    def retrieve(self, question: str, top_k: int = 3) -> list[FewShotExample]:
        """检索最相似的 few-shot 案例"""
        if not self._examples:
            return []

        # 优先使用向量检索
        if self._use_vector and self._vector_store:
            return self._retrieve_by_vector(question, top_k)

        # 降级：使用关键词匹配
        return self._retrieve_by_keyword(question, top_k)

    def _retrieve_by_vector(self, question: str, top_k: int) -> list[FewShotExample]:
        """使用向量相似度检索"""
        try:
            results = self._vector_store.similarity_search(question, top_k)
            retrieved = []
            for r in results:
                metadata = r.get("metadata", {})
                # 在本地examples中找到匹配的示例
                for ex in self._examples:
                    if ex.question == r.get("document"):
                        retrieved.append(ex)
                        break
                else:
                    # 如果没找到，创建新的示例
                    retrieved.append(FewShotExample(
                        question=r.get("document", ""),
                        sql=metadata.get("sql", ""),
                        table_names=metadata.get("table_names", "").split(",") if metadata.get("table_names") else [],
                        category=QuestionCategory(metadata.get("category", "data_query")),
                    ))
            return retrieved[:top_k]
        except Exception:
            # 向量检索失败，降级到关键词
            return self._retrieve_by_keyword(question, top_k)

    def _retrieve_by_keyword(self, question: str, top_k: int) -> list[FewShotExample]:
        """基于关键词共现的简单相似度检索"""
        scored = [
            (self._similarity(question, ex.question), ex)
            for ex in self._examples
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:top_k]]

    def _similarity(self, q1: str, q2: str) -> float:
        """基于关键词共现的简单相似度"""
        # 提取中文关键词（去掉停用词）
        stops = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
                 "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
                 "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "什么",
                 "吗", "啊", "呢", "吧", "怎么", "如何", "请", "帮", "为"}
        words1 = {w for w in q1 if w not in stops and len(w) > 1}
        words2 = {w for w in q2 if w not in stops and len(w) > 1}
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        return len(intersection) / max(len(words1), len(words2))

    def add_example(self, example: FewShotExample) -> None:
        """动态添加案例"""
        self._examples.append(example)
        # 如果有向量存储，添加到向量库
        if self._use_vector and self._vector_store:
            self._vector_store.add_texts(
                ids=[f"example_{len(self._examples) - 1}"],
                texts=[example.question],
                metadatas=[{
                    "sql": example.sql,
                    "table_names": ",".join(example.table_names),
                    "category": example.category.value,
                }],
            )

    def rebuild_index(self) -> None:
        """重建向量索引"""
        if self._use_vector:
            self._init_vector_store()

    @property
    def count(self) -> int:
        return len(self._examples)

    @property
    def use_vector(self) -> bool:
        return self._use_vector and self._vector_store is not None
