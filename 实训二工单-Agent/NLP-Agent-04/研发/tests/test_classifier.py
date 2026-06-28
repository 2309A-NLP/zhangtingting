"""问题分类器测试"""

from __future__ import annotations

import pytest


class TestQuestionClassifier:
    """问题分类器测试"""

    def test_data_query_keywords(self):
        """测试数据查询关键词匹配"""
        from src.core.engine.classifier import QuestionClassifier
        from src.core.models import QuestionCategory

        classifier = QuestionClassifier()

        test_cases = [
            ("基金净值是多少？", QuestionCategory.DATA_QUERY),
            ("2021年涨幅最大的股票", QuestionCategory.DATA_QUERY),
            ("基金的持仓占比", QuestionCategory.DATA_QUERY),
            ("债券名称是什么？", QuestionCategory.DATA_QUERY),
        ]

        for question, expected in test_cases:
            result = classifier.classify(question)
            assert result == expected, f"Question: {question!r}, Expected {expected}, got {result}"

    def test_text_comprehension(self):
        """测试文本理解分类"""
        from src.core.engine.classifier import QuestionClassifier
        from src.core.models import QuestionCategory

        classifier = QuestionClassifier()

        test_cases = [
            ("招股书中提到的主要风险", QuestionCategory.TEXT_COMPREHENSION),
            ("PDF文件中的内容", QuestionCategory.TEXT_COMPREHENSION),
        ]

        for question, expected in test_cases:
            result = classifier.classify(question)
            assert result == expected, f"Question: {question!r}, Expected {expected}, got {result}"
