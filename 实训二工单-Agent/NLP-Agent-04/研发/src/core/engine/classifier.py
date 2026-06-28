"""问题分类器 — 判断用户问题类型"""

from __future__ import annotations

import re
from typing import Optional

from src.core.models import QuestionCategory
from src.services.llm_service import LLMService

# 关键词 → 分类映射
_KEYWORD_MAP: list[tuple[re.Pattern, QuestionCategory]] = [
    (re.compile(r"招股书|说明书|文本|PDF|文件"), QuestionCategory.TEXT_COMPREHENSION),
    (re.compile(r"对比|分析|趋势|变化|波动|走势"), QuestionCategory.COMPLEX_ANALYSIS),
]

# 数据查询关键词（兜底）
_DATA_QUERY_KEYWORDS = re.compile(
    r"基金|净值|持仓|涨幅|跌幅|收益率|规模|份额|申购|赎回|"
    r"股票|债券|可转债|行业|行情|收盘|开盘|代码|名称|"
    r"占比|排名|前.*大|多少|几个|哪些|是什么"
)

# 数据库中没有的财务指标关键词 → 返回不支持提示
_OUT_OF_SCOPE_KEYWORDS = re.compile(
    r"存货|周转率|利润表|利润率|资产负债表|现金流|应收|应付|"
    r"毛利率|净利率|ROE|ROA|每股收益|市盈率|"
    r"营收|总收入|净利润|负债|资产总额|"
    r"股东持股明细|公司财务指标|招股书|"
    r"首发.*配售|战略配售"
)


class QuestionClassifier:
    """问题分类器：基于关键词 + LLM 兜底"""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self._llm = llm_service

    def classify(self, question: str) -> QuestionCategory:
        """三步分类策略"""
        # Step 0: 检查是否超出数据库范围
        if _OUT_OF_SCOPE_KEYWORDS.search(question):
            return QuestionCategory.UNKNOWN

        # Step 1: 关键词规则匹配
        for pattern, category in _KEYWORD_MAP:
            if pattern.search(question):
                return category

        # Step 2: 数据查询关键词判断
        if _DATA_QUERY_KEYWORDS.search(question):
            return QuestionCategory.DATA_QUERY

        # Step 3: LLM 兜底
        if self._llm:
            return self._classify_with_llm(question)

        return QuestionCategory.DATA_QUERY  # 默认走 NL2SQL

    def _classify_with_llm(self, question: str) -> QuestionCategory:
        """使用 LLM 分类"""
        prompt = f"""判断以下用户问题的类型，仅返回类型名称：
- data_query：涉及数据库查询（基金/股票/债券/行情/规模等结构化数据）
- text_comprehension：涉及文本理解（招股书、文档等）
- complex_analysis：涉及对比分析、趋势分析等复杂推理
- unknown：无法确定

问题：{question}

类型："""
        try:
            result = self._llm.generate(prompt, max_tokens=20).strip().lower()
            return QuestionCategory(result)
        except Exception:
            return QuestionCategory.DATA_QUERY
