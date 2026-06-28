"""Schema 选择器 — 根据问题确定需要哪些数据库表"""

from __future__ import annotations

import re
from typing import Optional

from src.core.models import TableSchema
from src.services.llm_service import LLMService
from src.services.db_service import DatabaseService

# 关键词 → 候选表映射（使用实际中文表名）
_KEYWORD_TABLE_MAP: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"持仓|重仓|持有.*股|持有.*债|占比|前.*大"), [
        "基金股票持仓明细", "基金债券持仓明细",
        "基金可转债持仓明细",
    ]),
    (re.compile(r"净值|收益率|日行情|涨跌幅|回报|资产净值"), [
        "基金日行情表",
    ]),
    (re.compile(r"规模|份额|申购|赎回|份额变动"), [
        "基金规模变动表",
    ]),
    (re.compile(r"持有人|机构|个人|户数"), [
        "基金份额持有人结构",
    ]),
    (re.compile(r"股票|A股|行情|开盘|收盘|涨跌|涨停|跌停"), [
        "A股票日行情表", "A股公司行业划分表",
    ]),
    (re.compile(r"港股|香港"), [
        "港股票日行情表",
    ]),
    (re.compile(r"行业|中信|一级行业|二级行业"), [
        "A股公司行业划分表", "A股票日行情表",
    ]),
    (re.compile(r"债券|国债|企业债|可转债"), [
        "基金债券持仓明细", "基金可转债持仓明细",
    ]),
    (re.compile(r"规模|费率|经理|成立|类型|管理人|托管"), [
        "基金基本信息",
    ]),
]

# 默认包含基础信息表
_DEFAULT_TABLES = ["基金基本信息"]


class SchemaSelector:
    """Schema 选择器：关键词匹配 + LLM 精排"""

    def __init__(
        self,
        db_service: DatabaseService,
        llm_service: Optional[LLMService] = None,
    ):
        self._db = db_service
        self._llm = llm_service

    def select(self, question: str) -> list[TableSchema]:
        """选择问题所需的表（最多 3 张）"""
        # Step 1: 关键词匹配
        matched_tables: set[str] = set(_DEFAULT_TABLES)
        for pattern, tables in _KEYWORD_TABLE_MAP:
            if pattern.search(question):
                matched_tables.update(tables)

        # Step 2: 全量获取 Schema
        all_schemas = {s.name: s for s in self._db.get_all_table_schemas()}

        # Step 3: LLM 精排（超过 3 张表时精简）
        matched_list = [t for t in matched_tables if t in all_schemas]
        if len(matched_list) > 3 and self._llm:
            matched_list = self._refine_with_llm(question, matched_list)

        return [all_schemas[t] for t in matched_list[:3] if t in all_schemas]

    def _refine_with_llm(self, question: str, candidates: list[str]) -> list[str]:
        """用 LLM 从候选表中选出最相关的 1-4 张表"""
        tables_str = "\n".join(f"- {t}" for t in candidates)
        prompt = f"""用户问题：{question}

候选数据库表：
{tables_str}

请选出解决问题最需要的表（最多 4 张），仅返回表名，逗号分隔："""
        try:
            result = self._llm.generate(prompt, max_tokens=100).strip()
            selected = [t.strip() for t in result.replace("，", ",").split(",") if t.strip()]
            return [t for t in selected if t in candidates][:4]
        except Exception:
            return candidates[:4]

    def format_schema_prompt(self, schemas: list[TableSchema]) -> str:
        """将 Schema 格式化为 Prompt 可用的文本"""
        parts: list[str] = []
        for table in schemas:
            cols = "\n  ".join(
                f"- {c.name} ({c.data_type}): {c.description or c.name}"
                for c in table.columns
            )
            parts.append(f"表名: {table.name}\n  字段:\n  {cols}")
        return "\n\n".join(parts)
