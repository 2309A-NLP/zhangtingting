"""SQL 生成器 — 使用 LLM 从自然语言生成 SQL"""

from __future__ import annotations

import re
from typing import Optional

from src.core.models import TableSchema, FewShotExample
from src.services.llm_service import LLMService
from src.services.db_service import DatabaseService

_SQL_BLOCK_RE = re.compile(r"```sql\s*\n?(.*?)\n?```", re.DOTALL)
_COLUMN_WITH_PARENS_RE = re.compile(r'(?<![a-zA-Z0-9"_])(\b[\u4e00-\u9fa5_a-zA-Z]+)\(([\u4e00-\u9fa5a-zA-Z]+)\)(?![a-zA-Z0-9")_])')
'''
(?<![a-zA-Z0-9"_])    # ① 负向后顾：前面不能是字母/数字/下划线/双引号
(                      # ② 捕获组1：函数名
    \b                 # 单词边界
    [\u4e00-\u9fa5_a-zA-Z]+  # 中文/英文/下划线（至少1个）
)                      # ③ 结束捕获组1
\(                     # 匹配左括号 ( 字面量
(                      # ④ 捕获组2：参数值
    [\u4e00-\u9fa5a-zA-Z]+   # 中文/英文字母（至少1个）
)                      # ⑤ 结束捕获组2
\)                     # 匹配右括号 ) 字面量
(?![a-zA-Z0-9")_])    # ⑥ 负向前瞻：后面不能是字母/数字/双引号/下划线/右括号
'''
_SELECT_RE = re.compile(r"^\s*SELECT\s+", re.IGNORECASE)


def _fix_column_quotes(sql: str) -> str:
    """修复带括号的列名，添加双引号包裹"""
    def replacer(match):
        col_name = match.group(1)
        unit = match.group(2)
        return f'"{col_name}({unit})"'
    
    return _COLUMN_WITH_PARENS_RE.sub(replacer, sql)
_DANGEROUS_RE = re.compile(
    r"\b(DROP|ALTER|DELETE|INSERT|UPDATE|CREATE|TRUNCATE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)

_PROMPT_TEMPLATE = """你是一个金融数据库SQL专家。根据用户问题生成SQLite查询语句。

【数据库Schema】
{schema_text}

【重要规则】
1. 只使用上述Schema中存在的表名和字段名
2. 日期格式: 'YYYYMMDD'（如 20210331，不带横杠）
3. 涨跌幅计算: (收盘价 - 昨收盘) / 昨收盘 * 100，结果保留两位小数
4. 表名和字段名使用数据库实际名称
5. 重要：包含括号的列名必须用双引号包裹，如："收盘价(元)"、"昨收盘(元)"
6. 【必须】ORDER BY 查询必须添加 LIMIT 限制返回条数，如 LIMIT 10、LIMIT 1
7. 【重要】JOIN 行业表时必须同时指定交易日日期条件，避免笛卡尔积！例如：JOIN A股公司行业划分表 i ON a.股票代码 = i.股票代码 AND a.交易日 = i.交易日期
8. 只返回SQL语句，不要多余解释

{examples_text}

【用户问题】
{question}

SQL:"""


class SQLGenerator:
    """NL2SQL 核心生成器"""

    def __init__(
        self,
        llm_service: LLMService,
        db_service: DatabaseService,
    ):
        self._llm = llm_service
        self._db = db_service

    def generate(
        self,
        question: str,
        schemas: list[TableSchema],
        few_shot_examples: Optional[list[FewShotExample]] = None,
    ) -> str:
        """生成 SQL"""
        schema_text = self._format_schemas(schemas)
        examples_text = self._format_examples(few_shot_examples or [])

        prompt = _PROMPT_TEMPLATE.format(
            schema_text=schema_text,
            examples_text=examples_text,
            question=question,
        )

        raw_sql = self._llm.generate(prompt)
        return self._clean_sql(raw_sql)

    def validate(self, sql: str) -> tuple[bool, str]:
        """验证 SQL 安全性"""
        if not sql:
            return False, "SQL 为空"

        if not _SELECT_RE.match(sql):
            return False, "仅支持 SELECT 查询"

        if _DANGEROUS_RE.search(sql):
            return False, f"SQL 包含危险关键字: {_DANGEROUS_RE.search(sql).group()}"

        return True, ""

    def _clean_sql(self, raw: str) -> str:
        """从 LLM 输出中提取并清理 SQL"""
        # 提取 markdown 代码块
        match = _SQL_BLOCK_RE.search(raw)
        if match:
            sql = match.group(1).strip()
        else:
            sql = raw.strip()

        # 去掉多余的前缀/后缀文字
        lines = sql.split("\n")
        sql_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.upper().startswith(("SELECT", "WITH", "/*")):
                sql_lines.append(line)
            elif sql_lines and not stripped.endswith(";"):
                sql_lines.append(line)

        sql = "\n".join(sql_lines).strip().rstrip(";") + ";"
        
        # 修复带括号的列名（如收盘价(元) → "收盘价(元)"）
        sql = _fix_column_quotes(sql)
        
        # 自动添加 LIMIT（防止全表排序超时）
        if "ORDER BY" in sql.upper() and "LIMIT" not in sql.upper():
            sql = sql.rstrip(";") + " LIMIT 10;"
        
        return sql

    def _format_schemas(self, schemas: list[TableSchema]) -> str:
        parts = []
        for t in schemas:
            cols = ", ".join(
                f"{c.name}" + (f"  # {c.description}" if c.description else "")
                for c in t.columns
            )
            parts.append(f"{t.name}: {cols}")
        return "\n".join(parts)

    def _format_examples(self, examples: list[FewShotExample]) -> str:
        if not examples:
            return ""
        parts = []
        for i, ex in enumerate(examples, 1):
            parts.append(f"例{i}: {ex.question} -> {ex.sql}")
        return "【参考】\n" + "\n".join(parts)
