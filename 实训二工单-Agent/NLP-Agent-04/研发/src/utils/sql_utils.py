"""SQL 相关工具函数"""

from __future__ import annotations

import re


def extract_table_names(sql: str) -> list[str]:
    """从 SQL 中提取涉及的表名"""
    pattern = re.compile(
        r"\b(?:FROM|JOIN)\s+[\"']?(\w+)[\"']?",
        re.IGNORECASE,
    )
    return list(dict.fromkeys(pattern.findall(sql)))


def format_sql(sql: str) -> str:
    """简单格式化 SQL"""
    keywords = ["SELECT", "FROM", "WHERE", "AND", "OR", "ORDER BY", "GROUP BY",
                 "HAVING", "LIMIT", "JOIN", "LEFT JOIN", "RIGHT JOIN",
                 "INNER JOIN", "ON", "AS", "DISTINCT", "CASE", "WHEN", "THEN",
                 "ELSE", "END", "IN", "NOT", "BETWEEN", "LIKE", "IS", "NULL"]
    result = sql.strip()
    for kw in sorted(keywords, key=len, reverse=True):
        result = re.sub(
            rf"\b{re.escape(kw)}\b",
            f"\n{kw}",
            result,
            flags=re.IGNORECASE,
        )
    return result.strip()
