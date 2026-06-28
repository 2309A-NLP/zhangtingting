"""SQL 生成器测试"""

from __future__ import annotations

import pytest


class TestSQLGenerator:
    """SQL 生成器单元测试"""

    def test_clean_sql_simple(self):
        """测试简单 SQL 清理"""
        from src.core.engine.sql_generator import SQLGenerator
        from src.services.llm_service import LLMService
        from src.services.db_service import DatabaseService

        # 这里只测 _clean_sql 方法，不依赖 LLM
        gen = SQLGenerator.__new__(SQLGenerator)

        test_cases = [
            ("SELECT * FROM fund_basic_info;", "SELECT * FROM fund_basic_info;"),
            (
                "```sql\nSELECT name FROM fund;\n```",
                "SELECT name FROM fund;",
            ),
            (
                "好的，这是SQL：\nSELECT count(*) FROM fund;",
                "SELECT count(*) FROM fund;",
            ),
        ]

        for raw, expected in test_cases:
            result = gen._clean_sql(raw)
            assert result == expected, f"Expected {expected!r}, got {result!r}"

    def test_validate_select_only(self):
        """测试 SQL 安全校验"""
        from src.core.engine.sql_generator import SQLGenerator

        gen = SQLGenerator.__new__(SQLGenerator)

        safe, msg = gen.validate("SELECT * FROM fund;")
        assert safe is True
        assert msg == ""

        unsafe_sqls = [
            "DROP TABLE fund;",
            "DELETE FROM fund;",
            "INSERT INTO fund VALUES (1);",
            "ALTER TABLE fund ADD COLUMN x;",
        ]

        for sql in unsafe_sqls:
            safe, msg = gen.validate(sql)
            assert safe is False, f"Should reject: {sql}"
            assert msg != "", "Should have error message"
