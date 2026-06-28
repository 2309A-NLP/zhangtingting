"""测试工具函数"""

from __future__ import annotations

from src.utils.sql_utils import extract_table_names, format_sql


def test_extract_table_names():
    sql = "SELECT * FROM fund_basic_info JOIN fund_daily_market ON ..."
    tables = extract_table_names(sql)
    assert "fund_basic_info" in tables
    assert "fund_daily_market" in tables


def test_format_sql():
    sql = "SELECT * FROM fund WHERE name = 'test'"
    formatted = format_sql(sql)
    assert "SELECT" in formatted
    assert "FROM" in formatted
    assert "WHERE" in formatted
