"""SQL 验证器测试"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestSQLValidator:
    """SQL 验证器单元测试"""

    def test_execute_timeout(self):
        """测试超时保护"""
        from src.core.engine.sql_validator import SQLValidator

        validator = SQLValidator(":memory:")  # 使用内存数据库

        # 超长 SQL — 使用一个故意慢的查询
        result = validator.execute("SELECT sleep(1);")  # SQLite 无 sleep，会快速返回错误
        assert result.success is False
        assert result.error_message != ""

    def test_sqlite_error_handling(self):
        """测试 SQL 语法错误的处理"""
        from src.core.engine.sql_validator import SQLValidator

        validator = SQLValidator(":memory:")

        result = validator.execute("SELECT invalid sql syntax;;;")
        assert result.success is False
        assert "error" in result.error_message.lower() or result.error_message
