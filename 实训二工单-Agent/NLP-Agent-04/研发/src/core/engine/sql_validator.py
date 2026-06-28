"""SQL 验证与执行器"""

from __future__ import annotations

import sqlite3
import threading
import time
from typing import Any

from config import settings
from src.core.models import SqlResult


class SQLValidator:
    """SQL 语法验证 + 安全执行"""

    def __init__(self, db_path: str):
        self._db_path = db_path

    def execute(self, sql: str) -> SqlResult:
        """执行 SQL 并返回结果"""
        start = time.perf_counter()

        # 异步执行 + 超时保护
        result_container: list[SqlResult] = []
        exception_container: list[Exception] = []

        def _run():
            try:
                conn = sqlite3.connect(self._db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(sql)
                rows = cursor.fetchmany(settings.SQL_MAX_ROWS)
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                result_container.append(SqlResult(
                    success=True,
                    columns=columns,
                    rows=[list(r) for r in rows],
                    row_count=len(rows),
                ))
                conn.close()
            except Exception as e:
                exception_container.append(e)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=settings.SQL_TIMEOUT_SECONDS)

        elapsed = (time.perf_counter() - start) * 1000

        if thread.is_alive():
            return SqlResult(
                success=False,
                error_message=f"SQL 执行超时 ({settings.SQL_TIMEOUT_SECONDS}s)",
                latency_ms=elapsed,
            )

        if exception_container:
            return SqlResult(
                success=False,
                error_message=str(exception_container[0]),
                latency_ms=elapsed,
            )

        result = result_container[0]
        result.latency_ms = elapsed
        return result

    def quick_check_only(self, sql: str) -> tuple[bool, str]:
        """仅检查语法，不执行（用于预检）"""
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
            conn.close()
            return True, ""
        except Exception as e:
            return False, str(e)
