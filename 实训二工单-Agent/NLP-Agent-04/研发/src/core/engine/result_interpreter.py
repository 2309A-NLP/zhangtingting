"""结果解释器 — 将 SQL 执行结果转为自然语言"""

from __future__ import annotations

from typing import Any, Optional

from src.core.models import SqlResult
from src.services.llm_service import LLMService


class ResultInterpreter:
    """SQL 结果 → 自然语言"""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self._llm = llm_service

    def interpret(
        self,
        question: str,
        sql: str,
        sql_result: SqlResult,
        use_llm: bool = True,
    ) -> str:
        """将执行结果解释为自然语言"""
        if not sql_result.success:
            return f"查询执行出错：{sql_result.error_message}"

        if sql_result.row_count == 0:
            return "未查询到相关数据，请检查查询条件是否正确。"

        if sql_result.row_count == 1 and len(sql_result.columns) == 1:
            # 单一值直接返回
            value = sql_result.rows[0][0]
            return f"{value}"

        # 使用 LLM 生成自然语言回答
        if self._llm and use_llm:
            return self._interpret_with_llm(question, sql, sql_result)

        # 兜底：格式化表格
        return self._format_table(sql_result)

    def _interpret_with_llm(
        self, question: str, sql: str, result: SqlResult
    ) -> str:
        """用 LLM 生成自然语言回答"""
        result_str = self._result_to_text(result)
        prompt = f"""用户问题：{question}

查询 SQL：{sql}

查询结果（{result.row_count} 行，字段：{', '.join(result.columns)}）：
{result_str}

请根据查询结果用自然语言回答用户的问题，语言简洁准确。"""

        try:
            return self._llm.generate(prompt, max_tokens=500).strip()
        except Exception:
            return self._format_table(result)

    def _format_table(self, result: SqlResult) -> str:
        """格式化表格输出"""
        lines: list[str] = []
        for row in result.rows:
            pairs = []
            for i, col in enumerate(result.columns):
                val = row[i] if i < len(row) else ""
                pairs.append(f"{col}: {val}")
            lines.append(" | ".join(pairs))
        return "\n".join(lines)

    def _result_to_text(self, result: SqlResult) -> str:
        """将结果转为文本（用于 LLM Prompt）"""
        header = " | ".join(result.columns)
        rows = [" | ".join(str(v) for v in row) for row in result.rows[:20]]
        return f"【表头】{header}\n" + "\n".join(rows)
