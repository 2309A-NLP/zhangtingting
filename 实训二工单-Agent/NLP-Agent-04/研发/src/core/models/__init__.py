"""领域模型定义"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QuestionCategory(str, Enum):
    """问题类型"""
    DATA_QUERY = "data_query"           # 数据查询（NL2SQL）
    TEXT_COMPREHENSION = "text_comprehension"  # 文本理解
    COMPLEX_ANALYSIS = "complex_analysis"      # 复杂分析
    UNKNOWN = "unknown"                 # 无法分类


@dataclass
class ColumnInfo:
    """字段信息"""
    name: str
    data_type: str
    description: str = ""
    example_values: str = ""
    is_primary_key: bool = False
    is_foreign_key: bool = False
    fk_ref_table: str = ""
    fk_ref_column: str = ""


@dataclass
class TableSchema:
    """表结构"""
    name: str
    description: str = ""
    columns: list[ColumnInfo] = field(default_factory=list)


@dataclass
class FewShotExample:
    """Few-shot 示例"""
    question: str
    sql: str
    table_names: list[str]
    category: QuestionCategory = QuestionCategory.DATA_QUERY
    embedding: list[float] | None = None


@dataclass
class SqlResult:
    """SQL 执行结果"""
    success: bool
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    row_count: int = 0
    error_message: str = ""
    latency_ms: float = 0.0


@dataclass
class AnswerResult:
    """最终问答结果"""
    question: str
    answer: str
    sql: str = ""
    sql_result: SqlResult | None = None
    tables_used: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    model_used: str = ""
    category: QuestionCategory = QuestionCategory.DATA_QUERY
    success: bool = True
    error_message: str = ""


@dataclass
class ChatRequest:
    """问答请求"""
    question: str
    model: str = ""
    temperature: float | None = None
    enable_few_shot: bool = True
    session_id: str = ""


@dataclass
class BatchTask:
    """批处理任务"""
    batch_id: str
    total: int
    completed: int = 0
    failed: int = 0
    status: str = "pending"  # pending / processing / completed / failed
    results: list[AnswerResult] = field(default_factory=list)
