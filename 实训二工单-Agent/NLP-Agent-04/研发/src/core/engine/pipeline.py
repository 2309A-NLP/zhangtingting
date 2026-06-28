"""NL2SQL 主流程编排器"""

from __future__ import annotations

import time
from typing import Optional

from config import settings
from src.core.engine.classifier import QuestionClassifier
from src.core.engine.schema_selector import SchemaSelector
from src.core.engine.sql_generator import SQLGenerator
from src.core.engine.sql_validator import SQLValidator
from src.core.engine.result_interpreter import ResultInterpreter
from src.core.models import (
    AnswerResult,
    ChatRequest,
    FewShotExample,
    QuestionCategory,
)
from src.core.retriever.few_shot import FewShotRetriever
from src.services.llm_service import LLMService
from src.services.db_service import DatabaseService
from src.services.cache_service import CacheService


class NL2SQLPipeline:
    """NL2SQL 主流程编排"""

    def __init__(
        self,
        llm_service: LLMService,
        db_service: DatabaseService,
        cache_service: Optional[CacheService] = None,
        few_shot_retriever: Optional[FewShotRetriever] = None,
    ):
        self._llm = llm_service
        self._db = db_service
        self._cache = cache_service

        self._classifier = QuestionClassifier(llm_service)
        self._schema_selector = SchemaSelector(db_service, llm_service)
        self._sql_generator = SQLGenerator(llm_service, db_service)
        self._sql_validator = SQLValidator(settings.DB_PATH)
        self._interpreter = ResultInterpreter(llm_service)
        self._few_shot = few_shot_retriever

    def run(self, request: ChatRequest) -> AnswerResult:
        """执行完整 NL2SQL Pipeline"""
        start = time.perf_counter()
        question = request.question.strip()

        # Step 0: 检查缓存
        if self._cache:
            cached = self._cache.get(question)
            if cached:
                cached.latency_ms = (time.perf_counter() - start) * 1000
                return cached

        # Step 1: 问题分类
        category = self._classifier.classify(question)

        # 非数据查询类型，给出提示
        if category == QuestionCategory.TEXT_COMPREHENSION:
            return AnswerResult(
                question=question,
                answer="该问题涉及文本理解（招股书等），该功能将在后续版本支持。",
                category=category,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        if category == QuestionCategory.UNKNOWN:
            return AnswerResult(
                question=question,
                answer="该问题涉及公司财务指标（如存货、周转率、利润表等），当前数据库仅包含基金/股票行情数据，无法回答。",
                category=category,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # Step 2: Schema 选择
        schemas = self._schema_selector.select(question)
        table_names = [s.name for s in schemas]

        # Step 3: Few-shot 检索
        few_shot_examples: list[FewShotExample] = []
        if request.enable_few_shot and self._few_shot:
            few_shot_examples = self._few_shot.retrieve(question)

        # Step 4: SQL 生成（最多重试 SQL_MAX_RETRIES 次）
        sql = ""
        sql_result = None
        for attempt in range(settings.SQL_MAX_RETRIES + 1):
            sql = self._sql_generator.generate(question, schemas, few_shot_examples)

            # 安全检查
            is_safe, error_msg = self._sql_generator.validate(sql)
            if not is_safe:
                if attempt < settings.SQL_MAX_RETRIES:
                    few_shot_examples = self._build_retry_few_shot(
                        f"SQL 安全问题: {error_msg}", few_shot_examples
                    )
                    continue
                return AnswerResult(
                    question=question,
                    answer=f"无法生成安全的 SQL 查询：{error_msg}",
                    sql=sql,
                    category=category,
                    tables_used=table_names,
                    success=False,
                    error_message=error_msg,
                    latency_ms=(time.perf_counter() - start) * 1000,
                )

            # Step 5: SQL 执行
            sql_result = self._sql_validator.execute(sql)
            if sql_result.success:
                break

            # 失败后重试
            if attempt < settings.SQL_MAX_RETRIES:
                few_shot_examples = self._build_retry_few_shot(
                    f"SQL 执行错误: {sql_result.error_message}", few_shot_examples
                )

        # Step 6: 结果解释
        if sql_result and sql_result.success:
            answer = self._interpreter.interpret(question, sql, sql_result)
        else:
            error = sql_result.error_message if sql_result else "SQL 生成失败"
            answer = f"查询失败：{error}"

        result = AnswerResult(
            question=question,
            answer=answer,
            sql=sql,
            sql_result=sql_result,
            tables_used=table_names,
            latency_ms=(time.perf_counter() - start) * 1000,
            model_used=settings.LLM_MODEL_NAME,
            category=category,
            success=bool(sql_result and sql_result.success),
            error_message=sql_result.error_message if sql_result and not sql_result.success else "",
        )

        # 写入缓存
        if self._cache:
            self._cache.set(question, result)

        return result

    def _build_retry_few_shot(
        self, error_info: str, existing: list[FewShotExample]
    ) -> list[FewShotExample]:
        """构建包含错误反馈的 few-shot"""
        error_example = FewShotExample(
            question=f"注意：之前的 SQL 有错误（{error_info}），请修正。",
            sql="-- 请根据错误信息修正 SQL",
            table_names=[],
        )
        return [error_example] + existing[:2]
