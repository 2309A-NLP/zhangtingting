"""批量执行引擎"""

from __future__ import annotations

import uuid
from threading import Lock
from typing import Optional

from src.core.engine.pipeline import NL2SQLPipeline
from src.core.models import BatchTask, ChatRequest, AnswerResult


class BatchRunner:
    """批量问答执行器"""

    def __init__(self, pipeline: NL2SQLPipeline):
        self._pipeline = pipeline
        self._tasks: dict[str, BatchTask] = {}
        self._lock = Lock()

    def submit(self, questions: list[str]) -> BatchTask:
        """提交批量任务（同步执行）"""
        batch_id = f"batch_{uuid.uuid4().hex[:12]}"
        task = BatchTask(
            batch_id=batch_id,
            total=len(questions),
            status="processing",
        )
        self._tasks[batch_id] = task

        # 同步执行（后续可改为异步 + 进程池）
        for i, q in enumerate(questions):
            try:
                result = self._pipeline.run(ChatRequest(question=q))
                task.results.append(result)
                if result.success:
                    task.completed += 1
                else:
                    task.failed += 1
            except Exception:
                task.failed += 1

        task.status = "completed"
        return task

    def get_task(self, batch_id: str) -> Optional[BatchTask]:
        return self._tasks.get(batch_id)
