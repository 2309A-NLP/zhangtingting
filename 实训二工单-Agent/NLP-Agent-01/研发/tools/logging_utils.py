# 工单编号：人工智能NLP-Agent数字人项目-记账本任务
# 来源：北京八维信息集团 · 八维文化与产业研究院

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def log_tool_call(tool_name: str, params: dict[str, Any], result: dict[str, Any]) -> None:
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "tool": tool_name,
        "params": params,
        "success": result.get("success", result.get("exists", False)),
        "message": result.get("message", ""),
        "record_id": result.get("id", None),
    }
    logger.info(json.dumps(log_entry, ensure_ascii=False))
