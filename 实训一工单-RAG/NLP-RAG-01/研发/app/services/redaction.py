# 工单: 人工智能NLP-RAG-基于PDF文档的问答系统

from __future__ import annotations

import re
from typing import Dict, Match, Tuple


REDACTION_PATTERNS = {
    "id_number": re.compile(r"\b\d{17}[\dXx]\b"),
    "secret_unit": re.compile(r"单位[A-Z]"),
    "named_person_field": re.compile(
        r"(?P<label>法定代表人|签字注册会计师|保荐代表人|经办注册会计师|经办资产评估师)[:：]?\s*(?P<name>[\u4e00-\u9fff]{2,4})"
    ),
}


def redact_sensitive_text(text: str) -> Tuple[str, Dict[str, int]]:
    stats = {"id_number": 0, "secret_unit": 0, "person_name": 0, "amount": 0}
    redacted = text
    redacted, secret_count = REDACTION_PATTERNS["secret_unit"].subn("[REDACTED]", redacted)
    stats["secret_unit"] = secret_count
    redacted, id_count = REDACTION_PATTERNS["id_number"].subn("[REDACTED]", redacted)
    stats["id_number"] = id_count

    def person_replacer(match: Match[str]) -> str:
        stats["person_name"] += 1
        return f"{match.group('label')} [REDACTED]"

    redacted = REDACTION_PATTERNS["named_person_field"].sub(person_replacer, redacted)

    def amount_replacer(match: Match[str]) -> str:
        stats["amount"] += 1
        unit = match.group("unit") or ""
        return f"[REDACTED]{unit}"

    redacted = re.sub(
        r"(?P<value>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>亿元|万元|元|万股|股)",
        amount_replacer,
        redacted,
    )
    return redacted, stats
