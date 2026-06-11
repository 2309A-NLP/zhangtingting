# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def strip_company_query_prefixes(query: str) -> str:
    text = normalize_text(query)
    patterns = [
        r"^(根据|按照|请问|请结合|结合|关于)\s*",
        r"^(根据|按照).{0,20}?(招股意向书|招股说明书)[，,：:\s]*",
        r"^(在|从).{0,30}?(招股意向书|招股说明书|报告期内|图中|表中)[，,：:\s]*",
    ]
    prev = None
    while prev != text:
        prev = text
        for p in patterns:
            text = re.sub(p, "", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；])|\n+", str(text or ""))
    return [normalize_text(p) for p in parts if normalize_text(p)]
