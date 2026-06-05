# 工单: 人工智能NLP-RAG-基于PDF文档的问答系统

from __future__ import annotations

import re
from typing import List

from app.schemas import QueryIntent


INTENT_RULES = {
    "definition": ["是什么", "定义", "含义", "what is", "define"],
    "statistics": ["多少", "占比", "金额", "收入", "数量", "how much", "ratio"],
    "comparison": ["区别", "对比", "比较", "versus", "compare"],
    "entity_lookup": ["谁", "哪家", "哪个", "法定代表人", "供应商", "客户", "who", "which"],
}


def detect_language(query: str) -> str:
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", query)
    return "zh" if chinese_chars else "en"


def infer_intent(query: str) -> str:
    lowered = query.lower()
    for intent, patterns in INTENT_RULES.items():
        if any(token in query or token in lowered for token in patterns):
            return intent
    return "factoid"


def rewrite_query(query: str) -> str:
    compact = re.sub(r"\s+", " ", query).strip()
    return compact.replace("？", "?").replace("，", ",")


def find_ambiguities(query: str) -> List[str]:
    notes: List[str] = []
    if "页" in query and not re.search(r"\d+", query):
        notes.append("Query references pages without a concrete page number.")
    if any(token in query for token in ["它", "该公司", "this company"]):
        notes.append("Pronoun detected; answer assumes the company is the prospectus issuer.")
    return notes


def split_complex_question(query: str) -> List[str]:
    separators = ["以及", "并且", "和", " and ", "；", ";"]
    parts = [query]
    for sep in separators:
        refined: List[str] = []
        for item in parts:
            if sep in item:
                refined.extend([piece.strip() for piece in item.split(sep) if piece.strip()])
            else:
                refined.append(item)
        parts = refined
    return parts if len(parts) > 1 else []


def analyze_query(query: str) -> QueryIntent:
    language = detect_language(query)
    return QueryIntent(
        language=language,
        intent=infer_intent(query),
        rewritten_query=rewrite_query(query),
        ambiguities=find_ambiguities(query),
        sub_questions=split_complex_question(query),
    )
