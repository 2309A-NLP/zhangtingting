# 工单: 人工智能NLP-RAG-基于PDF文档的问答系统
from __future__ import annotations

import re
from typing import List

from app.schemas import QueryIntent
from app.services.text_utils import dedupe_preserve_order


INTENT_RULES = {
    "definition": ["是什么", "定义", "含义", "what is", "define"],
    "statistics": ["多少", "占比", "金额", "收入", "数量", "how much", "ratio"],
    "comparison": ["区别", "对比", "比较", "versus", "compare"],
    "entity_lookup": ["谁", "哪家", "哪个", "法定代表人", "供应商", "客户", "who", "which"],
}

QUERY_EXPANSIONS = {
    "法定代表人": ["公司法定代表人", "发行人法定代表人", "企业法定代表人"],
    "注册资本": ["公司注册资本", "发行人注册资本", "企业注册资本"],
    "补充流动资金": ["募集资金补充流动资金", "补充流动资金金额", "募集资金用途 补充流动资金"],
    "军用领域收入": ["军用收入", "军用领域 营业收入", "军品收入"],
    "军用收入占比": ["军用收入 占比", "军用领域收入 占比", "军品收入 占比"],
    "技术标准": ["参与制定 技术标准", "参与制定 标准", "参与制定 规范"],
    "重要供应商": ["主要供应商", "重要供应商领域", "供应商领域"],
    "上游企业": ["上游企业", "上游厂商", "上游供应商"],
    "下游行业": ["下游行业", "下游应用行业", "终端应用行业"],
    "一等奖工程": ["国家科技进步一等奖", "一等奖工程", "科技进步一等奖工程"],
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
    compact = compact.replace("：", ":").replace("，", ",")
    return compact


def find_ambiguities(query: str) -> List[str]:
    notes: List[str] = []
    if "页" in query and not re.search(r"\d+", query):
        notes.append("Query references pages without a concrete page number.")
    if any(token in query for token in ["公司", "该公司", "this company"]):
        notes.append("Pronoun or company shorthand detected; answer assumes the prospectus issuer.")
    return notes


def split_complex_question(query: str) -> List[str]:
    separators = ["以及", "并且", "和", " and ", "，", ";"]
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


def build_search_queries(query: str) -> List[str]:
    rewritten = rewrite_query(query)
    expanded: List[str] = [rewritten]

    for key, values in QUERY_EXPANSIONS.items():
        if key in rewritten:
            expanded.extend(values)

    if "占比" in rewritten and "收入" in rewritten:
        expanded.extend(["收入 占比", "营业收入 占比"])
    if "金额" in rewritten and "募集资金" in rewritten:
        expanded.extend(["募集资金 金额", "募集资金用途 金额"])
    if "标准" in rewritten or "规范" in rewritten:
        expanded.extend(["技术标准", "标准 规范"])

    expanded.extend(split_complex_question(rewritten))
    return dedupe_preserve_order([item for item in expanded if item.strip()])


def analyze_query(query: str) -> QueryIntent:
    language = detect_language(query)
    rewritten = rewrite_query(query)
    sub_questions = split_complex_question(rewritten)
    search_queries = build_search_queries(rewritten)
    return QueryIntent(
        language=language,
        intent=infer_intent(query),
        rewritten_query=rewritten,
        ambiguities=find_ambiguities(query),
        sub_questions=sub_questions,
        search_queries=search_queries,
    )
