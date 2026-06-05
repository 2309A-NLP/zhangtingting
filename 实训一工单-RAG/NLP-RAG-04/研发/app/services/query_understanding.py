from __future__ import annotations

# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
import re
from typing import Dict, List

from app.schemas import QueryIntent
from app.services.text_utils import dedupe_preserve_order


COMPANY_PATTERN = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9()（）]+?(?:股份有限公司|有限责任公司|集团有限公司|集团股份有限公司)"
)


INTENT_RULES: Dict[str, List[str]] = {
    "definition": ["是什么", "定义", "含义", "what is", "define"],
    "statistics": ["多少", "占比", "比例", "金额", "收入", "数量", "股数", "发行", "几家", "几项", "how much", "ratio"],
    "comparison": ["区别", "对比", "比较", "versus", "compare"],
    "entity_lookup": ["谁", "哪个", "哪些", "哪家", "法定代表人", "供应商", "客户", "关联方", "which", "who"],
}


FIELD_EXPANSIONS: Dict[str, List[str]] = {
    "法定代表人": ["公司法定代表人", "发行人法定代表人", "企业法定代表人"],
    "注册资本": ["公司注册资本", "发行人注册资本", "企业注册资本"],
    "本次发行股数": ["本次发行股数", "本次发行数量", "公开发行股数", "股份数量"],
    "发行后总股本比例": ["占发行后总股本的比例", "发行后总股本比例"],
    "募集资金项目": ["募集资金投资项目", "募投项目", "募集资金用途", "本次募集资金拟投资项目"],
    "补充流动资金": ["补充流动资金金额", "募集资金补充流动资金"],
    "军用领域收入": ["来自军用领域的收入", "军用收入", "国防客户销售额", "军方市场收入"],
    "军用收入占比": ["来自军用领域收入占主营业务收入比重", "军用收入占比", "国防客户销售额占比"],
    "技术标准": ["参与制定技术标准", "参与制定标准", "参与制定规范"],
    "重要供应商领域": ["重要供应商领域", "主要供应商领域", "供应商领域"],
    "上游行业": ["行业上游", "上游行业", "上游涉及行业", "上游涉及企业"],
    "下游行业": ["行业下游", "下游行业", "下游主要行业", "下游应用行业"],
    "一等奖工程": ["国家科技进步一等奖工程", "荣获国家科技进步一等奖的工程", "一等奖工程"],
    "关联方": ["关联方企业", "关联企业", "关联方名单", "关联方情况"],
    "控制关系关联方": ["存在控制关系的关联方", "受控制关联方", "控股股东关联方"],
    "非控制关系关联方": ["不存在控制关系的关联方", "不受同一控制关联方", "非控制关系关联方"],
    "持股比例": ["持股比例", "持股情况", "股权比例"],
    "组织结构图": ["组织结构图", "组织架构图", "组织机构图"],
    "销售部构成": ["销售部由几个部门构成", "销售部下属部门", "销售部部门构成"],
    "大客户销售处构成": ["大客户销售部由几个销售处构成", "大客户销售部下属销售处", "大客户销售部销售处构成"],
    "增长最快行业": ["增长率最快的行业", "增速最快的行业", "增长最快的行业"],
    "负增长行业": ["负增长的行业", "增长率为负的行业"],
    "市场应用结构图": ["市场应用结构与增长图", "应用结构与增长图", "增长图"],
}


CHAPTER_HINTS: Dict[str, List[str]] = {
    "发行": ["本次发行概况", "发行基本情况", "发行方案"],
    "募集资金": ["募集资金运用", "募集资金投资项目", "募投项目"],
    "关联方": ["关联方", "关联关系", "关联交易", "财务附注 关联方"],
    "供应商": ["主要供应商", "前五大供应商", "供应商情况"],
    "客户": ["主要客户", "前五大客户", "客户情况"],
    "军用": ["重大事项提示", "风险因素", "行业及客户集中度较高的风险"],
    "技术标准": ["发行人在行业中的竞争地位", "核心技术优势", "研发实力"],
    "组织结构图": ["组织结构图", "组织架构图", "组织机构图"],
    "增长图": ["市场应用结构与增长图", "增长图", "行业发展情况"],
}


def detect_language(query: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", query or "") else "en"


def rewrite_query(query: str) -> str:
    compact = re.sub(r"\s+", " ", query or "").strip()
    return compact.replace("（", "(").replace("）", ")")


def infer_intent(query: str) -> str:
    lowered = (query or "").lower()
    for intent, patterns in INTENT_RULES.items():
        if any(token in query or token in lowered for token in patterns):
            return intent
    return "factoid"


def detect_target_company(query: str) -> str:
    matches = COMPANY_PATTERN.findall(query or "")
    if not matches:
        return ""
    matches.sort(key=len, reverse=True)
    company = matches[0].strip()
    company = re.sub(r"^(与|和|针对)", "", company).strip()
    return company


def find_ambiguities(query: str, target_company: str) -> List[str]:
    notes: List[str] = []
    if any(token in query for token in ["该公司", "发行人", "该企业"]) and not target_company:
        notes.append("问题使用了代称，但未明确给出公司全称。")
    if "第" in query and "页" in query and not re.search(r"第\s*\d+\s*页", query):
        notes.append("问题提到了页码，但格式不完整。")
    return notes


def split_complex_question(query: str) -> List[str]:
    parts = [rewrite_query(query)]
    separators = ["以及", "并且", "同时", "还有", "，并", " and ", ";", "；"]
    for separator in separators:
        refined: List[str] = []
        for item in parts:
            if separator in item:
                refined.extend([piece.strip("，。；; ") for piece in item.split(separator) if piece.strip("，。；; ")])
            else:
                refined.append(item)
        parts = refined
    return dedupe_preserve_order([item for item in parts if item and item != rewrite_query(query)])


def classify_rewrite_strategy(query: str) -> str:
    normalized = rewrite_query(query)
    if len(split_complex_question(normalized)) >= 2:
        return "decomposed"
    if any(token in normalized for token in ["分别", "哪些", "名单", "列表", "项目", "构成", "上游", "下游", "关联方"]):
        return "expanded"
    return "simple"


def detect_field_keys(query: str) -> List[str]:
    normalized = rewrite_query(query)
    rules = {
        "法定代表人": ["法定代表人"],
        "注册资本": ["注册资本"],
        "本次发行股数": ["发行股数", "发行数量", "公开发行股数", "本次发行股数"],
        "发行后总股本比例": ["占发行后总股本的比例", "发行后总股本比例", "总股本比例"],
        "募集资金项目": ["募集资金", "募投项目", "投资哪些项目", "拟投资项目"],
        "补充流动资金": ["补充流动资金"],
        "军用领域收入": ["来自军用领域的收入", "军用收入", "国防客户销售额", "军方市场收入"],
        "军用收入占比": ["占主营业务收入的比重", "收入占比", "比重分别", "军用收入占比"],
        "技术标准": ["技术标准", "标准", "规范", "参与制定"],
        "重要供应商领域": ["重要供应商", "主要供应商", "供应商领域"],
        "上游行业": ["上游"],
        "下游行业": ["下游"],
        "一等奖工程": ["一等奖", "国家科技进步一等奖", "工程"],
        "关联方": ["关联方", "关联企业", "关联关系"],
        "控制关系关联方": ["存在控制关系", "控制关系的关联方", "控股股东"],
        "非控制关系关联方": ["不存在控制关系", "不受同一控制", "非控制关系"],
        "持股比例": ["持股比例", "股权比例"],
        "组织结构图": ["组织结构图", "组织架构图", "组织机构图"],
        "销售部构成": ["销售部", "部门构成", "下属部门", "几个部门"],
        "大客户销售处构成": ["大客户销售部", "销售处构成", "下属销售处", "几个销售处"],
        "增长最快行业": ["增长率最快", "增速最快", "增长最快", "增长率最高"],
        "负增长行业": ["负增长", "增长率为负", "下降行业"],
        "市场应用结构图": ["应用结构与增长图", "市场应用结构与增长图", "增长图"],
    }
    hits: List[str] = []
    for key, markers in rules.items():
        if any(marker in normalized for marker in markers):
            hits.append(key)
    if "非控制关系关联方" in hits and ("不存在控制关系" in normalized or "不受同一控制" in normalized):
        hits = [item for item in hits if item != "控制关系关联方"]
    return dedupe_preserve_order(hits)


def build_search_queries(query: str, target_company: str, rewrite_strategy: str) -> List[str]:
    normalized = rewrite_query(query)
    field_keys = detect_field_keys(normalized)
    expanded: List[str] = [normalized]

    if target_company:
        expanded.append(target_company)
        expanded.append(f"{target_company} 招股意向书")

    for field_key in field_keys:
        expansions = FIELD_EXPANSIONS.get(field_key, [])
        expanded.extend(expansions)
        if target_company:
            expanded.extend([f"{target_company} {value}" for value in expansions])

    for hint_key, hint_queries in CHAPTER_HINTS.items():
        if hint_key in normalized or hint_key in "".join(field_keys):
            expanded.extend(hint_queries)
            if target_company:
                expanded.extend([f"{target_company} {item}" for item in hint_queries])

    if rewrite_strategy in {"expanded", "decomposed"}:
        if "募集资金项目" in field_keys:
            expanded.extend(["募集资金投资项目 表格", "募投项目 表格", "募集资金运用 表格"])
            if target_company:
                expanded.extend(
                    [
                        f"{target_company} 募集资金投资项目 表格",
                        f"{target_company} 募投项目 表格",
                        f"{target_company} 募集资金运用 表格",
                    ]
                )
        if any(key in field_keys for key in ["关联方", "控制关系关联方", "非控制关系关联方", "持股比例"]):
            expanded.extend(["关联方 表格", "关联关系 表格", "持股比例"])
            if target_company:
                expanded.extend(
                    [
                        f"{target_company} 关联方 表格",
                        f"{target_company} 关联关系 表格",
                        f"{target_company} 持股比例",
                    ]
                )
        if any(key in field_keys for key in ["上游行业", "下游行业"]):
            expanded.extend(["电子信息行业 上游", "电子信息行业 下游", "发行人所处行业基本情况"])
            if target_company:
                expanded.extend(
                    [
                        f"{target_company} 电子信息行业 上游",
                        f"{target_company} 电子信息行业 下游",
                        f"{target_company} 发行人所处行业基本情况",
                    ]
                )
        if "一等奖工程" in field_keys:
            expanded.extend(["国家科技进步一等奖 工程", "荣获国家科技进步一等奖"])
            if target_company:
                expanded.extend(
                    [
                        f"{target_company} 国家科技进步一等奖 工程",
                        f"{target_company} 荣获国家科技进步一等奖",
                    ]
                )

    return dedupe_preserve_order([item.strip() for item in expanded if item.strip()])


def build_decomposed_questions(query: str, target_company: str) -> List[str]:
    normalized = rewrite_query(query)
    company_prefix = f"{target_company} " if target_company else ""
    sub_questions = split_complex_question(normalized)

    if "发行股数" in normalized and "发行后总股本" in normalized:
        sub_questions.extend(
            [
                f"{company_prefix}本次发行股数是多少",
                f"{company_prefix}占发行后总股本的比例是多少",
            ]
        )

    if "募集资金" in normalized and "项目" in normalized:
        sub_questions.extend(
            [
                f"{company_prefix}本次募集资金拟投资哪些项目",
                f"{company_prefix}募集资金投资项目表包含哪些项目",
            ]
        )

    if "存在控制关系" in normalized and "关联方" in normalized:
        sub_questions.extend(
            [
                f"{company_prefix}与公司存在控制关系的关联方是谁",
                f"{company_prefix}该关联方持股比例是多少",
                f"{company_prefix}该关联方与公司的关系是什么",
            ]
        )

    if "上游" in normalized:
        sub_questions.extend(
            [
                f"{company_prefix}电子信息行业的上游涉及哪些行业",
                f"{company_prefix}电子信息行业的上游涉及哪些企业",
            ]
        )

    if "下游" in normalized:
        sub_questions.extend(
            [
                f"{company_prefix}电子信息行业的下游主要包括哪些行业",
                f"{company_prefix}电子信息行业的下游应用行业有哪些",
            ]
        )

    return dedupe_preserve_order([item.strip("，。；; ") for item in sub_questions if item.strip("，。；; ")])


def detect_preferred_sections(query: str, field_keys: List[str]) -> List[str]:
    normalized = rewrite_query(query)
    sections: List[str] = []
    for key, hints in CHAPTER_HINTS.items():
        if key in normalized:
            sections.extend(hints)

    section_map = {
        "本次发行股数": ["本次发行概况", "发行基本情况", "发行方案"],
        "发行后总股本比例": ["本次发行概况", "发行基本情况", "发行方案"],
        "募集资金项目": ["募集资金运用", "募集资金投资项目", "募投项目"],
        "补充流动资金": ["募集资金运用", "募集资金投资项目", "募投项目"],
        "关联方": ["关联方", "关联关系", "关联交易", "财务附注 关联方"],
        "控制关系关联方": ["关联方", "关联关系", "关联交易", "财务附注 关联方"],
        "非控制关系关联方": ["关联方", "关联关系", "关联交易", "财务附注 关联方"],
        "重要供应商领域": ["主要供应商", "供应商情况"],
        "军用领域收入": ["重大事项提示", "风险因素", "行业及客户集中度较高的风险"],
        "军用收入占比": ["重大事项提示", "风险因素", "行业及客户集中度较高的风险"],
        "技术标准": ["发行人在行业中的竞争地位", "核心技术优势", "研发实力"],
        "组织结构图": ["组织结构图", "组织架构图"],
        "销售部构成": ["组织结构图", "组织架构图"],
        "大客户销售处构成": ["组织结构图", "组织架构图"],
        "增长最快行业": ["市场应用结构与增长图", "增长图", "行业发展情况"],
        "负增长行业": ["市场应用结构与增长图", "增长图", "行业发展情况"],
        "市场应用结构图": ["市场应用结构与增长图", "增长图", "行业发展情况"],
        "上游行业": ["发行人所处行业基本情况", "行业基本情况"],
        "下游行业": ["发行人所处行业基本情况", "行业基本情况"],
        "一等奖工程": ["发行人在行业中的竞争地位", "核心技术优势", "研发实力"],
    }
    for field_key in field_keys:
        sections.extend(section_map.get(field_key, []))
    return dedupe_preserve_order([item for item in sections if item])


def detect_query_tags(query: str, field_keys: List[str]) -> List[str]:
    normalized = rewrite_query(query)
    tags: List[str] = []

    if any(token in normalized for token in ["表格", "名单", "项目", "包括", "分别", "哪些", "列表"]):
        tags.append("list")
    if any(token in normalized for token in ["股数", "比例", "金额", "收入", "数量", "募集资金", "持股比例"]):
        tags.append("table")
    if "募集资金" in normalized:
        tags.append("fundraising")
    if "发行" in normalized:
        tags.append("issuance")
    if "关联方" in normalized:
        tags.append("related_party")
    if "不存在控制关系" in normalized or "不受同一控制" in normalized:
        tags.append("non_control_related_party")
    if any(token in normalized for token in ["组织结构图", "组织架构图", "组织机构图", "销售部", "销售处", "构成"]):
        tags.append("org_chart")
    if any(token in normalized for token in ["增长率最快", "增速最快", "增长最快", "负增长", "增长图", "图中可以看出"]):
        tags.append("chart_analysis")
    if any(token in normalized for token in ["技术标准", "参与制定", "规范", "标准"]):
        tags.append("technical_standard")
    if any(token in normalized for token in ["军用领域收入", "国防客户销售额", "军方市场收入"]):
        tags.append("military_revenue")

    if any(key in field_keys for key in ["募集资金项目", "补充流动资金"]):
        tags.extend(["table", "fundraising"])
    if any(key in field_keys for key in ["本次发行股数", "发行后总股本比例"]):
        tags.extend(["table", "issuance"])
    if any(key in field_keys for key in ["关联方", "控制关系关联方", "非控制关系关联方"]):
        tags.extend(["table", "list", "related_party"])
    if any(key in field_keys for key in ["组织结构图", "销售部构成", "大客户销售处构成"]):
        tags.extend(["org_chart", "list"])
    if any(key in field_keys for key in ["增长最快行业", "负增长行业", "市场应用结构图"]):
        tags.extend(["chart_analysis", "table"])
    if any(key in field_keys for key in ["军用领域收入", "军用收入占比"]):
        tags.extend(["table", "military_revenue"])
    if "技术标准" in field_keys:
        tags.append("technical_standard")

    return dedupe_preserve_order(tags)


def detect_question_type(query: str, field_keys: List[str], query_tags: List[str]) -> str:
    normalized = rewrite_query(query)
    if any(key in field_keys for key in ["法定代表人", "注册资本", "补充流动资金", "技术标准"]):
        return "field_lookup"
    if "org_chart" in query_tags:
        return "org_structure"
    if "chart_analysis" in query_tags:
        return "chart_trend"
    if "募集资金项目" in field_keys:
        return "table_list"
    if "table" in query_tags and "list" in query_tags:
        return "table_list"
    if "table" in query_tags and any(token in normalized for token in ["多少", "比例", "占比", "金额", "收入", "股数", "数量"]):
        return "table_numeric"
    return "fact_text"


def detect_preferred_block_types(question_type: str, query_tags: List[str]) -> List[str]:
    if question_type == "field_lookup":
        return ["form", "table", "text"]
    if question_type in {"table_numeric", "table_list"}:
        return ["table", "form", "text"]
    if question_type == "org_structure":
        return ["figure", "form", "text"]
    if question_type == "chart_trend":
        return ["figure", "table", "text"]
    if "table" in query_tags:
        return ["table", "text", "form"]
    return ["text", "form", "table", "figure"]


def analyze_query(query: str) -> QueryIntent:
    language = detect_language(query)
    rewritten = rewrite_query(query)
    target_company = detect_target_company(rewritten)
    rewrite_strategy = classify_rewrite_strategy(rewritten)
    field_keys = detect_field_keys(rewritten)
    preferred_sections = detect_preferred_sections(rewritten, field_keys)
    query_tags = detect_query_tags(rewritten, field_keys)
    question_type = detect_question_type(rewritten, field_keys, query_tags)
    preferred_block_types = detect_preferred_block_types(question_type, query_tags)
    sub_questions = (
        build_decomposed_questions(rewritten, target_company)
        if rewrite_strategy == "decomposed"
        else split_complex_question(rewritten)
    )
    search_queries = build_search_queries(rewritten, target_company, rewrite_strategy)

    return QueryIntent(
        language=language,
        intent=infer_intent(query),
        rewritten_query=rewritten,
        rewrite_strategy=rewrite_strategy,
        question_type=question_type,
        target_company=target_company,
        ambiguities=find_ambiguities(query, target_company),
        sub_questions=sub_questions,
        search_queries=search_queries,
        field_keys=field_keys,
        preferred_sections=preferred_sections,
        query_tags=query_tags,
        preferred_block_types=preferred_block_types,
    )
