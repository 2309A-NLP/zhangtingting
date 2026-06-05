from __future__ import annotations

# 工单编号：人工智能 NLP-RAG-PDF 文档的表格解析及检索优化
import re
from typing import Dict, List

from app.schemas import QueryIntent
from app.services.text_utils import dedupe_preserve_order


INTENT_RULES = {
    "definition": ["是什么", "定义", "含义", "what is", "define"],
    "statistics": ["多少", "占比", "比例", "金额", "收入", "数量", "股数", "发行", "how much", "ratio"],
    "comparison": ["区别", "对比", "比较", "versus", "compare"],
    "entity_lookup": ["谁", "哪家", "哪个", "有哪些", "法定代表人", "供应商", "客户", "关联方", "who", "which"],
}

FIELD_EXPANSIONS: Dict[str, List[str]] = {
    "法定代表人": ["公司法定代表人", "发行人法定代表人", "企业法定代表人"],
    "注册资本": ["公司注册资本", "发行人注册资本", "企业注册资本"],
    "本次发行股数": ["本次发行股数", "本次发行数量", "本次公开发行数量", "发行股数", "发行数量"],
    "发行后总股本比例": ["占发行后总股本的比例", "发行后总股本比例", "本次发行后总股本占比"],
    "募集资金项目": ["本次募集资金拟投资项目", "募集资金投资项目", "募投项目", "募集资金用途项目"],
    "补充流动资金": ["募集资金补充流动资金", "补充流动资金金额", "募集资金用途 补充流动资金"],
    "军用领域收入": [
        "军用收入",
        "军用领域营业收入",
        "军品收入",
        "国防客户销售额",
        "来自军方市场的收入",
        "来自军队用户的营业收入",
        "军品销售额",
    ],
    "军用收入占比": [
        "军用收入 占比",
        "军用领域收入 占比",
        "军品收入 占比",
        "国防客户销售额 占主营业务收入比重",
        "国防客户销售额 占比",
        "来自军方市场收入 占主营业务收入比重",
        "军品销售额 占主营业务收入比重",
    ],
    "技术标准": [
        "参与制定 技术标准",
        "参与制定 标准",
        "参与制定 规范",
        "视频指挥系统技术标准",
        "全军第一个 视频指挥系统技术标准",
    ],
    "重要供应商": ["主要供应商", "重要供应商领域", "供应商领域"],
    "上游行业": ["上游企业", "上游厂商", "上游供应商", "行业上游"],
    "下游行业": ["下游行业", "下游应用行业", "终端应用行业"],
    "一等奖工程": ["国家科技进步一等奖", "一等奖工程", "科技进步一等奖工程"],
    "关联方": ["关联方企业", "关联企业", "关联方名单", "关联方情况"],
    "不存在控制关系": ["不存在控制关系", "非控制关系", "不受同一控制", "无控制关系"],
}

CHAPTER_HINTS: Dict[str, List[str]] = {
    "发行": ["本次发行概况", "发行基本情况", "发行方案", "发行人基本情况"],
    "募集资金": ["募集资金运用", "募集资金投资项目", "募投项目", "本次募集资金用途"],
    "关联方": ["关联方", "关联交易", "关联关系", "财务附注 关联方"],
    "供应商": ["主要供应商", "前五大供应商", "供应商情况"],
    "客户": ["主要客户", "前五大客户", "客户情况"],
    "军用": ["重大事项提示", "风险因素", "行业及客户集中度较高的风险"],
    "技术标准": ["发行人在行业中的竞争地位", "核心技术优势", "研发实力"],
}

COMPANY_PATTERN = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9（）()]+?(?:股份有限公司|有限责任公司|集团有限公司|集团股份有限公司)"
)


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


def detect_target_company(query: str) -> str:
    matches = COMPANY_PATTERN.findall(query)
    if not matches:
        return ""
    matches.sort(key=len, reverse=True)
    company = matches[0].strip()
    company = re.sub(r"^(与|和|及|对|针对)", "", company).strip()
    return company


def find_ambiguities(query: str, target_company: str) -> List[str]:
    notes: List[str] = []
    if "页" in query and not re.search(r"\d+", query):
        notes.append("问题提到了页码，但没有给出具体页码。")
    if any(token in query for token in ["公司", "该公司", "发行人"]) and not target_company:
        notes.append("问题存在公司简称或代称，后续检索可能需要结合上下文识别主体。")
    return notes


def split_complex_question(query: str) -> List[str]:
    separators = ["以及", "并且", "同时", "还有", "和", " and ", "；", ";", "，并", ", and "]
    parts = [query]
    for separator in separators:
        refined: List[str] = []
        for item in parts:
            if separator in item:
                refined.extend([piece.strip("，。；; ") for piece in item.split(separator) if piece.strip("，。；; ")])
            else:
                refined.append(item)
        parts = refined
    return dedupe_preserve_order([item for item in parts if item and item != query])


def classify_rewrite_strategy(query: str) -> str:
    query = rewrite_query(query)
    has_multiple_targets = len(split_complex_question(query)) >= 2
    composite_markers = ["分别", "以及", "并且", "同时", "占发行后总股本", "不存在控制关系"]
    list_markers = ["哪些", "项目", "名单", "表", "列表", "分别包括", "拟投资"]
    table_markers = ["募集资金", "股数", "比例", "项目", "前五大", "关联方"]

    if has_multiple_targets or any(marker in query for marker in composite_markers):
        return "decomposed"
    if any(marker in query for marker in list_markers) or sum(marker in query for marker in table_markers) >= 2:
        return "expanded"
    return "simple"


def detect_field_keys(query: str) -> List[str]:
    field_map = {
        "法定代表人": ["法定代表人"],
        "注册资本": ["注册资本"],
        "本次发行股数": ["发行股数", "发行数量", "公开发行数量", "本次发行"],
        "发行后总股本比例": ["发行后总股本比例", "占发行后总股本", "总股本比例"],
        "募集资金项目": ["募集资金", "募投项目", "投资项目", "拟投资项目"],
        "补充流动资金": ["补充流动资金"],
        "军用领域收入": ["军用领域收入", "军用收入", "军品收入", "来自军用领域的收入", "国防客户销售额", "军方市场收入"],
        "军用收入占比": ["收入占比", "占主营业务收入", "占比", "主营业务收入的比重", "销售额占比", "收入比重"],
        "技术标准": ["技术标准", "标准", "规范", "参与制定"],
        "重要供应商": ["重要供应商", "主要供应商", "供应商领域"],
        "上游行业": ["上游"],
        "下游行业": ["下游"],
        "一等奖工程": ["一等奖", "科技进步一等奖", "工程"],
        "关联方": ["关联方", "关联企业", "关联关系"],
        "不存在控制关系": ["不存在控制关系", "非控制关系", "不受同一控制"],
    }
    hits: List[str] = []
    for key, markers in field_map.items():
        if any(marker in query for marker in markers):
            hits.append(key)
    return dedupe_preserve_order(hits)


def build_search_queries(query: str, target_company: str, rewrite_strategy: str) -> List[str]:
    rewritten = rewrite_query(query)
    field_keys = detect_field_keys(rewritten)
    expanded: List[str] = [rewritten]

    if target_company:
        expanded.append(target_company)
        for key in field_keys:
            expanded.append(f"{target_company} {key}")

    for key in field_keys:
        expanded.extend(FIELD_EXPANSIONS.get(key, []))
        if target_company:
            expanded.extend(f"{target_company} {value}" for value in FIELD_EXPANSIONS.get(key, []))

    for hint_key, hint_queries in CHAPTER_HINTS.items():
        if hint_key in rewritten:
            expanded.extend(hint_queries)
            if target_company:
                expanded.extend(f"{target_company} {hint_query}" for hint_query in hint_queries)

    if rewrite_strategy in {"expanded", "decomposed"}:
        if "募集资金" in rewritten and "项目" in rewritten:
            expanded.extend(["募集资金投资项目 表", "募投项目 表", "募集资金运用"])
            if target_company:
                expanded.extend(
                    [
                        f"{target_company} 募集资金投资项目 表",
                        f"{target_company} 募投项目 表",
                        f"{target_company} 募集资金运用",
                    ]
                )
        if "关联方" in rewritten:
            expanded.extend(["关联方名单", "关联关系", "关联交易"])
            if "不存在控制关系" in rewritten:
                expanded.extend(["不存在控制关系 关联方", "非控制关系 关联方"])
            if target_company:
                expanded.extend(
                    [
                        f"{target_company} 关联方名单",
                        f"{target_company} 关联关系",
                        f"{target_company} 关联交易",
                    ]
                )

    return dedupe_preserve_order([item.strip() for item in expanded if item.strip()])


def build_decomposed_questions(query: str, target_company: str) -> List[str]:
    rewritten = rewrite_query(query)
    company_prefix = f"{target_company} " if target_company else ""
    sub_questions = split_complex_question(rewritten)

    if "发行" in rewritten and "股数" in rewritten and "占发行后总股本" in rewritten:
        sub_questions.extend(
            [
                f"{company_prefix}本次发行股数是多少",
                f"{company_prefix}占发行后总股本的比例是多少",
            ]
        )

    if "募集资金" in rewritten and "项目" in rewritten:
        sub_questions.extend(
            [
                f"{company_prefix}本次募集资金拟投资哪些项目",
                f"{company_prefix}募集资金投资项目有哪些",
                f"{company_prefix}募投项目表有哪些项目",
            ]
        )

    if "关联方" in rewritten and "不存在控制关系" in rewritten:
        sub_questions.extend(
            [
                f"{company_prefix}关联方企业有哪些",
                f"{company_prefix}不存在控制关系的关联方有哪些",
                f"{company_prefix}关联方中哪些企业不存在控制关系",
            ]
        )

    return dedupe_preserve_order([item.strip("，。；; ") for item in sub_questions if item.strip("，。；; ")])


def detect_preferred_sections(query: str, field_keys: List[str]) -> List[str]:
    sections: List[str] = []
    normalized = rewrite_query(query)
    for key, section_hints in CHAPTER_HINTS.items():
        if key in normalized:
            sections.extend(section_hints)

    section_map = {
        "本次发行股数": ["本次发行概况", "发行基本情况", "发行方案"],
        "发行后总股本比例": ["本次发行概况", "发行基本情况", "发行方案"],
        "募集资金项目": ["募集资金运用", "募集资金投资项目", "募投项目"],
        "补充流动资金": ["募集资金运用", "募集资金投资项目", "募投项目"],
        "关联方": ["关联方", "关联关系", "关联交易", "财务附注 关联方"],
        "不存在控制关系": ["关联方", "关联关系", "关联交易", "财务附注 关联方"],
        "重要供应商": ["主要供应商", "供应商情况"],
        "军用领域收入": ["重大事项提示", "风险因素", "行业及客户集中度较高的风险"],
        "军用收入占比": ["重大事项提示", "风险因素", "行业及客户集中度较高的风险"],
        "技术标准": ["发行人在行业中的竞争地位", "核心技术优势", "研发实力"],
    }
    for field_key in field_keys:
        sections.extend(section_map.get(field_key, []))
    return dedupe_preserve_order([section for section in sections if section])


def detect_query_tags(query: str, field_keys: List[str]) -> List[str]:
    tags: List[str] = []
    normalized = rewrite_query(query)

    if any(marker in normalized for marker in ["表", "项目", "名单", "前五大", "分别", "包括", "有哪些"]):
        tags.append("list")
    if any(marker in normalized for marker in ["表", "股数", "比例", "金额", "项目", "募投", "关联方"]):
        tags.append("table")
    if "募集资金" in normalized:
        tags.append("fundraising")
    if "发行" in normalized:
        tags.append("issuance")
    if "关联方" in normalized:
        tags.append("related_party")
    if "不存在控制关系" in normalized:
        tags.append("non_control_related_party")

    if "募集资金项目" in field_keys or "补充流动资金" in field_keys:
        tags.extend(["table", "fundraising"])
    if "本次发行股数" in field_keys or "发行后总股本比例" in field_keys:
        tags.extend(["table", "issuance"])
    if "关联方" in field_keys or "不存在控制关系" in field_keys:
        tags.extend(["table", "list", "related_party"])
    if "军用领域收入" in field_keys or "军用收入占比" in field_keys:
        tags.extend(["table", "list", "military_revenue"])
    if "技术标准" in field_keys:
        tags.extend(["technical_standard"])

    return dedupe_preserve_order(tags)


def analyze_query(query: str) -> QueryIntent:
    language = detect_language(query)
    rewritten = rewrite_query(query)
    target_company = detect_target_company(rewritten)
    rewrite_strategy = classify_rewrite_strategy(rewritten)
    field_keys = detect_field_keys(rewritten)
    preferred_sections = detect_preferred_sections(rewritten, field_keys)
    query_tags = detect_query_tags(rewritten, field_keys)
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
        target_company=target_company,
        ambiguities=find_ambiguities(query, target_company),
        sub_questions=sub_questions,
        search_queries=search_queries,
        field_keys=field_keys,
        preferred_sections=preferred_sections,
        query_tags=query_tags,
    )
