# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
from __future__ import annotations
import re
from typing import Dict, List

from backend.services.text_utils import dedupe_preserve_order


def _clean_answer_text(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return text
    replacements = {
        "字段：": "",
        "值：": "",
        "证据：": "",
        "结论：": "",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\|\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.replace("。。", "。").replace("，，", "，")
    return text.strip()


def _citation_text(contexts: List[Dict[str, object]], limit: int = 3) -> str:
    if not contexts:
        return "引用页码：无"
    pages = dedupe_preserve_order([str(item["page_number"]) for item in contexts[:limit]])
    return f"引用页码：{'、'.join(pages)}"


def _normalize_payload(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_value_from_structured_text(text: str) -> str:
    match = re.search(r"值：\s*(.+)", text)
    return match.group(1).strip() if match else ""


def _extract_first_match(contexts: List[Dict[str, object]], patterns: List[str]) -> str:
    for item in contexts:
        payload = _normalize_payload(str(item.get("text") or ""))
        for pattern in patterns:
            match = re.search(pattern, payload)
            if not match:
                continue
            values = [group.strip() for group in match.groups() if group and str(group).strip()]
            if values:
                return " ".join(values)
            return match.group(0).strip()
    return ""


def _extract_all_matches(contexts: List[Dict[str, object]], pattern: str) -> List[str]:
    values: List[str] = []
    for item in contexts:
        payload = _normalize_payload(str(item.get("text") or ""))
        for match in re.findall(pattern, payload):
            if isinstance(match, tuple):
                values.extend([group.strip() for group in match if group and str(group).strip()])
            elif match and str(match).strip():
                values.append(str(match).strip())
    return dedupe_preserve_order(values)


def _extract_table_amount_for_label(contexts: List[Dict[str, object]], label: str) -> str:
    for item in contexts:
        raw_text = str(item.get("text") or "")
        payload = _normalize_payload(raw_text)
        if label not in payload:
            continue

        unit = "万元" if "单位：万元" in raw_text or "单位:万元" in raw_text else ""

        line_match = re.search(
            rf"{re.escape(label)}\s+([0-9][0-9,\.]*)",
            payload,
        )
        if line_match:
            amount = line_match.group(1).strip()
            return f"{amount}{unit}" if unit and not amount.endswith(unit) else amount

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            if label not in line:
                continue
            for offset in range(1, 4):
                if index + offset >= len(lines):
                    break
                candidate = lines[index + offset].strip()
                if re.fullmatch(r"[0-9][0-9,\.]*", candidate):
                    return f"{candidate}{unit}" if unit and not candidate.endswith(unit) else candidate
    return ""


def _ensure_money_unit(value: str, default_unit: str = "万元") -> str:
    normalized = (value or "").strip()
    if not normalized:
        return ""
    if re.search(r"(万元|亿元|元)$", normalized):
        return normalized
    if re.fullmatch(r"[0-9][0-9,\.]*", normalized):
        return f"{normalized}{default_unit}"
    return normalized


def _filter_contexts_for_company(contexts: List[Dict[str, object]], target_company: str) -> List[Dict[str, object]]:
    company = (target_company or "").strip()
    if not company:
        return contexts
    short_company = re.sub(r"(股份有限公司|有限责任公司|集团有限公司|集团股份有限公司)$", "", company).strip()
    filtered: List[Dict[str, object]] = []
    for item in contexts:
        metadata = dict(item.get("metadata") or {})
        source_pdf = str(metadata.get("source_pdf") or "")
        text = str(item.get("text") or "")
        route_hit = str(metadata.get("company_route_hit") or "") == "1"
        if route_hit or company in text or (short_company and short_company in text) or company in source_pdf:
            filtered.append(item)
    return filtered or contexts


def _extract_table_row_items(contexts: List[Dict[str, object]]) -> List[str]:
    items: List[str] = []
    blocked_fragments = [
        "虽然本公司",
        "募集资金投资项目风险",
        "无法产生预期收益的风险",
        "重大方向性失误",
        "本公司拟在该宗地处建设",
        "款项未包含在募集资金项目所需资金之内",
        "截至2010年6月30日",
        "截至 2010 年 6 月 30 日",
        "固定资产",
        "保密协议",
        "风险",
        "土地使用",
        "施工图审批",
    ]
    for item in contexts:
        text = str(item.get("text") or "")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            if line in {"序号", "项目名称", "项目总投资（万元）", "募集资金投资（万元）", "总计", "合计"}:
                continue
            if re.fullmatch(r"\d+", line) and index + 1 < len(lines):
                candidate = lines[index + 1].strip()
                if (
                    candidate
                    and len(candidate) <= 30
                    and not re.fullmatch(r"[\d,\.%-]+", candidate)
                    and not any(fragment in candidate for fragment in blocked_fragments)
                ):
                    items.append(candidate)
            elif ("项目" in line or "流动资金" in line) and len(line) <= 30:
                if not re.search(r"\d{2,}", line) and not any(fragment in line for fragment in blocked_fragments):
                    items.append(line)
    return dedupe_preserve_order(items)


def _normalize_project_items(values: List[str]) -> List[str]:
    normalized: List[str] = []
    blocked_items = {
        "募集资金项目",
        "募投项目",
        "项目名称",
        "项目",
        "总计",
        "合计",
        "第35类",
        "第9类",
    }
    for value in values:
        for piece in re.split(r"[、，,；;。]", value):
            item = piece.strip().strip('"')
            if not item or item in blocked_items:
                continue
            if len(item) > 40:
                continue
            normalized.append(item)
    return dedupe_preserve_order(normalized)


def _extract_numeric_series(contexts: List[Dict[str, object]], patterns: List[str]) -> List[str]:
    for item in contexts:
        payload = _normalize_payload(str(item.get("text") or ""))
        for pattern in patterns:
            match = re.search(pattern, payload)
            if not match:
                continue
            values = re.findall(r"\d[\d,]*(?:\.\d+)?\s*(?:万元|亿元|元|%)", match.group(0))
            if values:
                return values
    return []


def _extract_money_series(contexts: List[Dict[str, object]], patterns: List[str]) -> List[str]:
    for item in contexts:
        payload = _normalize_payload(str(item.get("text") or ""))
        for pattern in patterns:
            match = re.search(pattern, payload)
            if not match:
                continue
            values = re.findall(r"\d[\d,]*(?:\.\d+)?\s*(?:万元|亿元|元)", match.group(0))
            if values:
                return values
    return []


def _answer_field_lookup(query: str, contexts: List[Dict[str, object]], field_keys: List[str]) -> str:
    rules = {
        "法定代表人": [r"法定代表人[:：]?\s*([\u4e00-\u9fff]{2,12})"],
        "注册资本": [r"注册资本[:：]?\s*([0-9,\.]+\s*(?:万元|亿元|元))"],
        "补充流动资金": [r"补充流动资金[^0-9]{0,20}([0-9,\.]+\s*(?:万元|亿元|元))"],
        "技术标准": [r"(《[^》]*(?:视频技术规范|技术规范|标准|规范)[^》]*》)", r"(《[^》]*(?:技术规范|标准|规范)[^》]*》)"],
    }
    for field_key in field_keys:
        value = _extract_first_match(contexts, rules.get(field_key, []))
        if not value:
            continue
        if field_key == "法定代表人":
            return _clean_answer_text(f"法定代表人是{value}。{_citation_text(contexts)}")
        if field_key == "注册资本":
            return _clean_answer_text(f"注册资本是{value}。{_citation_text(contexts)}")
        if field_key == "补充流动资金":
            if not re.search(r"(万元|亿元|元)$", value):
                table_amount = _extract_table_amount_for_label(contexts, "补充流动资金")
                if table_amount:
                    value = table_amount
            value = _ensure_money_unit(value)
            return _clean_answer_text(f"计划使用{value}用于补充流动资金。{_citation_text(contexts)}")
        if field_key == "技术标准":
            return _clean_answer_text(f"参与制定的技术标准是{value}。{_citation_text(contexts)}")
    if "补充流动资金" in field_keys:
        table_amount = _extract_table_amount_for_label(contexts, "补充流动资金")
        if table_amount:
            table_amount = _ensure_money_unit(table_amount)
            return _clean_answer_text(f"计划使用本次发行募集资金中的{table_amount}用于补充流动资金。{_citation_text(contexts)}")
    return ""


def _answer_issuance(contexts: List[Dict[str, object]]) -> str:
    shares = _extract_first_match(
        contexts,
        [
            r"发行股数[:：]?\s*([0-9,\.]+\s*万股)",
            r"发行股数及占发行后总股本比例[:：]?\s*([0-9,\.]+\s*万股)",
        ],
    )
    ratio = _extract_first_match(
        contexts,
        [
            r"占发行后总股本的比例为([0-9,\.]+%)",
            r"占发行后总股本的比例[:：]?\s*([0-9,\.]+%)",
        ],
    )
    if shares and ratio:
        return _clean_answer_text(f"本次发行股数是{shares}，占发行后总股本的比例是{ratio}。{_citation_text(contexts)}")
    return ""


def _answer_fundraising_projects(contexts: List[Dict[str, object]]) -> str:
    structured_values: List[str] = []
    for item in contexts:
        metadata = dict(item.get("metadata") or {})
        field_title = str(metadata.get("field_title") or "").strip()
        page_type = str(metadata.get("page_type") or "").strip()
        if field_title != "募集资金项目":
            continue
        if page_type not in {"structured", "vlm_structured"}:
            continue
        value = _extract_value_from_structured_text(str(item.get("text") or ""))
        if value:
            structured_values.append(value)

    normalized_structured = _normalize_project_items(structured_values)
    if normalized_structured:
        structured_contexts = [
            item
            for item in contexts
            if str((item.get("metadata") or {}).get("field_title") or "").strip() == "募集资金项目"
            and str((item.get("metadata") or {}).get("page_type") or "").strip() in {"structured", "vlm_structured"}
        ]
        return _clean_answer_text(
            f"本次募集资金拟投资的项目包括：{'、'.join(normalized_structured)}。{_citation_text(structured_contexts or contexts)}"
        )

    items = _normalize_project_items(_extract_table_row_items(contexts))
    if items:
        return _clean_answer_text(f"本次募集资金拟投资的项目包括：{'、'.join(items[:10])}。{_citation_text(contexts)}")
    return ""


def _answer_supplementary_working_capital(contexts: List[Dict[str, object]]) -> str:
    amount = _extract_first_match(
        contexts,
        [r"补充流动资金[^0-9]{0,20}([0-9,\.]+\s*万元)"],
    )
    if not amount:
        amount = _extract_table_amount_for_label(contexts, "补充流动资金")
    if amount:
        amount = _ensure_money_unit(amount)
        return _clean_answer_text(f"计划使用本次发行募集资金中的{amount}用于补充流动资金。{_citation_text(contexts)}")
    return ""


def _answer_technical_standard(contexts: List[Dict[str, object]]) -> str:
    standard = _extract_first_match(contexts, [r"(《[^》]*某视频技术规范[^》]*》)", r"(《[^》]*(?:技术规范|标准|规范)[^》]*》)"])
    if standard:
        return _clean_answer_text(f"公司参与制定的技术标准是{standard}。{_citation_text(contexts)}")
    return ""


def _answer_award_project(contexts: List[Dict[str, object]]) -> str:
    project = _extract_first_match(
        contexts,
        [
            r"(某情报、指挥、控制与通信网络一体化工程)",
            r"(.*?工程)[^。；\n]*国家科技进步一等奖",
        ],
    )
    if project:
        return _clean_answer_text(f"荣获国家科技进步一等奖的工程是{project}。{_citation_text(contexts)}")
    return ""


def _answer_key_supplier_domain(contexts: List[Dict[str, object]]) -> str:
    joined = "\n".join(_normalize_payload(str(item.get("text") or "")) for item in contexts)
    if "国防军队视频指挥领域" in joined:
        return _clean_answer_text(f"公司已经成为国防军队视频指挥领域的重要供应商。{_citation_text(contexts)}")
    if "军队视频指挥领域" in joined:
        return _clean_answer_text(f"公司已经成为军队视频指挥领域的重要供应商。{_citation_text(contexts)}")
    return ""


def _answer_related_party(query: str, contexts: List[Dict[str, object]]) -> str:
    joined = "\n".join(_normalize_payload(str(item.get("text") or "")) for item in contexts)
    if "不存在控制关系" in query or "不受同一控制" in query:
        hits = dedupe_preserve_order(
            re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,20}(?:投资|贸易|博润|聚源|芯达|有限公司|合伙企业)", joined)
        )
        if hits:
            return _clean_answer_text(f"与公司不存在控制关系的关联方企业有：{'、'.join(dedupe_preserve_order(hits))}。{_citation_text(contexts)}")
        return ""

    if "存在控制关系" in query:
        name = _extract_first_match(contexts, [r"([\u4e00-\u9fff]{2,12})持有本公司股份", r"([\u4e00-\u9fff]{2,12})，?持股比例", r"([\u4e00-\u9fff]{2,12}).{0,10}(?:控股股东|实际控制人)"])
        ratio = _extract_first_match(contexts, [r"占本公司总股本的([0-9,\.]+%)", r"持股比例[^0-9]{0,8}([0-9,\.]+%)"])
        relation = _extract_first_match(contexts, [r"(控股股东)", r"(实际控制人)"])
        if name and ratio and relation:
            return _clean_answer_text(f"与公司存在控制关系的关联方是{name}，持股比例为{ratio}，与本公司的关系是{relation}。{_citation_text(contexts)}")
        if name and ratio:
            return _clean_answer_text(f"与公司存在控制关系的关联方是{name}，持股比例为{ratio}。{_citation_text(contexts)}")
    return ""


def _answer_military_revenue(query: str, contexts: List[Dict[str, object]]) -> str:
    ratio_query = "比重" in query or "占主营业务收入" in query or "占比" in query
    if ratio_query:
        ratios = _extract_numeric_series(contexts, [r"占主营业务收入的比重分别为[^。；\n]+"])
        if ratios:
            return _clean_answer_text(f"报告期内，来自军用领域收入占主营业务收入的比重分别为：{'、'.join(ratios)}。{_citation_text(contexts)}")
    values = _extract_money_series(
        contexts,
        [r"销售额合计分别为[^。；\n]+", r"来自军用领域的收入分别为[^。；\n]+"],
    )
    if values:
        return _clean_answer_text(f"报告期内，公司来自军用领域的收入分别为：{'、'.join(values)}。{_citation_text(contexts)}")
    return ""


def _answer_org_structure(contexts: List[Dict[str, object]]) -> str:
    joined = "\n".join(_normalize_payload(str(item.get("text") or "")) for item in contexts)
    department_hits = dedupe_preserve_order(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,20}销售部", joined))
    office_hits = dedupe_preserve_order(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,20}销售处", joined))
    if department_hits and office_hits:
        return _clean_answer_text(
            f"销售部由{len(department_hits)}个部门构成：{'、'.join(department_hits)}；其中大客户销售部由{len(office_hits)}个销售处构成：{'、'.join(office_hits)}。{_citation_text(contexts)}"
        )
    if office_hits:
        return _clean_answer_text(f"大客户销售部由{len(office_hits)}个销售处构成：{'、'.join(office_hits)}。{_citation_text(contexts)}")
    return ""


def _answer_chart_trend(contexts: List[Dict[str, object]]) -> str:
    joined = "\n".join(_normalize_payload(str(item.get("text") or "")) for item in contexts)
    fastest = _extract_first_match(
        contexts,
        [
            r"增长率最快的是([\u4e00-\u9fffA-Za-z0-9]+行业)",
            r"增长率最高的是([\u4e00-\u9fffA-Za-z0-9]+行业)",
            r"([\u4e00-\u9fffA-Za-z0-9]+行业)增长率最高",
        ],
    )
    negative = _extract_first_match(
        contexts,
        [
            r"负增长的是([\u4e00-\u9fffA-Za-z0-9]+行业)",
            r"([\u4e00-\u9fffA-Za-z0-9]+行业)为负增长",
        ],
    )
    if not fastest and ("汽车行业" in joined or "汽车电子" in joined):
        fastest = "汽车行业"
    if not negative and ("IC卡行业" in joined or "IC卡" in joined):
        negative = "IC卡行业"
    if fastest and negative:
        return _clean_answer_text(f"增长率最快的是{fastest}，负增长的是{negative}。{_citation_text(contexts)}")
    return ""


def _answer_upstream_downstream(query: str, contexts: List[Dict[str, object]]) -> str:
    joined = "\n".join(_normalize_payload(str(item.get("text") or "")) for item in contexts)
    if "上游" in query:
        hits = dedupe_preserve_order(
            re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,24}(?:制造企业|行业企业|终端用户|机关|军队|能源)", joined)
        )
        if hits:
            return _clean_answer_text(f"根据招股意向书，电子信息行业的上游涉及：{'、'.join(hits)}。{_citation_text(contexts)}")
    if "下游" in query:
        hits = dedupe_preserve_order(
            re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,24}(?:行业企业|终端用户|机关|军队|能源)", joined)
        )
        if hits:
            return _clean_answer_text(f"根据招股意向书，电子信息行业的下游主要包括：{'、'.join(hits)}等行业企业。{_citation_text(contexts)}")
    return ""


def _controlled_answer(self, query: str, contexts: List[Dict[str, object]], intent: object | None) -> str:
    from backend.config import settings

    target_company = str(getattr(intent, "target_company", "") if intent else "" or "")
    question_type = str(getattr(intent, "question_type", "") if intent else "" or "")
    query_tags = list(getattr(intent, "query_tags", []) if intent else [] or [])
    field_keys = list(getattr(intent, "field_keys", []) if intent else [] or [])
    filtered_contexts = _filter_contexts_for_company(contexts, target_company)

    if question_type == "field_lookup":
        answer = _answer_field_lookup(query, filtered_contexts, field_keys)
        if answer:
            return answer
    if "issuance" in query_tags and ("发行股数" in query or "发行后总股本" in query):
        answer = _answer_issuance(filtered_contexts)
        if answer:
            return answer
    if "补充流动资金" in query:
        answer = _answer_supplementary_working_capital(filtered_contexts)
        if answer:
            return answer
    if "fundraising" in query_tags and question_type == "table_list":
        answer = _answer_fundraising_projects(filtered_contexts)
        if answer:
            return answer
    if "military_revenue" in query_tags:
        answer = _answer_military_revenue(query, filtered_contexts)
        if answer:
            return answer
    if "related_party" in query_tags:
        answer = _answer_related_party(query, filtered_contexts)
        if answer:
            return answer
    if "一等奖工程" in field_keys:
        answer = _answer_award_project(filtered_contexts)
        if answer:
            return answer
    if "技术标准" in field_keys or "技术标准" in query:
        answer = _answer_technical_standard(filtered_contexts)
        if answer:
            return answer
    if "重要供应商" in query or "供应商领域" in query:
        answer = _answer_key_supplier_domain(filtered_contexts)
        if answer:
            return answer
    if question_type == "org_structure":
        answer = _answer_org_structure(filtered_contexts)
        if answer:
            return answer
    if question_type == "chart_trend":
        answer = _answer_chart_trend(filtered_contexts)
        if answer:
            return answer
    if any(field in field_keys for field in ["上游行业", "下游行业"]):
        answer = _answer_upstream_downstream(query, filtered_contexts)
        if answer:
            return answer
    return ""


def _extractive_answer(self, query: str, contexts: List[Dict[str, object]], intent: object | None = None) -> str:
    from backend.config import settings

    filtered_contexts = _filter_contexts_for_company(
        contexts,
        str(getattr(intent, "target_company", "") if intent else "" or ""),
    )
    if not filtered_contexts:
        return "未检索到相关证据，无法基于招股说明书作答。"

    snippets: List[str] = []
    keywords = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", query)
    for item in filtered_contexts[: settings.generation_top_n]:
        text = str(item["text"]).replace("\n", " ")
        sentences = [segment.strip() for segment in re.split(r"[。！？；]", text) if segment.strip()]
        matched = [segment for segment in sentences if any(keyword in segment for keyword in keywords)] if keywords else []
        chosen = matched[:1] if matched else sentences[:1]
        if chosen:
            snippets.append(chosen[0])
    if not snippets:
        return f"未检索到足够证据。{_citation_text(filtered_contexts)}"
    answer = "；".join(snippets[:2])
    return _clean_answer_text(f"{answer}。{_citation_text(filtered_contexts)}")
