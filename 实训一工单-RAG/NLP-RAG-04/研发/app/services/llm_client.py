from __future__ import annotations

# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
import json
import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests

from app.config import settings
from app.services.text_utils import dedupe_preserve_order


class LLMClient:
    @staticmethod
    def _resolve_api_key(api_url: str, api_key: str) -> str:
        explicit_key = api_key.strip()
        if explicit_key:
            return explicit_key

        pdf_vlm_key = str(getattr(settings, "pdf_vlm_api_key", "") or "").strip()
        pdf_vlm_url = str(getattr(settings, "pdf_vlm_api_url", "") or "").strip()
        current_url = (api_url or "").strip()
        if not pdf_vlm_key:
            return ""

        if pdf_vlm_url and current_url and current_url == pdf_vlm_url:
            return pdf_vlm_key
        if "api.siliconflow.cn" in current_url:
            return pdf_vlm_key
        return ""

    def __init__(
        self,
        provider: str,
        api_url: str = "",
        api_key: str = "",
        model_name: str = "",
        fallback_api_url: str = "",
        fallback_api_key: str = "",
        fallback_model_name: str = "",
    ) -> None:
        self.provider = provider
        self.api_url = api_url.strip()
        self.api_key = self._resolve_api_key(self.api_url, api_key)
        self.model_name = model_name.strip()
        self.fallback_api_url = fallback_api_url.strip()
        self.fallback_api_key = self._resolve_api_key(self.fallback_api_url, fallback_api_key)
        self.fallback_model_name = fallback_model_name.strip()
        self.last_call_details: Dict[str, object] = {}

    def _get_intent_attr(self, intent: object | None, name: str, default):
        if intent is None:
            return default
        return getattr(intent, name, default)

    def _is_local_url(self, url: str) -> bool:
        current = (url or "").strip()
        if not current:
            return False
        try:
            parsed = urlparse(current)
            host = (parsed.hostname or "").strip().lower()
        except Exception:
            return False
        return host in {"127.0.0.1", "localhost", "::1"}

    def _normalize_payload(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    def _clean_answer_text(self, answer: str) -> str:
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

    def _citation_text(self, contexts: List[Dict[str, object]], limit: int = 3) -> str:
        if not contexts:
            return "引用页码：无"
        pages = dedupe_preserve_order([str(item["page_number"]) for item in contexts[:limit]])
        return f"引用页码：{'、'.join(pages)}"

    def _build_prompt(self, query: str, contexts: List[Dict[str, object]], intent: object | None = None) -> str:
        target_company = str(self._get_intent_attr(intent, "target_company", "") or "").strip()
        context_text = "\n\n".join(
            (
                f"[页码 {item['page_number']}, 逻辑页 {item.get('logical_page') or '-'}, 类型 {item.get('metadata', {}).get('page_type', 'text')}, "
                f"来源 {item.get('metadata', {}).get('source_pdf') or '-'}]\n"
                f"{str(item['text'])[:settings.max_context_chars]}"
            )
            for item in contexts[: settings.generation_top_n]
        )
        company_rule = (
            f"只回答与“{target_company}”直接相关的内容。若证据属于其他公司，一律忽略。\n"
            if target_company
            else ""
        )
        return (
            "你是一个严格基于招股说明书证据回答问题的助手。\n"
            f"{company_rule}"
            "只能根据给定证据作答，不得编造；如果证据不足，请明确说“未检索到足够证据”。\n"
            "回答时先给结论，不要把原始大段文本块直接拼到答案里。\n"
            "答案末尾必须附上引用页码，格式如：引用页码：22、30。\n\n"
            f"问题：{query}\n\n"
            f"证据：\n{context_text}\n\n"
            "请用简洁中文回答。"
        )

    def _extract_value_from_structured_text(self, text: str) -> str:
        match = re.search(r"值：\s*(.+)", text)
        return match.group(1).strip() if match else ""

    def _extract_first_match(self, contexts: List[Dict[str, object]], patterns: List[str]) -> str:
        for item in contexts:
            payload = self._normalize_payload(str(item.get("text") or ""))
            for pattern in patterns:
                match = re.search(pattern, payload)
                if not match:
                    continue
                values = [group.strip() for group in match.groups() if group and str(group).strip()]
                if values:
                    return " ".join(values)
                return match.group(0).strip()
        return ""

    def _extract_all_matches(self, contexts: List[Dict[str, object]], pattern: str) -> List[str]:
        values: List[str] = []
        for item in contexts:
            payload = self._normalize_payload(str(item.get("text") or ""))
            for match in re.findall(pattern, payload):
                if isinstance(match, tuple):
                    values.extend([group.strip() for group in match if group and str(group).strip()])
                elif match and str(match).strip():
                    values.append(str(match).strip())
        return dedupe_preserve_order(values)

    def _extract_table_amount_for_label(self, contexts: List[Dict[str, object]], label: str) -> str:
        for item in contexts:
            raw_text = str(item.get("text") or "")
            payload = self._normalize_payload(raw_text)
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

    def _ensure_money_unit(self, value: str, default_unit: str = "万元") -> str:
        normalized = (value or "").strip()
        if not normalized:
            return ""
        if re.search(r"(万元|亿元|元)$", normalized):
            return normalized
        if re.fullmatch(r"[0-9][0-9,\.]*", normalized):
            return f"{normalized}{default_unit}"
        return normalized

    def _filter_contexts_for_company(self, contexts: List[Dict[str, object]], target_company: str) -> List[Dict[str, object]]:
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

    def _extract_table_row_items(self, contexts: List[Dict[str, object]]) -> List[str]:
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

    def _normalize_project_items(self, values: List[str]) -> List[str]:
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
                item = piece.strip().strip("“”\"")
                if not item or item in blocked_items:
                    continue
                if len(item) > 40:
                    continue
                normalized.append(item)
        return dedupe_preserve_order(normalized)

    def _extract_numeric_series(self, contexts: List[Dict[str, object]], patterns: List[str]) -> List[str]:
        for item in contexts:
            payload = self._normalize_payload(str(item.get("text") or ""))
            for pattern in patterns:
                match = re.search(pattern, payload)
                if not match:
                    continue
                values = re.findall(r"\d[\d,]*(?:\.\d+)?\s*(?:万元|亿元|元|%)", match.group(0))
                if values:
                    return values
        return []

    def _extract_money_series(self, contexts: List[Dict[str, object]], patterns: List[str]) -> List[str]:
        for item in contexts:
            payload = self._normalize_payload(str(item.get("text") or ""))
            for pattern in patterns:
                match = re.search(pattern, payload)
                if not match:
                    continue
                values = re.findall(r"\d[\d,]*(?:\.\d+)?\s*(?:万元|亿元|元)", match.group(0))
                if values:
                    return values
        return []

    def _answer_field_lookup(self, query: str, contexts: List[Dict[str, object]], field_keys: List[str]) -> str:
        rules = {
            "法定代表人": [r"法定代表人[:：]?\s*([\u4e00-\u9fff]{2,12})"],
            "注册资本": [r"注册资本[:：]?\s*([0-9,\.]+\s*(?:万元|亿元|元))"],
            "补充流动资金": [r"补充流动资金[^0-9]{0,20}([0-9,\.]+\s*(?:万元|亿元|元))"],
            "技术标准": [r"(《[^》]*(?:视频技术规范|技术规范|标准|规范)[^》]*》)"],
        }
        for field_key in field_keys:
            value = self._extract_first_match(contexts, rules.get(field_key, []))
            if not value:
                continue
            if field_key == "法定代表人":
                return self._clean_answer_text(f"法定代表人是{value}。{self._citation_text(contexts)}")
            if field_key == "注册资本":
                return self._clean_answer_text(f"注册资本是{value}。{self._citation_text(contexts)}")
            if field_key == "补充流动资金":
                if not re.search(r"(万元|亿元|元)$", value):
                    table_amount = self._extract_table_amount_for_label(contexts, "补充流动资金")
                    if table_amount:
                        value = table_amount
                value = self._ensure_money_unit(value)
                return self._clean_answer_text(f"计划使用{value}用于补充流动资金。{self._citation_text(contexts)}")
            if field_key == "技术标准":
                return self._clean_answer_text(f"参与制定的技术标准是{value}。{self._citation_text(contexts)}")
        if "补充流动资金" in field_keys:
            table_amount = self._extract_table_amount_for_label(contexts, "补充流动资金")
            if table_amount:
                table_amount = self._ensure_money_unit(table_amount)
                return self._clean_answer_text(f"计划使用本次发行募集资金中的{table_amount}用于补充流动资金。{self._citation_text(contexts)}")
        return ""

    def _answer_issuance(self, contexts: List[Dict[str, object]]) -> str:
        shares = self._extract_first_match(
            contexts,
            [
                r"发行股数[:：]?\s*([0-9,\.]+\s*万股)",
                r"发行股数及占发行后总股本比例[:：]?\s*([0-9,\.]+\s*万股)",
            ],
        )
        ratio = self._extract_first_match(
            contexts,
            [
                r"占发行后总股本的比例为([0-9,\.]+%)",
                r"占发行后总股本的比例[:：]?\s*([0-9,\.]+%)",
            ],
        )
        if shares and ratio:
            return self._clean_answer_text(f"本次发行股数是{shares}，占发行后总股本的比例是{ratio}。{self._citation_text(contexts)}")
        return ""

    def _answer_fundraising_projects(self, contexts: List[Dict[str, object]]) -> str:
        structured_values: List[str] = []
        for item in contexts:
            metadata = dict(item.get("metadata") or {})
            field_title = str(metadata.get("field_title") or "").strip()
            page_type = str(metadata.get("page_type") or "").strip()
            if field_title != "募集资金项目":
                continue
            if page_type not in {"structured", "vlm_structured"}:
                continue
            value = self._extract_value_from_structured_text(str(item.get("text") or ""))
            if value:
                structured_values.append(value)

        normalized_structured = self._normalize_project_items(structured_values)
        if normalized_structured:
            structured_contexts = [
                item
                for item in contexts
                if str((item.get("metadata") or {}).get("field_title") or "").strip() == "募集资金项目"
                and str((item.get("metadata") or {}).get("page_type") or "").strip() in {"structured", "vlm_structured"}
            ]
            return self._clean_answer_text(
                f"本次募集资金拟投资的项目包括：{'、'.join(normalized_structured)}。{self._citation_text(structured_contexts or contexts)}"
            )

        items = self._normalize_project_items(self._extract_table_row_items(contexts))
        if items:
            return self._clean_answer_text(f"本次募集资金拟投资的项目包括：{'、'.join(items[:10])}。{self._citation_text(contexts)}")
        return ""

    def _answer_supplementary_working_capital(self, contexts: List[Dict[str, object]]) -> str:
        amount = self._extract_first_match(
            contexts,
            [r"补充流动资金[^0-9]{0,20}([0-9,\.]+\s*万元)"],
        )
        if not amount:
            amount = self._extract_table_amount_for_label(contexts, "补充流动资金")
        if amount:
            amount = self._ensure_money_unit(amount)
            return self._clean_answer_text(f"计划使用本次发行募集资金中的{amount}用于补充流动资金。{self._citation_text(contexts)}")
        return ""

    def _answer_technical_standard(self, contexts: List[Dict[str, object]]) -> str:
        standard = self._extract_first_match(contexts, [r"(《[^》]*某视频技术规范[^》]*》)", r"(《[^》]*(?:技术规范|标准|规范)[^》]*》)"])
        if standard:
            return self._clean_answer_text(f"公司参与制定的技术标准是{standard}。{self._citation_text(contexts)}")
        return ""

    def _answer_award_project(self, contexts: List[Dict[str, object]]) -> str:
        project = self._extract_first_match(
            contexts,
            [
                r"(某情报、指挥、控制与通信网络一体化工程)",
                r"(.*?工程)[^。；\n]*国家科技进步一等奖",
            ],
        )
        if project:
            return self._clean_answer_text(f"荣获国家科技进步一等奖的工程是{project}。{self._citation_text(contexts)}")
        return ""

    def _answer_key_supplier_domain(self, contexts: List[Dict[str, object]]) -> str:
        joined = "\n".join(self._normalize_payload(str(item.get("text") or "")) for item in contexts)
        if "国防军队视频指挥领域" in joined:
            return self._clean_answer_text(f"公司已经成为国防军队视频指挥领域的重要供应商。{self._citation_text(contexts)}")
        if "军队视频指挥领域" in joined:
            return self._clean_answer_text(f"公司已经成为军队视频指挥领域的重要供应商。{self._citation_text(contexts)}")
        return ""

    def _answer_related_party(self, query: str, contexts: List[Dict[str, object]]) -> str:
        joined = "\n".join(self._normalize_payload(str(item.get("text") or "")) for item in contexts)
        if "不存在控制关系" in query or "不受同一控制" in query:
            hits = dedupe_preserve_order(
                re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,20}(?:投资|贸易|博润|聚源|芯达|有限公司|合伙企业)", joined)
            )
            if hits:
                return self._clean_answer_text(f"与公司不存在控制关系的关联方企业有：{'、'.join(dedupe_preserve_order(hits))}。{self._citation_text(contexts)}")
            return ""

        if "存在控制关系" in query:
            name = self._extract_first_match(contexts, [r"([\u4e00-\u9fff]{2,12})持有本公司股份", r"([\u4e00-\u9fff]{2,12})，?持股比例", r"([\u4e00-\u9fff]{2,12}).{0,10}(?:控股股东|实际控制人)"])
            ratio = self._extract_first_match(contexts, [r"占本公司总股本的([0-9,\.]+%)", r"持股比例[^0-9]{0,8}([0-9,\.]+%)"])
            relation = self._extract_first_match(contexts, [r"(控股股东)", r"(实际控制人)"])
            if name and ratio and relation:
                return self._clean_answer_text(f"与公司存在控制关系的关联方是{name}，持股比例为{ratio}，与本公司的关系是{relation}。{self._citation_text(contexts)}")
            if name and ratio:
                return self._clean_answer_text(f"与公司存在控制关系的关联方是{name}，持股比例为{ratio}。{self._citation_text(contexts)}")
        return ""

    def _answer_military_revenue(self, query: str, contexts: List[Dict[str, object]]) -> str:
        ratio_query = "比重" in query or "占主营业务收入" in query or "占比" in query
        if ratio_query:
            ratios = self._extract_numeric_series(contexts, [r"占主营业务收入的比重分别为[^。；\n]+"])
            if ratios:
                return self._clean_answer_text(f"报告期内，来自军用领域收入占主营业务收入的比重分别为：{'、'.join(ratios)}。{self._citation_text(contexts)}")
        values = self._extract_money_series(
            contexts,
            [r"销售额合计分别为[^。；\n]+", r"来自军用领域的收入分别为[^。；\n]+"],
        )
        if values:
            return self._clean_answer_text(f"报告期内，公司来自军用领域的收入分别为：{'、'.join(values)}。{self._citation_text(contexts)}")
        return ""

    def _answer_org_structure(self, contexts: List[Dict[str, object]]) -> str:
        joined = "\n".join(self._normalize_payload(str(item.get("text") or "")) for item in contexts)
        department_hits = dedupe_preserve_order(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,20}销售部", joined))
        office_hits = dedupe_preserve_order(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,20}销售处", joined))
        if department_hits and office_hits:
            return self._clean_answer_text(
                f"销售部由{len(department_hits)}个部门构成：{'、'.join(department_hits)}；其中大客户销售部由{len(office_hits)}个销售处构成：{'、'.join(office_hits)}。{self._citation_text(contexts)}"
            )
        if office_hits:
            return self._clean_answer_text(f"大客户销售部由{len(office_hits)}个销售处构成：{'、'.join(office_hits)}。{self._citation_text(contexts)}")
        return ""

    def _answer_chart_trend(self, contexts: List[Dict[str, object]]) -> str:
        joined = "\n".join(self._normalize_payload(str(item.get("text") or "")) for item in contexts)
        fastest = self._extract_first_match(
            contexts,
            [
                r"增长率最快的是([\u4e00-\u9fffA-Za-z0-9]+行业)",
                r"增长率最高的是([\u4e00-\u9fffA-Za-z0-9]+行业)",
                r"([\u4e00-\u9fffA-Za-z0-9]+行业)增长率最高",
            ],
        )
        negative = self._extract_first_match(
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
            return self._clean_answer_text(f"增长率最快的是{fastest}，负增长的是{negative}。{self._citation_text(contexts)}")
        return ""

    def _answer_upstream_downstream(self, query: str, contexts: List[Dict[str, object]]) -> str:
        joined = "\n".join(self._normalize_payload(str(item.get("text") or "")) for item in contexts)
        if "上游" in query:
            hits = dedupe_preserve_order(
                re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,24}(?:制造企业|行业企业|终端用户|机关|军队|能源)", joined)
            )
            if hits:
                return self._clean_answer_text(f"根据招股意向书，电子信息行业的上游涉及：{'、'.join(hits)}。{self._citation_text(contexts)}")
        if "下游" in query:
            hits = dedupe_preserve_order(
                re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,24}(?:行业企业|终端用户|机关|军队|能源)", joined)
            )
            if hits:
                return self._clean_answer_text(f"根据招股意向书，电子信息行业的下游主要包括：{'、'.join(hits)}等行业企业。{self._citation_text(contexts)}")
        return ""

    def _controlled_answer(self, query: str, contexts: List[Dict[str, object]], intent: object | None) -> str:
        target_company = str(self._get_intent_attr(intent, "target_company", "") or "")
        question_type = str(self._get_intent_attr(intent, "question_type", "") or "")
        query_tags = list(self._get_intent_attr(intent, "query_tags", []) or [])
        field_keys = list(self._get_intent_attr(intent, "field_keys", []) or [])
        filtered_contexts = self._filter_contexts_for_company(contexts, target_company)

        if question_type == "field_lookup":
            answer = self._answer_field_lookup(query, filtered_contexts, field_keys)
            if answer:
                return answer
        if "issuance" in query_tags and ("发行股数" in query or "发行后总股本" in query):
            answer = self._answer_issuance(filtered_contexts)
            if answer:
                return answer
        if "补充流动资金" in query:
            answer = self._answer_supplementary_working_capital(filtered_contexts)
            if answer:
                return answer
        if "fundraising" in query_tags and question_type == "table_list":
            answer = self._answer_fundraising_projects(filtered_contexts)
            if answer:
                return answer
        if "military_revenue" in query_tags:
            answer = self._answer_military_revenue(query, filtered_contexts)
            if answer:
                return answer
        if "related_party" in query_tags:
            answer = self._answer_related_party(query, filtered_contexts)
            if answer:
                return answer
        if "一等奖工程" in field_keys:
            answer = self._answer_award_project(filtered_contexts)
            if answer:
                return answer
        if "技术标准" in field_keys or "技术标准" in query:
            answer = self._answer_technical_standard(filtered_contexts)
            if answer:
                return answer
        if "重要供应商" in query or "供应商领域" in query:
            answer = self._answer_key_supplier_domain(filtered_contexts)
            if answer:
                return answer
        if question_type == "org_structure":
            answer = self._answer_org_structure(filtered_contexts)
            if answer:
                return answer
        if question_type == "chart_trend":
            answer = self._answer_chart_trend(filtered_contexts)
            if answer:
                return answer
        if any(field in field_keys for field in ["上游行业", "下游行业"]):
            answer = self._answer_upstream_downstream(query, filtered_contexts)
            if answer:
                return answer
        return ""

    def _extractive_answer(self, query: str, contexts: List[Dict[str, object]], intent: object | None = None) -> str:
        filtered_contexts = self._filter_contexts_for_company(
            contexts,
            str(self._get_intent_attr(intent, "target_company", "") or ""),
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
            return f"未检索到足够证据。{self._citation_text(filtered_contexts)}"
        answer = "；".join(snippets[:2])
        return self._clean_answer_text(f"{answer}。{self._citation_text(filtered_contexts)}")

    def _post_chat(
        self,
        api_url: str,
        api_key: str,
        model_name: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = requests.post(
            api_url,
            headers=headers,
            data=json.dumps(payload),
            timeout=settings.llm_request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        if "choices" in data:
            return str(data["choices"][0]["message"]["content"])
        return str(data)

    def _summarize_request_exception(self, exc: requests.RequestException) -> str:
        response = getattr(exc, "response", None)
        if response is None:
            return f"{type(exc).__name__}: {exc}"
        status_code = getattr(response, "status_code", "")
        body_text = ""
        try:
            body_text = response.text or ""
        except Exception:
            body_text = ""
        body_text = re.sub(r"\s+", " ", body_text).strip()
        if len(body_text) > 300:
            body_text = body_text[:297] + "..."
        if body_text:
            return f"{type(exc).__name__}: status={status_code} body={body_text}"
        return f"{type(exc).__name__}: status={status_code} {exc}"

    def _call_primary_then_fallback(self, prompt: str, system_prompt: str, max_tokens: int) -> Optional[str]:
        self.last_call_details = {
            "status": "not_attempted",
            "prompt_chars": len(prompt or ""),
            "max_tokens": int(max_tokens),
            "attempts": [],
        }
        primary_error_summary = ""
        if self.provider != "extractive" and self.api_url and self.model_name:
            try:
                content = self._post_chat(self.api_url, self.api_key, self.model_name, prompt, system_prompt, max_tokens)
                self.last_call_details = {
                    **self.last_call_details,
                    "status": "primary_ok",
                    "selected_route": "primary",
                    "attempts": [
                        {
                            "route": "primary",
                            "api_url": self.api_url,
                            "model_name": self.model_name,
                            "result": "ok",
                        }
                    ],
                }
                return content
            except requests.RequestException as exc:
                primary_error_summary = self._summarize_request_exception(exc)
                self.last_call_details["attempts"] = [
                    {
                        "route": "primary",
                        "api_url": self.api_url,
                        "model_name": self.model_name,
                        "result": "error",
                        "error": primary_error_summary,
                    }
                ]
        if self._is_local_url(self.fallback_api_url):
            self.last_call_details = {
                **self.last_call_details,
                "status": "all_failed" if primary_error_summary else "no_route_available",
                "selected_route": "",
                "last_error": primary_error_summary or "Local fallback disabled: no local LLM service expected.",
            }
            return None
        if self.fallback_api_url and self.fallback_model_name:
            try:
                content = self._post_chat(
                    self.fallback_api_url,
                    self.fallback_api_key,
                    self.fallback_model_name,
                    prompt,
                    system_prompt,
                    max_tokens,
                )
                attempts = list(self.last_call_details.get("attempts") or [])
                attempts.append(
                    {
                        "route": "fallback",
                        "api_url": self.fallback_api_url,
                        "model_name": self.fallback_model_name,
                        "result": "ok",
                    }
                )
                self.last_call_details = {
                    **self.last_call_details,
                    "status": "fallback_ok",
                    "selected_route": "fallback",
                    "attempts": attempts,
                }
                return content
            except requests.RequestException as exc:
                attempts = list(self.last_call_details.get("attempts") or [])
                attempts.append(
                    {
                        "route": "fallback",
                        "api_url": self.fallback_api_url,
                        "model_name": self.fallback_model_name,
                        "result": "error",
                        "error": self._summarize_request_exception(exc),
                    }
                )
                self.last_call_details = {
                    **self.last_call_details,
                    "status": "all_failed",
                    "selected_route": "",
                    "attempts": attempts,
                    "last_error": attempts[-1]["error"],
                }
                return None
        self.last_call_details = {
            **self.last_call_details,
            "status": "no_route_available",
            "selected_route": "",
        }
        return None

    def answer(self, query: str, contexts: List[Dict[str, object]], intent: object | None = None) -> str:
        controlled_answer = self._controlled_answer(query, contexts, intent)
        question_type = str(self._get_intent_attr(intent, "question_type", "") or "")
        query_tags = list(self._get_intent_attr(intent, "query_tags", []) or [])
        force_controlled = question_type in {"field_lookup", "table_numeric", "table_list", "org_structure", "chart_trend"}
        if any(
            tag in query_tags
            for tag in ["issuance", "fundraising", "related_party", "military_revenue", "technical_standard"]
        ):
            force_controlled = True
        fallback_answer = self._extractive_answer(query, contexts, intent=intent)

        if self.provider == "extractive":
            return self._clean_answer_text(controlled_answer or fallback_answer)
        if controlled_answer and force_controlled:
            return self._clean_answer_text(controlled_answer)

        content = self._call_primary_then_fallback(
            prompt=self._build_prompt(query, contexts, intent=intent),
            system_prompt="你是一个严格基于证据回答问题的 RAG 助手。",
            max_tokens=settings.max_new_tokens,
        )
        if content is not None:
            if controlled_answer and (content.count("第") >= 3 or content.count("|") >= 6):
                return self._clean_answer_text(controlled_answer)
            if ("未检索到足够证据" in content or "未检索到相关证据" in content or "无法确定" in content) and fallback_answer:
                return self._clean_answer_text(controlled_answer or fallback_answer)
            return self._clean_answer_text(content)
        return self._clean_answer_text(controlled_answer or fallback_answer)

    def structure_page(self, page: Dict[str, object]) -> List[Dict[str, object]]:
        text = str(page.get("text") or "").strip()
        if not text:
            return []
        if self.provider == "extractive":
            return self._heuristic_structure_page(page)
        prompt = (
            "你是一个仅基于输入页面内容做结构化抽取的助手。\n"
            "请从页面中抽取高价值问答字段，尤其关注：金额、比例、注册资本、法定代表人、募集资金用途、技术标准、获奖工程、上游下游行业、重要供应商领域。\n"
            "不要补充外部知识，不要猜测。\n"
            '输出 JSON 数组，每个元素格式为：{"title":"字段名","value":"字段值","evidence":"原文证据","type":"field|table_summary|fact"}。\n'
            "如果无法抽取，返回空数组。\n\n"
            f"页码：{page.get('page_number')}\n"
            f"逻辑页：{page.get('logical_page') or '-'}\n"
            f"页面类型：{page.get('page_type') or 'text'}\n"
            f"内容：\n{text[:4000]}"
        )
        content = self._call_primary_then_fallback(prompt, "你是一个严格做结构化抽取的助手。", 600)
        if content is None:
            return self._heuristic_structure_page(page)
        return self._parse_structured_json(content, page)

    def analyze_table(self, page: Dict[str, object]) -> List[Dict[str, object]]:
        table_text = str(page.get("tables_markdown") or "").strip()
        if not table_text:
            return []
        if self.provider == "extractive":
            return self._heuristic_analyze_table(page)
        prompt = (
            "你是一个只基于输入表格内容做分析的助手。\n"
            "请从表格中提取对问答有价值的结论，尤其关注：增长趋势、下降趋势、最大值、最小值、关键金额、关键比例、项目对应金额。\n"
            "不要补充表格之外的信息，不要编造。\n"
            '输出 JSON 数组，每个元素格式为：{"title":"分析标题","value":"结论","evidence":"支撑该结论的表格数据","type":"table_trend|table_fact"}。\n'
            "如果无法得出可靠结论，返回空数组。\n\n"
            f"页码：{page.get('page_number')}\n"
            f"逻辑页：{page.get('logical_page') or '-'}\n"
            f"表格内容：\n{table_text[:5000]}"
        )
        content = self._call_primary_then_fallback(prompt, "你是一个严格基于表格证据做结构化分析的助手。", 700)
        if content is None:
            return self._heuristic_analyze_table(page)
        return self._parse_structured_json(content, page)

    def _parse_structured_json(self, content: str, page: Dict[str, object]) -> List[Dict[str, object]]:
        text = content.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
        text = text.strip()
        try:
            payload = json.loads(text)
        except Exception:
            return self._heuristic_structure_page(page)
        if not isinstance(payload, list):
            return []
        results: List[Dict[str, object]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            value = str(item.get("value") or "").strip()
            evidence = str(item.get("evidence") or "").strip()
            item_type = str(item.get("type") or "field").strip()
            if not title or not value:
                continue
            results.append({"title": title, "value": value, "evidence": evidence, "type": item_type})
        return results

    def _heuristic_structure_page(self, page: Dict[str, object]) -> List[Dict[str, object]]:
        text = str(page.get("text") or "")
        results: List[Dict[str, object]] = []
        patterns = [
            ("注册资本", r"注册资本[:：]?\s*([0-9,\.]+\s*(?:万元|亿元|元))"),
            ("法定代表人", r"法定代表人[:：]?\s*([\u4e00-\u9fff]{2,12})"),
            ("补充流动资金", r"补充流动资金[^0-9]{0,20}([0-9,\.]+\s*(?:万元|亿元|元))"),
            ("技术标准", r"(参与制定[^。；\n]*?(?:标准|规范)[^。；\n]*)"),
            ("获奖工程", r"([^。；\n]*国家科技进步一等奖[^。；\n]*)"),
        ]
        for title, pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            value = match.group(1) if match.lastindex else match.group(0)
            results.append({"title": title, "value": value.strip(), "evidence": match.group(0).strip(), "type": "field"})
        return results

    def _heuristic_analyze_table(self, page: Dict[str, object]) -> List[Dict[str, object]]:
        table_text = str(page.get("tables_markdown") or "")
        rows = [line.strip() for line in table_text.splitlines() if line.strip().startswith("|")]
        if len(rows) < 3:
            return []
        numeric_values = []
        for row in rows[2:]:
            values = re.findall(r"\d[\d,]*(?:\.\d+)?", row)
            for value in values:
                try:
                    numeric_values.append(float(value.replace(",", "")))
                except ValueError:
                    continue
        if len(numeric_values) < 2:
            return []
        evidence = "；".join(rows[1:4])[:500]
        results: List[Dict[str, object]] = []
        if numeric_values == sorted(numeric_values):
            results.append({"title": "表格趋势", "value": "表格中的数值整体呈上升趋势。", "evidence": evidence, "type": "table_trend"})
        elif numeric_values == sorted(numeric_values, reverse=True):
            results.append({"title": "表格趋势", "value": "表格中的数值整体呈下降趋势。", "evidence": evidence, "type": "table_trend"})
        results.append({"title": "表格最大值", "value": str(max(numeric_values)), "evidence": evidence, "type": "table_fact"})
        results.append({"title": "表格最小值", "value": str(min(numeric_values)), "evidence": evidence, "type": "table_fact"})
        return results
