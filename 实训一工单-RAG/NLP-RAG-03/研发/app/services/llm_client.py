# 工单编号：人工智能 NLP-RAG-PDF 文档的表格解析及检索优化
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

import requests

from app.config import settings
from app.services.text_utils import dedupe_preserve_order


class LLMClient:
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
        self.api_key = api_key.strip()
        self.model_name = model_name.strip()
        self.fallback_api_url = fallback_api_url.strip()
        self.fallback_api_key = fallback_api_key.strip()
        self.fallback_model_name = fallback_model_name.strip()

    def _build_prompt(self, query: str, contexts: List[Dict[str, object]]) -> str:
        context_text = "\n\n".join(
            (
                f"[页码 {item['page_number']}, 逻辑页 {item.get('logical_page') or '-'}, "
                f"类型 {item.get('metadata', {}).get('page_type', 'text')}]\n"
                f"{str(item['text'])[:settings.max_context_chars]}"
            )
            for item in contexts[: settings.generation_top_n]
        )
        return (
            "你是一个严格基于招股说明书证据回答问题的助手。\n"
            "只能根据给定证据作答，不得编造；如果证据不足，请明确说明“未检索到足够证据”。\n"
            "优先提取原文中的金额、比例、名称和表格字段，不要自行扩写背景。\n"
            "回答末尾必须附上引用页码，格式如：引用页码：128、129。\n\n"
            f"问题：{query}\n\n"
            f"证据：\n{context_text}\n\n"
            "请用简洁中文回答。"
        )

    def _extract_value_from_structured_text(self, text: str) -> str:
        match = re.search(r"值[:：]\s*(.+)", text)
        return match.group(1).strip() if match else ""

    def _compress_table_rows(self, text: str) -> str:
        if settings.answer_include_table_markdown:
            return text[:260]
        rows = [line.strip() for line in text.splitlines() if line.strip()]
        table_rows = [row for row in rows if row.startswith("|")]
        if table_rows:
            useful_rows = [row for row in table_rows if row != "| --- |" and "---" not in row][:3]
            if useful_rows:
                return "；".join(useful_rows)[:220]
        return text[:220]

    def _extractive_answer(self, query: str, contexts: List[Dict[str, object]]) -> str:
        if not contexts:
            return "未检索到相关证据，无法基于招股说明书作答。"

        structured_hits = [
            item
            for item in contexts[: settings.generation_top_n + 2]
            if str(item.get("metadata", {}).get("page_type")) in {"structured", "vlm_structured"}
        ]
        if structured_hits:
            primary = structured_hits[0]
            field_title = str(primary.get("metadata", {}).get("field_title") or "").strip()
            value = self._extract_value_from_structured_text(str(primary["text"]))
            if field_title and value:
                pages = "、".join(dedupe_preserve_order([str(item["page_number"]) for item in structured_hits[:2]]))
                return f"{field_title}是{value}。引用页码：{pages}。"

        snippets: List[str] = []
        keywords = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", query)
        for item in contexts[: settings.generation_top_n]:
            text = str(item["text"])
            page_type = str(item.get("metadata", {}).get("page_type") or "")

            if page_type in {"table_analysis", "structured", "vlm_structured"}:
                compact = text.replace("\n", " ")
                snippets.append(f"第{item['page_number']}页：{compact[:180]}。")
                continue

            if "|" in text:
                compact = self._compress_table_rows(text)
                snippets.append(f"第{item['page_number']}页：{compact}。")
                continue

            normalized = text.replace("\n", " ")
            sentences = [segment.strip() for segment in re.split(r"[。！？；]", normalized) if segment.strip()]
            matched = [segment for segment in sentences if any(keyword in segment for keyword in keywords)] if keywords else []
            chosen = matched[:2] if matched else sentences[:2]
            if chosen:
                snippets.append(f"第{item['page_number']}页：{'；'.join(chosen)}。")

        if not snippets:
            snippets = [f"第{item['page_number']}页：{str(item['text'])[:180]}..." for item in contexts[:2]]
        citation_pages = "、".join(dedupe_preserve_order([str(item["page_number"]) for item in contexts[:3]]))
        return f"{' '.join(snippets[:3])} 引用页码：{citation_pages}。"

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
            return data["choices"][0]["message"]["content"]
        return str(data)

    def _call_primary_then_fallback(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
    ) -> Optional[str]:
        if self.provider != "extractive" and self.api_url and self.model_name:
            try:
                return self._post_chat(
                    api_url=self.api_url,
                    api_key=self.api_key,
                    model_name=self.model_name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                )
            except requests.RequestException:
                pass

        if self.fallback_api_url and self.fallback_model_name:
            try:
                return self._post_chat(
                    api_url=self.fallback_api_url,
                    api_key=self.fallback_api_key,
                    model_name=self.fallback_model_name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                )
            except requests.RequestException:
                pass

        return None

    def answer(self, query: str, contexts: List[Dict[str, object]]) -> str:
        if self.provider == "extractive":
            return self._extractive_answer(query, contexts)
        content = self._call_primary_then_fallback(
            prompt=self._build_prompt(query, contexts),
            system_prompt="你是一个严格基于证据回答问题的 RAG 助手。",
            max_tokens=settings.max_new_tokens,
        )
        if content is not None:
            return content
        return self._extractive_answer(query, contexts)

    def structure_page(self, page: Dict[str, object]) -> List[Dict[str, object]]:
        text = str(page.get("text") or "").strip()
        if not text:
            return []
        if self.provider == "extractive":
            return self._heuristic_structure_page(page)

        prompt = (
            "你是一个仅基于输入页面内容做结构化抽取的助手。\n"
            "请从页面中抽取高价值问答字段，尤其关注：金额、比例、注册资本、法定代表人、"
            "募集资金用途、技术标准、获奖工程、上下游行业、重要供应商领域。\n"
            "不要补充外部知识，不要猜测。\n"
            "输出 JSON 数组，每个元素格式为："
            '{"title":"字段名","value":"字段值","evidence":"原文证据","type":"field|table_summary|fact"}。\n'
            "如果无法抽取，返回空数组。\n\n"
            f"页码：{page.get('page_number')}\n"
            f"逻辑页：{page.get('logical_page') or '-'}\n"
            f"页面类型：{page.get('page_type') or 'text'}\n"
            f"内容：\n{text[:4000]}"
        )
        content = self._call_primary_then_fallback(
            prompt=prompt,
            system_prompt="你是一个严格做结构化抽取的助手。",
            max_tokens=600,
        )
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
            "请从表格中提取对问答有价值的结论，尤其关注：增长趋势、下降趋势、最大值、最小值、"
            "关键金额、关键比例、项目对应金额。\n"
            "不要补充表格之外的信息，不要编造。\n"
            "输出 JSON 数组，每个元素格式为："
            '{"title":"分析标题","value":"结论","evidence":"支撑该结论的表格数据","type":"table_trend|table_fact"}。\n'
            "如果无法得出可靠结论，返回空数组。\n\n"
            f"页码：{page.get('page_number')}\n"
            f"逻辑页：{page.get('logical_page') or '-'}\n"
            f"表格内容：\n{table_text[:5000]}"
        )
        content = self._call_primary_then_fallback(
            prompt=prompt,
            system_prompt="你是一个严格基于表格证据做结构化分析的助手。",
            max_tokens=700,
        )
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
            results.append(
                {
                    "title": title,
                    "value": value,
                    "evidence": evidence,
                    "type": item_type,
                }
            )
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
            results.append(
                {
                    "title": title,
                    "value": value.strip(),
                    "evidence": match.group(0).strip(),
                    "type": "field",
                }
            )
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
            results.append(
                {
                    "title": "表格趋势",
                    "value": "表格中的数值整体呈上升趋势。",
                    "evidence": evidence,
                    "type": "table_trend",
                }
            )
        elif numeric_values == sorted(numeric_values, reverse=True):
            results.append(
                {
                    "title": "表格趋势",
                    "value": "表格中的数值整体呈下降趋势。",
                    "evidence": evidence,
                    "type": "table_trend",
                }
            )

        results.append(
            {
                "title": "表格最大值",
                "value": str(max(numeric_values)),
                "evidence": evidence,
                "type": "table_fact",
            }
        )
        results.append(
            {
                "title": "表格最小值",
                "value": str(min(numeric_values)),
                "evidence": evidence,
                "type": "table_fact",
            }
        )
        return results
