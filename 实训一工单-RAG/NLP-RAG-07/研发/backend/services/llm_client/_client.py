# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
from __future__ import annotations
from backend.services.llm_client._answers import (
    _clean_answer_text,
    _controlled_answer,
    _extractive_answer,
)

import json
import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests

from backend.config import settings


class LLMClient:
    _clean_answer_text = staticmethod(_clean_answer_text)
    _controlled_answer = _controlled_answer
    _extractive_answer = _extractive_answer

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

    def _build_prompt(
        self,
        query: str,
        contexts: List[Dict[str, object]],
        intent: object | None = None,
        candidate_answer: str = "",
    ) -> str:
        target_company = str(self._get_intent_attr(intent, "target_company", "") or "").strip()
        context_text = "\n\n".join(
            (
                f"[页码 {item['page_number']}, 逻辑页 {item.get('logical_page') or '-'}, 类型 {item.get('metadata', {}).get('page_type', 'text')}, "
                f"来源 {item.get('metadata', {}).get('source_pdf') or '-'}]\n"
                f"{self._clean_answer_text(str(item['text']))[:settings.max_context_chars]}"
            )
            for item in contexts[: settings.generation_top_n]
        )
        company_rule = (
            f'只回答与"{target_company}"直接相关的内容。若证据属于其他公司，一律忽略。\n'
            if target_company
            else ""
        )
        candidate_block = (
            f"可参考的规则抽取候选结论：{self._clean_answer_text(candidate_answer)}\n"
            "你必须核对证据后再决定是否采用，不能无条件照抄。\n"
            if candidate_answer.strip()
            else ""
        )
        return (
            "你是一个严格基于招股说明书证据回答问题的助手。\n"
            f"{company_rule}"
            '只能根据给定证据作答，不得编造；如果证据不足，请明确说"未检索到足够证据"。\n'
            "回答时先给结论，不要把原始大段文本块直接拼到答案里。\n"
            "不要输出任何内部检索标记、结构化字段名、调试字段或上下文标签。\n"
            "答案末尾必须附上引用页码，格式如：引用页码：22、30。\n\n"
            f"问题：{query}\n\n"
            f"{candidate_block}"
            f"证据：\n{context_text}\n\n"
            "请用简洁中文回答。"
        )

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
        self.last_call_details = {
            "status": "not_attempted",
            "answer_source": "",
        }
        controlled_answer = self._controlled_answer(query, contexts, intent)
        question_type = str(self._get_intent_attr(intent, "question_type", "") or "")
        query_tags = list(self._get_intent_attr(intent, "query_tags", []) or [])
        fallback_answer = self._extractive_answer(query, contexts, intent=intent)

        if self.provider == "extractive":
            self.last_call_details = {
                "status": "extractive_only",
                "answer_source": "controlled" if controlled_answer else "extractive_fallback",
            }
            return self._clean_answer_text(controlled_answer or fallback_answer)
        content = self._call_primary_then_fallback(
            prompt=self._build_prompt(query, contexts, intent=intent, candidate_answer=controlled_answer),
            system_prompt="你是一个严格基于证据回答问题的 RAG 助手。",
            max_tokens=settings.max_new_tokens,
        )
        if content is not None:
            cleaned_content = self._clean_answer_text(content)
            low_confidence_markers = ("未检索到足够证据", "未检索到相关证据")
            if controlled_answer and (content.count("第") >= 3 or content.count("|") >= 6):
                self.last_call_details = {
                    **self.last_call_details,
                    "answer_source": "controlled_after_bad_llm_format",
                }
                return self._clean_answer_text(controlled_answer)
            if any(marker in cleaned_content for marker in low_confidence_markers) and fallback_answer:
                self.last_call_details = {
                    **self.last_call_details,
                    "answer_source": "extractive_after_llm_no_evidence",
                }
                return self._clean_answer_text(controlled_answer or fallback_answer)
            self.last_call_details = {
                **self.last_call_details,
                "answer_source": "llm",
            }
            return cleaned_content
        self.last_call_details = {
            **self.last_call_details,
            "answer_source": "controlled_fallback" if controlled_answer else "extractive_fallback",
        }
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
