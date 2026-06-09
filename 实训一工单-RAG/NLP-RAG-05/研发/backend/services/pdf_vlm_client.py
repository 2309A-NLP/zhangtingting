from __future__ import annotations

# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
import base64
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import requests

from backend.config import settings


logger = logging.getLogger(__name__)


class PDFVLMClient:
    def __init__(self) -> None:
        self.provider = settings.pdf_vlm_provider
        self.api_url = settings.pdf_vlm_api_url.strip()
        self.api_key = settings.pdf_vlm_api_key.strip()
        self.model_name = settings.pdf_vlm_model_name.strip()

    def _raw_cache_path(self, cache_dir: Path, page_number: int) -> Path:
        return cache_dir / f"page_{page_number}.raw.json"

    def is_enabled(self) -> bool:
        return (
            self.provider == "openai_compatible"
            and bool(self.api_url)
            and bool(self.api_key)
            and bool(self.model_name)
        )

    def enhance_page(
        self,
        page_number: int,
        logical_page: str | None,
        local_text: str,
        table_markdown: str,
        image_bytes: bytes,
        cache_dir: Path,
        mode: str = "full",
        force_items: bool = False,
        bypass_cache: bool = False,
        cache_variant: str = "",
    ) -> Dict[str, object]:
        if not self.is_enabled():
            return {"items": [], "from_cache": False, "status": "disabled"}

        cache_dir.mkdir(parents=True, exist_ok=True)
        suffix = "" if mode == "full" else f".{mode}"
        if cache_variant:
            suffix = f"{suffix}.{cache_variant}" if suffix else f".{cache_variant}"
        cache_path = cache_dir / f"page_{page_number}{suffix}.json"
        raw_cache_path = self._raw_cache_path(cache_dir, page_number)
        if suffix:
            raw_cache_path = cache_dir / f"page_{page_number}{suffix}.raw.json"

        if cache_path.exists() and not bypass_cache:
            try:
                return {
                    "items": json.loads(cache_path.read_text(encoding="utf-8")),
                    "from_cache": True,
                    "status": "cache_hit",
                }
            except Exception:
                cache_path.unlink(missing_ok=True)

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt = self._build_prompt(
            page_number=page_number,
            logical_page=logical_page,
            local_text=local_text,
            table_markdown=table_markdown,
            mode=mode,
            force_items=force_items,
        )
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是一个严格基于 PDF 页面截图和本地解析结果做结构化增强抽取的助手。"
                        "不能编造，不要复述整页正文，只输出有检索价值的结构化结论。"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                            },
                        },
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": 900,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        for attempt in range(1, settings.pdf_vlm_retry_count + 1):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=settings.pdf_vlm_request_timeout,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"] if "choices" in data else "[]"
                raw_cache_path.write_text(
                    json.dumps(
                        {
                            "page_number": page_number,
                            "logical_page": logical_page,
                            "mode": mode,
                            "response": data,
                            "content": content,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                parsed = self._parse_json(content)
                cache_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
                return {"items": parsed, "from_cache": False, "status": "api_success"}
            except requests.RequestException as exc:
                last_error = exc
                raw_cache_path.write_text(
                    json.dumps(
                        {
                            "page_number": page_number,
                            "logical_page": logical_page,
                            "mode": mode,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                logger.warning(
                    "PDF VLM enhancement failed on page %s (attempt %s/%s): %s",
                    page_number,
                    attempt,
                    settings.pdf_vlm_retry_count,
                    exc,
                )
                if attempt < settings.pdf_vlm_retry_count:
                    time.sleep(settings.pdf_vlm_retry_backoff * attempt)

        if settings.pdf_vlm_strict_mode and last_error is not None:
            raise RuntimeError(f"PDF VLM enhancement failed on page {page_number}: {last_error}") from last_error

        return {"items": [], "from_cache": False, "status": "failed"}

    def _build_prompt(
        self,
        *,
        page_number: int,
        logical_page: str | None,
        local_text: str,
        table_markdown: str,
        mode: str,
        force_items: bool,
    ) -> str:
        common_header = (
            "请只输出 JSON 数组。每个元素格式必须为："
            '{"title":"字段名或图表主题","value":"精炼结论","evidence":"直接证据","type":"field|table_fact|table_summary|layout_fix|org_chart_summary|org_chart_relation|chart_fact|chart_summary"}。'
        )
        common_rules = (
            "优先抽取：法定代表人、注册资本、金额、比例、募集资金用途、技术标准、上游下游行业、重要供应商领域、"
            "组织结构图层级、图表中的最高/最低/负增长/最快增长项。"
        )
        page_info = f"页码：{page_number}\n逻辑页：{logical_page or '-'}"
        empty_rule = (
            "如果页面没有明确可抽取的高价值信息，返回空数组 []。"
            if not force_items
            else "请尽量返回至少 1 条最有把握的结构化结果。"
        )

        if mode == "image_only":
            return "\n".join(
                [
                    "你现在只允许依据页面截图本身抽取信息，不要参考任何额外文本。",
                    common_header,
                    common_rules,
                    "如果页面是组织结构图，请优先输出部门层级、部门数量、下设销售处等关系。",
                    "如果页面是增长图/应用结构图，请优先输出增长最快行业、负增长行业、最大值、最小值等。",
                    empty_rule,
                    page_info,
                ]
            )

        return "\n".join(
            [
                "你要基于页面截图 + 本地正文 + 本地表格，补强本地解析最容易错漏的信息。",
                common_header,
                common_rules,
                "不要复述整页全文，不要生成和证据无关的总结。",
                "如果页面是组织结构图，请优先输出父子部门关系、部门数量、销售处数量。",
                "如果页面是增长图/应用结构图，请优先输出增长最快行业、负增长行业、具体数值和结论。",
                empty_rule,
                page_info,
                f"本地正文：\n{local_text[:2500]}",
                f"本地表格：\n{table_markdown[:2500]}",
            ]
        )

    def _parse_json(self, content: str) -> List[Dict[str, str]]:
        text = content.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
        text = text.strip()

        try:
            payload = json.loads(text)
        except Exception:
            return []
        if not isinstance(payload, list):
            return []

        results: List[Dict[str, str]] = []
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
