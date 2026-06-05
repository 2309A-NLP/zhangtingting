# 工单: 人工智能NLP-RAG-基于PDF文档的问答系统
from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import requests

from app.config import settings


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
        if mode == "image_only":
            prompt = (
                "你是一个用于招股说明书页面增强解析的多模态助手。\n"
                "本次只允许依据页面截图本身提取信息，不要参考任何外部文本。\n"
                "请只输出 JSON 数组，每个元素格式为："
                '{"title":"字段名或表格主题","value":"精炼结论或字段值","evidence":"页面中的原文或表格证据","type":"field|table_fact|table_summary|layout_fix"}。\n'
                "优先提取：法定代表人、注册资本、金额、比例、募集资金用途、技术标准、上下游行业、供应商领域、复杂表格关键值。\n"
                + ("必须尽量输出至少 1 条最有把握的结构化结果；不要轻易返回空数组。\n" if force_items else "不要输出整页全文，不要编造；若图片中没有高价值补充内容，返回空数组 []。\n")
                + "\n"
                f"页码：{page_number}\n"
                f"逻辑页：{logical_page or '-'}"
            )
        else:
            prompt = (
                "你是一个用于招股说明书页面增强解析的多模态助手。\n"
                "输入包括：页面截图、本地解析到的正文、本地提取的表格 Markdown。\n"
                "你的任务不是复述整页，而是补充本地解析中最重要、最容易错漏的内容。\n"
                "请只输出 JSON 数组，每个元素格式为："
                '{"title":"字段名或表格主题","value":"精炼结论或字段值","evidence":"页面中的原文或表格证据","type":"field|table_fact|table_summary|layout_fix"}。\n'
                "优先补充：法定代表人、注册资本、金额、比例、募集资金用途、技术标准、上下游行业、供应商领域、复杂表格关键值。\n"
                + ("必须尽量输出至少 1 条最有把握的结构化结果；不要轻易返回空数组。\n" if force_items else "不要输出整页全文，不要编造；若没有高价值补充内容，返回空数组 []。\n")
                + "\n"
                f"页码：{page_number}\n"
                f"逻辑页：{logical_page or '-'}\n"
                f"本地正文：\n{local_text[:2500]}\n\n"
                f"本地表格：\n{table_markdown[:2500]}"
            )
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个严格基于页面图像和本地解析结果做增强抽取的助手。",
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
                return {
                    "items": parsed,
                    "from_cache": False,
                    "status": "api_success",
                }
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

        return {
            "items": [],
            "from_cache": False,
            "status": "failed",
        }

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
