# 工单: 人工智能NLP-RAG-基于PDF文档的问答系统

from __future__ import annotations

import json
from typing import Dict, List

import requests

from app.config import settings
from app.services.text_utils import dedupe_preserve_order


class LLMClient:
    def __init__(self, provider: str, api_url: str = "", api_key: str = "", model_name: str = "") -> None:
        self.provider = provider
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name

    def _build_prompt(self, query: str, contexts: List[Dict[str, object]]) -> str:
        context_text = "\n\n".join(
            f"[页码 {item['page_number']}, 逻辑页 {item.get('logical_page') or '-'}]\n{str(item['text'])[:settings.max_context_chars]}"
            for item in contexts[: settings.generation_top_n]
        )
        return (
            "你是一个严格基于招股说明书证据回答问题的助手。"
            "只能根据证据作答，不得编造。若证据不足，请明确说明。"
            f"\n\n问题：{query}\n\n证据：\n{context_text}\n\n请用简洁中文回答，并附引用页码。"
        )

    def _extractive_answer(self, query: str, contexts: List[Dict[str, object]]) -> str:
        if not contexts:
            return "未检索到相关证据，无法基于招股说明书作答。"
        snippets = []
        keywords = [token for token in query.replace("？", "").replace("?", "").split() if token]
        for item in contexts[: settings.generation_top_n]:
            text = str(item["text"])
            sentences = [segment.strip() for segment in text.replace("\n", " ").split("。") if segment.strip()]
            matched = [segment for segment in sentences if any(keyword in segment for keyword in keywords)] if keywords else sentences[:2]
            chosen = matched[:2] if matched else sentences[:2]
            if chosen:
                snippets.append(f"第{item['page_number']}页：{'；'.join(chosen)}。")
        if not snippets:
            snippets = [f"第{item['page_number']}页：{str(item['text'])[:180]}..." for item in contexts[:2]]
        citation_pages = "、".join(dedupe_preserve_order([str(item["page_number"]) for item in contexts[:3]]))
        return f"{' '.join(snippets[:3])} 以上内容来自招股说明书第{citation_pages}页。"

    def answer(self, query: str, contexts: List[Dict[str, object]]) -> str:
        if self.provider == "extractive" or not self.api_url:
            return self._extractive_answer(query, contexts)
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "你是一个严格基于证据回答问题的RAG助手。"},
                    {"role": "user", "content": self._build_prompt(query, contexts)},
                ],
                "temperature": 0.1,
                "max_tokens": settings.max_new_tokens,
            }
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=30)
            response.raise_for_status()
            data = response.json()
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            return str(data)
        except requests.RequestException:
            return self._extractive_answer(query, contexts)
