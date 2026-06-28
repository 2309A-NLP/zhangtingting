"""LLM 服务封装 — 支持多 Provider"""

from __future__ import annotations

import time
from typing import Optional

from config import settings


class LLMService:
    """大语言模型调用封装"""

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self._provider = provider or settings.LLM_PROVIDER
        self._api_key = api_key or settings.LLM_API_KEY
        self._model_name = model_name or settings.LLM_MODEL_NAME
        self._base_url = base_url or settings.LLM_BASE_URL
        self._client = None

    def _lazy_init(self):
        """延迟初始化 LLM 客户端"""
        if self._client is not None:
            return

        if self._provider == "local":
            # 本地模型（预留）
            raise NotImplementedError("本地模型部署暂未实现，请使用 API 模式")
        else:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """生成文本（带自动重试）"""
        self._lazy_init()
        start = time.perf_counter()
        last_error = None

        for attempt in range(3):
            try:
                kwargs = {
                    "model": self._model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": settings.LLM_TEMPERATURE,
                    "max_tokens": max_tokens or settings.LLM_MAX_TOKENS,
                    "timeout": settings.LLM_TIMEOUT,
                }

                response = self._client.chat.completions.create(**kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                content = response.choices[0].message.content or ""
                return content

            except Exception as e:
                last_error = e
                if attempt < 2:
                    time.sleep(2 ** attempt)  # 1s, 2s 退避
                    continue
                raise last_error

    def generate_stream(self, prompt: str):
        """流式生成"""
        self._lazy_init()
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
