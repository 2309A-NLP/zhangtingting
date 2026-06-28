"""文本处理工具"""

from __future__ import annotations

import re


def remove_extra_whitespace(text: str) -> str:
    """去除多余空白"""
    return re.sub(r"\s+", " ", text).strip()


def truncate(text: str, max_length: int = 500) -> str:
    """截断文本"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
