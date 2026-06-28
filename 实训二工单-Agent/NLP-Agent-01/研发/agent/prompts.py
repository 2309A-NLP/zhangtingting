# 工单编号：人工智能NLP-Agent数字人项目-记账本任务
# 来源：北京八维信息集团 · 八维文化与产业研究院

from __future__ import annotations

import re
from pathlib import Path

from config import DEFAULT_WELCOME, PROMPT_PATH


def _strip_markdown(text: str) -> str:
    # 1. 删除代码块
    text = re.sub(r"```[\s\S]*?```", "", text)
    # 2. 删除标题标记（#、##、###等）
    # flags=re.MULTILINE 的作用：让 ^ 匹配每一行的开头，而不只是字符串开头
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # 3. 删除引用标记（>）
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    # 4. 删除分割线（---、***等）
    text = re.sub(r"\n-{3,}\n", "\n", text)
    return text.strip()


def load_system_prompt() -> str:
    path = Path(PROMPT_PATH)
    if path.exists():
        raw = path.read_text(encoding="utf-8-sig")
        cleaned = _strip_markdown(raw)
        return cleaned or DEFAULT_WELCOME
    return DEFAULT_WELCOME
# encoding="utf-8-sig" 的作用：
# 标准 utf-8 会保留文件开头的 BOM（\ufeff）
# utf-8-sig 自动识别并移除 BOM

SYSTEM_PROMPT = load_system_prompt()
