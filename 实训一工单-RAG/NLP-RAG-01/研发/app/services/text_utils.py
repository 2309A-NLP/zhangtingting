# 工单: 人工智能NLP-RAG-基于PDF文档的问答系统

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_number: int
    logical_page: str | None
    metadata: Dict[str, str]


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def stable_chunk_id(page_number: int, offset: int, text: str) -> str:
    digest = hashlib.md5(f"{page_number}:{offset}:{text}".encode("utf-8")).hexdigest()
    return f"chunk-{digest[:16]}"


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(
    pages: Sequence[Dict[str, object]],
    chunk_size: int,
    overlap: int,
) -> List[Chunk]:
    results: List[Chunk] = []
    for page in pages:
        page_text = str(page["text"]).strip()
        if not page_text:
            continue
        segments = split_text(page_text, chunk_size, overlap)
        offset = 0
        for segment in segments:
            chunk_id = stable_chunk_id(int(page["page_number"]), offset, segment)
            results.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=segment,
                    page_number=int(page["page_number"]),
                    logical_page=page.get("logical_page"),
                    metadata={
                        "page_number": str(page["page_number"]),
                        "logical_page": str(page.get("logical_page") or ""),
                    },
                )
            )
            offset += max(1, len(segment) - overlap)
    return results


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
