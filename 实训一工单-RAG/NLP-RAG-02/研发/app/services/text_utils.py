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

    paragraphs = [segment.strip() for segment in re.split(r"\n{2,}", text) if segment.strip()]
    if not paragraphs:
        return _sliding_window(text, chunk_size, overlap)

    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= chunk_size:
            current = paragraph
            continue

        sentence_parts = _split_long_paragraph(paragraph, chunk_size, overlap)
        chunks.extend(sentence_parts[:-1])
        current = sentence_parts[-1]

    if current:
        chunks.append(current)

    return _merge_small_chunks(chunks, chunk_size, overlap)


def _sliding_window(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _split_long_paragraph(text: str, chunk_size: int, overlap: int) -> List[str]:
    sentences = [segment.strip() for segment in re.split(r"(?<=[。！？；;.!?])", text) if segment.strip()]
    if len(sentences) <= 1:
        return _sliding_window(text, chunk_size, overlap)

    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        candidate = sentence if not current else f"{current}{sentence}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(sentence) > chunk_size:
            chunks.extend(_sliding_window(sentence, chunk_size, overlap))
            current = ""
        else:
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def _merge_small_chunks(chunks: List[str], chunk_size: int, overlap: int) -> List[str]:
    if not chunks:
        return []
    merged: List[str] = []
    current = chunks[0]
    for chunk in chunks[1:]:
        candidate = f"{current}\n\n{chunk}"
        if len(current) < max(120, overlap * 2) and len(candidate) <= chunk_size:
            current = candidate
        else:
            merged.append(current)
            current = chunk
    merged.append(current)
    return merged


def build_chunks(
    pages: Sequence[Dict[str, object]],
    chunk_size: int,
    overlap: int,
) -> List[Chunk]:
    results: List[Chunk] = []
    for page in pages:
        page_text = str(page.get("text") or "").strip()
        if not page_text:
            continue

        page_type = str(page.get("page_type") or "text")
        section_title = str(page.get("section_title") or "")
        segments = split_text(page_text, chunk_size, overlap)
        offset = 0

        for segment in segments:
            normalized_segment = normalize_whitespace(segment)
            if not normalized_segment:
                continue
            chunk_id = stable_chunk_id(int(page["page_number"]), offset, normalized_segment)
            metadata = {
                "page_number": str(page["page_number"]),
                "logical_page": str(page.get("logical_page") or ""),
                "page_type": page_type,
                "section_title": section_title,
                "has_table": "1" if page.get("tables_markdown") else "0",
                "has_ocr": "1" if page.get("handwriting") else "0",
                "source": str(page.get("source") or "builtin"),
            }
            results.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=normalized_segment,
                    page_number=int(page["page_number"]),
                    logical_page=page.get("logical_page"),
                    metadata=metadata,
                )
            )
            offset += max(1, len(normalized_segment) - overlap)

        table_text = normalize_whitespace(str(page.get("tables_markdown") or ""))
        if table_text:
            table_segments = split_text(table_text, chunk_size, overlap)
            table_offset = 500000
            for segment in table_segments:
                normalized_segment = normalize_whitespace(segment)
                if not normalized_segment:
                    continue
                chunk_id = stable_chunk_id(int(page["page_number"]), table_offset, normalized_segment)
                metadata = {
                    "page_number": str(page["page_number"]),
                    "logical_page": str(page.get("logical_page") or ""),
                    "page_type": "table_markdown",
                    "section_title": section_title,
                    "has_table": "1",
                    "has_ocr": "1" if page.get("handwriting") else "0",
                    "source": str(page.get("source") or "builtin"),
                }
                results.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=normalized_segment,
                        page_number=int(page["page_number"]),
                        logical_page=page.get("logical_page"),
                        metadata=metadata,
                    )
                )
                table_offset += max(1, len(normalized_segment) - overlap)
    return results


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def keyword_overlap_score(query: str, text: str) -> float:
    query_terms = {term for term in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{1,}", query) if term.strip()}
    text_terms = {term for term in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{1,}", text) if term.strip()}
    if not query_terms or not text_terms:
        return 0.0
    hits = len(query_terms & text_terms)
    return hits / max(1, len(query_terms))
