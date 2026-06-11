from __future__ import annotations

# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CHUNK_SIZE = 750
DEFAULT_CHUNK_OVERLAP = 75
DEFAULT_SEED_PAGES = 2
DEFAULT_APPEND_THRESHOLD = 400
DEFAULT_INPUT_DIR_NAME = "stage2_precise_extraction_rewire_test"
DEFAULT_OUTPUT_DIR_NAME = "stage3_text_chunking"
MARKER_PATTERN = re.compile(r"\[(?:TABLE|FIGURE|IMAGE):[^\]\n]+\]")
CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百零〇0-9]+章")
SECTION_RE = re.compile(r"^第[一二三四五六七八九十百零〇0-9]+节")
CN_ENUM_RE = re.compile(r"^[一二三四五六七八九十]+、")
CN_PAREN_ENUM_RE = re.compile(r"^[（(][一二三四五六七八九十百零〇0-9]+[）)]")
NUMERIC_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,3})(?:[、.．]|\s+)([^\d].*)$")


@dataclass
class HeadingRef:
    level: int
    title: str


@dataclass
class FlowBlock:
    block_id: str
    page_index: int
    text: str
    block_type: str
    char_count: int
    heading_level: int = 0
    heading_title: str = ""
    heading_path: tuple[HeadingRef, ...] = ()


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    payload = f"[stage3 {timestamp}] {message}"
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe_payload = payload.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe_payload, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heading-aware chunking for page text flow.")
    parser.add_argument("--artifact-dir", type=str, default="", help="Directory containing page_text_flow.json or page_text_flow_resolved.json")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory for chunk artifacts")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Target chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap in characters")
    parser.add_argument("--seed-pages", type=int, default=DEFAULT_SEED_PAGES, help="Reserved compatibility arg")
    parser.add_argument("--append-threshold", type=int, default=DEFAULT_APPEND_THRESHOLD, help="Reserved compatibility arg")
    parser.add_argument(
        "--source-mode",
        choices=("auto", "resolved", "raw"),
        default="auto",
        help="Choose resolved page text flow first, only raw, or auto fallback",
    )
    return parser.parse_args()


def load_page_text_flow(artifact_dir: Path, source_mode: str) -> tuple[list[dict[str, Any]], Path]:
    resolved_path = artifact_dir / "page_text_flow_resolved.json"
    raw_path = artifact_dir / "page_text_flow.json"

    if source_mode == "resolved":
        candidates = [resolved_path]
    elif source_mode == "raw":
        candidates = [raw_path]
    else:
        candidates = [resolved_path, raw_path]

    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload, path
    raise FileNotFoundError(f"No valid page text flow found under {artifact_dir}")


def normalize_page_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "page_index": int(entry.get("page_index") or 0),
        "page_text_flow": str(entry.get("page_text_flow") or "").strip(),
        "page_type": str(entry.get("page_type") or ""),
        "sub_type": str(entry.get("sub_type") or ""),
        "object_flow": entry.get("object_flow") if isinstance(entry.get("object_flow"), list) else [],
    }


def split_page_into_blocks(text: str) -> list[str]:
    if not text.strip():
        return []
    parts = re.split(r"\n\s*\n+", text)
    return [part.strip() for part in parts if part and part.strip()]


def is_marker_only(text: str) -> bool:
    return bool(MARKER_PATTERN.fullmatch(text.strip()))


def looks_like_sentence(text: str) -> bool:
    return any(token in text for token in ["。", "；", "！", "？", "："])


def digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    digit_count = sum(1 for char in text if char.isdigit())
    return digit_count / max(1, len(text))


def contains_table_like_units(text: str) -> bool:
    return any(token in text for token in ["万元", "亿元", "%", "比例", "增长率", "金额"])


def infer_heading_level(text: str) -> int:
    candidate = text.strip()
    if not candidate or is_marker_only(candidate):
        return 0
    if len(candidate) > 80 and not CHAPTER_RE.match(candidate) and not SECTION_RE.match(candidate):
        return 0
    if CHAPTER_RE.match(candidate):
        return 1
    if SECTION_RE.match(candidate):
        return 2
    if CN_ENUM_RE.match(candidate):
        return 2
    if CN_PAREN_ENUM_RE.match(candidate):
        return 3
    numeric_match = NUMERIC_HEADING_RE.match(candidate)
    if numeric_match:
        prefix = numeric_match.group(1).strip()
        suffix = numeric_match.group(2).strip()
        if len(candidate) <= 40 and len(suffix) <= 28 and not looks_like_sentence(candidate):
            return min(4, prefix.count(".") + 2)
        return 0
    if (
        len(candidate) <= 18
        and not looks_like_sentence(candidate)
        and digit_ratio(candidate) <= 0.25
        and not contains_table_like_units(candidate)
    ):
        return 2
    return 0


def to_heading_chain(path: tuple[HeadingRef, ...]) -> list[dict[str, Any]]:
    return [{"level": ref.level, "title": ref.title} for ref in path]


def heading_key(path: tuple[HeadingRef, ...]) -> str:
    if not path:
        return ""
    return " > ".join(ref.title for ref in path)


def common_heading_prefix(paths: list[tuple[HeadingRef, ...]]) -> tuple[HeadingRef, ...]:
    valid_paths = [path for path in paths if path]
    if not valid_paths:
        return ()
    prefix: list[HeadingRef] = []
    for index in range(min(len(path) for path in valid_paths)):
        first = valid_paths[0][index]
        if all(path[index].title == first.title and path[index].level == first.level for path in valid_paths[1:]):
            prefix.append(first)
        else:
            break
    return tuple(prefix)


def dominant_heading_path(blocks: list[FlowBlock]) -> tuple[HeadingRef, ...]:
    weights: dict[str, int] = {}
    path_map: dict[str, tuple[HeadingRef, ...]] = {}
    for block in blocks:
        if not block.heading_path:
            continue
        key = heading_key(block.heading_path)
        weights[key] = weights.get(key, 0) + max(1, block.char_count)
        path_map[key] = block.heading_path
    if not weights:
        return ()
    best_key = max(weights.items(), key=lambda item: item[1])[0]
    return path_map[best_key]


def build_flow_blocks(pages: list[dict[str, Any]]) -> tuple[list[FlowBlock], dict[str, Any]]:
    blocks: list[FlowBlock] = []
    heading_stack: list[HeadingRef] = []
    page_block_counts: dict[int, int] = {}
    heading_count = 0

    for page in pages:
        page_index = int(page["page_index"])
        page_blocks = split_page_into_blocks(str(page.get("page_text_flow") or ""))
        page_block_counts[page_index] = len(page_blocks)
        for block_index, block_text in enumerate(page_blocks, start=1):
            level = infer_heading_level(block_text)
            block_type = "marker" if is_marker_only(block_text) else ("heading" if level > 0 else "content")
            if level > 0:
                while heading_stack and heading_stack[-1].level >= level:
                    heading_stack.pop()
                heading_stack.append(HeadingRef(level=level, title=block_text))
                heading_count += 1
                current_path = tuple(heading_stack)
                heading_title = block_text
            else:
                current_path = tuple(heading_stack)
                heading_title = current_path[-1].title if current_path else ""
            blocks.append(
                FlowBlock(
                    block_id=f"p{page_index}_b{block_index}",
                    page_index=page_index,
                    text=block_text,
                    block_type=block_type,
                    char_count=len(block_text),
                    heading_level=level,
                    heading_title=heading_title,
                    heading_path=current_path,
                )
            )

    debug = {
        "block_count": len(blocks),
        "heading_count": heading_count,
        "page_block_counts": page_block_counts,
    }
    return blocks, debug


def join_block_texts(blocks: list[FlowBlock]) -> str:
    return "\n\n".join(block.text for block in blocks if block.text.strip()).strip()


def select_chunk_end(blocks: list[FlowBlock], start_index: int, chunk_size: int) -> int:
    min_target = max(180, int(chunk_size * 0.55))
    soft_limit = max(chunk_size + 1, int(chunk_size * 1.35))
    hard_limit = max(soft_limit + 1, int(chunk_size * 1.55))

    char_count = 0
    cursor = start_index
    active_path = blocks[start_index].heading_path

    while cursor < len(blocks):
        block = blocks[cursor]
        addition = block.char_count if char_count == 0 else block.char_count + 2
        projected = char_count + addition
        same_section = bool(active_path) and heading_key(block.heading_path) == heading_key(active_path)

        if cursor > start_index and block.block_type == "heading" and char_count >= min_target:
            break
        if projected <= chunk_size:
            char_count = projected
            if block.heading_path:
                active_path = block.heading_path
            cursor += 1
            continue
        if same_section and projected <= soft_limit:
            char_count = projected
            cursor += 1
            continue
        if char_count < min_target and projected <= hard_limit:
            char_count = projected
            cursor += 1
            continue
        break

    return max(start_index + 1, cursor)


def compute_next_start(blocks: list[FlowBlock], start_index: int, end_index: int, chunk_overlap: int) -> int:
    if end_index <= start_index + 1:
        return end_index

    overlap_chars = 0
    cursor = end_index - 1
    while cursor > start_index and overlap_chars < chunk_overlap:
        overlap_chars += blocks[cursor].char_count + 2
        cursor -= 1

    next_start = max(start_index + 1, cursor + 1)
    if next_start >= end_index:
        next_start = end_index
    return next_start


def collect_markers(text: str) -> list[str]:
    return [match.group(0) for match in MARKER_PATTERN.finditer(text)]


def build_chunk_record(
    *,
    chunk_index: int,
    selected_blocks: list[FlowBlock],
    chunk_size_target: int,
    chunk_overlap: int,
    is_final_chunk: bool,
) -> dict[str, Any]:
    normalized_text = join_block_texts(selected_blocks)
    marker_list = collect_markers(normalized_text)
    source_pages = sorted({block.page_index for block in selected_blocks})
    non_marker_blocks = [block for block in selected_blocks if block.block_type != "marker"]
    path_candidates = [block.heading_path for block in non_marker_blocks if block.heading_path]
    primary_path = common_heading_prefix(path_candidates) if path_candidates else ()
    dominant_path = dominant_heading_path(non_marker_blocks)
    heading_blocks = [block for block in selected_blocks if block.block_type == "heading"]
    block_type_counts: dict[str, int] = {}
    for block in selected_blocks:
        block_type_counts[block.block_type] = block_type_counts.get(block.block_type, 0) + 1

    return {
        "chunk_id": f"chunk_{chunk_index:05d}",
        "chunk_index": chunk_index,
        "text": normalized_text,
        "char_count": len(normalized_text),
        "source_pages": source_pages,
        "source_page_count": len(source_pages),
        "marker_count": len(marker_list),
        "markers": marker_list,
        "chunk_size_target": chunk_size_target,
        "chunk_overlap": chunk_overlap,
        "is_final_chunk": is_final_chunk,
        "heading_chain": to_heading_chain(dominant_path),
        "heading_key": heading_key(dominant_path),
        "parent_heading_chain": to_heading_chain(dominant_path[:-1]) if dominant_path else [],
        "parent_heading_key": heading_key(dominant_path[:-1]) if dominant_path else "",
        "primary_heading_chain": to_heading_chain(primary_path),
        "primary_heading_key": heading_key(primary_path),
        "root_heading": dominant_path[0].title if dominant_path else "",
        "leaf_heading": dominant_path[-1].title if dominant_path else "",
        "heading_block_titles": [block.text for block in heading_blocks],
        "heading_levels_in_chunk": sorted({block.heading_level for block in heading_blocks if block.heading_level > 0}),
        "block_ids": [block.block_id for block in selected_blocks],
        "block_count": len(selected_blocks),
        "block_type_counts": block_type_counts,
        "contains_heading": bool(heading_blocks),
        "same_heading_prev_chunk_id": "",
        "same_heading_next_chunk_id": "",
    }


def merge_chunk_payload(base: dict[str, Any], tail: dict[str, Any]) -> dict[str, Any]:
    merged_text = "\n\n".join([str(base.get("text") or "").strip(), str(tail.get("text") or "").strip()]).strip()
    merged_markers = list(dict.fromkeys([*list(base.get("markers") or []), *list(tail.get("markers") or [])]))
    merged_pages = sorted({*list(base.get("source_pages") or []), *list(tail.get("source_pages") or [])})
    merged_heading_titles = list(dict.fromkeys([*list(base.get("heading_block_titles") or []), *list(tail.get("heading_block_titles") or [])]))
    merged_block_ids = [*list(base.get("block_ids") or []), *list(tail.get("block_ids") or [])]
    block_type_counts = dict(base.get("block_type_counts") or {})
    for key, value in dict(tail.get("block_type_counts") or {}).items():
        block_type_counts[key] = int(block_type_counts.get(key, 0)) + int(value or 0)

    merged = dict(base)
    merged["text"] = merged_text
    merged["char_count"] = len(merged_text)
    merged["source_pages"] = merged_pages
    merged["source_page_count"] = len(merged_pages)
    merged["markers"] = merged_markers
    merged["marker_count"] = len(merged_markers)
    merged["heading_block_titles"] = merged_heading_titles
    merged["heading_levels_in_chunk"] = sorted(
        {int(level) for level in [*list(base.get("heading_levels_in_chunk") or []), *list(tail.get("heading_levels_in_chunk") or [])] if int(level) > 0}
    )
    merged["block_ids"] = merged_block_ids
    merged["block_count"] = len(merged_block_ids)
    merged["block_type_counts"] = block_type_counts
    merged["contains_heading"] = bool(merged_heading_titles)
    merged["is_final_chunk"] = bool(tail.get("is_final_chunk"))
    return merged


def squash_tiny_chunks(chunks: list[dict[str, Any]], min_chars: int) -> list[dict[str, Any]]:
    if not chunks:
        return []

    squashed: list[dict[str, Any]] = []
    for chunk in chunks:
        char_count = int(chunk.get("char_count") or 0)
        if char_count < min_chars and squashed:
            squashed[-1] = merge_chunk_payload(squashed[-1], chunk)
            continue
        squashed.append(dict(chunk))

    for index, chunk in enumerate(squashed, start=1):
        chunk["chunk_index"] = index
        chunk["chunk_id"] = f"chunk_{index:05d}"
        chunk["same_heading_prev_chunk_id"] = ""
        chunk["same_heading_next_chunk_id"] = ""
        chunk["is_final_chunk"] = index == len(squashed)
    return squashed


def chunk_blocks(
    blocks: list[FlowBlock],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    start_index = 0
    chunk_index = 1

    while start_index < len(blocks):
        end_index = select_chunk_end(blocks, start_index, chunk_size)
        selected_blocks = blocks[start_index:end_index]
        is_final_chunk = end_index >= len(blocks)
        chunk = build_chunk_record(
            chunk_index=chunk_index,
            selected_blocks=selected_blocks,
            chunk_size_target=chunk_size,
            chunk_overlap=chunk_overlap,
            is_final_chunk=is_final_chunk,
        )
        chunks.append(chunk)
        log(
            f"emit_chunk chunk={chunk_index} pages={chunk['source_pages']} "
            f"chars={chunk['char_count']} blocks={chunk['block_count']} "
            f"heading={chunk['leaf_heading'] or '-'}"
        )
        chunk_index += 1

        next_start = compute_next_start(blocks, start_index, end_index, chunk_overlap)
        if next_start <= start_index:
            next_start = end_index
        start_index = next_start

    return squash_tiny_chunks(chunks, min_chars=max(80, int(chunk_size * 0.18)))


def attach_heading_links(chunks: list[dict[str, Any]]) -> None:
    bucket: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        key = str(chunk.get("heading_key") or "")
        if not key:
            continue
        bucket.setdefault(key, []).append(chunk)

    for items in bucket.values():
        items.sort(key=lambda item: int(item.get("chunk_index") or 0))
        for index, item in enumerate(items):
            item["same_heading_prev_chunk_id"] = str(items[index - 1]["chunk_id"]) if index > 0 else ""
            item["same_heading_next_chunk_id"] = (
                str(items[index + 1]["chunk_id"]) if index + 1 < len(items) else ""
            )


def build_heading_outline(blocks: list[FlowBlock], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    node_map: dict[str, dict[str, Any]] = {}
    for block in blocks:
        for depth in range(1, len(block.heading_path) + 1):
            path = block.heading_path[:depth]
            key = heading_key(path)
            parent_path = path[:-1]
            parent_key = heading_key(parent_path)
            node = node_map.setdefault(
                key,
                {
                    "heading_key": key,
                    "title": path[-1].title,
                    "level": path[-1].level,
                    "parent_heading_key": parent_key,
                    "heading_chain": to_heading_chain(path),
                    "page_indexes": [],
                    "block_ids": [],
                    "chunk_ids": [],
                    "child_heading_keys": [],
                },
            )
            if block.page_index not in node["page_indexes"]:
                node["page_indexes"].append(block.page_index)
            node["block_ids"].append(block.block_id)
            if parent_key:
                parent_node = node_map.setdefault(
                    parent_key,
                    {
                        "heading_key": parent_key,
                        "title": parent_path[-1].title,
                        "level": parent_path[-1].level,
                        "parent_heading_key": heading_key(parent_path[:-1]),
                        "heading_chain": to_heading_chain(parent_path),
                        "page_indexes": [],
                        "block_ids": [],
                        "chunk_ids": [],
                        "child_heading_keys": [],
                    },
                )
                if key not in parent_node["child_heading_keys"]:
                    parent_node["child_heading_keys"].append(key)

    for chunk in chunks:
        for key_name in ["heading_key", "primary_heading_key", "parent_heading_key"]:
            key = str(chunk.get(key_name) or "")
            if key and key in node_map and chunk["chunk_id"] not in node_map[key]["chunk_ids"]:
                node_map[key]["chunk_ids"].append(chunk["chunk_id"])

    outline = list(node_map.values())
    outline.sort(key=lambda item: (int(item.get("level") or 0), str(item.get("heading_key") or "")))
    return outline


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_summary(
    *,
    source_path: Path,
    output_dir: Path,
    pages: list[dict[str, Any]],
    blocks: list[FlowBlock],
    chunks: list[dict[str, Any]],
    heading_outline: list[dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
    seed_pages: int,
    append_threshold: int,
    flow_debug: dict[str, Any],
) -> dict[str, Any]:
    avg_chunk_chars = round(sum(chunk["char_count"] for chunk in chunks) / len(chunks), 2) if chunks else 0
    max_chunk_chars = max((chunk["char_count"] for chunk in chunks), default=0)
    min_chunk_chars = min((chunk["char_count"] for chunk in chunks), default=0)
    return {
        "source_path": str(source_path),
        "output_dir": str(output_dir),
        "page_count": len(pages),
        "block_count": len(blocks),
        "chunk_count": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "seed_pages": seed_pages,
        "append_threshold": append_threshold,
        "average_chunk_chars": avg_chunk_chars,
        "max_chunk_chars": max_chunk_chars,
        "min_chunk_chars": min_chunk_chars,
        "chunks_with_markers": sum(1 for chunk in chunks if chunk["marker_count"] > 0),
        "chunks_with_headings": sum(1 for chunk in chunks if chunk.get("heading_key")),
        "heading_node_count": len(heading_outline),
        "chunking_mode": "heading_aware_block_chunking",
        "flow_debug": flow_debug,
    }


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve() if args.artifact_dir else (PROJECT_ROOT / "artifacts" / DEFAULT_INPUT_DIR_NAME)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (artifact_dir / DEFAULT_OUTPUT_DIR_NAME)

    pages_payload, source_path = load_page_text_flow(artifact_dir, args.source_mode)
    pages = [normalize_page_entry(item) for item in pages_payload if isinstance(item, dict)]
    pages = [item for item in pages if item["page_index"] > 0 and item["page_text_flow"]]

    log(f"artifact_dir={artifact_dir}")
    log(f"source_path={source_path}")
    log(
        f"config chunk_size={args.chunk_size} overlap={args.chunk_overlap} "
        f"seed_pages={args.seed_pages} append_threshold={args.append_threshold}"
    )
    log(f"page_count={len(pages)}")

    blocks, flow_debug = build_flow_blocks(pages)
    log(f"block_count={len(blocks)} heading_count={flow_debug['heading_count']}")

    chunks = chunk_blocks(
        blocks=blocks,
        chunk_size=max(1, args.chunk_size),
        chunk_overlap=max(0, args.chunk_overlap),
    )
    attach_heading_links(chunks)
    heading_outline = build_heading_outline(blocks, chunks)
    summary = build_summary(
        source_path=source_path,
        output_dir=output_dir,
        pages=pages,
        blocks=blocks,
        chunks=chunks,
        heading_outline=heading_outline,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        seed_pages=args.seed_pages,
        append_threshold=args.append_threshold,
        flow_debug=flow_debug,
    )

    chunks_json_path = output_dir / "text_chunks.json"
    chunks_jsonl_path = output_dir / "text_chunks.jsonl"
    manifest_path = output_dir / "chunk_manifest.json"
    preview_path = output_dir / "chunk_preview.json"
    outline_path = output_dir / "heading_outline.json"
    block_preview_path = output_dir / "flow_block_preview.json"

    write_json(chunks_json_path, chunks)
    write_jsonl(chunks_jsonl_path, chunks)
    write_json(manifest_path, summary)
    write_json(preview_path, chunks[:20])
    write_json(outline_path, heading_outline)
    write_json(
        block_preview_path,
        [
            {
                "block_id": block.block_id,
                "page_index": block.page_index,
                "block_type": block.block_type,
                "heading_level": block.heading_level,
                "heading_title": block.heading_title,
                "heading_chain": to_heading_chain(block.heading_path),
                "text_preview": block.text[:240],
            }
            for block in blocks[:120]
        ],
    )

    log(f"chunks_written={len(chunks)}")
    log(f"heading_nodes={len(heading_outline)}")
    log(f"manifest_path={manifest_path}")
    log(f"chunk_jsonl_path={chunks_jsonl_path}")


if __name__ == "__main__":
    main()
