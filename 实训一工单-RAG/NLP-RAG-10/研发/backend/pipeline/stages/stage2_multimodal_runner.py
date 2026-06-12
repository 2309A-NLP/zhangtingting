from __future__ import annotations

# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
import argparse
import base64
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings


SCHEMA_VERSION = "1.0"
RESULT_FILE_NAME = "vlm_results.jsonl"
RAW_DIR_NAME = "vlm_raw_responses"
TASK_TYPES = {
    "single_table_understanding",
    "cross_page_table_merge",
    "figure_or_image_understanding",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[vlm-runner] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute stage2 multimodal tasks and write vlm_results.jsonl")
    parser.add_argument("--artifact-dir", type=str, default="", help="Artifact dir containing multimodal_tasks.jsonl")
    parser.add_argument("--task-id", type=str, default="", help="Run only one task_id")
    parser.add_argument("--task-type", type=str, default="", help="Filter by task_type")
    parser.add_argument("--page-index", type=int, default=0, help="Filter by page index")
    parser.add_argument("--limit", type=int, default=0, help="Run at most N pending tasks")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing result lines for same final_object_id")
    parser.add_argument("--dry-run", action="store_true", help="Do not call API, only validate and print pending tasks")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def digest_prompt(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{payload}"


def ensure_vlm_enabled() -> None:
    if not settings.pdf_vlm_api_url or not settings.pdf_vlm_api_key or not settings.pdf_vlm_model_name:
        raise RuntimeError("PDF VLM is not fully configured. Check PDF_VLM_API_URL / PDF_VLM_API_KEY / PDF_VLM_MODEL_NAME.")


def build_response_schema_text(task_type: str) -> str:
    if task_type in {"single_table_understanding", "cross_page_table_merge"}:
        return (
            "只输出一个 JSON 对象，字段必须固定为："
            '{"content":"一句中文结论","structured_content":{"title":"表格标题或空字符串","merge_decision":"single_page|same_table|not_same_table",'
            '"headers":["列1","列2"],"rows":[["值1","值2"]],"notes":["可选说明"]}}。'
            "不要输出 markdown，不要输出代码块，不要省略字段。"
        )
    return (
        "只输出一个 JSON 对象，字段必须固定为："
        '{"content":"一句中文结论","structured_content":{"summary":"图片/图表摘要","labels":["标签1"],'
        '"numbers":["数值1"],"relations":["关系1"],"notes":["可选说明"]}}。'
        "不要输出 markdown，不要输出代码块，不要省略字段。"
    )


def build_visual_response_schema_text() -> str:
    return (
        'Only output one JSON object with fixed fields: '
        '{"content":"一句中文结论","structured_content":{"summary":"页面级摘要","is_same_visual":"yes|no|partial",'
        '"groups":[{"group_name":"组1","region_ids":["r1"],"is_same_object":true,"summary":"该组摘要"}],'
        '"labels":["标签1"],"numbers":["数字1"],"relations":["关系1"],"notes":["可选说明"]}}. '
        "Do not output markdown, do not output code fences, and do not omit fields."
    )


def build_task_prompt(task: dict[str, Any]) -> str:
    task_type = str(task.get("task_type") or "")
    base_prompt = str(task.get("prompt") or "").strip()
    if task_type == "single_table_understanding":
        return "\n\n".join(
            [
                base_prompt,
                build_response_schema_text(task_type),
                "以下是机器提取结果，请以图片为准进行校正：",
                json.dumps(
                    {
                        "source_region_id": task.get("source_region_id"),
                        "page_index": task.get("page_index"),
                        "backend": task.get("backend"),
                        "strategy": task.get("strategy"),
                        "quality_score": task.get("quality_score"),
                        "complexity_score": task.get("complexity_score"),
                        "complexity_reasons": task.get("complexity_reasons"),
                        "headers": task.get("headers"),
                        "rows": task.get("rows"),
                    },
                    ensure_ascii=False,
                ),
            ]
        )
    if task_type == "cross_page_table_merge":
        fragments = []
        for fragment in task.get("fragments") or []:
            fragments.append(
                {
                    "fragment_object_id": fragment.get("fragment_object_id"),
                    "source_region_id": fragment.get("source_region_id"),
                    "page_index": fragment.get("page_index"),
                    "backend": fragment.get("backend"),
                    "strategy": fragment.get("strategy"),
                    "quality_score": fragment.get("quality_score"),
                    "complexity_score": fragment.get("complexity_score"),
                    "headers": fragment.get("headers"),
                    "rows": fragment.get("rows"),
                }
            )
        return "\n\n".join(
            [
                base_prompt,
                build_response_schema_text(task_type),
                "以下是跨页关系与每页机器提取结果，请结合图片合并：",
                json.dumps(
                    {
                        "chain_id": task.get("chain_id"),
                        "fragment_region_ids": task.get("fragment_region_ids"),
                        "fragment_pages": task.get("fragment_pages"),
                        "relation": task.get("relation"),
                        "fragments": fragments,
                    },
                    ensure_ascii=False,
                ),
            ]
        )
    return "\n\n".join(
        [
            base_prompt,
            build_visual_response_schema_text(),
            "请严格基于图片内容作答。",
        ]
    )


def build_visual_response_schema_text() -> str:
    return (
        'Only output one JSON object with fixed fields: '
        '{"content":"2-4句中文结论","structured_content":{"summary":"页面级摘要","visual_type":"flowchart|data_viz|org_chart|mixed_visual|image|unknown",'
        '"is_same_visual":"yes|no|partial","groups":[{"group_name":"组1","region_ids":["r1"],"is_same_object":true,"summary":"该组摘要"}],'
        '"detailed_description":"对图表整体内容的详细描述","flow_description":{"start":"起点","steps":["步骤1","步骤2"],"decision_points":["判断点1"],"end":"终点"},'
        '"chart_analysis":{"chart_type":"line|bar|pie|combo|scatter|area|unknown","x_axis":"横轴含义","y_axis":"纵轴含义","series":["系列1"],'
        '"trend_summary":"总体趋势","max_points":["最大值点"],"min_points":["最小值点"],"turning_points":["拐点"],"comparison_points":["对比结论"]},'
        '"labels":["标签1"],"numbers":["数字1"],"relations":["关系1"],"key_observations":["观察1"],"notes":["可选说明"]}}. '
        "Do not output markdown, do not output code fences, and do not omit fields."
    )


def build_table_response_schema_text() -> str:
    return (
        'Only output one JSON object with fixed fields: '
        '{"content":"2-4句中文结论","structured_content":{"title":"表格标题或空字符串","merge_decision":"single_page|same_table|not_same_table",'
        '"summary":"表格整体摘要","key_columns":["列1"],"key_findings":["发现1"],"headers":["列1","列2"],'
        '"rows":[["值1","值2"]],"notes":["补充说明"],"quantitative_highlights":["关键数值结论"]}}. '
        "Do not output markdown, do not output code fences, and do not omit fields."
    )


def build_visual_specific_guidance(task: dict[str, Any]) -> str:
    regions = task.get("regions") or []
    sub_types = sorted({str(item.get("sub_type") or "") for item in regions if str(item.get("sub_type") or "")})
    region_types = sorted({str(item.get("region_type") or "") for item in regions if str(item.get("region_type") or "")})
    hints: list[str] = [
        f"Region types: {', '.join(region_types) if region_types else 'unknown'}",
        f"Detected sub types: {', '.join(sub_types) if sub_types else 'unknown'}",
        "If the visual is a flowchart, org chart, or process diagram, describe the overall structure, the main nodes, the direction of the flow, and the step-by-step path in Chinese.",
        "If the visual is a chart with numeric information, analyze the trend direction, highest point, lowest point, turning points, series comparison, and any explicit values visible in the figure.",
        "If the page contains multiple cropped pieces that belong to one chart, merge them conceptually before answering.",
    ]
    return "\n".join(hints)


def build_task_prompt(task: dict[str, Any]) -> str:
    task_type = str(task.get("task_type") or "")
    base_prompt = str(task.get("prompt") or "").strip()
    if task_type == "single_table_understanding":
        return "\n\n".join(
            [
                base_prompt,
                build_table_response_schema_text(),
                "Use the image as the primary truth source and the machine extracted rows only as auxiliary reference.",
                json.dumps(
                    {
                        "source_region_id": task.get("source_region_id"),
                        "page_index": task.get("page_index"),
                        "backend": task.get("backend"),
                        "strategy": task.get("strategy"),
                        "quality_score": task.get("quality_score"),
                        "complexity_score": task.get("complexity_score"),
                        "complexity_reasons": task.get("complexity_reasons"),
                        "headers": task.get("headers"),
                        "rows": task.get("rows"),
                    },
                    ensure_ascii=False,
                ),
            ]
        )
    if task_type == "cross_page_table_merge":
        fragments = []
        for fragment in task.get("fragments") or []:
            fragments.append(
                {
                    "fragment_object_id": fragment.get("fragment_object_id"),
                    "source_region_id": fragment.get("source_region_id"),
                    "page_index": fragment.get("page_index"),
                    "backend": fragment.get("backend"),
                    "strategy": fragment.get("strategy"),
                    "quality_score": fragment.get("quality_score"),
                    "complexity_score": fragment.get("complexity_score"),
                    "headers": fragment.get("headers"),
                    "rows": fragment.get("rows"),
                }
            )
        return "\n\n".join(
            [
                base_prompt,
                build_table_response_schema_text(),
                "Judge whether these fragments belong to the same cross-page table, merge them when appropriate, and summarize the important quantitative conclusions in Chinese.",
                json.dumps(
                    {
                        "chain_id": task.get("chain_id"),
                        "fragment_region_ids": task.get("fragment_region_ids"),
                        "fragment_pages": task.get("fragment_pages"),
                        "relation": task.get("relation"),
                        "fragments": fragments,
                    },
                    ensure_ascii=False,
                ),
            ]
        )
    return "\n\n".join(
        [
            base_prompt,
            build_visual_response_schema_text(),
            build_visual_specific_guidance(task),
            "Answer strictly based on the visual evidence and produce a richer Chinese explanation instead of a short label list.",
        ]
    )


def image_dimensions(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except Exception:
        return (0, 0)
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return (0, 0)


def is_valid_visual_region(region: dict[str, Any]) -> bool:
    crop_path = Path(str(region.get("crop_path") or ""))
    if not crop_path.exists():
        return False
    width, height = image_dimensions(crop_path)
    if width <= 0 or height <= 0:
        return False
    bbox = region.get("bbox") or [0, 0, 0, 0]
    try:
        bbox_width = float(bbox[2]) - float(bbox[0])
        bbox_height = float(bbox[3]) - float(bbox[1])
    except Exception:
        bbox_width = 0.0
        bbox_height = 0.0
    sub_type = str(region.get("sub_type") or "")
    region_type = str(region.get("region_type") or "")
    if height < 8 or bbox_height < 3.0:
        return False
    if region_type == "image" and sub_type == "inline_image" and (height < 20 or bbox_height < 8.0):
        return False
    if width < 24 or bbox_width < 8.0:
        return False
    return True


def filter_visual_regions(task: dict[str, Any]) -> list[dict[str, Any]]:
    return [region for region in (task.get("regions") or []) if is_valid_visual_region(region)]


def build_messages(task: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = build_task_prompt(task)
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    task_type = str(task.get("task_type") or "")
    if task_type == "cross_page_table_merge":
        for fragment in task.get("fragments") or []:
            crop_path = Path(str(fragment.get("crop_path") or ""))
            if crop_path.exists():
                content.append(
                    {
                        "type": "text",
                        "text": f"片段页码={fragment.get('page_index')} source_region_id={fragment.get('source_region_id')}",
                    }
                )
                content.append({"type": "image_url", "image_url": {"url": image_to_data_url(crop_path)}})
    elif task_type == "figure_or_image_understanding" and task.get("regions"):
        for region in filter_visual_regions(task):
            crop_path = Path(str(region.get("crop_path") or ""))
            if crop_path.exists():
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"region_id={region.get('region_id')} "
                            f"region_type={region.get('region_type')} "
                            f"sub_type={region.get('sub_type') or ''} "
                            f"bbox={region.get('bbox')}"
                        ),
                    }
                )
                content.append({"type": "image_url", "image_url": {"url": image_to_data_url(crop_path)}})
    else:
        crop_path = Path(str(task.get("crop_path") or ""))
        if crop_path.exists():
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(crop_path)}})
    return [
        {
            "role": "system",
            "content": "你是一个严格的 PDF 多模态结构化抽取助手。只能基于图片和给定结构化上下文输出 JSON，不得编造。",
        },
        {
            "role": "user",
            "content": content,
        },
    ]


def request_vlm(task: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ensure_vlm_enabled()
    messages = build_messages(task)
    payload = {
        "model": settings.pdf_vlm_model_name,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 2400,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.pdf_vlm_api_key}",
    }
    response = requests.post(
        settings.pdf_vlm_api_url,
        headers=headers,
        data=json.dumps(payload),
        timeout=settings.pdf_vlm_request_timeout,
    )
    response.raise_for_status()
    data = response.json()
    message = data["choices"][0]["message"]["content"] if "choices" in data else "{}"
    return data, {"message_content": message, "prompt_digest": digest_prompt(json.dumps(messages, ensure_ascii=False))}


def strip_code_fence(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```json"):
        value = value[len("```json") :]
    if value.startswith("```"):
        value = value[len("```") :]
    if value.endswith("```"):
        value = value[: -3]
    return value.strip()


def sanitize_watermark_text(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    if "水印" not in value:
        return value

    lines = [line.strip() for line in re.split(r"\n+", value) if line.strip()]
    cleaned_lines: list[str] = []
    sentence_splitter = re.compile(r"(?<=[。！？；;.!?])")
    for line in lines:
        if "水印" not in line:
            cleaned_lines.append(line)
            continue
        sentences = [segment.strip() for segment in sentence_splitter.split(line) if segment.strip()]
        kept = [sentence for sentence in sentences if "水印" not in sentence]
        cleaned_line = "".join(kept).strip()
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    return "\n".join(cleaned_lines).strip()


def sanitize_watermark_list(values: list[Any]) -> list[str]:
    cleaned: list[str] = []
    for item in values:
        text = sanitize_watermark_text(str(item or "").strip())
        if text:
            cleaned.append(text)
    return cleaned


def normalize_table_structured(payload: dict[str, Any]) -> dict[str, Any]:
    structured = payload.get("structured_content") or {}
    if not isinstance(structured, dict):
        structured = {}
    headers = structured.get("headers") if isinstance(structured.get("headers"), list) else []
    rows = structured.get("rows") if isinstance(structured.get("rows"), list) else []
    notes = structured.get("notes") if isinstance(structured.get("notes"), list) else []
    return {
        "title": str(structured.get("title") or "").strip(),
        "merge_decision": str(structured.get("merge_decision") or "single_page").strip(),
        "headers": [str(item) for item in headers],
        "rows": [[str(cell) for cell in row] for row in rows if isinstance(row, list)],
        "notes": [str(item) for item in notes],
    }


def normalize_figure_structured(payload: dict[str, Any]) -> dict[str, Any]:
    structured = payload.get("structured_content") or {}
    if not isinstance(structured, dict):
        structured = {}
    groups = []
    for item in (structured.get("groups") or []):
        if not isinstance(item, dict):
            continue
        groups.append(
            {
                "group_name": sanitize_watermark_text(str(item.get("group_name") or "").strip()),
                "region_ids": [str(region_id) for region_id in (item.get("region_ids") or []) if str(region_id).strip()],
                "is_same_object": bool(item.get("is_same_object")),
                "summary": sanitize_watermark_text(str(item.get("summary") or "").strip()),
            }
        )
    return {
        "summary": str(structured.get("summary") or "").strip(),
        "is_same_visual": str(structured.get("is_same_visual") or "").strip(),
        "groups": groups,
        "labels": [str(item) for item in (structured.get("labels") or []) if str(item).strip()],
        "numbers": [str(item) for item in (structured.get("numbers") or []) if str(item).strip()],
        "relations": [str(item) for item in (structured.get("relations") or []) if str(item).strip()],
        "notes": [str(item) for item in (structured.get("notes") or []) if str(item).strip()],
    }


def normalize_table_structured(payload: dict[str, Any]) -> dict[str, Any]:
    structured = payload.get("structured_content") or {}
    if not isinstance(structured, dict):
        structured = {}
    headers = structured.get("headers") if isinstance(structured.get("headers"), list) else []
    rows = structured.get("rows") if isinstance(structured.get("rows"), list) else []
    notes = structured.get("notes") if isinstance(structured.get("notes"), list) else []
    return {
        "title": str(structured.get("title") or "").strip(),
        "merge_decision": str(structured.get("merge_decision") or "single_page").strip(),
        "summary": str(structured.get("summary") or "").strip(),
        "key_columns": [str(item) for item in (structured.get("key_columns") or []) if str(item).strip()],
        "key_findings": [str(item) for item in (structured.get("key_findings") or []) if str(item).strip()],
        "headers": [str(item) for item in headers],
        "rows": [[str(cell) for cell in row] for row in rows if isinstance(row, list)],
        "notes": [str(item) for item in notes],
        "quantitative_highlights": [
            str(item) for item in (structured.get("quantitative_highlights") or []) if str(item).strip()
        ],
    }


def normalize_figure_structured(payload: dict[str, Any]) -> dict[str, Any]:
    structured = payload.get("structured_content") or {}
    if not isinstance(structured, dict):
        structured = {}
    groups = []
    for item in (structured.get("groups") or []):
        if not isinstance(item, dict):
            continue
        groups.append(
            {
                "group_name": str(item.get("group_name") or "").strip(),
                "region_ids": [str(region_id) for region_id in (item.get("region_ids") or []) if str(region_id).strip()],
                "is_same_object": bool(item.get("is_same_object")),
                "summary": str(item.get("summary") or "").strip(),
            }
        )
    flow_description = structured.get("flow_description") or {}
    if not isinstance(flow_description, dict):
        flow_description = {}
    chart_analysis = structured.get("chart_analysis") or {}
    if not isinstance(chart_analysis, dict):
        chart_analysis = {}
    return {
        "summary": sanitize_watermark_text(str(structured.get("summary") or "").strip()),
        "visual_type": str(structured.get("visual_type") or "").strip(),
        "is_same_visual": str(structured.get("is_same_visual") or "").strip(),
        "groups": groups,
        "detailed_description": sanitize_watermark_text(str(structured.get("detailed_description") or "").strip()),
        "flow_description": {
            "start": sanitize_watermark_text(str(flow_description.get("start") or "").strip()),
            "steps": sanitize_watermark_list(list(flow_description.get("steps") or [])),
            "decision_points": sanitize_watermark_list(list(flow_description.get("decision_points") or [])),
            "end": sanitize_watermark_text(str(flow_description.get("end") or "").strip()),
        },
        "chart_analysis": {
            "chart_type": str(chart_analysis.get("chart_type") or "").strip(),
            "x_axis": sanitize_watermark_text(str(chart_analysis.get("x_axis") or "").strip()),
            "y_axis": sanitize_watermark_text(str(chart_analysis.get("y_axis") or "").strip()),
            "series": sanitize_watermark_list(list(chart_analysis.get("series") or [])),
            "trend_summary": sanitize_watermark_text(str(chart_analysis.get("trend_summary") or "").strip()),
            "max_points": sanitize_watermark_list(list(chart_analysis.get("max_points") or [])),
            "min_points": sanitize_watermark_list(list(chart_analysis.get("min_points") or [])),
            "turning_points": sanitize_watermark_list(list(chart_analysis.get("turning_points") or [])),
            "comparison_points": sanitize_watermark_list(list(chart_analysis.get("comparison_points") or [])),
        },
        "labels": sanitize_watermark_list(list(structured.get("labels") or [])),
        "numbers": sanitize_watermark_list(list(structured.get("numbers") or [])),
        "relations": sanitize_watermark_list(list(structured.get("relations") or [])),
        "key_observations": sanitize_watermark_list(list(structured.get("key_observations") or [])),
        "notes": sanitize_watermark_list(list(structured.get("notes") or [])),
    }


def parse_vlm_payload(task: dict[str, Any], content: str) -> tuple[str, dict[str, Any]]:
    text = strip_code_fence(content)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("VLM response must be a JSON object")
    answer = str(payload.get("content") or "").strip()
    task_type = str(task.get("task_type") or "")
    if task_type in {"single_table_understanding", "cross_page_table_merge"}:
        structured = normalize_table_structured(payload)
    else:
        structured = normalize_figure_structured(payload)
    return answer, structured


def build_result_record(
    task: dict[str, Any],
    *,
    status: str,
    content: str = "",
    structured_content: dict[str, Any] | None = None,
    model: str = "",
    provider: str = "",
    raw_response_path: str = "",
    error: str | None = None,
    prompt_digest_value: str = "",
) -> dict[str, Any]:
    source_region_ids = list(task.get("fragment_region_ids") or [])
    if not source_region_ids and task.get("source_region_id"):
        source_region_ids = [str(task.get("source_region_id"))]
    if not source_region_ids and task.get("source_region_ids"):
        source_region_ids = [str(item) for item in (task.get("source_region_ids") or []) if str(item)]
    source_pages = list(task.get("fragment_pages") or [])
    if not source_pages and task.get("page_index"):
        source_pages = [int(task.get("page_index"))]
    references = []
    if task.get("task_type") == "cross_page_table_merge":
        for fragment in task.get("fragments") or []:
            references.append(
                {
                    "source_region_id": fragment.get("source_region_id"),
                    "page_index": fragment.get("page_index"),
                    "crop_path": fragment.get("crop_path"),
                }
            )
    elif task.get("task_type") == "figure_or_image_understanding" and task.get("regions"):
        for region in task.get("regions") or []:
            references.append(
                {
                    "source_region_id": region.get("region_id"),
                    "page_index": task.get("page_index"),
                    "crop_path": region.get("crop_path"),
                    "region_type": region.get("region_type"),
                    "sub_type": region.get("sub_type"),
                }
            )
    else:
        references.append(
            {
                "source_region_id": task.get("source_region_id") or task.get("region_id"),
                "page_index": task.get("page_index"),
                "crop_path": task.get("crop_path"),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": str(task.get("task_id") or ""),
        "final_object_id": str(task.get("final_object_id") or ""),
        "task_type": str(task.get("task_type") or ""),
        "source_region_ids": source_region_ids,
        "source_pages": source_pages,
        "model": model,
        "provider": provider,
        "status": status,
        "content": content,
        "structured_content": structured_content or {},
        "references": references,
        "raw_response_path": raw_response_path,
        "error": error,
        "created_at": now_iso(),
        "version": 1,
        "prompt_digest": prompt_digest_value,
    }


def task_filter(tasks: list[dict[str, Any]], task_id: str, task_type: str, page_index: int, limit: int) -> list[dict[str, Any]]:
    selected = [task for task in tasks if str(task.get("task_type") or "") in TASK_TYPES]
    if task_id:
        selected = [task for task in selected if str(task.get("task_id") or "") == task_id]
    if task_type:
        selected = [task for task in selected if str(task.get("task_type") or "") == task_type]
    if page_index > 0:
        selected = [task for task in selected if int(task.get("page_index") or 0) == page_index]
    if limit > 0:
        selected = selected[:limit]
    return selected


def update_registry_and_resolved_text(artifact_dir: Path, result_map: dict[str, dict[str, Any]]) -> None:
    registry_path = artifact_dir / "object_registry.jsonl"
    registry = load_jsonl(registry_path)
    registry_by_final = {str(item.get("final_object_id") or ""): item for item in registry}
    for final_object_id, result in result_map.items():
        entry = registry_by_final.get(final_object_id)
        if entry is None:
            continue
        entry["status"] = "resolved" if result.get("status") == "success" else result.get("status")
        entry["latest_content"] = result.get("content")
        entry["latest_structured_content"] = result.get("structured_content")
        entry["content_version"] = int(result.get("version") or 1)
        entry["updated_at"] = result.get("created_at")
    write_jsonl(registry_path, list(registry_by_final.values()))

    source_to_final: dict[str, str] = {}
    for entry in registry_by_final.values():
        final_object_id = str(entry.get("final_object_id") or "")
        for source_id in entry.get("source_ids") or []:
            source_to_final[str(source_id)] = final_object_id

    page_text_flow_path = artifact_dir / "page_text_flow.json"
    if not page_text_flow_path.exists():
        return
    page_text_flow = json.loads(page_text_flow_path.read_text(encoding="utf-8"))
    resolved: list[dict[str, Any]] = []
    for page in page_text_flow:
        object_flow: list[dict[str, Any]] = []
        parts: list[str] = []
        for item in page.get("object_flow", []):
            current = dict(item)
            marker = current.get("marker")
            if marker:
                region_id = str(current.get("region_id") or "")
                final_object_id = source_to_final.get(region_id, region_id)
                registry_entry = registry_by_final.get(final_object_id, {})
                current["final_object_id"] = final_object_id
                current["resolved_status"] = registry_entry.get("status", "unknown")
                current["resolved_preview"] = str(registry_entry.get("latest_content") or "")[:120]
                current["marker"] = f"[{str(current.get('region_type') or '').upper()}:{final_object_id}]"
                parts.append(current["marker"])
            else:
                parts.append(str(current.get("text") or ""))
            object_flow.append(current)
        resolved.append(
            {
                "page_index": page.get("page_index"),
                "page_type": page.get("page_type"),
                "sub_type": page.get("sub_type"),
                "object_flow": object_flow,
                "page_text_flow": "\n\n".join(part for part in parts if part).strip(),
            }
        )
    (artifact_dir / "page_text_flow_resolved.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def compact_table_fragment(fragment: dict[str, Any]) -> dict[str, Any]:
    rows = fragment.get("rows") or []
    preview_rows = []
    if isinstance(rows, list):
        preview_rows = rows[:2]
        if len(rows) > 2:
            preview_rows += rows[-1:]
    return {
        "fragment_object_id": fragment.get("fragment_object_id"),
        "source_region_id": fragment.get("source_region_id"),
        "page_index": fragment.get("page_index"),
        "backend": fragment.get("backend"),
        "strategy": fragment.get("strategy"),
        "quality_score": fragment.get("quality_score"),
        "complexity_score": fragment.get("complexity_score"),
        "col_count": fragment.get("col_count"),
        "row_count": fragment.get("row_count"),
        "headers": fragment.get("headers"),
        "row_preview": preview_rows,
        "complexity_reasons": fragment.get("complexity_reasons"),
    }


def build_task_prompt(task: dict[str, Any]) -> str:
    task_type = str(task.get("task_type") or "")
    base_prompt = str(task.get("prompt") or "").strip()
    if task_type == "single_table_understanding":
        return "\n\n".join(
            [
                base_prompt,
                build_response_schema_text(task_type),
                "Use the image as the primary truth source and the machine extracted rows only as auxiliary reference.",
                json.dumps(
                    {
                        "source_region_id": task.get("source_region_id"),
                        "page_index": task.get("page_index"),
                        "backend": task.get("backend"),
                        "strategy": task.get("strategy"),
                        "quality_score": task.get("quality_score"),
                        "complexity_score": task.get("complexity_score"),
                        "complexity_reasons": task.get("complexity_reasons"),
                        "headers": task.get("headers"),
                        "rows": task.get("rows"),
                    },
                    ensure_ascii=False,
                ),
            ]
        )
    if task_type == "cross_page_table_merge":
        fragments = [compact_table_fragment(fragment) for fragment in (task.get("fragments") or [])]
        return "\n\n".join(
            [
                base_prompt,
                build_response_schema_text(task_type),
                "Below is the cross-page relation and each page's machine extraction. Return only the necessary merge result and do not restate the full table row by row.",
                json.dumps(
                    {
                        "chain_id": task.get("chain_id"),
                        "fragment_region_ids": task.get("fragment_region_ids"),
                        "fragment_pages": task.get("fragment_pages"),
                        "relation": task.get("relation"),
                        "fragments": fragments,
                    },
                    ensure_ascii=False,
                ),
            ]
        )
    return "\n\n".join(
        [
            base_prompt,
            build_visual_response_schema_text(),
            build_visual_specific_guidance(task),
            "Answer strictly based on the visual evidence and produce a richer Chinese explanation instead of a short label list.",
        ]
    )


def parse_vlm_payload(task: dict[str, Any], content: str) -> tuple[str, dict[str, Any]]:
    text = strip_code_fence(content)
    try:
        payload = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            payload = json.loads(text[start : end + 1])
        else:
            raise
    if not isinstance(payload, dict):
        raise ValueError("VLM response must be a JSON object")
    answer = sanitize_watermark_text(str(payload.get("content") or "").strip())
    task_type = str(task.get("task_type") or "")
    if task_type in {"single_table_understanding", "cross_page_table_merge"}:
        structured = normalize_table_structured(payload)
    else:
        structured = normalize_figure_structured(payload)
    return answer, structured


request_vlm_single_pass = request_vlm


def parse_json_object_loose(text: str) -> dict[str, Any]:
    value = strip_code_fence(text).strip().replace("\ufeff", "")
    candidates: list[str] = []
    payload: Any = None
    if value:
        candidates.append(value)

    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        candidates.append(value[start : end + 1])

    balanced = extract_balanced_json_object(value)
    if balanced:
        candidates.append(balanced)

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        for cleaned in generate_json_cleanup_candidates(candidate):
            try:
                payload = json.loads(cleaned)
                break
            except Exception:
                payload = None
        if payload is not None:
            break
    else:
        raise ValueError("Unable to parse JSON object from VLM response")
    if not isinstance(payload, dict):
        raise ValueError("VLM response must be a JSON object")
    return payload


def extract_balanced_json_object(text: str) -> str:
    in_string = False
    escaped = False
    depth = 0
    start = -1
    for index, char in enumerate(text):
        if start < 0:
            if char == "{":
                start = index
                depth = 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def cleanup_common_json_damage(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("\r\n", "\n")
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    return cleaned


def generate_json_cleanup_candidates(text: str) -> list[str]:
    candidates = [text]
    cleaned = cleanup_common_json_damage(text)
    if cleaned != text:
        candidates.append(cleaned)
    balanced = extract_balanced_json_object(cleaned)
    if balanced and balanced not in candidates:
        candidates.append(balanced)
    return candidates


def call_vlm_messages(messages: list[dict[str, Any]], *, max_tokens: int) -> tuple[dict[str, Any], dict[str, Any]]:
    ensure_vlm_enabled()
    payload = {
        "model": settings.pdf_vlm_model_name,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.pdf_vlm_api_key}",
    }
    response = requests.post(
        settings.pdf_vlm_api_url,
        headers=headers,
        data=json.dumps(payload),
        timeout=settings.pdf_vlm_request_timeout,
    )
    response.raise_for_status()
    data = response.json()
    message = data["choices"][0]["message"]["content"] if "choices" in data else "{}"
    return data, {"message_content": message, "prompt_digest": digest_prompt(json.dumps(messages, ensure_ascii=False))}


def build_cross_page_fragment_messages(
    task: dict[str, Any],
    fragment: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    current_headers = (state.get("headers") or [])[:8]
    last_rows = (state.get("last_rows") or [])[-1:]
    relation = {
        "chain_id": task.get("chain_id"),
        "fragment_pages": task.get("fragment_pages"),
        "current_page": fragment.get("page_index"),
        "previous_headers": current_headers,
        "previous_last_rows": last_rows,
        "accumulated_row_count": state.get("row_count", 0),
    }
    schema_text = (
        'Only output one JSON object with fixed fields: '
        '{"content":"1-2句中文结论","structured_content":{"page_index":1,"title":"当前页表格标题或空字符串",'
        '"headers":["列1","列2"],"rows":[["值1","值2"]],"repeated_header":true,'
        '"continuation_from_previous":true,"notes":["备注1"]}}. '
        "Return the full cleaned rows for the current fragment only. Do not output markdown."
    )
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "\n\n".join(
                [
                    "You are processing one fragment of a cross-page table.",
                    "This is an ordered multi-step merge workflow. Do not rewrite the whole final table.",
                    "Only normalize the current fragment, keep row order, remove duplicated repeated header rows when appropriate, and return valid JSON.",
                    "If the image is unclear, still output the best structured rows for the current fragment instead of failing.",
                    schema_text,
                    json.dumps(
                        {
                            "relation": relation,
                            "fragment_meta": compact_table_fragment(fragment),
                        },
                        ensure_ascii=False,
                    ),
                ]
            ),
        }
    ]
    crop_path = Path(str(fragment.get("crop_path") or ""))
    if crop_path.exists():
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(crop_path)}})
    return [
        {
            "role": "system",
            "content": "You are a strict PDF table normalization assistant. Output JSON only and preserve table structure.",
        },
        {
            "role": "user",
            "content": content,
        },
    ]


def normalize_fragment_rows(rows: list[Any], headers: list[str]) -> list[list[str]]:
    normalized: list[list[str]] = []
    header_norm = [str(item).strip() for item in headers]
    for row in rows:
        if not isinstance(row, list):
            continue
        values = [str(cell) for cell in row]
        if header_norm and [item.strip() for item in values] == header_norm:
            continue
        if header_norm and len(values) < len(header_norm):
            values = values + [""] * (len(header_norm) - len(values))
        normalized.append(values)
    return normalized


def get_response_finish_reason(raw_response: dict[str, Any]) -> str:
    try:
        return str((((raw_response.get("choices") or [{}])[0]).get("finish_reason") or "")).strip()
    except Exception:
        return ""


def fallback_fragment_payload(
    fragment: dict[str, Any],
    state: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    previous_headers = [str(item).strip() for item in (state.get("headers") or []) if str(item).strip()]
    fragment_headers = [str(item).strip() for item in (fragment.get("headers") or []) if str(item).strip()]
    headers = fragment_headers or previous_headers
    rows = normalize_fragment_rows(fragment.get("rows") or [], headers)
    repeated_header = False
    if rows and headers and [str(cell).strip() for cell in rows[0]] == headers:
        rows = rows[1:]
        repeated_header = True
    return {
        "page_index": int(fragment.get("page_index") or 0),
        "title": str(fragment.get("title") or "").strip(),
        "headers": headers,
        "rows": rows,
        "repeated_header": repeated_header,
        "continuation_from_previous": bool(state.get("row_count")),
        "notes": [f"local_fallback:{reason}"],
    }


def normalize_cross_page_fragment_result(
    fragment: dict[str, Any],
    state: dict[str, Any],
    structured: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[list[str]]]:
    fragment_headers = [str(item).strip() for item in (structured.get("headers") or []) if str(item).strip()]
    if not fragment_headers:
        fragment_headers = [str(item).strip() for item in (fragment.get("headers") or []) if str(item).strip()]
    normalized_rows = normalize_fragment_rows(structured.get("rows") or [], fragment_headers)
    if not normalized_rows and fragment.get("rows"):
        normalized_rows = normalize_fragment_rows(fragment.get("rows") or [], fragment_headers)
    result = {
        "page_index": int(structured.get("page_index") or fragment.get("page_index") or 0),
        "title": str(structured.get("title") or fragment.get("title") or "").strip(),
        "headers": fragment_headers,
        "rows": normalized_rows,
        "repeated_header": bool(structured.get("repeated_header")),
        "continuation_from_previous": bool(structured.get("continuation_from_previous") or state.get("row_count")),
        "notes": [str(item) for item in (structured.get("notes") or []) if str(item).strip()],
    }
    return result, fragment_headers, normalized_rows


def is_noisy_table_header_cell(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    noisy_tokens = (
        "招股意向书",
        "招股说明书",
        "武汉兴图新科电子股份有限公司",
        "武汉力源信息技术股份有限公司",
    )
    if any(token in text for token in noisy_tokens):
        return True
    if len(text) >= 18 and re.search(r"股份有限公司", text):
        return True
    return False


def score_header_candidate(headers: list[str]) -> tuple[int, int, int]:
    clean = [str(item).strip() for item in headers if str(item).strip()]
    useful = [item for item in clean if not is_noisy_table_header_cell(item)]
    duplicate_penalty = len(useful) - len(set(useful))
    semantic_hits = sum(1 for item in useful if re.search(r"(金额|占比|比例|数量|项目|名称|单位|年度|收入|成本)", item))
    return (semantic_hits, len(useful), -duplicate_penalty)


def choose_cross_page_headers(fragments: list[dict[str, Any]]) -> list[str]:
    best: list[str] = []
    best_score = (-1, -1, -999)
    for fragment in fragments:
        raw_headers = [str(item).strip() for item in (fragment.get("headers") or [])]
        filtered = [item for item in raw_headers if item and not is_noisy_table_header_cell(item)]
        candidate = filtered or [item for item in raw_headers if item]
        if not candidate:
            continue
        score = score_header_candidate(candidate)
        if score > best_score:
            best = candidate
            best_score = score
    return best


def merge_cross_page_locally(task: dict[str, Any], reason: str) -> dict[str, Any]:
    fragments = sorted(
        list(task.get("fragments") or []),
        key=lambda item: (int(item.get("page_index") or 0), str(item.get("source_region_id") or "")),
    )
    merged_headers = choose_cross_page_headers(fragments)
    merged_rows: list[list[str]] = []
    segment_results: list[dict[str, Any]] = []
    state: dict[str, Any] = {"headers": merged_headers, "last_rows": [], "row_count": 0}

    for fragment in fragments:
        headers = [str(item).strip() for item in (fragment.get("headers") or []) if str(item).strip()]
        headers = [item for item in headers if not is_noisy_table_header_cell(item)] or headers or merged_headers
        if headers and (not merged_headers or score_header_candidate(headers) > score_header_candidate(merged_headers)):
            merged_headers = headers
        rows = normalize_fragment_rows(fragment.get("rows") or [], merged_headers or headers)
        if rows and merged_headers and [str(cell).strip() for cell in rows[0]] == merged_headers:
            rows = rows[1:]
        merged_rows.extend(rows)
        segment_results.append(
            {
                "page_index": int(fragment.get("page_index") or 0),
                "title": str(fragment.get("title") or "").strip(),
                "headers": merged_headers or headers,
                "rows": rows,
                "repeated_header": False,
                "continuation_from_previous": bool(state.get("row_count")),
                "notes": [f"local_cross_page_merge:{reason}"],
            }
        )
        state = {"headers": merged_headers, "last_rows": merged_rows[-2:], "row_count": len(merged_rows)}

    if not merged_headers and merged_rows:
        merged_headers = [f"col_{index+1}" for index in range(max(len(row) for row in merged_rows))]
        merged_rows = normalize_fragment_rows(merged_rows, merged_headers)

    return build_cross_page_final_payload(task, merged_headers, merged_rows, segment_results)


def build_cross_page_final_payload(
    task: dict[str, Any],
    headers: list[str],
    rows: list[list[str]],
    segment_results: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = f"跨页表格已按顺序处理并在本地拼接，共 {len(segment_results)} 个片段、{len(rows)} 行、{max(len(headers), 0)} 列。"
    return {
        "content": summary,
        "structured_content": {
            "title": str(next((item.get('title') for item in segment_results if item.get('title')), "") or ""),
            "merge_decision": "same_table",
            "summary": summary,
            "key_columns": headers[: min(5, len(headers))],
            "key_findings": [
                f"片段页码顺序: {', '.join(str(item.get('page_index')) for item in segment_results)}",
                f"最终拼接后总行数 {len(rows)}",
            ],
            "headers": headers,
            "rows": rows,
            "notes": [note for item in segment_results for note in (item.get("notes") or []) if str(note).strip()],
            "quantitative_highlights": [],
        },
    }


def table_result_quality_ok(structured: dict[str, Any], finish_reason: str) -> bool:
    if finish_reason == "length":
        return False
    headers = [str(item).strip() for item in (structured.get("headers") or []) if str(item).strip()]
    rows = structured.get("rows") or []
    if not headers or not rows:
        return False
    expected_cols = max(1, len(headers))
    valid_rows = 0
    for row in rows:
        if not isinstance(row, list):
            continue
        width = len([cell for cell in row if str(cell).strip()]) or len(row)
        if width >= max(1, expected_cols - 1):
            valid_rows += 1
    return valid_rows >= max(1, min(3, len(rows)))


def chunk_rows(rows: list[list[str]], chunk_size: int = 12) -> list[list[list[str]]]:
    if not rows:
        return []
    return [rows[index : index + chunk_size] for index in range(0, len(rows), chunk_size)]


def build_single_table_segment_messages(
    task: dict[str, Any],
    chunk: list[list[str]],
    chunk_index: int,
    total_chunks: int,
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    schema_text = (
        'Only output one JSON object with fixed fields: '
        '{"content":"1-2句中文结论","structured_content":{"chunk_index":1,"title":"当前表格标题或空字符串",'
        '"headers":["列1","列2"],"rows":[["值1","值2"]],"continued_from_previous":true,"notes":["备注1"]}}. '
        "Return only the cleaned rows for the current chunk. Do not output markdown."
    )
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "\n\n".join(
                [
                    "You are repairing one chunk of a single-page table after the full-table attempt failed.",
                    "Use the image as primary evidence, use the chunk row hints only as guidance, and output only the rows for the current chunk in order.",
                    schema_text,
                    json.dumps(
                        {
                            "chunk_index": chunk_index,
                            "total_chunks": total_chunks,
                            "previous_headers": state.get("headers") or [],
                            "previous_last_rows": state.get("last_rows") or [],
                            "current_chunk_row_hints": chunk,
                            "table_meta": {
                                "source_region_id": task.get("source_region_id"),
                                "page_index": task.get("page_index"),
                                "backend": task.get("backend"),
                                "strategy": task.get("strategy"),
                                "quality_score": task.get("quality_score"),
                                "complexity_score": task.get("complexity_score"),
                                "headers_hint": task.get("headers") or [],
                            },
                        },
                        ensure_ascii=False,
                    ),
                ]
            ),
        }
    ]
    crop_path = Path(str(task.get("crop_path") or ""))
    if crop_path.exists():
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(crop_path)}})
    return [
        {
            "role": "system",
            "content": "You are a strict PDF table normalization assistant. Output JSON only and preserve table structure.",
        },
        {
            "role": "user",
            "content": content,
        },
    ]


def build_single_table_final_payload(
    task: dict[str, Any],
    headers: list[str],
    rows: list[list[str]],
    segment_results: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = f"单页表格分段修复完成，共 {len(segment_results)} 个片段、{len(rows)} 行、{len(headers)} 列。"
    return {
        "content": summary,
        "structured_content": {
            "title": str(next((item.get("title") for item in segment_results if item.get("title")), "") or ""),
            "merge_decision": "single_page",
            "summary": summary,
            "key_columns": headers[: min(5, len(headers))],
            "key_findings": [
                f"分段数 {len(segment_results)}",
                f"最终行数 {len(rows)}",
            ],
            "headers": headers,
            "rows": rows,
            "notes": [note for item in segment_results for note in (item.get("notes") or []) if str(note).strip()],
            "quantitative_highlights": [],
        },
    }


def request_vlm_single_table_segmented(task: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_rows = task.get("rows") or []
    if not isinstance(source_rows, list):
        source_rows = []
    row_chunks = chunk_rows(source_rows, chunk_size=12)
    if not row_chunks:
        row_chunks = [[[]]]

    state: dict[str, Any] = {"headers": [str(item) for item in (task.get("headers") or []) if str(item).strip()], "last_rows": []}
    merged_headers: list[str] = list(state["headers"])
    merged_rows: list[list[str]] = []
    segment_results: list[dict[str, Any]] = []
    raw_segments: list[dict[str, Any]] = []

    for index, chunk in enumerate(row_chunks, start=1):
        messages = build_single_table_segment_messages(task, chunk, index, len(row_chunks), state)
        raw_response, meta = call_vlm_messages(messages, max_tokens=1400)
        raw_segments.append(
            {
                "chunk_index": index,
                "raw_response": raw_response,
                "prompt_digest": meta.get("prompt_digest"),
            }
        )
        payload = parse_json_object_loose(str(meta.get("message_content") or ""))
        structured = payload.get("structured_content") or {}
        if not isinstance(structured, dict):
            structured = {}
        fragment_headers = [str(item) for item in (structured.get("headers") or []) if str(item).strip()]
        if fragment_headers and (not merged_headers or len(fragment_headers) > len(merged_headers)):
            merged_headers = fragment_headers
        normalized_rows = normalize_fragment_rows(structured.get("rows") or [], merged_headers or fragment_headers)
        merged_rows.extend(normalized_rows)
        segment_results.append(
            {
                "chunk_index": index,
                "title": str(structured.get("title") or "").strip(),
                "headers": fragment_headers,
                "rows": normalized_rows,
                "notes": [str(item) for item in (structured.get("notes") or []) if str(item).strip()],
            }
        )
        state = {"headers": merged_headers, "last_rows": merged_rows[-2:]}

    final_payload = build_single_table_final_payload(task, merged_headers, merged_rows, segment_results)
    synthetic_response = {
        "object": "segmented_single_table_repair",
        "model": settings.pdf_vlm_model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps(final_payload, ensure_ascii=False)},
                "finish_reason": "stop",
            }
        ],
        "segments": raw_segments,
    }
    return synthetic_response, {
        "message_content": json.dumps(final_payload, ensure_ascii=False),
        "prompt_digest": digest_prompt(
            json.dumps({"task_id": task.get("task_id"), "segments": [item.get("prompt_digest") for item in raw_segments]}, ensure_ascii=False)
        ),
    }


def request_vlm(task: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    task_type = str(task.get("task_type") or "")
    if task_type == "figure_or_image_understanding":
        valid_regions = filter_visual_regions(task)
        if not valid_regions:
            final_payload = {
                "content": "当前页仅检测到无效或过薄的图像碎片，已跳过多模态分析。",
                "structured_content": {
                    "summary": "未执行图像理解：检测到的区域过薄或疑似噪声碎片。",
                    "visual_type": "unknown",
                    "is_same_visual": "no",
                    "groups": [],
                    "detailed_description": "",
                    "flow_description": {"start": "", "steps": [], "decision_points": [], "end": ""},
                    "chart_analysis": {
                        "chart_type": "unknown",
                        "x_axis": "",
                        "y_axis": "",
                        "series": [],
                        "trend_summary": "",
                        "max_points": [],
                        "min_points": [],
                        "turning_points": [],
                        "comparison_points": [],
                    },
                    "labels": [],
                    "numbers": [],
                    "relations": [],
                    "key_observations": ["检测到的视觉区域高度过小，疑似分割噪声，已自动跳过。"],
                    "notes": [],
                },
            }
            synthetic_response = {
                "object": "skipped_invalid_visual_task",
                "model": settings.pdf_vlm_model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": json.dumps(final_payload, ensure_ascii=False)},
                        "finish_reason": "stop",
                    }
                ],
            }
            return synthetic_response, {
                "message_content": json.dumps(final_payload, ensure_ascii=False),
                "prompt_digest": digest_prompt(
                    json.dumps({"task_id": task.get("task_id"), "skip": "invalid_visual_regions"}, ensure_ascii=False)
                ),
            }
    if task_type == "single_table_understanding":
        try:
            raw_response, meta = request_vlm_single_pass(task)
            payload = parse_json_object_loose(str(meta.get("message_content") or ""))
            structured = normalize_table_structured(payload)
            finish_reason = get_response_finish_reason(raw_response)
            if table_result_quality_ok(structured, finish_reason):
                return raw_response, meta
        except Exception:
            pass
        return request_vlm_single_table_segmented(task)

    if task_type != "cross_page_table_merge":
        return request_vlm_single_pass(task)

    try:
        raw_response, meta = request_vlm_single_pass(task)
        payload = parse_json_object_loose(str(meta.get("message_content") or ""))
        structured = normalize_table_structured(payload)
        finish_reason = get_response_finish_reason(raw_response)
        if table_result_quality_ok(structured, finish_reason):
            return raw_response, meta
        log(
            "cross_page_single_pass_degraded "
            f"task_id={task.get('task_id')} finish_reason={finish_reason or 'unknown'} "
            f"headers={len(structured.get('headers') or [])} rows={len(structured.get('rows') or [])}"
        )
    except Exception as exc:
        log(f"cross_page_single_pass_fallback task_id={task.get('task_id')} reason={type(exc).__name__}:{exc}")

    try:
        fragments = sorted(
            list(task.get("fragments") or []),
            key=lambda item: (int(item.get("page_index") or 0), str(item.get("source_region_id") or "")),
        )
        state: dict[str, Any] = {"headers": [], "last_rows": [], "row_count": 0}
        merged_headers: list[str] = []
        merged_rows: list[list[str]] = []
        segment_results: list[dict[str, Any]] = []
        raw_segments: list[dict[str, Any]] = []

        for fragment in fragments:
            try:
                messages = build_cross_page_fragment_messages(task, fragment, state)
                raw_response, meta = call_vlm_messages(messages, max_tokens=1200)
                raw_segments.append(
                    {
                        "source_region_id": fragment.get("source_region_id"),
                        "page_index": fragment.get("page_index"),
                        "raw_response": raw_response,
                        "prompt_digest": meta.get("prompt_digest"),
                    }
                )
                payload = parse_json_object_loose(str(meta.get("message_content") or ""))
                structured = payload.get("structured_content") or {}
                if not isinstance(structured, dict):
                    structured = {}
                finish_reason = get_response_finish_reason(raw_response)
                if finish_reason == "length":
                    raise ValueError("fragment response truncated by model")
                segment_result, fragment_headers, normalized_rows = normalize_cross_page_fragment_result(fragment, state, structured)
                if not fragment_headers and fragment.get("headers"):
                    raise ValueError("fragment headers missing after normalization")
            except Exception as exc:
                fallback_reason = f"{type(exc).__name__}:{exc}"
                log(
                    "cross_page_fragment_fallback "
                    f"task_id={task.get('task_id')} page={fragment.get('page_index')} "
                    f"region={fragment.get('source_region_id')} reason={fallback_reason}"
                )
                raw_segments.append(
                    {
                        "source_region_id": fragment.get("source_region_id"),
                        "page_index": fragment.get("page_index"),
                        "raw_response": {"fallback": True, "reason": fallback_reason},
                        "prompt_digest": "",
                    }
                )
                segment_result = fallback_fragment_payload(fragment, state, fallback_reason)
                fragment_headers = segment_result["headers"]
                normalized_rows = segment_result["rows"]
            if fragment_headers and (not merged_headers or len(fragment_headers) > len(merged_headers)):
                merged_headers = fragment_headers
            merged_rows.extend(normalized_rows)
            segment_results.append(segment_result)
            state = {
                "headers": merged_headers,
                "last_rows": merged_rows[-2:],
                "row_count": len(merged_rows),
            }

        final_payload = build_cross_page_final_payload(task, merged_headers, merged_rows, segment_results)
        synthetic_response = {
            "object": "segmented_cross_page_merge",
            "model": settings.pdf_vlm_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(final_payload, ensure_ascii=False),
                    },
                    "finish_reason": "stop",
                }
            ],
            "segments": raw_segments,
        }
        return synthetic_response, {
            "message_content": json.dumps(final_payload, ensure_ascii=False),
            "prompt_digest": digest_prompt(
                json.dumps({"task_id": task.get("task_id"), "segments": [item.get("prompt_digest") for item in raw_segments]}, ensure_ascii=False)
            ),
        }
    except Exception as exc:
        fallback_reason = f"{type(exc).__name__}:{exc}"
        log(f"cross_page_task_fallback task_id={task.get('task_id')} reason={fallback_reason}")
        final_payload = merge_cross_page_locally(task, fallback_reason)
        synthetic_response = {
            "object": "local_cross_page_merge_fallback",
            "model": settings.pdf_vlm_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(final_payload, ensure_ascii=False),
                    },
                    "finish_reason": "stop",
                }
            ],
            "segments": [],
        }
        return synthetic_response, {
            "message_content": json.dumps(final_payload, ensure_ascii=False),
            "prompt_digest": digest_prompt(
                json.dumps({"task_id": task.get("task_id"), "fallback": fallback_reason}, ensure_ascii=False)
            ),
        }


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve() if args.artifact_dir else (settings.artifact_root_dir / "stage2_precise_extraction")
    task_path = artifact_dir / "multimodal_tasks.jsonl"
    result_path = artifact_dir / RESULT_FILE_NAME
    raw_dir = artifact_dir / RAW_DIR_NAME
    raw_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_jsonl(task_path)
    if not tasks:
        raise FileNotFoundError(f"No multimodal tasks found: {task_path}")
    selected = task_filter(tasks, args.task_id, args.task_type, args.page_index, args.limit)
    existing_rows = load_jsonl(result_path)
    existing_map = {str(item.get("final_object_id") or ""): item for item in existing_rows}
    pending = []
    for task in selected:
        final_object_id = str(task.get("final_object_id") or "")
        if not args.overwrite and final_object_id in existing_map and existing_map[final_object_id].get("status") == "success":
            continue
        pending.append(task)

    log(f"artifact_dir={artifact_dir}")
    log(f"tasks_total={len(tasks)} pending={len(pending)} dry_run={args.dry_run}")
    if args.dry_run:
        for task in pending[:20]:
            log(f"pending task_id={task.get('task_id')} final_object_id={task.get('final_object_id')} task_type={task.get('task_type')}")
        return

    ensure_vlm_enabled()
    new_results: dict[str, dict[str, Any]] = {}
    for task in pending:
        task_id = str(task.get("task_id") or "")
        final_object_id = str(task.get("final_object_id") or "")
        task_type = str(task.get("task_type") or "")
        log(f"run task_id={task_id} final_object_id={final_object_id} task_type={task_type}")
        try:
            raw_response, meta = request_vlm(task)
            raw_path = raw_dir / f"{task_id}.raw.json"
            raw_path.write_text(json.dumps(raw_response, ensure_ascii=False, indent=2), encoding="utf-8")
            content, structured_content = parse_vlm_payload(task, meta["message_content"])
            result = build_result_record(
                task,
                status="success",
                content=content,
                structured_content=structured_content,
                model=settings.pdf_vlm_model_name,
                provider=settings.pdf_vlm_provider,
                raw_response_path=str(raw_path),
                error=None,
                prompt_digest_value=meta["prompt_digest"],
            )
        except Exception as exc:
            result = build_result_record(
                task,
                status="failed",
                content="",
                structured_content={},
                model=settings.pdf_vlm_model_name,
                provider=settings.pdf_vlm_provider,
                raw_response_path="",
                error=f"{type(exc).__name__}: {exc}",
                prompt_digest_value="",
            )
        existing_map[final_object_id] = result
        new_results[final_object_id] = result

    final_rows = list(existing_map.values())
    final_rows.sort(key=lambda item: (str(item.get("final_object_id") or ""), str(item.get("created_at") or "")))
    write_jsonl(result_path, final_rows)
    update_registry_and_resolved_text(artifact_dir, new_results)
    log(f"done results_written={len(new_results)} result_path={result_path}")


if __name__ == "__main__":
    main()
