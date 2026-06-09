from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "vlm_model_probe_isolated"


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[vlm-probe {timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated VLM probe for testing a new multimodal model on local images.")
    parser.add_argument("--image", type=str, default="", help="Single image file to test")
    parser.add_argument("--image-dir", type=str, default="", help="Directory containing images to test")
    parser.add_argument("--output-dir", type=str, default="", help="Directory to save isolated probe outputs")
    parser.add_argument(
        "--prompt-profile",
        choices=("chart_page", "generic_visual"),
        default="chart_page",
        help="Prompt style for the probe",
    )
    parser.add_argument("--max-tokens", type=int, default=1800, help="Max tokens for the VLM response")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N images")
    return parser.parse_args()


def ensure_vlm_configured() -> None:
    if not settings.pdf_vlm_api_url or not settings.pdf_vlm_api_key or not settings.pdf_vlm_model_name:
        raise RuntimeError("PDF VLM is not fully configured. Check PDF_VLM_API_URL / PDF_VLM_API_KEY / PDF_VLM_MODEL_NAME.")


def resolve_images(args: argparse.Namespace) -> list[Path]:
    images: list[Path] = []
    if args.image:
        image_path = Path(args.image).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        images.append(image_path)
    elif args.image_dir:
        image_dir = Path(args.image_dir).resolve()
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        for suffix in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
            images.extend(sorted(image_dir.glob(suffix)))
    else:
        raise ValueError("Provide either --image or --image-dir.")

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in images:
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path.resolve())

    if not deduped:
        raise FileNotFoundError("No images found for probing.")
    if args.limit and args.limit > 0:
        deduped = deduped[: args.limit]
    return deduped


def image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime = mime_map.get(suffix, "image/png")
    payload = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{payload}"


def strip_code_fence(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    return cleaned


def parse_json_object_loose(text: str) -> dict[str, Any]:
    cleaned = strip_code_fence(text)
    try:
        payload = json.loads(cleaned)
        return payload if isinstance(payload, dict) else {"raw_text": cleaned}
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            payload = json.loads(candidate)
            return payload if isinstance(payload, dict) else {"raw_text": cleaned}
        except Exception:
            pass
    return {"raw_text": cleaned}


def build_prompt(profile: str) -> str:
    if profile == "chart_page":
        return "\n".join(
            [
                "你是一个严格的图表结构化分析助手。请只基于图片内容作答，不要猜测看不清的内容。",
                "如果页面里包含多个图表，必须先区分每个图表，再分别分析，禁止把不同图表的数据混在一起。",
                "对于条形图/折线图/柱状图：请逐项建立“标签 -> 数值”的对应关系；如果无法确认对应关系，就写 uncertain，不要错配。",
                "对于饼图：请逐项提取“标签 -> 数值/占比”。",
                "最后输出一个 JSON 对象，不要输出 markdown，不要输出代码块。",
                '{"content":"2-4句中文结论","structured_content":{"page_type":"chart_page","chart_count":0,'
                '"charts":[{"chart_index":1,"chart_type":"pie|bar|line|combo|unknown","title":"图表标题或空字符串",'
                '"items":[{"label":"标签","value":"数值","unit":"%|万元|空字符串","confidence":"high|medium|low"}],'
                '"max_item":"最大项或空字符串","min_item":"最小项或空字符串","negative_items":["负值项"],'
                '"trend_summary":"趋势或结构摘要","notes":["补充说明"]}],"cross_chart_findings":["跨图结论"],"warnings":["不确定点"]}}',
            ]
        )
    return "\n".join(
        [
            "你是一个严格的视觉理解助手。请只基于图片内容作答，不要编造。",
            "输出一个 JSON 对象，不要输出 markdown，不要输出代码块。",
            '{"content":"2-4句中文结论","structured_content":{"summary":"图片摘要","labels":["标签1"],"numbers":["数字1"],"relations":["关系1"],"warnings":["不确定点"]}}',
        ]
    )


def build_messages(image_path: Path, profile: str) -> list[dict[str, Any]]:
    prompt = build_prompt(profile)
    return [
        {
            "role": "system",
            "content": "你是一个严格的多模态结构化抽取助手。只能基于图片输出 JSON，不得编造。",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
            ],
        },
    ]


def call_vlm(messages: list[dict[str, Any]], *, max_tokens: int, temperature: float) -> tuple[dict[str, Any], str]:
    payload = {
        "model": settings.pdf_vlm_model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.pdf_vlm_api_key}",
    }
    response = requests.post(
        settings.pdf_vlm_api_url,
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False),
        timeout=settings.pdf_vlm_request_timeout,
    )
    response.raise_for_status()
    raw_json = response.json()
    message_content = ""
    if "choices" in raw_json:
        try:
            message_content = str(raw_json["choices"][0]["message"]["content"] or "")
        except Exception:
            message_content = ""
    return raw_json, message_content


def safe_slug(text: str) -> str:
    value = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", text).strip("_")
    return value or "image"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_vlm_configured()
    images = resolve_images(args)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    log(f"output_dir={output_dir}")
    log(f"image_count={len(images)}")
    log(f"vlm_model={settings.pdf_vlm_model_name}")
    log(f"vlm_api_url={settings.pdf_vlm_api_url}")

    for index, image_path in enumerate(images, start=1):
        item_started = time.time()
        slug = f"{index:03d}_{safe_slug(image_path.stem)}"
        item_dir = output_dir / slug
        item_dir.mkdir(parents=True, exist_ok=True)

        log(f"probing image {index}/{len(images)} -> {image_path.name}")
        messages = build_messages(image_path, args.prompt_profile)
        prompt_text = ""
        for block in messages[1]["content"]:
            if isinstance(block, dict) and block.get("type") == "text":
                prompt_text = str(block.get("text") or "")
                break
        write_text(item_dir / "prompt.txt", prompt_text)
        write_json(
            item_dir / "request_meta.json",
            {
                "image_path": str(image_path),
                "prompt_profile": args.prompt_profile,
                "model_name": settings.pdf_vlm_model_name,
                "api_url": settings.pdf_vlm_api_url,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            },
        )

        try:
            raw_json, message_content = call_vlm(messages, max_tokens=args.max_tokens, temperature=args.temperature)
            parsed = parse_json_object_loose(message_content)
            write_json(item_dir / "raw_response.json", raw_json)
            write_text(item_dir / "message_content.txt", message_content)
            write_json(item_dir / "parsed_response.json", parsed)
            status = "success"
            error = ""
        except Exception as exc:
            raw_json = {}
            message_content = ""
            parsed = {}
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            write_text(item_dir / "error.txt", error)

        manifest_rows.append(
            {
                "image_path": str(image_path),
                "output_dir": str(item_dir),
                "status": status,
                "error": error,
                "elapsed_seconds": round(time.time() - item_started, 3),
            }
        )

    write_json(output_dir / "manifest.json", {"items": manifest_rows})
    log(f"done manifest={output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
