from __future__ import annotations

# 工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统
import argparse
import json
import sys
from pathlib import Path

import fitz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.services.pdf_parser import PDFParser
from app.services.pdf_vlm_client import PDFVLMClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-page PDF VLM smoke test")
    parser.add_argument("--page", type=int, required=True, help="Physical page number in PDF, starting from 1")
    parser.add_argument(
        "--pdf",
        type=str,
        default="",
        help="Optional PDF path. Defaults to config pdf_path.",
    )
    parser.add_argument(
        "--render-scale",
        type=float,
        default=settings.pdf_vlm_render_scale,
        help="Override render scale for page image.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "image_only"],
        default="full",
        help="full = image + local text + table, image_only = image only",
    )
    parser.add_argument(
        "--force-items",
        action="store_true",
        help="Ask the VLM to avoid empty arrays when possible.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve() if args.pdf else settings.pdf_path
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if args.page < 1:
        raise ValueError("--page must be >= 1")

    pdf_parser = PDFParser(ocr_lang=settings.ocr_lang)
    pages = pdf_parser.parse(pdf_path)
    if args.page > len(pages):
        raise ValueError(f"PDF has only {len(pages)} parsed pages, got page={args.page}")

    page = pages[args.page - 1]
    client = PDFVLMClient()
    if not client.is_enabled():
        raise RuntimeError("PDF VLM is not enabled. Check PDF_VLM_API_URL / KEY / MODEL_NAME.")

    cache_dir = settings.artifact_dir / "pdf_vlm_smoke_test" / pdf_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    try:
        pdf_page = doc.load_page(args.page - 1)
        pix = pdf_page.get_pixmap(
            matrix=fitz.Matrix(args.render_scale, args.render_scale),
            alpha=False,
        )
        result = client.enhance_page(
            page_number=int(page["page_number"]),
            logical_page=page.get("logical_page"),
            local_text=str(page.get("text") or ""),
            table_markdown=str(page.get("tables_markdown") or ""),
            image_bytes=pix.tobytes("png"),
            cache_dir=cache_dir,
            mode=args.mode,
            force_items=args.force_items,
            bypass_cache=args.force_items,
            cache_variant="force" if args.force_items else "",
        )
    finally:
        doc.close()

    summary = {
        "pdf": str(pdf_path),
        "page": int(page["page_number"]),
        "logical_page": page.get("logical_page"),
        "page_type": page.get("page_type"),
        "mode": args.mode,
        "force_items": args.force_items,
        "status": result.get("status"),
        "item_count": len(list(result.get("items") or [])),
        "cache_dir": str(cache_dir),
        "local_text_preview": str(page.get("text") or "")[:600],
        "table_markdown_preview": str(page.get("tables_markdown") or "")[:600],
    }
    force_suffix = ".force" if args.force_items else ""
    summary_suffix = "" if args.mode == "full" else f".{args.mode}"
    summary_suffix = f"{summary_suffix}{force_suffix}"
    summary_path = cache_dir / f"page_{args.page}{summary_suffix}.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nRaw response file: {cache_dir / f'page_{args.page}{summary_suffix}.raw.json'}")
    print(f"Parsed items file: {cache_dir / f'page_{args.page}{summary_suffix}.json'}")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()
