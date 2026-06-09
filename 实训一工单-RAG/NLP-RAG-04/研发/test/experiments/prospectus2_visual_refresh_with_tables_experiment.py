from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "prospectus2_parse4_full"
DEFAULT_STAGE1_PATH = DEFAULT_ARTIFACT_DIR / "stage1_layout" / "stage1_layout_analysis.json"
DEFAULT_VISUAL_COLLECTION = "prospectus2_chunks_04_visual"
DEFAULT_MONGO_COLLECTION = "prospectus2_tables_04"

FLOWCHART_KEYWORDS = [
    "组织结构",
    "组织架构",
    "股权结构",
    "流程图",
    "委员会",
    "董事会",
    "监事会",
    "总经理",
    "销售部",
    "事业部",
    "分公司",
    "销售处",
    "研发中心",
]


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[visual-refresh+tables {timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild prospectus2 visual tasks from existing stage1 layout, and include same-page table crops into merged visual tasks."
    )
    parser.add_argument("--artifact-dir", type=str, default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--stage1-path", type=str, default="")
    parser.add_argument("--run-vlm", action="store_true")
    parser.add_argument("--run-vectorize", action="store_true")
    parser.add_argument("--run-persist", action="store_true")
    parser.add_argument("--overwrite-vlm", action="store_true")
    parser.add_argument("--visual-collection", type=str, default=DEFAULT_VISUAL_COLLECTION)
    parser.add_argument("--mongo-collection", type=str, default=DEFAULT_MONGO_COLLECTION)
    parser.add_argument("--limit-pages", type=int, default=0, help="Only rebuild the first N pages that produce visual tasks")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    text = str(text or "").replace("\u3000", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clamp_bbox(bbox: list[float], page_rect: fitz.Rect) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    x0 = max(page_rect.x0, min(page_rect.x1, x0))
    x1 = max(page_rect.x0, min(page_rect.x1, x1))
    y0 = max(page_rect.y0, min(page_rect.y1, y0))
    y1 = max(page_rect.y0, min(page_rect.y1, y1))
    if x1 <= x0:
        x1 = min(page_rect.x1, x0 + 1.0)
    if y1 <= y0:
        y1 = min(page_rect.y1, y0 + 1.0)
    return [x0, y0, x1, y1]


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def union_bboxes(boxes: list[list[float]]) -> list[float]:
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    return [float(x0), float(y0), float(x1), float(y1)]


def expand_bbox(bbox: list[float], page_rect: fitz.Rect, margin_x: float, margin_y: float) -> list[float]:
    expanded = [
        float(bbox[0]) - margin_x,
        float(bbox[1]) - margin_y,
        float(bbox[2]) + margin_x,
        float(bbox[3]) + margin_y,
    ]
    return clamp_bbox(expanded, page_rect)


def overlaps(a: list[float], b: list[float]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def resolve_stage1_path(args: argparse.Namespace, artifact_dir: Path) -> Path:
    if args.stage1_path:
        return Path(args.stage1_path).resolve()
    return artifact_dir / "stage1_layout" / "stage1_layout_analysis.json"


def resolve_pdf_path(artifact_dir: Path, stage1: dict[str, Any]) -> Path:
    manifest_path = artifact_dir / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        pdf_path = Path(str(manifest.get("pdf_path") or ""))
        if pdf_path.exists():
            return pdf_path
    pdf_path = Path(str(stage1.get("pdf") or ""))
    if pdf_path.exists():
        return pdf_path
    pdfs = sorted((PROJECT_ROOT / "data").glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError("No PDF found under data directory")
    return pdfs[0]


def extract_drawing_boxes(page: fitz.Page) -> list[list[float]]:
    boxes: list[list[float]] = []
    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if rect is None:
            continue
        bbox = [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
        if bbox_area(bbox) <= 4:
            continue
        boxes.append(bbox)
    return boxes


def looks_like_short_box_text(region: dict[str, Any]) -> bool:
    if str(region.get("region_type") or "") != "text":
        return False
    text = normalize_text(str(region.get("text") or ""))
    if not text:
        return False
    compact = text.replace(" ", "").replace("\n", "")
    if len(compact) == 0 or len(compact) > 24:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 5:
        return False
    if any(keyword in compact for keyword in FLOWCHART_KEYWORDS):
        return True
    if len(lines) >= 2:
        return True
    return len(compact) <= 10


def page_looks_like_diagram(page_info: dict[str, Any], short_text_regions: list[dict[str, Any]], drawing_boxes: list[list[float]]) -> bool:
    page_tags = [
        normalize_text(str(page_info.get("page_type") or "")),
        normalize_text(str(page_info.get("sub_type") or "")),
    ]
    joined_text = " ".join(normalize_text(str(region.get("text") or "")) for region in short_text_regions)
    if any("flowchart" in tag or "org" in tag for tag in page_tags):
        return True
    if any(keyword in joined_text for keyword in FLOWCHART_KEYWORDS):
        return True
    if len(short_text_regions) >= 8 and len(drawing_boxes) >= 6:
        return True
    return len(short_text_regions) >= 12


def crop_region_image(page: fitz.Page, bbox: list[float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rect = fitz.Rect(*clamp_bbox(bbox, page.rect))
    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), clip=rect, alpha=False)
    pix.save(output_path)


def build_component_crop(page: fitz.Page, region: dict[str, Any], component_dir: Path) -> dict[str, Any]:
    region_id = str(region.get("region_id") or "")
    crop_path = component_dir / f"{region_id}.png"
    crop_region_image(page, list(region["bbox"]), crop_path)
    return {
        "region_id": region_id,
        "page_index": int(region.get("page_index") or 0),
        "region_type": str(region.get("region_type") or ""),
        "sub_type": str(region.get("sub_type") or ""),
        "bbox": list(region["bbox"]),
        "crop_path": str(crop_path),
        "context_text": normalize_text(str(region.get("text") or "")),
    }


def build_existing_table_component(region: dict[str, Any], crop_path: Path) -> dict[str, Any]:
    return {
        "region_id": str(region.get("region_id") or ""),
        "page_index": int(region.get("page_index") or 0),
        "region_type": "table",
        "sub_type": str(region.get("sub_type") or "data_table"),
        "bbox": list(region["bbox"]),
        "crop_path": str(crop_path),
        "context_text": normalize_text(str(region.get("text") or "")),
    }


def find_same_page_table_regions(artifact_dir: Path, regions: list[dict[str, Any]]) -> list[tuple[dict[str, Any], Path]]:
    table_dir = artifact_dir / "crops" / "tables"
    matches: list[tuple[dict[str, Any], Path]] = []
    for region in regions:
        if str(region.get("region_type") or "") != "table":
            continue
        region_id = str(region.get("region_id") or "")
        if not region_id:
            continue
        crop_path = table_dir / f"{region_id}.png"
        if crop_path.exists():
            matches.append((region, crop_path))
    return matches


def build_visual_tasks_from_stage1(
    artifact_dir: Path,
    stage1: dict[str, Any],
    pdf_path: Path,
    limit_pages: int = 0,
) -> list[dict[str, Any]]:
    crop_root = artifact_dir / "crops" / "visual_refresh_with_tables"
    tasks: list[dict[str, Any]] = []

    with fitz.open(str(pdf_path)) as doc:
        for page_info in stage1.get("page_details", []):
            page_index = int(page_info["page_index"])
            page = doc.load_page(page_index - 1)
            regions = list(page_info.get("regions") or [])
            visual_regions = [region for region in regions if str(region.get("region_type") or "") in {"figure", "image"}]
            short_text_regions = [region for region in regions if looks_like_short_box_text(region)]
            drawing_boxes = extract_drawing_boxes(page)
            diagram_like = page_looks_like_diagram(page_info, short_text_regions, drawing_boxes)

            if not visual_regions and not diagram_like:
                continue

            same_page_table_regions = find_same_page_table_regions(artifact_dir, regions)
            if same_page_table_regions:
                log(
                    f"page={page_index} include_same_page_tables "
                    f"count={len(same_page_table_regions)} ids={[region.get('region_id') for region, _ in same_page_table_regions]}"
                )

            candidate_regions: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            for region in [*visual_regions, *short_text_regions, *(region for region, _ in same_page_table_regions)]:
                region_id = str(region.get("region_id") or "")
                if not region_id or region_id in seen_ids:
                    continue
                seen_ids.add(region_id)
                candidate_regions.append(region)

            if not candidate_regions:
                continue

            candidate_boxes = [list(region["bbox"]) for region in candidate_regions]
            merged_bbox = union_bboxes(candidate_boxes)
            expanded_bbox = expand_bbox(merged_bbox, page.rect, margin_x=24.0, margin_y=24.0)

            linked_drawing_boxes = [
                box
                for box in drawing_boxes
                if overlaps(expanded_bbox, expand_bbox(box, page.rect, margin_x=8.0, margin_y=8.0))
            ]
            if linked_drawing_boxes:
                expanded_bbox = expand_bbox(
                    union_bboxes([expanded_bbox, *linked_drawing_boxes]),
                    page.rect,
                    margin_x=12.0,
                    margin_y=12.0,
                )

            merged_region_id = f"p{page_index}_visual_refresh_tables_merged"
            merged_crop_path = crop_root / "merged" / f"{merged_region_id}.png"
            crop_region_image(page, expanded_bbox, merged_crop_path)

            component_dir = crop_root / "components" / f"p{page_index:04d}"
            component_regions: list[dict[str, Any]] = []

            existing_visual_components = [
                build_component_crop(page, region, component_dir)
                for region in candidate_regions
                if str(region.get("region_type") or "") in {"figure", "image"}
            ]
            component_regions.extend(existing_visual_components)

            table_component_map = {
                str(region.get("region_id") or ""): build_existing_table_component(region, crop_path)
                for region, crop_path in same_page_table_regions
            }
            component_regions.extend(table_component_map.values())

            if not existing_visual_components and not table_component_map:
                text_candidates = sorted(candidate_regions, key=lambda item: bbox_area(list(item["bbox"])), reverse=True)[:8]
                component_regions.extend(build_component_crop(page, region, component_dir) for region in text_candidates)

            context_text = "\n".join(
                text
                for text in (normalize_text(str(region.get("text") or "")) for region in candidate_regions)
                if text
            )[:2000]

            sub_type = "flowchart" if diagram_like else (
                str(visual_regions[0].get("sub_type") or "") if visual_regions else "mixed_visual"
            )
            merged_region = {
                "region_id": merged_region_id,
                "page_index": page_index,
                "region_type": "figure",
                "sub_type": sub_type or "mixed_visual",
                "bbox": expanded_bbox,
                "crop_path": str(merged_crop_path),
                "context_text": context_text,
            }

            task_regions = [merged_region, *component_regions]
            source_region_ids = [str(region.get("region_id") or "") for region in candidate_regions if str(region.get("region_id") or "")]
            crop_paths = [str(region["crop_path"]) for region in task_regions]

            task = {
                "group_id": f"visual_refresh_tables_page_{page_index:04d}",
                "page_index": page_index,
                "region_type": "visual_group",
                "object_type": "visual_group",
                "sub_type": "grouped_same_page_visuals_plus_tables_refresh",
                "route": "vlm_direct",
                "prompt_type": "figure_or_image_understanding",
                "task_type": "figure_or_image_understanding",
                "source_region_ids": source_region_ids,
                "crop_paths": crop_paths,
                "regions": task_regions,
                "region_count": len(task_regions),
                "final_object_id": f"VISUAL_FINAL_PAGE_{page_index:04d}",
                "context_text": context_text,
                "refresh_method": "merged_visual_region_plus_same_page_tables_from_stage1",
                "diagram_like": diagram_like,
                "included_table_region_ids": sorted(table_component_map.keys()),
            }
            tasks.append(task)
            log(
                f"rebuilt_visual_task page={page_index} diagram_like={diagram_like} "
                f"components={len(component_regions)} source_regions={len(source_region_ids)} "
                f"included_tables={len(table_component_map)}"
            )

            if limit_pages and len(tasks) >= limit_pages:
                break

    return tasks


def backup_file(path: Path, backup_root: Path) -> None:
    if not path.exists():
        return
    backup_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_root / path.name)


def rebuild_registry(existing_rows: list[dict[str, Any]], visual_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preserved = [
        row
        for row in existing_rows
        if str(row.get("prompt_type") or "") != "figure_or_image_understanding"
        and not str(row.get("final_object_id") or "").startswith("VISUAL_FINAL_PAGE_")
    ]
    for task in visual_tasks:
        preserved.append(
            {
                "final_object_id": str(task.get("final_object_id") or ""),
                "object_type": str(task.get("object_type") or "visual_group"),
                "source_ids": list(task.get("source_region_ids") or []),
                "status": "pending_vlm",
                "latest_content": None,
                "latest_structured_content": None,
                "content_version": 0,
                "prompt_type": "figure_or_image_understanding",
            }
        )
    return preserved


def rewrite_artifact_files(artifact_dir: Path, visual_tasks: list[dict[str, Any]]) -> None:
    figure_tasks_path = artifact_dir / "figure_tasks.jsonl"
    multimodal_tasks_path = artifact_dir / "multimodal_tasks.jsonl"
    object_registry_path = artifact_dir / "object_registry.jsonl"
    backup_root = artifact_dir / "visual_refresh_with_tables_backup" / datetime.now().strftime("%Y%m%d_%H%M%S")

    backup_file(figure_tasks_path, backup_root)
    backup_file(multimodal_tasks_path, backup_root)
    backup_file(object_registry_path, backup_root)

    existing_tasks = load_jsonl(multimodal_tasks_path)
    preserved_tasks = [task for task in existing_tasks if str(task.get("task_type") or "") != "figure_or_image_understanding"]
    preserved_tasks.extend(visual_tasks)
    write_jsonl(multimodal_tasks_path, preserved_tasks)
    write_jsonl(figure_tasks_path, visual_tasks)

    existing_registry = load_jsonl(object_registry_path)
    rebuilt_registry = rebuild_registry(existing_registry, visual_tasks)
    write_jsonl(object_registry_path, rebuilt_registry)

    refresh_manifest = {
        "artifact_dir": str(artifact_dir),
        "visual_task_count": len(visual_tasks),
        "backup_dir": str(backup_root),
        "updated_files": [
            str(figure_tasks_path),
            str(multimodal_tasks_path),
            str(object_registry_path),
        ],
    }
    write_json(artifact_dir / "visual_refresh_with_tables_manifest.json", refresh_manifest)
    log(f"visual_task_count={len(visual_tasks)} backup_dir={backup_root}")


def run_subprocess(command: list[str]) -> None:
    log("run: " + " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    stage1_path = resolve_stage1_path(args, artifact_dir)
    stage1 = load_json(stage1_path)
    pdf_path = resolve_pdf_path(artifact_dir, stage1)

    log(f"artifact_dir={artifact_dir}")
    log(f"stage1_path={stage1_path}")
    log(f"pdf_path={pdf_path}")

    visual_tasks = build_visual_tasks_from_stage1(
        artifact_dir=artifact_dir,
        stage1=stage1,
        pdf_path=pdf_path,
        limit_pages=max(0, int(args.limit_pages)),
    )
    if not visual_tasks:
        raise ValueError("No rebuilt visual tasks produced")

    rewrite_artifact_files(artifact_dir, visual_tasks)

    script_dir = PROJECT_ROOT / "test"
    if args.run_vlm:
        command = [
            sys.executable,
            str(script_dir / "run_stage2_multimodal_tasks.py"),
            "--artifact-dir",
            str(artifact_dir),
            "--task-type",
            "figure_or_image_understanding",
        ]
        if args.overwrite_vlm:
            command.append("--overwrite")
        run_subprocess(command)

    if args.run_vectorize:
        command = [
            sys.executable,
            str(script_dir / "stage4_vectorize_visual_results_smoke_test.py"),
            "--artifact-dir",
            str(artifact_dir),
        ]
        run_subprocess(command)

    if args.run_persist:
        command = [
            sys.executable,
            str(script_dir / "stage5_persist_outputs_smoke_test.py"),
            "--artifact-dir",
            str(artifact_dir),
            "--visual-collection",
            str(args.visual_collection),
            "--mongo-collection",
            str(args.mongo_collection),
            "--skip-text",
            "--skip-tables",
            "--skip-artifact-upload",
        ]
        run_subprocess(command)

    log("done")


if __name__ == "__main__":
    main()
