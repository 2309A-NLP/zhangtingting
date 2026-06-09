from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import fitz

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.profiles import ProspectusProfile, get_prospectus_profile


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[pipeline {timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full PDF-processing-to-ingestion pipeline for a prospectus profile.")
    parser.add_argument("--profile", type=str, default="prospectus1", help="Prospectus profile id")
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT), help="Project root directory")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python interpreter used to run stage scripts")
    parser.add_argument("--pdf-name", type=str, default="", help="Override PDF filename under data/")
    parser.add_argument("--doc-name", type=str, default="", help="Override logical document name written to storage")
    parser.add_argument("--artifact-dir", type=str, default="", help="Artifact root directory for this PDF")
    parser.add_argument("--parse-version", type=str, default="", help="Parse version tag written to storage")
    parser.add_argument("--text-collection", type=str, default="", help="Milvus text collection name")
    parser.add_argument("--visual-collection", type=str, default="", help="Milvus visual collection name")
    parser.add_argument("--mongo-collection", type=str, default="", help="MongoDB table collection name")
    parser.add_argument("--skip-stage0", action="store_true", help="Skip stage0 page cleanup analysis")
    parser.add_argument("--skip-visual-upload", action="store_true", help="Skip uploading visual crops to MinIO")
    parser.add_argument("--skip-artifact-upload", action="store_true", help="Skip uploading stage artifacts to MinIO")
    parser.add_argument("--recreate-text-collection", action="store_true", help="Drop and recreate the Milvus text collection")
    parser.add_argument("--recreate-visual-collection", action="store_true", help="Drop and recreate the Milvus visual collection")
    parser.add_argument("--overwrite-vlm", action="store_true", help="Overwrite existing VLM result lines during stage2 multimodal execution")
    parser.add_argument("--keep-artifacts", action="store_true", help="Do not clear the artifact directory before running")
    return parser.parse_args()


def ensure_exists(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def resolve_python(raw: str) -> Path:
    candidate = str(raw or "").strip()
    if not candidate:
        raise FileNotFoundError("Python interpreter is empty.")
    direct = Path(candidate)
    if direct.exists():
        return direct.resolve()
    resolved = shutil.which(candidate)
    if resolved:
        return Path(resolved).resolve()
    raise FileNotFoundError(f"Python interpreter not found: {candidate}")


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def clean_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def stage_script_path(project_root: Path, filename: str) -> Path:
    return ensure_exists(project_root / "app" / "pipeline" / "stages" / filename, f"stage script {filename}")


def run_stage0(project_root: Path, pdf_path: Path, output_dir: Path) -> Path:
    module_path = stage_script_path(project_root, "stage0_cleanup.py")
    stage0 = load_module(module_path, f"stage0_runner_{int(time.time() * 1000)}")
    stage0.PROJECT_ROOT = project_root

    output_dir.mkdir(parents=True, exist_ok=True)
    with fitz.open(str(pdf_path)) as doc:
        target_page_indices = list(range(len(doc)))
        preview_pages = [doc.load_page(index) for index in target_page_indices]
        repeated_lines = stage0.collect_repeated_lines(preview_pages)
        position_dense_counter = stage0.collect_position_dense_candidates(preview_pages)
        visual_template_boxes, visual_template_debug = stage0.learn_visual_watermark_template(
            preview_pages,
            learn_pages=min(5, len(preview_pages)),
        )
        watermark_patterns = stage0.load_watermark_patterns()

        results: list[dict[str, Any]] = []
        for index in target_page_indices:
            page = doc.load_page(index)
            stage0_result = stage0.stage0_for_page(
                page,
                repeated_lines,
                watermark_patterns,
                position_dense_counter,
                visual_template_boxes,
            )
            results.append(
                {
                    "page_index": index + 1,
                    "page_size": {
                        "width": float(page.rect.width),
                        "height": float(page.rect.height),
                    },
                    "page_text_preview": stage0_result["filtered_text_preview"][:500],
                    "stage0_result": stage0_result,
                }
            )

    payload = {
        "pdf": str(pdf_path),
        "tested_pages": [index + 1 for index in target_page_indices],
        "visual_watermark_template_boxes": visual_template_boxes,
        "visual_watermark_template_debug": visual_template_debug,
        "results": results,
    }
    output_path = output_dir / "stage0_selected_pages.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def run_stage1(project_root: Path, pdf_path: Path, stage1_output_dir: Path) -> Path:
    module_path = stage_script_path(project_root, "stage1_layout_analysis.py")
    stage1 = load_module(module_path, f"stage1_runner_{int(time.time() * 1000)}")
    stage1.PROJECT_ROOT = project_root
    stage1.CONFIG_PATH = project_root / "config" / "pdf_intelligence_config.json"
    stage1.OUTPUT_DIR = stage1_output_dir
    stage1.DEFAULT_PAGE_RANGE = []
    stage1.resolve_pdf_path = lambda config: pdf_path
    stage1_output_dir.mkdir(parents=True, exist_ok=True)
    stage1.main()
    return ensure_exists(stage1_output_dir / "stage1_layout_analysis.json", "stage1 output")


def run_subprocess(command: list[str], *, cwd: Path, env: dict[str, str], stage_name: str) -> None:
    rendered = " ".join(f'"{part}"' if " " in part else part for part in command)
    log(f"{stage_name} -> {rendered}")
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def append_step(manifest_steps: list[dict[str, Any]], step_name: str, started_at: float, outputs: dict[str, Any] | None = None) -> None:
    manifest_steps.append(
        {
            "step": step_name,
            "elapsed_seconds": round(time.time() - started_at, 3),
            "outputs": outputs or {},
        }
    )


def apply_profile_defaults(args: argparse.Namespace, profile: ProspectusProfile, project_root: Path) -> dict[str, Any]:
    pdf_name = args.pdf_name or profile.pdf_name
    doc_name = args.doc_name or profile.doc_name
    artifact_dir = Path(args.artifact_dir).resolve() if args.artifact_dir else profile.default_artifact_dir(project_root)
    parse_version = args.parse_version or profile.parse_version
    text_collection = args.text_collection or profile.text_collection_name
    visual_collection = args.visual_collection or profile.visual_collection_name
    mongo_collection = args.mongo_collection or profile.mongo_collection_name
    return {
        "pdf_name": pdf_name,
        "doc_name": doc_name,
        "artifact_dir": artifact_dir,
        "parse_version": parse_version,
        "text_collection": text_collection,
        "visual_collection": visual_collection,
        "mongo_collection": mongo_collection,
    }


def main() -> None:
    args = parse_args()
    started_at = time.time()

    project_root = Path(args.project_root).resolve()
    python_exe = resolve_python(args.python)
    profile = get_prospectus_profile(args.profile)
    options = apply_profile_defaults(args, profile, project_root)

    pdf_path = ensure_exists(project_root / "data" / options["pdf_name"], "prospectus PDF")
    artifact_dir = options["artifact_dir"]
    stage0_output_dir = artifact_dir / "stage0_cleanup"
    stage1_output_dir = artifact_dir / "stage1_layout"
    stage1_output_path = stage1_output_dir / "stage1_layout_analysis.json"
    stage5_manifest_path = artifact_dir / "stage5_persist_manifest.json"
    pipeline_manifest_path = artifact_dir / f"{profile.profile_id}_full_pipeline_manifest.json"

    if not args.keep_artifacts:
        log(f"clearing artifact dir: {artifact_dir}")
        clean_directory(artifact_dir)
    else:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        stage0_output_dir.mkdir(parents=True, exist_ok=True)
        stage1_output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["NLP_RAG_PROJECT_ROOT"] = str(project_root)

    manifest: dict[str, Any] = {
        "profile": profile.profile_id,
        "project_root": str(project_root),
        "pdf_path": str(pdf_path),
        "doc_name": options["doc_name"],
        "artifact_dir": str(artifact_dir),
        "parse_version": options["parse_version"],
        "text_collection": options["text_collection"],
        "visual_collection": options["visual_collection"],
        "mongo_collection": options["mongo_collection"],
        "python": str(python_exe),
        "steps": [],
    }

    log(f"profile={profile.profile_id}")
    log(f"project_root={project_root}")
    log(f"pdf_path={pdf_path}")
    log(f"artifact_dir={artifact_dir}")
    log(f"text_collection={options['text_collection']}")
    log(f"visual_collection={options['visual_collection']}")
    log(f"mongo_collection={options['mongo_collection']}")

    if not args.skip_stage0:
        step_started = time.time()
        stage0_output_path = run_stage0(project_root, pdf_path, stage0_output_dir)
        append_step(manifest["steps"], "stage0_cleanup", step_started, {"stage0_output_path": str(stage0_output_path)})

    step_started = time.time()
    run_stage1(project_root, pdf_path, stage1_output_dir)
    append_step(manifest["steps"], "stage1_layout", step_started, {"stage1_output_path": str(stage1_output_path)})

    step_started = time.time()
    run_subprocess(
        [
            str(python_exe),
            str(stage_script_path(project_root, "stage2_precise_extraction.py")),
            "--project-root",
            str(project_root),
            "--stage1-path",
            str(stage1_output_path),
            "--output-dir",
            str(artifact_dir),
        ],
        cwd=project_root,
        env=env,
        stage_name="stage2_precise_extraction",
    )
    append_step(
        manifest["steps"],
        "stage2_precise_extraction",
        step_started,
        {
            "manifest_path": str(artifact_dir / "manifest.json"),
            "multimodal_tasks_path": str(artifact_dir / "multimodal_tasks.jsonl"),
        },
    )

    step_started = time.time()
    vlm_command = [str(python_exe), str(stage_script_path(project_root, "stage2_multimodal_runner.py")), "--artifact-dir", str(artifact_dir)]
    if args.overwrite_vlm:
        vlm_command.append("--overwrite")
    run_subprocess(vlm_command, cwd=project_root, env=env, stage_name="stage2_multimodal")
    append_step(manifest["steps"], "stage2_multimodal", step_started, {"vlm_results_path": str(artifact_dir / "vlm_results.jsonl")})

    step_started = time.time()
    run_subprocess(
        [str(python_exe), str(stage_script_path(project_root, "stage3_text_chunking.py")), "--artifact-dir", str(artifact_dir)],
        cwd=project_root,
        env=env,
        stage_name="stage3_text_chunking",
    )
    append_step(manifest["steps"], "stage3_text_chunking", step_started, {"chunk_file": str(artifact_dir / "stage3_text_chunking" / "text_chunks.jsonl")})

    step_started = time.time()
    run_subprocess(
        [str(python_exe), str(stage_script_path(project_root, "stage4_vectorize_text.py")), "--artifact-dir", str(artifact_dir)],
        cwd=project_root,
        env=env,
        stage_name="stage4_vectorize_text",
    )
    append_step(manifest["steps"], "stage4_vectorize_text", step_started, {"vector_dir": str(artifact_dir / "stage4_vectorized_chunks")})

    step_started = time.time()
    run_subprocess(
        [str(python_exe), str(stage_script_path(project_root, "stage4_vectorize_visual.py")), "--artifact-dir", str(artifact_dir)],
        cwd=project_root,
        env=env,
        stage_name="stage4_vectorize_visual",
    )
    append_step(manifest["steps"], "stage4_vectorize_visual", step_started, {"vector_dir": str(artifact_dir / "stage4_vectorized_visuals")})

    step_started = time.time()
    stage5_command = [
        str(python_exe),
        str(stage_script_path(project_root, "stage5_persist_outputs.py")),
        "--artifact-dir",
        str(artifact_dir),
        "--output-manifest",
        str(stage5_manifest_path),
        "--doc-name",
        options["doc_name"],
        "--parse-version",
        options["parse_version"],
        "--text-collection",
        options["text_collection"],
        "--visual-collection",
        options["visual_collection"],
        "--mongo-collection",
        options["mongo_collection"],
    ]
    if args.skip_visual_upload:
        stage5_command.append("--skip-visual-upload")
    if args.skip_artifact_upload:
        stage5_command.append("--skip-artifact-upload")
    if args.recreate_text_collection:
        stage5_command.append("--recreate-text-collection")
    if args.recreate_visual_collection:
        stage5_command.append("--recreate-visual-collection")
    run_subprocess(stage5_command, cwd=project_root, env=env, stage_name="stage5_persist")
    append_step(manifest["steps"], "stage5_persist", step_started, {"stage5_manifest_path": str(stage5_manifest_path)})

    manifest["total_elapsed_seconds"] = round(time.time() - started_at, 3)
    pipeline_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"pipeline manifest written: {pipeline_manifest_path}")
    log(f"total elapsed={manifest['total_elapsed_seconds']}s")


if __name__ == "__main__":
    main()
