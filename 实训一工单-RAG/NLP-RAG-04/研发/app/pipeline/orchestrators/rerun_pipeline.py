from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.profiles import get_prospectus_profile


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[rerun {timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun multimodal, vectorization, and persistence stages for a prospectus profile.")
    parser.add_argument("--profile", type=str, default="prospectus1", help="Prospectus profile id")
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT), help="Project root directory")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python interpreter used to run stage scripts")
    parser.add_argument("--artifact-dir", type=str, default="", help="Artifact root directory for the profile")
    parser.add_argument("--doc-name", type=str, default="", help="Logical document name written to storage")
    parser.add_argument("--parse-version", type=str, default="", help="Parse version tag written to storage")
    parser.add_argument("--text-collection", type=str, default="", help="Milvus text collection name")
    parser.add_argument("--visual-collection", type=str, default="", help="Milvus visual collection name")
    parser.add_argument("--mongo-collection", type=str, default="", help="MongoDB table collection name")
    parser.add_argument("--overwrite-vlm", action="store_true", help="Overwrite existing VLM result lines")
    parser.add_argument("--recreate-text-collection", action="store_true", help="Drop and recreate the Milvus text collection")
    parser.add_argument("--recreate-visual-collection", action="store_true", help="Drop and recreate the Milvus visual collection")
    parser.add_argument("--skip-visual-upload", action="store_true", help="Skip uploading visual crops to MinIO")
    parser.add_argument("--skip-artifact-upload", action="store_true", help="Skip uploading artifacts to MinIO")
    return parser.parse_args()


def resolve_python(raw: str) -> Path:
    candidate = str(raw or "").strip()
    direct = Path(candidate)
    if direct.exists():
        return direct.resolve()
    resolved = shutil.which(candidate)
    if resolved:
        return Path(resolved).resolve()
    raise FileNotFoundError(f"Python interpreter not found: {candidate}")


def ensure_exists(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def stage_script_path(project_root: Path, filename: str) -> Path:
    return ensure_exists(project_root / "app" / "pipeline" / "stages" / filename, f"stage script {filename}")


def run_subprocess(command: list[str], *, cwd: Path, env: dict[str, str], stage_name: str) -> None:
    rendered = " ".join(f'"{part}"' if " " in part else part for part in command)
    log(f"{stage_name} -> {rendered}")
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def main() -> None:
    args = parse_args()
    started_at = time.time()

    project_root = Path(args.project_root).resolve()
    python_exe = resolve_python(args.python)
    profile = get_prospectus_profile(args.profile)
    artifact_dir = Path(args.artifact_dir).resolve() if args.artifact_dir else profile.default_artifact_dir(project_root)
    doc_name = args.doc_name or profile.doc_name
    parse_version = args.parse_version or profile.parse_version
    text_collection = args.text_collection or profile.text_collection_name
    visual_collection = args.visual_collection or profile.visual_collection_name
    mongo_collection = args.mongo_collection or profile.mongo_collection_name

    ensure_exists(artifact_dir, "artifact dir")
    ensure_exists(artifact_dir / "multimodal_tasks.jsonl", "multimodal tasks")
    ensure_exists(artifact_dir / "manifest.json", "stage2 manifest")

    env = os.environ.copy()
    env["NLP_RAG_PROJECT_ROOT"] = str(project_root)

    log(f"profile={profile.profile_id}")
    log(f"project_root={project_root}")
    log(f"artifact_dir={artifact_dir}")
    log(f"text_collection={text_collection}")
    log(f"visual_collection={visual_collection}")
    log(f"mongo_collection={mongo_collection}")

    vlm_command = [str(python_exe), str(stage_script_path(project_root, "stage2_multimodal_runner.py")), "--artifact-dir", str(artifact_dir)]
    if args.overwrite_vlm:
        vlm_command.append("--overwrite")
    run_subprocess(vlm_command, cwd=project_root, env=env, stage_name="stage2_multimodal")

    run_subprocess([str(python_exe), str(stage_script_path(project_root, "stage3_text_chunking.py")), "--artifact-dir", str(artifact_dir)], cwd=project_root, env=env, stage_name="stage3_text_chunking")
    run_subprocess([str(python_exe), str(stage_script_path(project_root, "stage4_vectorize_text.py")), "--artifact-dir", str(artifact_dir)], cwd=project_root, env=env, stage_name="stage4_vectorize_text")
    run_subprocess([str(python_exe), str(stage_script_path(project_root, "stage4_vectorize_visual.py")), "--artifact-dir", str(artifact_dir)], cwd=project_root, env=env, stage_name="stage4_vectorize_visual")

    stage5_command = [
        str(python_exe),
        str(stage_script_path(project_root, "stage5_persist_outputs.py")),
        "--artifact-dir",
        str(artifact_dir),
        "--doc-name",
        doc_name,
        "--parse-version",
        parse_version,
        "--text-collection",
        text_collection,
        "--visual-collection",
        visual_collection,
        "--mongo-collection",
        mongo_collection,
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

    log(f"total_elapsed={round(time.time() - started_at, 3)}s")


if __name__ == "__main__":
    main()
