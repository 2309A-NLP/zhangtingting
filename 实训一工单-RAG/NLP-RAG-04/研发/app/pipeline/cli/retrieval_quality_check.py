from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.pipeline.profiles import get_prospectus_profile


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Unified retrieval quality check entry for all prospectus profiles.")
    parser.add_argument("--profile", type=str, default="prospectus1", help="Prospectus profile id")
    parser.add_argument("--artifact-dir", type=str, default="", help="Optional artifact dir override")
    return parser.parse_known_args()


def has_cli_arg(argv: list[str], name: str) -> bool:
    return name in argv


def main() -> None:
    args, passthrough = parse_args()
    profile = get_prospectus_profile(args.profile)

    settings.collection_name = profile.collection_name
    settings.text_vector_collection_name = profile.text_collection_name
    settings.visual_vector_collection_name = profile.visual_collection_name
    settings.mongodb_table_collection = profile.mongo_collection_name

    target_script = PROJECT_ROOT / "app" / "retrieval" / "quality_check.py"
    passthrough = list(passthrough or [])
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    if not has_cli_arg(passthrough, "--artifact-dir"):
        artifact_dir = Path(args.artifact_dir).resolve() if args.artifact_dir else profile.default_artifact_dir(PROJECT_ROOT)
        passthrough.extend(["--artifact-dir", str(artifact_dir)])

    sys.argv = [str(target_script), *passthrough]
    runpy.run_path(str(target_script), run_name="__main__")


if __name__ == "__main__":
    main()
