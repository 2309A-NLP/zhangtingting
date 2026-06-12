from __future__ import annotations

import runpy
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings


DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "prospectus2_parse4_full"
DEFAULT_COLLECTION_NAME = "prospectus2_chunks_04"
DEFAULT_TEXT_COLLECTION = "prospectus2_chunks_04_text"
DEFAULT_VISUAL_COLLECTION = "prospectus2_chunks_04_visual"
DEFAULT_MONGO_COLLECTION = "prospectus2_tables_04"


def has_cli_arg(name: str) -> bool:
    return name in sys.argv[1:]


def main() -> None:
    settings.collection_name = DEFAULT_COLLECTION_NAME
    settings.text_vector_collection_name = DEFAULT_TEXT_COLLECTION
    settings.visual_vector_collection_name = DEFAULT_VISUAL_COLLECTION
    settings.mongodb_table_collection = DEFAULT_MONGO_COLLECTION

    if not has_cli_arg("--artifact-dir"):
        sys.argv.extend(["--artifact-dir", str(DEFAULT_ARTIFACT_DIR)])

    target_script = CURRENT_DIR / "simple_retrieval_quality_check.py"
    runpy.run_path(str(target_script), run_name="__main__")


if __name__ == "__main__":
    main()
