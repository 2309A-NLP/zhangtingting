from __future__ import annotations

import runpy
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent


def main() -> None:
    target_script = PROJECT_ROOT / "app" / "pipeline" / "stages" / "stage2_multimodal_runner.py"
    sys.argv = [str(target_script), *sys.argv[1:]]
    runpy.run_path(str(target_script), run_name="__main__")


if __name__ == "__main__":
    main()
