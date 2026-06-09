from __future__ import annotations

import runpy
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent


def main() -> None:
    target_script = CURRENT_DIR / "backend" / "pipeline" / "cli" / "retrieval_quality_check.py"
    sys.argv = [str(target_script), *sys.argv[1:]]
    runpy.run_path(str(target_script), run_name="__main__")


if __name__ == "__main__":
    main()
