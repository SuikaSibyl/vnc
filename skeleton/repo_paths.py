from __future__ import annotations

import sys
from pathlib import Path

SKELETON_DIR = Path(__file__).resolve().parent
REPO_ROOT = SKELETON_DIR.parent
EXTERNAL_ROOT = REPO_ROOT / "external"
SCENES_ROOT = REPO_ROOT / "scenes" / "skeleton"
OUTPUTS_ROOT = REPO_ROOT / "outputs" / "skeleton"
DROT_ROOT = EXTERNAL_ROOT / "DROT"


def setup_python_paths() -> None:
    for path in (SKELETON_DIR, DROT_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
