from __future__ import annotations

import sys
from pathlib import Path

POSE_DIR = Path(__file__).resolve().parent
REPO_ROOT = POSE_DIR.parent
EXTERNAL_ROOT = REPO_ROOT / "external"
SCENES_ROOT = REPO_ROOT / "scenes" / "pose"
OUTPUTS_ROOT = REPO_ROOT / "outputs" / "pose"
DROT_ROOT = EXTERNAL_ROOT / "DROT"
PRDPT_ROOT = EXTERNAL_ROOT / "prdpt"


def setup_python_paths() -> None:
    for path in (POSE_DIR, DROT_ROOT, PRDPT_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

