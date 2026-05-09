#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/haolin/.conda/envs/drot/bin/python}"
OUTPUT_ROOT="$REPO_ROOT/outputs/pose/furniture"

# mkdir -p "$OUTPUT_ROOT"

# "$PYTHON_BIN" pose/furniture.py 2-nvdiffrast \
#   --tag nvdiffrast \
#   --summary-name nvdiffrast_timing.json

# "$PYTHON_BIN" pose/furniture.py 2-rgbxy \
#   --tag rgbxy \
#   --summary-name rgbxy_timing.json

# "$PYTHON_BIN" pose/furniture.py 2-loir \
#   --tag loir \
#   --summary-name loir_timing.json

"$PYTHON_BIN" pose/furniture.py 2-video \
  --tag video_nvdiffrast \
  --summary-name video_nvdiffrast_timing.json

# "$PYTHON_BIN" pose/furniture.py 2 \
#   --tag prdpt \
#   --summary-name prdpt_timing.json

# "$PYTHON_BIN" - <<'PY' "$OUTPUT_ROOT"
# import json
# import sys
# from pathlib import Path

# base_dir = Path(sys.argv[1])
# summary_files = [
#     ("nvdiffrast", base_dir / "nvdiffrast_timing.json"),
#     ("rgbxy", base_dir / "rgbxy_timing.json"),
#     ("loir", base_dir / "loir_timing.json"),
#     ("video_nvdiffrast", base_dir / "video_nvdiffrast_timing.json"),
#     ("prdpt", base_dir / "prdpt_timing.json"),
# ]

# methods = {}
# for label, path in summary_files:
#     if not path.exists():
#         continue
#     data = json.loads(path.read_text())
#     if isinstance(data, dict):
#         method_map = data.get("methods", {})
#         if len(method_map) == 1:
#             methods[label] = float(next(iter(method_map.values())))
#         elif label in method_map:
#             methods[label] = float(method_map[label])

# summary = {"methods": dict(sorted(methods.items()))}
# summary_path = base_dir / "furniture_optimization_time_summary.json"
# summary_path.write_text(json.dumps(summary, indent=2))
# print(f"Saved: {summary_path}")
# PY
