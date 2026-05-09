#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_DIR="$SCRIPT_DIR/../scenes/material/teapot"
OUTPUT_ROOT="$SCRIPT_DIR/../outputs/material/teapot"
PYTHON_BIN="${PYTHON_BIN:-/home/haolin/.conda/envs/torch28/bin/python}"
RENDER_INTERVAL="${RENDER_INTERVAL:-1}"

mkdir -p "$OUTPUT_ROOT"

# rm -rf "$OUTPUT_ROOT/static"

# "$PYTHON_BIN" "$SCRIPT_DIR/teapot/render.py" \
#   --scene "$SCENE_DIR/scene.xml" \
#   --reference-mode static \
#   --variant cuda_ad_rgb \
#   --save-trajectory-video \
#   --trajectory-fps 12 \
#   --render-interval "$RENDER_INTERVAL" \
#   --out-dir "$OUTPUT_ROOT/static"

rm -rf "$OUTPUT_ROOT/video_gen"

"$PYTHON_BIN" "$SCRIPT_DIR/teapot/render.py" \
  --scene "$SCENE_DIR/scene.xml" \
  --reference-mode video \
  --video-path "$SCENE_DIR/videos/output.mp4" \
  --variant cuda_ad_rgb \
  --save-trajectory-video \
  --video-dedup-threshold 0.0005 \
  --trajectory-fps 12 \
  --render-interval "$RENDER_INTERVAL" \
  --out-dir "$OUTPUT_ROOT/video_gen"

# "$PYTHON_BIN" - <<'PY' "$OUTPUT_ROOT"
# import json
# import sys
# from pathlib import Path

# base_dir = Path(sys.argv[1])
# methods = {}
# for run_dir in (base_dir / "static", base_dir / "video_gen"):
#     summary_path = run_dir / "summary.json"
#     if not summary_path.exists():
#         continue
#     data = json.loads(summary_path.read_text())
#     total_time_sec = data.get("total_time_sec")
#     if isinstance(total_time_sec, (int, float)):
#         methods[run_dir.name] = float(total_time_sec)

# summary = {"methods": dict(sorted(methods.items()))}
# summary_path = base_dir / "teapot_optimization_time_summary.json"
# summary_path.write_text(json.dumps(summary, indent=2))
# print(f"Saved: {summary_path}")
# PY
