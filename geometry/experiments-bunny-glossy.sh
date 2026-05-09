#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_DIR="$SCRIPT_DIR/../scenes/geometry/bunny"
OUTPUT_ROOT="$SCRIPT_DIR/../outputs/geometry"
PYTHON_BIN="${PYTHON_BIN:-/home/haolin/.conda/envs/torch28/bin/python}"

mkdir -p "$OUTPUT_ROOT"

rm -rf "$OUTPUT_ROOT/bunny_glossy"

"$PYTHON_BIN" "$SCRIPT_DIR/optmizer_glossy.py" \
  --scene-dir "$SCENE_DIR" \
  --view-idxs 5 \
  --steps 1200 \
  --roughness 0.2 \
  --roughness-target 0.9 \
  --roughness-step-size 0.01 \
  --sh-order 4 \
  --render-interval 10 \
  --diffuse-weight 0.4 \
  --specular-weight 0.6 \
  --save-trajectory-video \
  --trajectory-fps 12 \
  --output-dir "$OUTPUT_ROOT/bunny_glossy"

# rm -rf "$OUTPUT_ROOT/bunny_glossy_video"

# "$PYTHON_BIN" "$SCRIPT_DIR/optmizer_glossy.py" \
#   --scene-dir "$SCENE_DIR" \
#   --view-idxs 5 \
#   --steps 1200 \
#   --roughness 0.2 \
#   --roughness-target 0.9 \
#   --roughness-step-size 0.01 \
#   --sh-order 4 \
#   --diffuse-weight 0.4 \
#   --render-interval 10 \
#   --video-path "$SCENE_DIR/videos/output_1.mp4" \
#   --video-ratio 0.5 \
#   --specular-weight 0.6 \
#   --save-trajectory-video \
#   --trajectory-fps 12 \
#   --output-dir "$OUTPUT_ROOT/bunny_glossy_video"

# rm -rf "$OUTPUT_ROOT/bunny_glossy_video2"

# "$PYTHON_BIN" "$SCRIPT_DIR/optmizer_glossy.py" \
#   --scene-dir "$SCENE_DIR" \
#   --view-idxs 5 \
#   --steps 1200 \
#   --roughness 0.2 \
#   --roughness-target 0.9 \
#   --roughness-step-size 0.01 \
#   --sh-order 4 \
#   --diffuse-weight 0.4 \
#   --render-interval 10 \
#   --video-path "$SCENE_DIR/videos/output.mp4" \
#   --video-ratio 0.5 \
#   --specular-weight 0.6 \
#   --save-trajectory-video \
#   --trajectory-fps 12 \
#   --output-dir "$OUTPUT_ROOT/bunny_glossy_video2"

# "$PYTHON_BIN" - <<'PY' "$OUTPUT_ROOT"
# import json
# import sys
# from pathlib import Path

# base_dir = Path(sys.argv[1])
# methods = {}
# for run_dir in (
#     base_dir / "bunny_glossy",
#     base_dir / "bunny_glossy_video",
#     base_dir / "bunny_glossy_video2",
# ):
#     summary_path = run_dir / "summary.json"
#     if not summary_path.exists():
#         continue
#     data = json.loads(summary_path.read_text())
#     total_time_sec = data.get("total_time_sec")
#     if isinstance(total_time_sec, (int, float)):
#         methods[run_dir.name] = float(total_time_sec)

# summary = {"methods": dict(sorted(methods.items()))}
# summary_path = base_dir / "bunny_glossy_optimization_time_summary.json"
# summary_path.write_text(json.dumps(summary, indent=2))
# print(f"Saved: {summary_path}")
# PY