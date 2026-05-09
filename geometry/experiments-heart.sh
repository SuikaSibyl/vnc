#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_DIR="$SCRIPT_DIR/../scenes/geometry/heart"
OUTPUT_ROOT="$SCRIPT_DIR/../outputs/geometry"
PYTHON_BIN="${PYTHON_BIN:-/home/haolin/.conda/envs/torch28/bin/python}"

mkdir -p "$OUTPUT_ROOT"

rm -rf "$OUTPUT_ROOT/heart"

"$PYTHON_BIN" "$SCRIPT_DIR/optmizer.py" \
  --scene-dir "$SCENE_DIR" \
  --view-idx 0 \
  --render-interval 50 \
  --steps 8999 \
  --remesh-interval 1000 \
  --gamma 2.2 \
  --black-background \
  --save-trajectory-video \
  --trajectory-fps 12 \
  --output-dir "$OUTPUT_ROOT/heart"

rm -rf "$OUTPUT_ROOT/heart_video"

"$PYTHON_BIN" "$SCRIPT_DIR/optmizer.py" \
  --scene-dir "$SCENE_DIR" \
  --view-idx 0 \
  --render-interval 50 \
  --steps 8999 \
  --remesh-interval 1000 \
  --gamma 2.2 \
  --black-background \
  --video-path "$SCENE_DIR/videos/output.mp4" \
  --video-ratio 0.6 \
  --save-trajectory-video \
  --trajectory-fps 12 \
  --output-dir "$OUTPUT_ROOT/heart_video"

"$PYTHON_BIN" - <<'PY' "$OUTPUT_ROOT"
import json
import sys
from pathlib import Path

base_dir = Path(sys.argv[1])
methods = {}
for run_dir in (base_dir / "heart", base_dir / "heart_video"):
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        continue
    data = json.loads(summary_path.read_text())
    total_time_sec = data.get("total_time_sec")
    if isinstance(total_time_sec, (int, float)):
        methods[run_dir.name] = float(total_time_sec)

summary = {"methods": dict(sorted(methods.items()))}
summary_path = base_dir / "heart_optimization_time_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"Saved: {summary_path}")
PY
