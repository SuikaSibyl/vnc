#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_DIR="$SCRIPT_DIR/../scenes/geometry/suzanne"
OUTPUT_ROOT="$SCRIPT_DIR/../outputs/geometry"
PYTHON_BIN="${PYTHON_BIN:-/home/haolin/.conda/envs/torch28/bin/python}"

mkdir -p "$OUTPUT_ROOT"

rm -rf "$OUTPUT_ROOT/suzanne_multiview"

"$PYTHON_BIN" "$SCRIPT_DIR/optmizer.py" \
  --scene-dir "$SCENE_DIR" \
  --view-idxs 1 5 9 \
  --render-interval 25 \
  --steps 4000 \
  --remesh-interval 2000 \
  --optimize-translation \
  --lambda 23 \
  --video-ratio 0.2 \
  --step-size 0.01 \
  --translation-step-size 0.02 \
  --gamma 2.2 \
  --save-trajectory-video \
  --trajectory-fps 12 \
  --output-dir "$OUTPUT_ROOT/suzanne_multiview"

rm -rf "$OUTPUT_ROOT/suzanne_video_multiview"

"$PYTHON_BIN" "$SCRIPT_DIR/optmizer.py" \
  --scene-dir "$SCENE_DIR" \
  --view-idxs 1 5 9 \
  --render-interval 25 \
  --steps 4000 \
  --remesh-interval 2000 \
  --optimize-translation \
  --lambda 21 \
  --video-paths \
    "$SCENE_DIR/videos/output_view01.mp4" \
    "$SCENE_DIR/videos/output_view05.mp4" \
    "$SCENE_DIR/videos/output_view09.mp4" \
  --video-ratio 0.2 \
  --step-size 0.01 \
  --translation-step-size 0.02 \
  --gamma 2.2 \
  --save-trajectory-video \
  --trajectory-fps 12 \
  --output-dir "$OUTPUT_ROOT/suzanne_video_multiview"

"$PYTHON_BIN" - <<'PY' "$OUTPUT_ROOT"
import json
import sys
from pathlib import Path

base_dir = Path(sys.argv[1])
methods = {}
for run_dir in (base_dir / "suzanne_multiview", base_dir / "suzanne_video_multiview"):
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        continue
    data = json.loads(summary_path.read_text())
    total_time_sec = data.get("total_time_sec")
    if isinstance(total_time_sec, (int, float)):
        methods[run_dir.name] = float(total_time_sec)

summary = {"methods": dict(sorted(methods.items()))}
summary_path = base_dir / "suzanne_optimization_time_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"Saved: {summary_path}")
PY
