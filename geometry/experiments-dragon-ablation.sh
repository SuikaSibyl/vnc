#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_DIR="$SCRIPT_DIR/../scenes/geometry/dragon"
OUTPUT_ROOT="$SCRIPT_DIR/../outputs/geometry"
PYTHON_BIN="${PYTHON_BIN:-/home/haolin/.conda/envs/torch28/bin/python}"

mkdir -p "$OUTPUT_ROOT"

run_video_ablation() {
  local name="$1"
  local video_path="$2"
  local output_dir="$OUTPUT_ROOT/dragon_video_${name}"

  rm -rf "$output_dir"

  "$PYTHON_BIN" "$SCRIPT_DIR/optmizer.py" \
    --scene-dir "$SCENE_DIR" \
    --view-idx 19 \
    --render-interval 50 \
    --steps 8999 \
    --remesh-interval 1000 \
    --video-path "$video_path" \
    --video-ratio 0.8 \
    --gamma 2.2 \
    --save-trajectory-video \
    --trajectory-fps 12 \
    --output-dir "$output_dir"
}

run_video_ablation "seedance" "$SCENE_DIR/videos/seedance.mp4"
run_video_ablation "veo3" "$SCENE_DIR/videos/veo3.mp4"
run_video_ablation "wan" "$SCENE_DIR/videos/wan.mp4"

"$PYTHON_BIN" - <<'PY' "$OUTPUT_ROOT"
import json
import sys
from pathlib import Path

base_dir = Path(sys.argv[1])
methods = {}
for run_dir in (
    base_dir / "dragon_video_seedance",
    base_dir / "dragon_video_veo3",
    base_dir / "dragon_video_wan",
):
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        continue
    data = json.loads(summary_path.read_text())
    total_time_sec = data.get("total_time_sec")
    if isinstance(total_time_sec, (int, float)):
        methods[run_dir.name] = float(total_time_sec)

summary = {"methods": dict(sorted(methods.items()))}
summary_path = base_dir / "dragon_ablation_optimization_time_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"Saved: {summary_path}")
PY
