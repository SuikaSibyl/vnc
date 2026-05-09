#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_DIR="$SCRIPT_DIR/../scenes/lights/floor"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/../outputs/lights/floor}"
PYTHON_BIN="${PYTHON_BIN:-/home/haolin/.conda/envs/drot/bin/python}"
TRAJECTORY_FPS="${TRAJECTORY_FPS:-12}"
TRAJECTORY_EVERY="${TRAJECTORY_EVERY:-10}"
PRDPT_SIGMAS="${PRDPT_SIGMAS:-0.05 0.1 0.3 0.5 1.0 2.0}"

mkdir -p "$OUTPUT_ROOT"

COMMON_ARGS=(
  --output-dir "$OUTPUT_ROOT"
  --n-iters 400
  --resolution 128
  --lr 0.1
  --trajectory-fps "$TRAJECTORY_FPS"
  --trajectory-every "$TRAJECTORY_EVERY"
)

"$PYTHON_BIN" "$SCRIPT_DIR/floor.py" \
  "${COMMON_ARGS[@]}" \
  --methods mse rgbxy loir

"$PYTHON_BIN" "$SCRIPT_DIR/floor.py" \
  "${COMMON_ARGS[@]}" \
  --methods video \
  --video-path "$SCENE_DIR/videos/output.mp4" \
  --video-ratio 0.95

for sigma in $PRDPT_SIGMAS; do
  sigma_tag="${sigma/./p}"
  "$PYTHON_BIN" "$SCRIPT_DIR/floor.py" \
    "${COMMON_ARGS[@]}" \
    --methods prdpt \
    --prdpt-sigma "$sigma" \
    --prdpt-sigma-min 0.0 \
    --prdpt-nsamples 8 \
    --output-dir "$OUTPUT_ROOT/prdpt_sigma${sigma_tag}"
done

"$PYTHON_BIN" - <<'PY' "$OUTPUT_ROOT"
import json
import sys
from pathlib import Path

base_dir = Path(sys.argv[1])
methods = {}
for path in sorted(base_dir.rglob("summary.json")):
    data = json.loads(path.read_text())
    total_time_sec = data.get("total_time_sec")
    if not isinstance(total_time_sec, (int, float)):
        continue
    rel = path.parent.relative_to(base_dir).as_posix()
    methods[rel] = float(total_time_sec)

summary = {"methods": dict(sorted(methods.items()))}
summary_path = base_dir / "floor_optimization_time_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"Saved: {summary_path}")
PY
