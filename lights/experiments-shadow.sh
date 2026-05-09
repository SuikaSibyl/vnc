#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_DIR="$SCRIPT_DIR/../scenes/lights/shadows"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/../outputs/lights/shadows_opt}"
PYTHON_BIN="${PYTHON_BIN:-/home/haolin/.conda/envs/torch28/bin/python}"
TRAJECTORY_FPS="${TRAJECTORY_FPS:-12}"
TRAJECTORY_EVERY="${TRAJECTORY_EVERY:-10}"
PRDPT_SIGMAS="${PRDPT_SIGMAS:-0.05 0.1 0.2 0.4 0.8}"

mkdir -p "$OUTPUT_ROOT"

COMMON_ARGS=(
  --mode optimize
  --out-dir "$OUTPUT_ROOT"
  --opt-width 256
  --opt-height 256
  --trajectory-fps "$TRAJECTORY_FPS"
  --trajectory-every "$TRAJECTORY_EVERY"
)

# "$PYTHON_BIN" "$SCRIPT_DIR/shadows.py" \
#   "${COMMON_ARGS[@]}" \
#   --method mse

# "$PYTHON_BIN" "$SCRIPT_DIR/shadows.py" \
#   "${COMMON_ARGS[@]}" \
#   --method loir

"$PYTHON_BIN" "$SCRIPT_DIR/shadows.py" \
  "${COMMON_ARGS[@]}" \
  --method video \
  --video-path "$SCENE_DIR/videos/output.mp4"

# for sigma in $PRDPT_SIGMAS; do
#   "$PYTHON_BIN" "$SCRIPT_DIR/shadows.py" \
#     "${COMMON_ARGS[@]}" \
#     --method prdpt \
#     --prdpt-sigma "$sigma"
# done

# "$PYTHON_BIN" - <<'PY' "$OUTPUT_ROOT"
# import json
# import sys
# from pathlib import Path

# base_dir = Path(sys.argv[1])
# methods = {}
# for path in sorted(base_dir.glob("*/summary.json")):
#     data = json.loads(path.read_text())
#     total_time_sec = data.get("total_time_sec")
#     if isinstance(total_time_sec, (int, float)):
#         methods[path.parent.name] = float(total_time_sec)

# summary = {"methods": dict(sorted(methods.items()))}
# summary_path = base_dir / "shadows_optimization_time_summary.json"
# summary_path.write_text(json.dumps(summary, indent=2))
# print(f"Saved: {summary_path}")
# PY
