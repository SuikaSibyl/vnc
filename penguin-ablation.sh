#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

VIDEO_PATH="scenes/pose/gugugaga/videos/output.mp4"
OUTPUT_ROOT="outputs/pose/penguin_ablation"
TRAJECTORY_FPS="${TRAJECTORY_FPS:-12}"

if [[ "$#" -gt 0 ]]; then
  FRAME_BUDGETS=("$@")
else
  FRAME_BUDGETS=(1 2 4 8 16 32)
fi

mkdir -p "$OUTPUT_ROOT"

run_penguin_ablation() {
  local label="$1"
  local max_frames="$2"
  local output_dir="$OUTPUT_ROOT/$label"

  rm -rf "$output_dir"

  python pose/gugugaga.py \
    --output-dir "$output_dir" \
    --methods video_nvdiffrast \
    --video-path "$VIDEO_PATH" \
    --video-max-frames "$max_frames" \
    --init-offset -0.25 \
    --target-offset 0.25 \
    --save-trajectory-video \
    --trajectory-fps "$TRAJECTORY_FPS"
}

for n in "${FRAME_BUDGETS[@]}"; do
  run_penguin_ablation "frames_${n}" "$n"
done

python - <<'PY' "$OUTPUT_ROOT" "${FRAME_BUDGETS[@]}"
import json
import sys
from pathlib import Path

base_dir = Path(sys.argv[1])
budgets = sys.argv[2:]
methods = {}
for budget in budgets:
    run_dir = base_dir / f"frames_{budget}"
    summary_path = run_dir / "optimization_time_summary.json"
    method_json_path = run_dir / "video_nvdiffrast.json"
    if method_json_path.exists():
        data = json.loads(method_json_path.read_text())
        total_time_sec = data.get("total_time_sec")
        if isinstance(total_time_sec, (int, float)):
            methods[run_dir.name] = float(total_time_sec)
    elif summary_path.exists():
        data = json.loads(summary_path.read_text())
        maybe = data.get("methods", {}).get("video_nvdiffrast")
        if isinstance(maybe, (int, float)):
            methods[run_dir.name] = float(maybe)

summary = {"methods": dict(sorted(methods.items()))}
summary_path = base_dir / "penguin_ablation_time_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"Saved: {summary_path}")
PY
