#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VHC_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LEGACY_GEOMETRY_DIR="$VHC_ROOT/codes/geometry"
LEGACY_SHADOW_SCRIPT="$LEGACY_GEOMETRY_DIR/shadow.py"
OUTPUT_ROOT="$SCRIPT_DIR/../outputs/geometry"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAJECTORY_FPS="${TRAJECTORY_FPS:-12}"

mkdir -p "$OUTPUT_ROOT"

run_curve() {
  local output_dir="$1"
  shift

  rm -rf "$output_dir"

  "$PYTHON_BIN" - <<'PY' "$LEGACY_SHADOW_SCRIPT" "$output_dir" "$@" 
import json
import subprocess
import sys
import time
from pathlib import Path

shadow_script = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
args = sys.argv[3:]

cmd = [sys.executable, str(shadow_script), "--output-dir", str(output_dir), *args]
print("Running:", " ".join(cmd))
t_start = time.perf_counter()
subprocess.run(cmd, check=True)
elapsed = time.perf_counter() - t_start

results_path = output_dir / "results.json"
results = {}
if results_path.exists():
    results = json.loads(results_path.read_text())

summary = {
    "name": output_dir.name,
    "total_time_sec": elapsed,
    "results_path": str(results_path),
    "reference_mode": results.get("reference_mode"),
    "video_ratio": results.get("video_ratio"),
    "max_iter": results.get("max_iter"),
}
summary_path = output_dir / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"Saved: {summary_path}")
PY
}

build_videos() {
  local output_dir="$1"

  "$PYTHON_BIN" - <<'PY' "$output_dir" "$TRAJECTORY_FPS"
import re
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

out_dir = Path(sys.argv[1])
fps = int(sys.argv[2])
target_size = (512, 256)


def parse_iter(path: Path) -> int:
    match = re.search(r"_(\d{4})", path.stem)
    if not match:
        return 0
    return int(match.group(1))


def annotate(frame: np.ndarray, iteration: int) -> np.ndarray:
    if frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_CUBIC)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    label = f"iter {iteration}"
    cv2.rectangle(bgr, (8, 8), (128, 42), (0, 0, 0), thickness=-1)
    cv2.putText(bgr, label, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_rgb(path: Path) -> np.ndarray:
    return imageio.imread(path)[..., :3]


def collect_sequence(paths: list[Path]) -> list[tuple[np.ndarray, int]]:
    frames = []
    for path in sorted(paths):
        frames.append((load_rgb(path), parse_iter(path)))
    return frames


def save_video(path: Path, frames: list[tuple[np.ndarray, int]]) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    annotated = [annotate(frame, iteration) for frame, iteration in frames]
    imageio.mimwrite(path, annotated, fps=fps)
    print(f"Saved: {path}")


render_frames = []
init_path = out_dir / "init.png"
if init_path.exists():
    render_frames.append((load_rgb(init_path), 0))
render_frames.extend(collect_sequence(list(out_dir.glob("render_iter_*.png"))))

ref_frames = []
reference0 = out_dir / "reference_iter_0000.png"
if reference0.exists():
    ref_frames.append((load_rgb(reference0), 0))
ref_frames.extend(collect_sequence(list(out_dir.glob("ref_iter_*.png"))))

pair_frames = collect_sequence(list((out_dir / "debug" / "pair").glob("pair_iter_*.png")))
shadow_frames = collect_sequence(list((out_dir / "debug" / "shadow").glob("shadow_iter_*.png")))

save_video(out_dir / "trajectory.mp4", render_frames)
save_video(out_dir / "trajectory_ref.mp4", ref_frames)
save_video(out_dir / "trajectory_pair.mp4", pair_frames)
save_video(out_dir / "trajectory_shadow.mp4", shadow_frames)
PY
}

run_curve \
  "$OUTPUT_ROOT/curve" \
  --max-iter 450 \
  --save-spp 16 \
  --sensor-spp 256 \
  --projective-sppi 1024 \
  --reference-mode split \
  --save-interval 10
build_videos "$OUTPUT_ROOT/curve"

run_curve \
  "$OUTPUT_ROOT/curve_video" \
  --max-iter 450 \
  --save-spp 16 \
  --sensor-spp 256 \
  --video-dedup-threshold 0.002 \
  --projective-sppi 1024 \
  --reference-mode video_then_split \
  --save-interval 10
build_videos "$OUTPUT_ROOT/curve_video"

"$PYTHON_BIN" - <<'PY' "$OUTPUT_ROOT"
import json
import sys
from pathlib import Path

base_dir = Path(sys.argv[1])
methods = {}
for run_dir in (base_dir / "curve", base_dir / "curve_video"):
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        continue
    data = json.loads(summary_path.read_text())
    total_time_sec = data.get("total_time_sec")
    if isinstance(total_time_sec, (int, float)):
        methods[run_dir.name] = float(total_time_sec)

summary = {"methods": dict(sorted(methods.items()))}
summary_path = base_dir / "curve_optimization_time_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"Saved: {summary_path}")
PY
