#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/skeleton/uprock}"
TRAJECTORY_FPS="${TRAJECTORY_FPS:-12}"
TRAJECTORY_EVERY="${TRAJECTORY_EVERY:-10}"
METHODS=(${METHODS:-video+rgbxy})

"$PYTHON_BIN" skeleton/optimize_skeleton_coeffs.py \
  --asset scenes/skeleton/uprock_surface.npz \
  --output-dir "$OUTPUT_DIR" \
  --methods "${METHODS[@]}" \
  --reference-image scenes/skeleton/uprock/target.png \
  --video-path scenes/skeleton/uprock/videos/output.mp4 \
  --azimuth 45 \
  --distance-scale 1.8 \
  --render-size 512 \
  --image-size 128 \
  --steps 800 \
  --lr 0.04 \
  --save-every 100 \
  --trajectory-every "$TRAJECTORY_EVERY" \
  --trajectory-fps "$TRAJECTORY_FPS"
