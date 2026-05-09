#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-outputs/pose/earth}"
VIDEO_PATH="${VIDEO_PATH:-scenes/pose/earth/videos/output.mp4}"
TRAJECTORY_FPS="${TRAJECTORY_FPS:-12}"
TRAJECTORY_RESOLUTION="${TRAJECTORY_RESOLUTION:-512}"
PRDPT_SIGMAS="${PRDPT_SIGMAS:-0.05 0.10 0.20 0.30 0.50}"

COMMON_ARGS=(
  --input scenes/pose/earth/planet_earth.glb
  --output-dir "$OUT_DIR"
  --resolution 256
  --loss-resolution 256
  --n-iters 1000
  --lr 0.01
  --log-every 50
  --seed 0
  --init-rotation 40 0 20
  --init-translation 0.0 2.0 2.0
  --target-rotation 0 -90 0
  --target-translation 0.0 -2.0 -2.0
  --camera-scale 2.5
  --save-trajectory-video
  --trajectory-fps "$TRAJECTORY_FPS"
  --trajectory-resolution "$TRAJECTORY_RESOLUTION"
)

python pose/earth.py \
  "${COMMON_ARGS[@]}" \
  --methods nvdiffrast rgbxy loir video \
  --video-path "$VIDEO_PATH" \
  --video-loss gaussian_pyramid \
  --video-loss-resolution 256 \
  --pyramid-levels 2

for SIGMA in $PRDPT_SIGMAS; do
  python pose/earth.py \
    "${COMMON_ARGS[@]}" \
    --methods prdpt \
    --prdpt-sigma "$SIGMA" \
    --prdpt-sigma-min 0 \
    --prdpt-nsamples 16
done
