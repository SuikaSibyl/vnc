#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_OUT="${BASE_OUT:-outputs/pose/gnome}"
VIDEO_PATH="${VIDEO_PATH:-scenes/pose/gnome/videos/output.mp4}"
TRAJECTORY_FPS="${TRAJECTORY_FPS:-12}"
TRAJECTORY_EVERY="${TRAJECTORY_EVERY:-10}"

COMMON_ARGS=(
  --init-rotation 0 0 100
  --n-iters 1000
  --log-every 100
  --n-samples 8
  --lr 0.06
  --translation-lr 0.0005
  --silhouette-weight 5.0
  --save-trajectory-video
  --trajectory-fps "$TRAJECTORY_FPS"
  --trajectory-every "$TRAJECTORY_EVERY"
)

python pose/gnome.py \
  --output-dir "${BASE_OUT}/nvdiffrecmc" \
  "${COMMON_ARGS[@]}"

python pose/gnome.py \
  --output-dir "${BASE_OUT}/video_nvdiffrecmc" \
  --video-path "$VIDEO_PATH" \
  "${COMMON_ARGS[@]}"
