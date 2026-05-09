#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-outputs/pose/mug}"
VIDEO_PATH="${VIDEO_PATH:-scenes/pose/mug/videos/output.mp4}"
TRAJECTORY_FPS="${TRAJECTORY_FPS:-12}"
TRAJECTORY_EVERY="${TRAJECTORY_EVERY:-10}"

COMMON_ARGS=(
  --output-dir "$OUT_DIR"
  --init-deg 30.0
  --target-deg -30.0
  --save-trajectory-video
  --trajectory-fps "$TRAJECTORY_FPS"
  --trajectory-every "$TRAJECTORY_EVERY"
)

python pose/mug.py \
  "${COMMON_ARGS[@]}" \
  --methods nvdiffrast video_nvdiffrast \
  --video-path "$VIDEO_PATH"

python pose/mug.py \
  "${COMMON_ARGS[@]}" \
  --methods prdpt \
  --prdpt-sigma 10 \
  --prdpt-sigma-min 1.0 \
  --prdpt-nsamples 16

python pose/mug.py \
  "${COMMON_ARGS[@]}" \
  --methods prdpt \
  --prdpt-sigma 5 \
  --prdpt-sigma-min 1.0 \
  --prdpt-nsamples 16

python pose/mug.py \
  "${COMMON_ARGS[@]}" \
  --methods prdpt+video \
  --video-path "$VIDEO_PATH" \
  --prdpt-sigma 5 \
  --prdpt-sigma-min 1.0 \
  --prdpt-nsamples 16
