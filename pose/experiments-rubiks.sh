#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-outputs/pose/rubiks}"
VIDEO_PATH="${VIDEO_PATH:-scenes/pose/rubiks/videos/output.mp4}"
TRAJECTORY_FPS="${TRAJECTORY_FPS:-12}"
PRDPT_SIGMAS="${PRDPT_SIGMAS:-10}"

COMMON_ARGS=(
  --output-dir "$OUT_DIR"
  --video-path "$VIDEO_PATH"
  --save-trajectory-video
  --trajectory-fps "$TRAJECTORY_FPS"
)

python pose/rubiks.py \
  "${COMMON_ARGS[@]}" \
  --methods nvdiffrast pytorch3d_sil_rgb rgbxy rgbxy_pytorch3d video_nvdiffrast

for SIGMA in $PRDPT_SIGMAS; do
  python pose/rubiks.py \
    "${COMMON_ARGS[@]}" \
    --methods prdpt \
    --prdpt-sigma "$SIGMA" \
    --prdpt-sigma-min 1 \
    --prdpt-nsamples 64
done
