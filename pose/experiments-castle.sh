#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="outputs/pose/castle"
VIDEO_PATH="scenes/pose/castle/videos/output.mp4"
TRAJECTORY_FPS="${TRAJECTORY_FPS:-12}"

run_castle() {
  python pose/castle.py "$@"
}

for sigma in 1 10 20 30 40 50 70 90; do
  args=(
    --output-dir "$OUTPUT_DIR" \
    --methods prdpt \
    --prdpt-sigma "$sigma" \
    --prdpt-sigma-min 1 \
    --prdpt-nsamples 64
  )

  if [[ "$sigma" == "70" ]]; then
    args+=(
      --save-trajectory-video
      --trajectory-fps "$TRAJECTORY_FPS"
    )
  fi

  run_castle "${args[@]}"
done

run_castle \
  --output-dir "$OUTPUT_DIR" \
  --methods nvdiffrast pytorch3d_sil_rgb rgbxy rgbxy_pytorch3d video_nvdiffrast \
  --video-path "$VIDEO_PATH" \
  --save-trajectory-video \
  --trajectory-fps "$TRAJECTORY_FPS"
