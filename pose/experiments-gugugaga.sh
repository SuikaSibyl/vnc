#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="outputs/pose/gugugaga"
VIDEO_PATH="scenes/pose/gugugaga/videos/output.mp4"
TRAJECTORY_FPS="${TRAJECTORY_FPS:-12}"

run_gugugaga() {
  python pose/gugugaga.py "$@"
}

for sigma in 0.1 0.3 0.5 1.0; do
  args=(
    --output-dir "$OUTPUT_DIR"
    --methods prdpt
    --prdpt-sigma "$sigma"
    --prdpt-sigma-min 0.01
    --prdpt-nsamples 64
    --init-offset -0.25
    --target-offset 0.25
  )

  if [[ "$sigma" == "1.0" ]]; then
    args+=(
      --save-trajectory-video
      --trajectory-fps "$TRAJECTORY_FPS"
    )
  fi

  run_gugugaga "${args[@]}"
done

run_gugugaga \
  --output-dir "$OUTPUT_DIR" \
  --methods nvdiffrast pytorch3d_sil_rgb rgbxy rgbxy_pytorch3d video_nvdiffrast \
  --video-path "$VIDEO_PATH" \
  --init-offset -0.25 \
  --target-offset 0.25 \
  --save-trajectory-video \
  --trajectory-fps "$TRAJECTORY_FPS"
