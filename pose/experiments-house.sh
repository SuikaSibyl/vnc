#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python pose/house.py \
  --methods prdpt \
  --prdpt-sigma 1 \
  --prdpt-sigma-min 1 \
  --prdpt-nsamples 64

python pose/house.py \
  --methods prdpt \
  --prdpt-sigma 10 \
  --prdpt-sigma-min 1 \
  --prdpt-nsamples 64

python pose/house.py \
  --methods prdpt \
  --prdpt-sigma 20 \
  --prdpt-sigma-min 1 \
  --prdpt-nsamples 64

python pose/house.py \
  --methods prdpt \
  --prdpt-sigma 30 \
  --prdpt-sigma-min 1 \
  --prdpt-nsamples 64 \
  --save-trajectory-video \
  --trajectory-fps 12

python pose/house.py \
  --methods prdpt \
  --prdpt-sigma 40 \
  --prdpt-sigma-min 1 \
  --prdpt-nsamples 64

python pose/house.py \
  --methods prdpt \
  --prdpt-sigma 50 \
  --prdpt-sigma-min 1 \
  --prdpt-nsamples 64

python pose/house.py \
  --methods prdpt \
  --prdpt-sigma 70 \
  --prdpt-sigma-min 1 \
  --prdpt-nsamples 64

python pose/house.py \
  --methods prdpt \
  --prdpt-sigma 90 \
  --prdpt-sigma-min 1 \
  --prdpt-nsamples 64

python pose/house.py \
  --methods nvdiffrast pytorch3d_sil_rgb rgbxy rgbxy_pytorch3d video_nvdiffrast \
  --video-path scenes/pose/house/videos/output.mp4 \
  --save-trajectory-video \
  --trajectory-fps 12