#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-vnc}"
PYTORCH_VERSION="2.10.0"
TORCHVISION_VERSION="0.25.0"
PYTORCH3D_COMMIT="9a3342e1a48d4b7b272f5975291ae3a7e470cd4f"
NVDIFFRAST_COMMIT="253ac4fcea7de5f396371124af597e6cc957bfae"

if [[ ! "$ENV_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid Conda environment name: $ENV_NAME" >&2
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda was not found on PATH." >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "Git was not found on PATH." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REMESH_DIR="$REPO_ROOT/geometry/largesteps/ext/botsch-kobbelt-remesher-libigl/build"

if conda env list | awk 'NF && $1 !~ /^#/ { print $1 }' | grep -Fxq "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' already exists; choose another name or remove it first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

conda create -y -n "$ENV_NAME" \
  -c nvidia -c conda-forge \
  python=3.10 \
  pip \
  git \
  cmake \
  ninja \
  "gcc_linux-64=11" \
  "gxx_linux-64=11" \
  "cuda-toolkit=12.8"

conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  "torch==${PYTORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
  "numpy==2.2.6" \
  "mitsuba==3.8.0" \
  "drjit==1.3.1" \
  "cholespy==2.2.0" \
  "geomloss==0.2.6" \
  fvcore \
  iopath \
  imageio \
  imageio-ffmpeg \
  matplotlib \
  opencv-python \
  pandas \
  pillow \
  scipy \
  scikit-image \
  trimesh \
  rtree \
  embreex \
  tqdm \
  jupyter \
  nbconvert \
  tensorboard \
  pytorch-ignite \
  lpips \
  glfw \
  PyOpenGL \
  xmltodict \
  PyGLM

BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/vnc-build.XXXXXX")"
trap 'rm -rf -- "$BUILD_ROOT"' EXIT

git clone https://github.com/jkxing/pytorch3d.git "$BUILD_ROOT/pytorch3d"
git -C "$BUILD_ROOT/pytorch3d" checkout "$PYTORCH3D_COMMIT"

git clone https://github.com/NVlabs/nvdiffrast.git "$BUILD_ROOT/nvdiffrast"
git -C "$BUILD_ROOT/nvdiffrast" checkout "$NVDIFFRAST_COMMIT"

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0;12.0+PTX}"
export LD_LIBRARY_PATH="$REMESH_DIR:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export VNC_REPO_ROOT="$REPO_ROOT"

python -m pip install --no-build-isolation "$BUILD_ROOT/pytorch3d"
python -m pip install --no-build-isolation "$BUILD_ROOT/nvdiffrast"

ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

{
  printf 'export _VNC_OLD_PYTHONPATH="${PYTHONPATH-}"\n'
  printf 'export _VNC_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"\n'
  printf 'export _VNC_OLD_CUDA_HOME="${CUDA_HOME-}"\n'
  printf 'export _VNC_OLD_TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST-}"\n'
  printf 'export VNC_REPO_ROOT=%q\n' "$REPO_ROOT"
  printf 'export CUDA_HOME=%q\n' "$CONDA_PREFIX"
  printf 'export TORCH_CUDA_ARCH_LIST=%q\n' "$TORCH_CUDA_ARCH_LIST"
  printf 'export PYTHONPATH=%q:%q:"${PYTHONPATH-}"\n' "$REPO_ROOT/external/DROT" "$REPO_ROOT/external/prdpt"
  printf 'export LD_LIBRARY_PATH=%q:%q:"${LD_LIBRARY_PATH-}"\n' "$REMESH_DIR" "$CONDA_PREFIX/lib"
} > "$ACTIVATE_DIR/vnc.sh"

{
  printf 'export PYTHONPATH="${_VNC_OLD_PYTHONPATH-}"\n'
  printf 'export LD_LIBRARY_PATH="${_VNC_OLD_LD_LIBRARY_PATH-}"\n'
  printf 'export CUDA_HOME="${_VNC_OLD_CUDA_HOME-}"\n'
  printf 'export TORCH_CUDA_ARCH_LIST="${_VNC_OLD_TORCH_CUDA_ARCH_LIST-}"\n'
  printf 'unset VNC_REPO_ROOT _VNC_OLD_PYTHONPATH _VNC_OLD_LD_LIBRARY_PATH\n'
  printf 'unset _VNC_OLD_CUDA_HOME _VNC_OLD_TORCH_CUDA_ARCH_LIST\n'
} > "$DEACTIVATE_DIR/vnc.sh"

python - <<'PY'
import sys
from pathlib import Path

import cholespy
import cv2
import drjit
import geomloss
import mitsuba
import nvdiffrast.torch
import pytorch3d
import torch

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access a CUDA GPU; check the NVIDIA driver.")

repo_root = Path(__import__("os").environ["VNC_REPO_ROOT"])
remesh_dir = repo_root / "geometry/largesteps/ext/botsch-kobbelt-remesher-libigl/build"
sys.path.insert(0, str(remesh_dir))
import pyremesh

mitsuba.set_variant("cuda_ad_rgb")
print(f"PyTorch {torch.__version__} / CUDA {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Mitsuba {mitsuba.__version__} / Dr.Jit {drjit.__version__}")
print(f"PyTorch3D: {Path(pytorch3d.__file__).parent}")
print(f"nvdiffrast: {Path(nvdiffrast.torch.__file__).parent}")
print(f"pyremesh: {Path(pyremesh.__file__)}")
PY

echo
echo "Environment '$ENV_NAME' is ready."
echo "Activate it with: conda activate $ENV_NAME"
