# vnc

code for SIGGRAPH Asia 2026 paper "Video Numerical Continuation for Differentiable Inverse Rendering".

project page: [https://suikasibyl.github.io/vnc](https://suikasibyl.github.io/vnc)

![](https://suikasibyl.github.io/files/vnc/teaser.webp "teaser")


## Setup

The experiments require Linux (maybe Windows works as well, I'm not sure), an NVIDIA GPU, a recent NVIDIA driver, Conda, and Git. Blender and FFmpeg must be available on `PATH` for some experiments that convert assets or write videos, but likely you can easily remove them with some hack.

Create the Python environment with:

```bash
bash scripts/create_vnc_env.sh
conda activate vnc
```

The optional first argument changes the environment name:

```bash
bash scripts/create_vnc_env.sh vnc-dev
```

The setup script installs PyTorch with CUDA 12.8, Mitsuba/Dr.Jit, and the pinned PyTorch3D and nvdiffrast revisions used for these experiments. It also configures the bundled DROT, PRDPT, and remeshing code on environment activation.

The layout should look like:

```text
vnc/
  external/
    DROT/
    prdpt/
  geometry/
  lights/
  material/
  pose/
  scenes/
  skeleton/
  scripts/
```

Scene assets and video references are included under `scenes/`; experiment outputs are written under `outputs/`.

## Experiments

- `geometry/`: single- and multi-view mesh optimization for Bob, bunny, dragon, heart, and Suzanne, including glossy and video-guided variants.
- `pose/`: rigid pose and scene-layout optimization for the castle, Earth, furniture, gnome, house, mug, penguin, and Rubik's cube scenes.
- `skeleton/`: articulated fitting for the breakdance and uprock sequences.
- `lights/`: light and shadow optimization with rasterization and Mitsuba.
- `material/`: Mitsuba-based teapot alpha/material optimization.

Run launchers from the repository root. For example:

```bash
bash geometry/experiments-dragon.sh
bash pose/experiments-mug.sh
bash skeleton/run-experiments-uprock.sh
bash material/experiments-teapot.sh
```

Some launchers retain a machine-specific default `PYTHON_BIN`. Override it with the active environment when needed:

```bash
PYTHON_BIN="$(which python)" bash geometry/experiments-dragon.sh
```

The full launchers can be long-running and require substantial GPU memory. Their shell variables (for example `OUTPUT_ROOT`, `TRAJECTORY_FPS`, and `PRDPT_SIGMAS`) can be overridden to select outputs or smaller sweeps.

`pose/gnome.py` downloads the Stanford-ORB baseline on first use. `geometry/experiments-curve.sh` and `pose/mirror.ipynb` still contain paths from the original development workspace and need to be adapted before running from a standalone clone.

