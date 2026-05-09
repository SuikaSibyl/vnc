#!/usr/bin/env python3
"""Differentiable 6-DOF pose optimisation for the gnome scene via NVDiffRecMC.

Uses the Stanford-ORB baseline Monte Carlo renderer (nvdiffrecmc) so we can
optimize pose against the same gnome scene with path-traced environment-map
lighting instead of the raster PBR path used in pose_gnome.py.

Usage:
    python pose/gnome.py
    python pose/gnome.py --init-rotation 0 45 0 --n-iters 300
    python pose/gnome.py --diffuse-only --n-samples 8
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import subprocess
import sys
import time
import types
from types import SimpleNamespace

_cuda_home = "/usr/local/cuda-12.9"
if os.path.isdir(_cuda_home):
    os.environ.setdefault("CUDA_HOME", _cuda_home)
    os.environ["PATH"] = f"{_cuda_home}/bin:{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = f"{_cuda_home}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0+PTX")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from repo_paths import OUTPUTS_ROOT, REPO_ROOT, SCENES_ROOT, setup_python_paths
from timing_summary import write_subdir_summary_timing_summary

setup_python_paths()

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SCENE_DIR = SCENES_ROOT / "gnome"
MESH_PATH = SCENE_DIR / "mesh_blender" / "mesh.obj"
XFORMS = SCENE_DIR / "transforms_test.json"
ENVMAP = SCENE_DIR / "0070.exr"
REF_0070 = SCENE_DIR / "ref_0070_800.png"
MASK_0070 = SCENE_DIR / "mask_0070.png"
DEFAULT_OUT = OUTPUTS_ROOT / "gnome_nvdiffrecmc"

NEAR = 0.1
FAR = 1000.0
ROUGHNESS_SCALE = 6.0
ROUGHNESS_BIAS = 0.0
METALLIC_SCALE = 0.45
# Fitted from iter_0000 ref/render in the raster baseline; reused here so the
# saved images stay in roughly the same brightness/chroma regime.
RENDER_RGB_GAIN = (0.714983996721, 0.719221476506, 0.616068042834)

BASELINE_CACHE = pathlib.Path("/tmp/Stanford-ORB-baselines")
BASELINE_BRANCH = "baselines"
BASELINE_REPO = "https://github.com/StanfordORB/Stanford-ORB"

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


_baseline: dict = {}


def _install_tinycudann_stub() -> None:
    """Provide a minimal stub so nvdiffrecmc can import without tinycudann.

    The gnome pose script only uses OBJ texture maps, not the MLPTexture3D path.
    Some Stanford-ORB modules still import mlptexture eagerly, which imports
    tinycudann at module import time. This stub keeps those imports alive while
    raising a clear error if code ever tries to instantiate the TCNN encoder.
    """
    if "tinycudann" in sys.modules:
        return

    stub = types.ModuleType("tinycudann")

    class _UnavailableEncoding:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "tinycudann is not installed. This pose script only supports "
                "texture-map materials; MLPTexture3D/tinycudann paths are unavailable."
            )

    def _free_temporary_memory() -> None:
        return None

    stub.Encoding = _UnavailableEncoding
    stub.free_temporary_memory = _free_temporary_memory
    sys.modules["tinycudann"] = stub


def _get_baseline():
    if _baseline:
        return _baseline
    if not BASELINE_CACHE.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                BASELINE_BRANCH,
                BASELINE_REPO,
                str(BASELINE_CACHE),
            ],
            check=True,
        )

    root = BASELINE_CACHE
    mc_root = root / "orb" / "third_party" / "nvdiffrecmc"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(mc_root))

    try:
        import tinycudann  # type: ignore  # noqa: F401
    except Exception:
        _install_tinycudann_stub()

    import orb  # type: ignore

    if not hasattr(orb, "third_party"):
        orb.third_party = SimpleNamespace()

    try:
        import nvdiffrast.torch as dr  # type: ignore
    except Exception:
        local_root = REPO_ROOT / "geometry" / "largesteps" / "ext" / "nvdiffrast"
        major, minor = sys.version_info[:2]
        build_dir = local_root / "build" / f"lib.linux-x86_64-cpython-{major}{minor}"
        if build_dir.exists():
            sys.path.insert(0, str(build_dir))
        sys.path.insert(0, str(local_root))
        import nvdiffrast.torch as dr  # type: ignore

    from orb.utils.env_map import env_map_to_cam_to_world_by_convention  # type: ignore
    from orb.third_party.nvdiffrecmc.render import light, mesh, optixutils, render, texture, util  # type: ignore

    _baseline.update(
        dr=dr,
        light=light,
        mesh=mesh,
        optixutils=optixutils,
        render=render,
        texture=texture,
        util=util,
        env2world=env_map_to_cam_to_world_by_convention,
    )
    return _baseline


def _rotation_x_torch(a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a), torch.sin(a)
    z = torch.zeros([], device=a.device, dtype=torch.float32)
    o = torch.ones([], device=a.device, dtype=torch.float32)
    return torch.stack([
        torch.stack([o, z, z, z]),
        torch.stack([z, c, -s, z]),
        torch.stack([z, s, c, z]),
        torch.stack([z, z, z, o]),
    ])


def _rotation_y_torch(a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a), torch.sin(a)
    z = torch.zeros([], device=a.device, dtype=torch.float32)
    o = torch.ones([], device=a.device, dtype=torch.float32)
    return torch.stack([
        torch.stack([c, z, s, z]),
        torch.stack([z, o, z, z]),
        torch.stack([-s, z, c, z]),
        torch.stack([z, z, z, o]),
    ])


def _rotation_z_torch(a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a), torch.sin(a)
    z = torch.zeros([], device=a.device, dtype=torch.float32)
    o = torch.ones([], device=a.device, dtype=torch.float32)
    return torch.stack([
        torch.stack([c, -s, z, z]),
        torch.stack([s, c, z, z]),
        torch.stack([z, z, o, z]),
        torch.stack([z, z, z, o]),
    ])


def _build_model_matrix(
    rotation_xyz: torch.Tensor,
    translation_xyz: torch.Tensor,
    center_t: torch.Tensor,
) -> torch.Tensor:
    Rx = _rotation_x_torch(rotation_xyz[0])
    Ry = _rotation_y_torch(rotation_xyz[1])
    Rz = _rotation_z_torch(rotation_xyz[2])
    R = Rz @ Ry @ Rx
    T_origin = torch.eye(4, device=device, dtype=torch.float32)
    T_origin[:3, 3] = -center_t
    T_back = torch.eye(4, device=device, dtype=torch.float32)
    T_back[:3, 3] = center_t
    T_world = torch.eye(4, device=device, dtype=torch.float32)
    T_world[:3, 3] = translation_xyz
    return T_world @ T_back @ R @ T_origin


def _rotation_mat_3x3(rotation_xyz: torch.Tensor) -> torch.Tensor:
    return (
        _rotation_z_torch(rotation_xyz[2])
        @ _rotation_y_torch(rotation_xyz[1])
        @ _rotation_x_torch(rotation_xyz[0])
    )[:3, :3]


class GnomeMC:
    """Gnome mesh with differentiable 6-DOF pose, rendered via NVDiffRecMC."""

    def __init__(self) -> None:
        self._mesh_template = None
        self._tmpl_v_pos: torch.Tensor | None = None
        self._camera_mvp: torch.Tensor | None = None
        self._campos: torch.Tensor | None = None
        self._lgt = None
        self._optix_ctx = None
        self._center_t: torch.Tensor | None = None
        self._resolution: int = 512
        self._translation_scale: float = 1.0
        self._flags = None
        self._render_mod = None
        self._mesh_mod = None
        self._ou_mod = None
        self._bsdf: str | None = None
        self._mc_seed: int = 0

        self.rotation_xyz: torch.Tensor | None = None
        self._translation_param: torch.Tensor | None = None
        self.optim: list[torch.Tensor] = []

    def load_scene(
        self,
        xforms_path: str = str(XFORMS),
        frame_idx: int = 9,
        envmap_path: str = str(ENVMAP),
        init_rotation: tuple = (0.0, 0.0, 0.0),
        init_translation: tuple = (0.0, 0.0, 0.0),
        resolution: int = 512,
        bsdf: str | None = None,
        n_samples: int = 4,
        spp: int = 1,
        decorrelated: bool = False,
        mc_seed: int = 0,
    ) -> None:
        bl = _get_baseline()
        light_mod = bl["light"]
        mesh_mod = bl["mesh"]
        ou_mod = bl["optixutils"]
        texture_mod = bl["texture"]
        util_mod = bl["util"]
        env2world = bl["env2world"]

        rm = mesh_mod.load_mesh(str(MESH_PATH))
        ks_data = rm.material["ks"].data.detach().clone()
        ks_data[..., 1] = (ks_data[..., 1] * ROUGHNESS_SCALE + ROUGHNESS_BIAS).clamp(0.08, 1.0)
        ks_data[..., 2] = (ks_data[..., 2] * METALLIC_SCALE).clamp(0.0, 1.0)
        rm.material["ks"] = texture_mod.Texture2D(ks_data)

        self._mesh_template = rm
        self._tmpl_v_pos = rm.v_pos.detach().to(device)
        self._mesh_mod = mesh_mod
        self._ou_mod = ou_mod
        self._render_mod = bl["render"]
        self._optix_ctx = ou_mod.OptiXContext()

        center_np = self._tmpl_v_pos.cpu().numpy().mean(axis=0)
        extents = (self._tmpl_v_pos.max(0).values - self._tmpl_v_pos.min(0).values).cpu().numpy()
        radius = 0.5 * float(np.linalg.norm(extents))
        print(f"  Gnome radius={radius:.4f}  center={np.round(center_np, 4).tolist()}")

        meta = json.loads(pathlib.Path(xforms_path).read_text())
        frame = meta["frames"][frame_idx]
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        fovx = float(frame.get("camera_angle_x", meta["camera_angle_x"]))
        w2c = np.linalg.inv(c2w).astype(np.float32)
        fovy = util_mod.fovx_to_fovy(float(fovx), 1.0)
        proj = util_mod.perspective(fovy, 1.0, NEAR, FAR, device=device)
        mv = torch.tensor(w2c, dtype=torch.float32, device=device)
        mvp = proj @ mv

        self._camera_mvp = mvp[None]
        self._campos = torch.tensor(c2w[:3, 3], dtype=torch.float32, device=device)[None]

        env_raw = imageio.imread(str(envmap_path))
        env_np = np.nan_to_num(
            env_raw[..., :3].astype(np.float32),
            nan=0.0,
            posinf=1e4,
            neginf=0.0,
        )
        env_world = env2world(env_np, c2w, "ours")
        env_tensor = torch.tensor(env_world, dtype=torch.float32, device=device)
        self._lgt = light_mod.EnvironmentLight(env_tensor)

        self._center_t = torch.tensor(center_np, dtype=torch.float32, device=device)
        self._resolution = resolution
        self._bsdf = bsdf
        self._mc_seed = mc_seed
        self._flags = SimpleNamespace(
            n_samples=n_samples,
            decorrelated=decorrelated,
            denoiser_demodulate=True,
            no_perturbed_nrm=False,
            layers=1,
            spp=spp,
        )

        self._translation_scale = max(radius, 1e-6)
        rot_rad = np.deg2rad(init_rotation).astype(np.float32)
        self.rotation_xyz = torch.tensor(
            rot_rad, dtype=torch.float32, device=device, requires_grad=True
        )
        init_trans_np = np.asarray(init_translation, dtype=np.float32)
        self._translation_param = torch.tensor(
            init_trans_np / self._translation_scale,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.optim = [self.rotation_xyz, self._translation_param]

    @property
    def translation_xyz(self) -> torch.Tensor:
        return self._translation_param * self._translation_scale

    def render(self, glctx, with_grad: bool = True) -> torch.Tensor:
        rot = self.rotation_xyz if with_grad else self.rotation_xyz.detach()
        tra = self.translation_xyz if with_grad else self.translation_xyz.detach()

        model = _build_model_matrix(rot, tra, self._center_t)
        R3 = model[:3, :3]
        t3 = model[:3, 3]

        posed_v_pos = self._tmpl_v_pos @ R3.t() + t3
        render_mesh = self._mesh_mod.Mesh(v_pos=posed_v_pos, base=self._mesh_template)
        render_mesh = self._mesh_mod.auto_normals(render_mesh)
        render_mesh = self._mesh_mod.compute_tangents(render_mesh)

        with torch.no_grad():
            self._ou_mod.optix_build_bvh(
                self._optix_ctx,
                render_mesh.v_pos.contiguous(),
                render_mesh.t_pos_idx.int(),
                rebuild=1,
            )

        # Keep MC sampling deterministic across iterations unless the caller
        # explicitly opts into decorrelated sampling. This dramatically reduces
        # optimization jitter for pose fitting.
        if not self._flags.decorrelated and hasattr(self._render_mod, "rnd_seed"):
            self._render_mod.rnd_seed = self._mc_seed

        bg = torch.zeros(1, self._resolution, self._resolution, 3, dtype=torch.float32, device=device)
        buffers = self._render_mod.render_mesh(
            self._flags,
            glctx,
            render_mesh,
            self._camera_mvp,
            self._campos,
            self._lgt,
            [self._resolution, self._resolution],
            spp=self._flags.spp,
            num_layers=1,
            msaa=True,
            background=bg,
            optix_ctx=self._optix_ctx,
            bsdf=self._bsdf,
            denoiser=None,
        )
        shaded = buffers["shaded"][0]
        gain = torch.tensor(RENDER_RGB_GAIN, dtype=shaded.dtype, device=shaded.device)
        return torch.cat([shaded[..., :3] * gain, shaded[..., 3:4]], dim=-1)


def bilinear_downsample(t: torch.Tensor, size: int) -> torch.Tensor:
    if t.shape[-3] == size and t.shape[-2] == size:
        return t
    squeezed = t.ndim == 3
    if squeezed:
        t = t.unsqueeze(0)
    out = F.interpolate(
        t.permute(0, 3, 1, 2).contiguous(),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    return out.squeeze(0) if squeezed else out


def _gauss_downsample(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    c = img.shape[-1]
    x = img.permute(2, 0, 1).unsqueeze(0)
    k = kernel.expand(c, 1, 5, 5)
    x = F.conv2d(x, k, padding=2, groups=c)
    x = x[:, :, ::2, ::2]
    return x.squeeze(0).permute(1, 2, 0)


def gaussian_pyramid_loss(pred: torch.Tensor, target: torch.Tensor, n_levels: int = 4) -> torch.Tensor:
    k1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=pred.device, dtype=pred.dtype)
    k1d = k1d / k1d.sum()
    kernel = (k1d[:, None] * k1d[None, :]).unsqueeze(0).unsqueeze(0)
    loss = torch.tensor(0.0, device=pred.device)
    p, t = pred, target
    for i in range(n_levels):
        loss = loss + torch.mean((p - t) ** 2)
        if i < n_levels - 1:
            p = _gauss_downsample(p, kernel)
            t = _gauss_downsample(t, kernel)
    return loss


def reference_foreground_mask(img: torch.Tensor, threshold: float) -> torch.Tensor:
    """Binary foreground mask from a linear-RGB reference using non-black pixels."""
    return (img.max(dim=-1, keepdim=True).values > threshold).to(img.dtype)


class VideoFrameScheduler:
    def __init__(
        self,
        video_path: str,
        n_iter: int,
        fallback: torch.Tensor,
        resolution: int,
        device: torch.device,
        video_ratio: float = 0.8,
    ):
        reader = imageio.get_reader(video_path)
        frames = []
        for frame in reader:
            t = torch.from_numpy(frame.astype(np.float32) / 255.0).to(device)[..., :3]
            # Match the static PNG reference path: video frames are sRGB and need
            # conversion to linear RGB before being used for the loss.
            t = t.pow(2.2)
            if t.shape[0] != resolution or t.shape[1] != resolution:
                t = bilinear_downsample(t, resolution)
            frames.append(t)
        reader.close()
        self.frames = frames
        self.fallback = fallback
        self.n_video_iters = int(n_iter * video_ratio)

    def get_ref(self, iteration: int) -> torch.Tensor:
        if iteration >= self.n_video_iters or not self.frames:
            return self.fallback
        frac = iteration / max(self.n_video_iters - 1, 1)
        idx = min(int(round(frac * (len(self.frames) - 1))), len(self.frames) - 1)
        return self.frames[idx]


def _save_png(path: pathlib.Path, img) -> None:
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8))
    print(f"  Saved {path}")


def _to_uint8_frame(img: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)


def _annotate_iteration(frame: np.ndarray, iteration: int) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    label = f"iter {iteration + 1}"
    x, y = 12, 10
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((x, y), label, font=font)
    draw.rectangle((bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4), fill=(0, 0, 0))
    draw.text((x, y), label, fill=(255, 255, 255), font=font)
    return np.array(img)


def _save_video(path: pathlib.Path, frames: list[tuple[np.ndarray, int]], fps: int) -> None:
    if not frames:
        return
    max_frames = max(1, int(round(fps * 3.0)))
    if len(frames) > max_frames:
        keep_idx = np.linspace(0, len(frames) - 1, num=max_frames, dtype=int)
        frames = [frames[idx] for idx in keep_idx]
    target_h, target_w = frames[0][0].shape[:2]
    normalized_frames: list[tuple[np.ndarray, int]] = []
    for frame, iteration in frames:
        if frame.shape[:2] != (target_h, target_w):
            frame = np.asarray(
                Image.fromarray(frame).resize((target_w, target_h), Image.Resampling.BILINEAR)
            )
        normalized_frames.append((frame, iteration))
    annotated_frames = [_annotate_iteration(frame, iteration) for frame, iteration in normalized_frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimwrite(path, annotated_frames, fps=fps, format="FFMPEG", codec="libx264")
    except Exception:
        imageio.mimwrite(path, annotated_frames, fps=fps)
    print(f"  Saved {path}")


def _save_debug_snapshot(
    snap_dir: pathlib.Path,
    iter_idx: int,
    ref_img: torch.Tensor,
    render_img: torch.Tensor,
    render_alpha: torch.Tensor,
    util_mod,
    silhouette_threshold: float = 1e-4,
) -> None:
    err_img = (render_img - ref_img).abs()
    ref_sil = reference_foreground_mask(ref_img, silhouette_threshold).repeat(1, 1, 3)
    render_sil = render_alpha.clamp(0.0, 1.0).repeat(1, 1, 3)
    sil_err = (render_sil - ref_sil).abs()
    top_row = torch.cat([ref_img, render_img, err_img], dim=1)
    bot_row = torch.cat([ref_sil, render_sil, sil_err], dim=1)
    panel_img = torch.cat([top_row, bot_row], dim=0)
    _save_png(
        snap_dir / f"iter_{iter_idx:04d}.png",
        util_mod.rgb_to_srgb(panel_img).detach().cpu().numpy(),
    )
    _save_png(
        snap_dir / f"iter_{iter_idx:04d}_render.png",
        util_mod.rgb_to_srgb(render_img).detach().cpu().numpy(),
    )
    _save_png(
        snap_dir / f"iter_{iter_idx:04d}_mask_diff.png",
        sil_err.detach().cpu().numpy(),
    )


def _geodesic_error_deg(R_pred: torch.Tensor, R_target: torch.Tensor) -> float:
    with torch.no_grad():
        R_rel = R_pred.t() @ R_target
        trace = torch.clamp((torch.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        return float(torch.rad2deg(torch.acos(trace)).item())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Differentiable 6-DOF pose optimisation for the gnome scene "
        "(Stanford-ORB NVDiffRecMC renderer)."
    )
    parser.add_argument("--xforms", default=str(XFORMS))
    parser.add_argument("--envmap", default=str(ENVMAP))
    parser.add_argument("--frame", type=int, default=9)
    parser.add_argument("--ref-image", default=str(REF_0070))
    parser.add_argument("--mask", default=str(MASK_0070))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--loss-resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--init-rotation", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("RX", "RY", "RZ"))
    parser.add_argument("--init-translation", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("TX", "TY", "TZ"))
    parser.add_argument("--target-rotation", type=float, nargs=3, default=None, metavar=("RX", "RY", "RZ"))
    parser.add_argument("--target-translation", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("TX", "TY", "TZ"))

    parser.add_argument("--n-iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--translation-lr", type=float, default=None, help="Optional separate learning rate for translation.")
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--rotation-only", action="store_true")
    parser.add_argument("--diffuse-only", action="store_true")
    parser.add_argument("--n-samples", type=int, default=4, help="MC env-light samples per pixel.")
    parser.add_argument("--spp", type=int, default=1, help="Raster visibility samples per pixel.")
    parser.add_argument("--decorrelated", action="store_true", help="Use decorrelated MC sampling.")
    parser.add_argument("--mc-seed", type=int, default=0, help="Fixed MC sampling seed for stable optimization.")
    parser.add_argument("--video-path", type=str, default=None, help="Optional reference video path.")
    parser.add_argument("--video-ratio", type=float, default=0.8, help="Fraction of iterations that use video frames before falling back.")
    parser.add_argument(
        "--save-trajectory-video",
        action="store_true",
        help="If set, save per-iteration optimization renders as a video for this run.",
    )
    parser.add_argument(
        "--trajectory-fps",
        type=int,
        default=12,
        help="Frames per second for saved optimization trajectory videos.",
    )
    parser.add_argument(
        "--trajectory-every",
        type=int,
        default=20,
        help="Save one trajectory video frame every N optimization iterations.",
    )
    parser.add_argument("--video-loss-resolution", type=int, default=None, help="Optional loss resolution for video frames; defaults to --loss-resolution.")
    parser.add_argument("--video-loss", choices=["tone_mse", "gaussian_pyramid"], default="gaussian_pyramid")
    parser.add_argument("--pyramid-levels", type=int, default=4)
    parser.add_argument("--silhouette-weight", type=float, default=0.0, help="Optional weight for non-black foreground silhouette loss.")
    parser.add_argument("--silhouette-threshold", type=float, default=1e-4, help="Linear-RGB threshold used to define non-black reference foreground.")

    args = parser.parse_args()
    set_seed(args.seed)
    use_video_ref = args.video_path is not None
    use_real_ref = bool(args.ref_image)
    if not use_real_ref and not use_video_ref and args.target_rotation is None:
        parser.error("--target-rotation is required when --ref-image is ''.")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for NVDiffRecMC pose optimisation.")

    loss_res = args.loss_resolution
    video_loss_res = args.video_loss_resolution or loss_res
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bl = _get_baseline()
    glctx = bl["dr"].RasterizeGLContext()

    scene_kw = dict(
        xforms_path=args.xforms,
        frame_idx=args.frame,
        envmap_path=args.envmap,
        resolution=args.resolution,
        bsdf="diffuse" if args.diffuse_only else None,
        n_samples=args.n_samples,
        spp=args.spp,
        decorrelated=args.decorrelated,
        mc_seed=args.mc_seed,
    )

    if use_real_ref:
        print(f"Loading reference: {args.ref_image}")
        ref_np = np.array(Image.open(args.ref_image)).astype(np.float32) / 255.0
        ref_np = ref_np[..., :3]
        ref_np = (ref_np ** 2.2).astype(np.float32)
        gt_img = torch.tensor(ref_np, dtype=torch.float32, device=device)
        _save_png(out_dir / "gt_raw.png", ref_np ** (1 / 2.2))
        print(f"  Size: {ref_np.shape[1]}x{ref_np.shape[0]}")
        has_known_target = False
        R_target, target_tra_t = None, None
    elif use_video_ref:
        print(f"Loading reference video: {args.video_path}")
        reader = imageio.get_reader(args.video_path)
        first_frame = None
        for frame in reader:
            first_frame = frame
            break
        reader.close()
        if first_frame is None:
            raise RuntimeError(f"Reference video has no frames: {args.video_path}")
        ref_np = first_frame.astype(np.float32) / 255.0
        ref_np = ref_np[..., :3]
        ref_np = (ref_np ** 2.2).astype(np.float32)
        if ref_np.shape[0] != args.resolution or ref_np.shape[1] != args.resolution:
            ref_t = torch.tensor(ref_np, dtype=torch.float32, device=device)
            ref_t = bilinear_downsample(ref_t, args.resolution)
            ref_np = ref_t.cpu().numpy()
        gt_img = torch.tensor(ref_np, dtype=torch.float32, device=device)
        _save_png(out_dir / "gt_raw.png", np.clip(ref_np, 0.0, 1.0) ** (1 / 2.2))
        print(f"  First frame size: {gt_img.shape[1]}x{gt_img.shape[0]}")
        has_known_target = False
        R_target, target_tra_t = None, None
    else:
        print("Rendering synthetic ground truth ...")
        with torch.no_grad():
            gt_model = GnomeMC()
            gt_model.load_scene(
                **scene_kw,
                init_rotation=args.target_rotation,
                init_translation=args.target_translation,
            )
            gt_raw = gt_model.render(glctx, with_grad=False)
        gt_img = gt_raw[..., :3]
        _save_png(out_dir / "gt.png", bl["util"].rgb_to_srgb(gt_img).detach().cpu().numpy())
        target_rot_rad = np.deg2rad(args.target_rotation).astype(np.float32)
        R_target = _rotation_mat_3x3(torch.tensor(target_rot_rad, device=device))[:3, :3].detach()
        target_tra_t = torch.tensor(args.target_translation, dtype=torch.float32, device=device)
        has_known_target = True

    if args.mask:
        mask_np = np.array(Image.open(args.mask)).astype(np.float32) / 255.0
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        mask_t = torch.tensor(mask_np, dtype=torch.float32, device=device)
        mask_t = F.interpolate(
            mask_t.unsqueeze(0).unsqueeze(0),
            size=(gt_img.shape[0], gt_img.shape[1]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0).unsqueeze(-1)
        gt_img = gt_img * mask_t
        print(
            f"  Mask applied: {mask_np.shape} -> {list(gt_img.shape[:2])}  "
            f"fg={float((mask_t > 0.5).float().mean()) * 100:.1f}%"
        )

    gt_img_loss = bilinear_downsample(gt_img, loss_res)
    gt_vis = gt_img.clamp(0, 1) ** (1 / 2.2)
    _save_png(out_dir / "gt.png", gt_vis.cpu().numpy())

    print("Rendering init ...")
    with torch.no_grad():
        init_model = GnomeMC()
        init_model.load_scene(
            **scene_kw,
            init_rotation=args.init_rotation,
            init_translation=args.init_translation,
        )
        init_raw = init_model.render(glctx, with_grad=False)
    _save_png(out_dir / "init.png", bl["util"].rgb_to_srgb(init_raw[..., :3]).detach().cpu().numpy())

    init_loss = bilinear_downsample(init_raw[..., :3], loss_res)
    diff = (gt_img_loss - init_loss).abs().mean().item()
    print(f"  Mean |ref - init| (at {loss_res}px): {diff:.4f}")

    print("\n=== Optimising 6-DOF pose with NVDiffRecMC ===")
    model = GnomeMC()
    model.load_scene(
        **scene_kw,
        init_rotation=args.init_rotation,
        init_translation=args.init_translation,
    )

    translation_lr = args.translation_lr if args.translation_lr is not None else args.lr
    optim_params = [model.rotation_xyz] if args.rotation_only else model.optim
    if args.rotation_only:
        print("  Rotation-only mode: translation fixed at init value.")
        optimizer = torch.optim.Adam(
            [{"params": [model.rotation_xyz], "lr": args.lr}]
        )
    else:
        optimizer = torch.optim.Adam(
            [
                {"params": [model.rotation_xyz], "lr": args.lr},
                {"params": [model._translation_param], "lr": translation_lr},
            ]
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.n_iters, 1), eta_min=args.lr_min
    )

    snap_dir = out_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    trajectory_frames: list[tuple[np.ndarray, int]] = []
    trajectory_ref_frames: list[tuple[np.ndarray, int]] = []
    t_start = time.perf_counter()
    video_scheduler = None
    if use_video_ref:
        video_scheduler = VideoFrameScheduler(
            video_path=args.video_path,
            n_iter=args.n_iters,
            fallback=gt_img[..., :3],
            resolution=args.resolution,
            device=device,
            video_ratio=args.video_ratio,
        )

    with torch.no_grad():
        init_snap = model.render(glctx, with_grad=False)
        init_snap_loss = bilinear_downsample(init_snap[..., :3], video_loss_res if use_video_ref else loss_res)
        init_ref = video_scheduler.get_ref(0) if video_scheduler is not None else gt_img[..., :3]
        init_ref_loss = bilinear_downsample(init_ref, video_loss_res if use_video_ref else loss_res)
    init_snap_alpha = bilinear_downsample(init_snap[..., 3:4], video_loss_res if use_video_ref else loss_res)
    _save_debug_snapshot(
        snap_dir, 0, init_ref_loss, init_snap_loss, init_snap_alpha, bl["util"], args.silhouette_threshold
    )

    pbar = tqdm(range(args.n_iters), desc="pose-mc")
    for it in pbar:
        pred = model.render(glctx, with_grad=True)
        current_ref = video_scheduler.get_ref(it) if video_scheduler is not None else gt_img[..., :3]
        current_loss_res = video_loss_res if video_scheduler is not None else loss_res
        pred_loss = bilinear_downsample(pred[..., :3], current_loss_res)
        pred_alpha = bilinear_downsample(pred[..., 3:4], current_loss_res)
        ref_loss = bilinear_downsample(current_ref, current_loss_res)
        if video_scheduler is not None and args.video_loss == "gaussian_pyramid":
            loss = gaussian_pyramid_loss(pred_loss, ref_loss, n_levels=args.pyramid_levels)
        elif video_scheduler is not None:
            pred_tm = pred_loss / (1.0 + pred_loss)
            ref_tm = ref_loss / (1.0 + ref_loss)
            loss = (pred_tm - ref_tm).pow(2).mean()
        else:
            loss = (pred_loss - ref_loss).pow(2).mean()

        silhouette_loss = torch.tensor(0.0, device=device)
        if args.silhouette_weight > 0:
            ref_fg = reference_foreground_mask(ref_loss, args.silhouette_threshold)
            silhouette_loss = (pred_alpha - ref_fg).pow(2).mean()
            loss = loss + args.silhouette_weight * silhouette_loss

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(optim_params, args.grad_clip)
        optimizer.step()
        scheduler.step()

        if args.save_trajectory_video and (it % max(args.trajectory_every, 1) == 0 or it == args.n_iters - 1):
            pred_vis = bl["util"].rgb_to_srgb(pred[..., :3]).detach().cpu().numpy()
            trajectory_frames.append((_to_uint8_frame(pred_vis), it))
            if video_scheduler is not None:
                ref_vis = current_ref.clamp(0, 1).pow(1 / 2.2).detach().cpu().numpy()
                trajectory_ref_frames.append((_to_uint8_frame(ref_vis), it))

        with torch.no_grad():
            rot_deg = np.rad2deg(model.rotation_xyz.detach().cpu().numpy()).tolist()
            tra_val = model.translation_xyz.detach().cpu().numpy().tolist()
            step: dict = {
                "iteration": it,
                "loss": float(loss.item()),
                "silhouette_loss": float(silhouette_loss.item()),
                "rotation_xyz_deg": rot_deg,
                "translation_xyz": tra_val,
            }
            desc = f"loss={loss.item():.5f}"
            if args.silhouette_weight > 0:
                desc += f"  sil={silhouette_loss.item():.5f}"
            if has_known_target:
                R_pred = _rotation_mat_3x3(model.rotation_xyz)[:3, :3]
                rot_err = _geodesic_error_deg(R_pred, R_target)
                trans_err = float((model.translation_xyz - target_tra_t).norm().item())
                step["rot_error_deg"] = rot_err
                step["trans_error"] = trans_err
                desc += f"  rot_err={rot_err:.1f}deg  trans_err={trans_err:.4f}"

        history.append(step)
        pbar.set_description(desc)

        if (it + 1) % args.log_every == 0:
            with torch.no_grad():
                snap = model.render(glctx, with_grad=False)
                snap_loss = bilinear_downsample(snap[..., :3], current_loss_res)
            snap_alpha = bilinear_downsample(snap[..., 3:4], current_loss_res)
            _save_debug_snapshot(
                snap_dir, it + 1, ref_loss, snap_loss, snap_alpha, bl["util"], args.silhouette_threshold
            )

    elapsed = time.perf_counter() - t_start

    with torch.no_grad():
        final_raw = model.render(glctx, with_grad=False)
    _save_png(out_dir / "final.png", bl["util"].rgb_to_srgb(final_raw[..., :3]).detach().cpu().numpy())

    summary: dict = {
        "final_rotation_xyz_deg": np.rad2deg(model.rotation_xyz.detach().cpu().numpy()).tolist(),
        "final_translation_xyz": model.translation_xyz.detach().cpu().numpy().tolist(),
        "total_time_sec": elapsed,
        "n_samples": args.n_samples,
        "spp": args.spp,
        "rotation_lr": args.lr,
        "translation_lr": translation_lr,
        "video_path": args.video_path,
        "steps": history,
    }
    if has_known_target and history:
        summary.update(
            target_rotation_deg=args.target_rotation,
            target_translation=args.target_translation,
            final_rot_error_deg=history[-1].get("rot_error_deg"),
            final_trans_error=history[-1].get("trans_error"),
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_subdir_summary_timing_summary(out_dir.parent)

    if args.save_trajectory_video:
        _save_video(out_dir / "trajectory.mp4", trajectory_frames, fps=args.trajectory_fps)
        if video_scheduler is not None:
            _save_video(out_dir / "trajectory_ref.mp4", trajectory_ref_frames, fps=args.trajectory_fps)

    if history:
        if has_known_target:
            final_line = (
                f"\nFinal  rot_err={summary['final_rot_error_deg']:.2f}deg"
                f"  trans_err={summary['final_trans_error']:.4f}"
            )
        else:
            final_line = f"\nFinal  loss={history[-1]['loss']:.5f}"
    else:
        final_line = "\nFinal  no optimization steps run"
    print(final_line + f"  ({elapsed:.1f}s)")
    print(f"Results in {out_dir}")


if __name__ == "__main__":
    main()
