"""rubiks.py – two-axis (Y + X) differentiable pose optimisation for the Rubik's cube scene.

Analogous to house.py but optimises both rotation_y and rotation_x simultaneously.
"""

from __future__ import annotations

import json
import os
import random
import sys
import pathlib
import argparse
import time

import imageio
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import nvdiffrast.torch as dr
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    Materials,
    OpenGLOrthographicCameras,
    PointLights,
    TexturesUV,
    look_at_rotation,
)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from repo_paths import OUTPUTS_ROOT, SCENES_ROOT, setup_python_paths

setup_python_paths()

from utils_fns import smoothFn

from renderer import (
    _normalize,
    look_at,
    orthographic,
    load_scene_mesh,
    _build_camera,
    _rotation_y_torch,
    _rotation_x_torch,
    _render,
)
from experiments.common import get_pytorch3d_renderer
from core.LossFunction import PointLossFunction
from core.NvDiffRastRenderer import NVDiffRastFullRenderer
from timing_summary import write_method_json_timing_summary
from video_guided_methods import VideoFrameScheduler

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


# ---------------------------------------------------------------------------
# Minimal logger stub
# ---------------------------------------------------------------------------
class SimpleLogger:
    def __init__(self, out_dir: pathlib.Path) -> None:
        self.out_dir = out_dir

    def add_image(self, name, img, flip=True, **kw):
        pass

    def add_scalar(self, *a, **kw):
        pass

    def save_img(self, name, img, flip=True):
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
        if flip:
            img = img[::-1]
        path = self.out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(path, np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8))

    def save_npy(self, name, arr, **kw):
        pass


# ---------------------------------------------------------------------------
# Rubiks – analogous to House but with two-axis rotation
# ---------------------------------------------------------------------------

class Rubiks:
    """Wraps the Rubik's cube GLB with differentiable Y and X rotation parameters.

    Exposes:
      - render_nvdiffrast()     – nvdiffrast render (RGBA, bottom-up)
      - gen_mesh_pytorch3d()    – pytorch3d Meshes with R_y @ R_x applied
      - scene_dict(material)    – DROT-compatible scene dict
      - render_drot(material)   – render via DROT NVDiffRastFullRenderer
      - rotation_y, rotation_x  – learnable parameters (scalars, radians)
      - optim                   – list[Tensor] for torch optimizer
    """

    def __init__(self) -> None:
        self._vertices:  torch.Tensor | None = None
        self._faces:     torch.Tensor | None = None
        self._uvs:       torch.Tensor | None = None
        self._tex:       torch.Tensor | None = None
        self._proj_view: torch.Tensor | None = None
        self._center_t:  torch.Tensor | None = None
        self._resolution: int = 512

        self._p3d_mesh_template: Meshes | None = None
        self._p3d_cameras: OpenGLOrthographicCameras | None = None
        self._p3d_lights:  PointLights | None = None

        self._sensors:     list | None = None
        self._nv_renderer: NVDiffRastFullRenderer | None = None

        self.rotation_y: torch.Tensor | None = None
        self.rotation_x: torch.Tensor | None = None
        self.optim: list[torch.Tensor] = []

    def load_scene(
        self,
        glb_path: str,
        init_y_deg: float = 0.0,
        init_x_deg: float = 0.0,
        resolution: int = 512,
        camera_scale: float | None = None,
    ) -> None:
        import trimesh

        mesh, texture = load_scene_mesh(glb_path)

        # --- camera parameters (compute first) ----------------------------------
        bounds     = mesh.bounds.astype(np.float32)
        center_np  = bounds.mean(axis=0)
        extents    = bounds[1] - bounds[0]
        radius     = 0.5 * float(np.linalg.norm(extents))

        # --- nvdiffrast tensors (use default 1.15x scale) ----------------------
        self._vertices = torch.from_numpy(mesh.vertices.astype(np.float32)).to(device)
        self._faces    = torch.from_numpy(mesh.faces.astype(np.int32)).to(device)
        self._uvs      = torch.from_numpy(mesh.visual.uv.astype(np.float32)).to(device)
        self._tex      = torch.from_numpy(texture.astype(np.float32)).to(device)
        self._proj_view, self._center_t, _ = _build_camera(mesh, None, device)  # None → uses 1.15x
        self._resolution = resolution

        # --- pytorch3d/DROT camera scale (match nvdiffrast)
        if camera_scale is None:
            camera_scale = 1.15 * radius

        view_dir = _normalize(np.asarray([1.0, 0.5, 1.0], dtype=np.float32))
        up_dir   = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        eye_np   = center_np + view_dir * (4.0 * radius)
        near     = max(0.1 * radius, 4.0 * radius - radius)
        far      = 4.0 * radius + radius

        view_np  = look_at(eye_np, center_np, up_dir)
        proj_np  = orthographic(camera_scale, near, far)
        vp_np    = proj_np @ view_np

        # --- pytorch3d Meshes template -----------------------------------------
        texture_p3d = np.flip(texture, axis=0).copy()
        tex_maps  = (
            torch.from_numpy(texture_p3d[..., :3].astype(np.float32))
            .unsqueeze(0).to(device)
        )
        faces_uvs = (
            torch.from_numpy(mesh.faces.astype(np.int64))
            .unsqueeze(0).to(device)
        )
        verts_uvs = (
            torch.from_numpy(mesh.visual.uv.astype(np.float32))
            .unsqueeze(0).to(device)
        )
        textures_uv = TexturesUV(maps=tex_maps, faces_uvs=faces_uvs, verts_uvs=verts_uvs)
        verts_p3d   = (
            torch.from_numpy(mesh.vertices.astype(np.float32))
            .unsqueeze(0).to(device)
        )
        faces_p3d   = (
            torch.from_numpy(mesh.faces.astype(np.int64))
            .unsqueeze(0).to(device)
        )
        self._p3d_mesh_template = Meshes(
            verts=verts_p3d, faces=faces_p3d, textures=textures_uv
        )

        # --- pytorch3d cameras -------------------------------------------------
        position = torch.tensor(eye_np, dtype=torch.float32).unsqueeze(0).to(device)
        at       = torch.tensor(center_np, dtype=torch.float32).unsqueeze(0).to(device)
        R        = look_at_rotation(position, at=at, device=device)
        R_p3d    = R.clone()
        R_p3d[:, :, 0] *= -1
        R_p3d[:, :, 2] *= -1
        T_p3d    = -torch.bmm(R_p3d.transpose(1, 2), position.unsqueeze(-1)).squeeze(-1)

        self._p3d_cameras = OpenGLOrthographicCameras(
            device=device, R=R_p3d, T=T_p3d,
            znear=near, zfar=far,
            top=camera_scale, bottom=-camera_scale,
            left=-camera_scale, right=camera_scale,
        )
        self._p3d_lights = PointLights(
            device=device,
            location=position.tolist(),
            ambient_color=[[1.0, 1.0, 1.0]],
            diffuse_color=[[1.0, 1.0, 1.0]],
            specular_color=[[1.0, 1.0, 1.0]],
        )

        # --- DROT sensors ------------------------------------------------------
        view_t  = torch.tensor(view_np, dtype=torch.float32).unsqueeze(0).to(device)
        proj_t  = torch.tensor(proj_np, dtype=torch.float32).unsqueeze(0).to(device)
        vp_t    = torch.tensor(vp_np,   dtype=torch.float32).unsqueeze(0).to(device)
        pos_t   = position
        cen_t   = at
        self._sensors = [{
            "position":            pos_t,
            "resolution":          (resolution, resolution),
            "matrix":              vp_t,
            "camera_matrix":       view_t,
            "perspective_matrix":  proj_t,
            "center":              cen_t,
            "init_position":       pos_t.clone(),
            "init_center":         cen_t.clone(),
        }]

        self._nv_renderer = NVDiffRastFullRenderer(
            device=device,
            settings={"background": "white", "shading": False, "light_power": 1.0},
            resolution=[resolution, resolution],
        )

        # --- learnable parameters ----------------------------------------------
        self.rotation_y = torch.tensor(
            np.deg2rad(init_y_deg), dtype=torch.float32, device=device, requires_grad=True
        )
        self.rotation_x = torch.tensor(
            np.deg2rad(init_x_deg), dtype=torch.float32, device=device, requires_grad=True
        )
        self.optim = [self.rotation_y, self.rotation_x]

    def render_nvdiffrast(self, glctx: dr.RasterizeCudaContext, with_grad: bool = True) -> torch.Tensor:
        """Returns [H, W, 4] RGBA on device (bottom-up convention)."""
        angle_y = self.rotation_y if with_grad else self.rotation_y.detach()
        angle_x = self.rotation_x if with_grad else self.rotation_x.detach()
        color = _render(
            glctx, self._vertices, self._faces, self._uvs, self._tex,
            self._proj_view, self._center_t, angle_y, self._resolution, device,
            angle_x_rad=angle_x,
        )
        return color[0]  # [H, W, 4]

    def gen_mesh_pytorch3d(self) -> Meshes:
        """pytorch3d Meshes with combined R_y @ R_x rotation applied (differentiable)."""
        src_mesh = self._p3d_mesh_template.clone()
        rot_y    = _rotation_y_torch(self.rotation_y, device)[:3, :3]
        rot_x    = _rotation_x_torch(self.rotation_x, device)[:3, :3]
        rot_mat  = rot_y @ rot_x
        center   = self._center_t
        src_mesh.offset_verts_(-center)
        src_mesh.transform_verts_(rot_mat)
        src_mesh.offset_verts_(center)
        return src_mesh

    def scene_dict(self, material: Materials) -> dict:
        return {
            "meshes":   [{"model": self.gen_mesh_pytorch3d()}],
            "sensors":  self._sensors,
            "material": material,
        }

    def render_drot(self, material: Materials, DcDt: bool = False) -> dict:
        return self._nv_renderer.render(self.scene_dict(material), DcDt=DcDt)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _composite_bg(img: torch.Tensor, bg_color: tuple = (0.961, 0.969, 0.980)) -> torch.Tensor:
    """Composite image over background color (#F5F7FA gray by default)."""
    bg_tensor = torch.tensor(bg_color, device=img.device, dtype=img.dtype).view(1, 1, -1)
    if img.shape[-1] == 4:
        rgb = img[..., :3]
        alpha = img[..., 3:4]
        return rgb + (1.0 - alpha) * bg_tensor
    return img


def _replace_white_bg(img: np.ndarray, bg_color: tuple = (0.961, 0.969, 0.980)) -> np.ndarray:
    """Replace pure white or near-white background pixels with gray color."""
    # Create mask for nearly-white pixels (all channels > 0.99)
    white_mask = np.all(img > 0.99, axis=-1)
    img_out = img.copy()
    img_out[white_mask] = np.array(bg_color)
    return img_out


def _gauss_downsample(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    C = img.shape[-1]
    x = img.permute(2, 0, 1).unsqueeze(0)
    k = kernel.expand(C, 1, 5, 5)
    x = torch.nn.functional.conv2d(x, k, padding=2, groups=C)
    x = x[:, :, ::2, ::2]
    return x.squeeze(0).permute(1, 2, 0)


def gaussian_pyramid_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_levels: int = 4,
) -> torch.Tensor:
    k1d = torch.tensor([1., 4., 6., 4., 1.], device=pred.device, dtype=pred.dtype)
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


# ---------------------------------------------------------------------------
# PRDPT render function
# ---------------------------------------------------------------------------
def render_smooth_rubiks(theta_p: torch.Tensor, update_fn, context_args: dict):
    """Batch render for PRDPT importance sampling (two-axis Y+X rotation).

    Args:
        theta_p:      shape [N, 2] — N perturbed (rotation_y, rotation_x) pairs (radians)
        update_fn:    unused (kept for API compatibility with prdpt)
        context_args: must contain 'glctx', 'model', 'gt_image'

    Returns:
        losses:  shape [N] — per-sample MSE vs gt
        avg_img: first sample's rendered image (for logging)
    """
    glctx  = context_args["glctx"]
    model  = context_args["model"]
    gt_img = context_args["gt_image"]  # [H, W, C]

    losses = []
    avg_img = None
    with torch.no_grad():
        for i in range(theta_p.shape[0]):
            angle_y = theta_p[i, 0]
            angle_x = theta_p[i, 1]
            color = _render(
                glctx,
                model._vertices, model._faces, model._uvs, model._tex,
                model._proj_view, model._center_t,
                angle_y, model._resolution, device,
                angle_x_rad=angle_x,
            )
            img = color[0]
            if img.shape[-1] == 4:
                img = torch.cat([img[..., :3], torch.ones_like(img[..., 3:4])], dim=-1)
            loss = torch.mean((img[..., :3] - gt_img[..., :3]) ** 2)
            losses.append(loss)
            if avg_img is None:
                avg_img = img.detach()

    return torch.stack(losses), avg_img


def _save_png(path: pathlib.Path, img_hwc: np.ndarray, flip: bool = True) -> None:
    if torch.is_tensor(img_hwc):
        img_hwc = img_hwc.detach().cpu().numpy()
    if flip:
        img_hwc = img_hwc[::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.clip(np.rint(img_hwc * 255.0), 0, 255).astype(np.uint8))
    print(f"Saved {path}")


def _to_uint8_frame(img_hwc: np.ndarray | torch.Tensor, flip: bool = True) -> np.ndarray:
    if torch.is_tensor(img_hwc):
        img_hwc = img_hwc.detach().cpu().numpy()
    if flip:
        img_hwc = img_hwc[::-1]
    return np.clip(np.rint(img_hwc * 255.0), 0, 255).astype(np.uint8)


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
    annotated_frames = [_annotate_iteration(frame, iteration) for frame, iteration in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, annotated_frames, fps=fps)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Differentiable two-axis (Y+X) pose optimisation for the Rubik's cube scene."
    )
    parser.add_argument(
        "--input",
        default=str(SCENES_ROOT / "rubiks" / "rubiks_cube.glb"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUTS_ROOT / "rubiks"),
    )
    parser.add_argument("--resolution",   type=int,   default=512)
    parser.add_argument("--camera-scale", type=float, default=None)
    parser.add_argument("--init-y",       type=float, default=0.0,
                        help="Initial Y-axis rotation in degrees.")
    parser.add_argument("--init-x",       type=float, default=0.0,
                        help="Initial X-axis rotation in degrees.")
    parser.add_argument("--target-y",     type=float, default=170.0,
                        help="Target Y-axis rotation in degrees.")
    parser.add_argument("--target-x",     type=float, default=-20.0,
                        help="Target X-axis rotation in degrees.")
    parser.add_argument("--n-iters",      type=int,   default=300)
    parser.add_argument("--lr",           type=float, default=0.05)
    parser.add_argument("--gamma",        type=float, default=1.0)
    parser.add_argument("--log-every",    type=int,   default=50)
    parser.add_argument("--video-path", type=str, default=str(SCENES_ROOT / "rubiks" / "videos" / "output.mp4"))
    parser.add_argument("--video-ratio",  type=float, default=0.9)
    parser.add_argument(
        "--save-trajectory-video",
        action="store_true",
        help="If set, save per-iteration optimization renders as a video for each method.",
    )
    parser.add_argument(
        "--trajectory-fps",
        type=int,
        default=12,
        help="Frames per second for saved optimization trajectory videos.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["nvdiffrast", "video_nvdiffrast"],
        choices=[
            "nvdiffrast",
            "pytorch3d_rgb",
            "pytorch3d_sil",
            "pytorch3d_sil_rgb",
            "random",
            "finetune",
            "rgbxy",
            "rgbxy_pytorch3d",
            "video_nvdiffrast",
            "video_pytorch3d",
            "prdpt",
        ],
    )
    parser.add_argument("--matching-weight",   type=float, default=1.0)
    parser.add_argument("--matching-interval", type=int,   default=0)
    # PRDPT hyperparameters
    parser.add_argument("--prdpt-sigma",     type=float, default=0.5,
                        help="Initial kernel bandwidth (radians) for PRDPT.")
    parser.add_argument("--prdpt-sigma-min", type=float, default=0.01,
                        help="Minimum sigma after annealing (0 = no annealing).")
    parser.add_argument("--prdpt-nsamples",  type=int,   default=4,
                        help="Monte Carlo samples per PRDPT iteration (antithetic doubles this).")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger  = SimpleLogger(out_dir)

    resolution = [args.resolution, args.resolution]
    material   = Materials(
        device=device,
        ambient_color=((0.5, 0.5, 0.5),),
        diffuse_color=((0.7, 0.7, 0.7),),
        specular_color=((0.2, 0.2, 0.2),),
    )
    albedo_material = Materials(
        device=device,
        ambient_color=((1.0, 1.0, 1.0),),
        diffuse_color=((0.0, 0.0, 0.0),),
        specular_color=((0.0, 0.0, 0.0),),
    )
    albedo_lights = PointLights(
        device=device,
        location=[[0.0, 0.0, 0.0]],
        ambient_color=[[1.0, 1.0, 1.0]],
        diffuse_color=[[0.0, 0.0, 0.0]],
        specular_color=[[0.0, 0.0, 0.0]],
    )

    gt_renderer, sil_renderer, _, _ = get_pytorch3d_renderer(
        bg="white",
        faces_per_pixel=1,
        resolution=args.resolution,
        persp=False,
    )

    glctx = dr.RasterizeCudaContext()

    # ------------------------------------------------------------------
    # 1. Ground-truth render  (target pose, no grad)
    # ------------------------------------------------------------------
    with torch.no_grad():
        gt_model = Rubiks()
        gt_model.load_scene(
            args.input,
            init_y_deg=args.target_y,
            init_x_deg=args.target_x,
            resolution=args.resolution,
            camera_scale=args.camera_scale,
        )
        gt_img_nv    = gt_model.render_nvdiffrast(glctx, with_grad=False)
        gt_img_drot  = gt_model.render_drot(material, DcDt=False)
        gt_mesh_p3d  = gt_model.gen_mesh_pytorch3d()
        gt_img_torch = gt_renderer(
            gt_mesh_p3d,
            cameras=gt_model._p3d_cameras,
            lights=albedo_lights,
            materials=albedo_material,
        )
        gt_sil_torch = sil_renderer(
            gt_mesh_p3d,
            cameras=gt_model._p3d_cameras,
            lights=gt_model._p3d_lights,
            materials=material,
        )

    _save_png(out_dir / "gt_nv.png",    _composite_bg(gt_img_nv).cpu().numpy())
    # DROT renders white background, replace with gray
    drot_img = _composite_bg(gt_img_drot["images"][0]).cpu().numpy()
    drot_img = _replace_white_bg(drot_img)
    _save_png(out_dir / "gt_drot.png",  drot_img)
    # pytorch3d renders white background, replace with gray
    torch_img = _composite_bg(gt_img_torch[0]).cpu().numpy()
    torch_img = _replace_white_bg(torch_img)
    _save_png(out_dir / "gt_torch.png", torch_img, flip=False)

    # ------------------------------------------------------------------
    # 2. Initial render
    # ------------------------------------------------------------------
    with torch.no_grad():
        init_model = Rubiks()
        init_model.load_scene(
            args.input,
            init_y_deg=args.init_y,
            init_x_deg=args.init_x,
            resolution=args.resolution,
            camera_scale=args.camera_scale,
        )
        init_img = init_model.render_nvdiffrast(glctx, with_grad=False)

    _save_png(out_dir / "init.png", init_img.cpu().numpy())

    # ------------------------------------------------------------------
    # 3. Method loop
    # ------------------------------------------------------------------
    matching_settings = {
        "matching_weight":   args.matching_weight,
        "matching_interval": args.matching_interval,
        "matcher":           "Sinkhorn",
    }

    for method in args.methods:
        method_tag = f"prdpt_s{args.prdpt_sigma:.3g}" if method == "prdpt" else method
        print(f"\n=== Method: {method_tag} ===")

        model = Rubiks()
        model.load_scene(
            args.input,
            init_y_deg=args.init_y,
            init_x_deg=args.init_x,
            resolution=args.resolution,
            camera_scale=args.camera_scale,
        )

        optimizer = torch.optim.Adam(model.optim, lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=args.gamma
        )

        loss_func = PointLossFunction(
            debug=False,
            resolution=resolution,
            settings=matching_settings,
            device=device,
            renderer=model._nv_renderer,
            num_views=1,
            logger=logger,
        )

        # PRDPT setup
        prdpt_ctx   = None
        smooth_render = None
        if method == "prdpt":
            prdpt_ctx = {
                "glctx":     glctx,
                "model":     model,
                "gt_image":  gt_img_nv,
                "sigma":     args.prdpt_sigma,
                "nsamples":  args.prdpt_nsamples,
                "antithetic": True,
                "sampler":   "importance",
                "update_fn": None,
                "device":    device,
            }
            smooth_render = smoothFn(render_smooth_rubiks, context_args=None, device=device)

        video_scheduler = None
        if method in {"video_nvdiffrast", "video_pytorch3d"}:
            if args.video_path is None:
                print(f"  [WARN] --video-path not set for {method}; skipping.")
                continue
            video_scheduler = VideoFrameScheduler(
                video_path=args.video_path,
                n_iter=args.n_iters,
                resolution=resolution,
                device=device,
                fallback=gt_img_drot["images"][0],
                video_ratio=args.video_ratio,
            )

        method_data: list[dict] = []
        trajectory_frames: list[tuple[np.ndarray, int]] = []
        trajectory_ref_frames: list[tuple[np.ndarray, int]] = []
        t_start = time.perf_counter()

        pbar = tqdm(range(args.n_iters))
        for iteration in pbar:

            scene = model.scene_dict(material)

            loss_all = torch.tensor(0.0, device=device)
            _mse_gt: float | None = None

            if method == "nvdiffrast":
                pred  = model.render_nvdiffrast(glctx, with_grad=True)
                loss  = torch.mean((pred - gt_img_nv) ** 2)

            elif method == "pytorch3d_rgb":
                pred_p3d = gt_renderer(
                    model.gen_mesh_pytorch3d(),
                    cameras=model._p3d_cameras,
                    lights=model._p3d_lights,
                    materials=material,
                )
                loss = torch.mean((pred_p3d[0, ..., :3] - gt_img_torch[0, ..., :3]) ** 2)

            elif method == "pytorch3d_sil":
                pred_sil = sil_renderer(
                    model.gen_mesh_pytorch3d(),
                    cameras=model._p3d_cameras,
                    lights=model._p3d_lights,
                    materials=material,
                )
                loss = torch.mean((pred_sil[0, ..., 3] - gt_sil_torch[0, ..., 3]) ** 2)

            elif method == "pytorch3d_sil_rgb":
                pred_soft = gt_renderer(
                    model.gen_mesh_pytorch3d(),
                    cameras=model._p3d_cameras,
                    lights=model._p3d_lights,
                    materials=material,
                )
                loss = torch.mean(
                    (pred_soft[0, ..., :3] - gt_img_torch[0, ..., :3]) ** 2
                )
                pred_sil = sil_renderer(
                    model.gen_mesh_pytorch3d(),
                    cameras=model._p3d_cameras,
                    lights=model._p3d_lights,
                    materials=material,
                )
                loss += torch.mean(
                    (pred_sil[0, ..., 3] - gt_sil_torch[0, ..., 3]) ** 2
                )

            elif method == "random":
                if random.randint(0, 1) == 0:
                    loss, _ = loss_func(gt_img_drot, iteration=iteration, scene=scene, view=0)
                else:
                    render_res = model._nv_renderer.render(scene, DcDt=True, view=0)
                    loss = torch.mean(
                        (render_res["images"][0] - gt_img_drot["images"][0]) ** 2
                    )

            elif method == "finetune":
                switch = int(args.n_iters * 0.75)
                if iteration < switch:
                    loss, _ = loss_func(gt_img_drot, iteration=iteration, scene=scene, view=0)
                else:
                    if iteration == switch:
                        optimizer = torch.optim.SGD(
                            model.optim, lr=args.lr * (args.gamma ** iteration)
                        )
                    render_res = model._nv_renderer.render(scene, DcDt=True, view=0)
                    loss = torch.mean(
                        (render_res["images"][0] - gt_img_drot["images"][0]) ** 2
                    )

            elif method == "rgbxy":
                loss, _ = loss_func(gt_img_drot, iteration=iteration, scene=scene, view=0)

            elif method == "rgbxy_pytorch3d":
                loss, _ = loss_func(gt_img_drot, iteration=iteration, scene=scene, view=0)
                pred_soft = gt_renderer(
                    model.gen_mesh_pytorch3d(),
                    cameras=model._p3d_cameras,
                    lights=model._p3d_lights,
                    materials=material,
                )
                loss += torch.mean(
                    (pred_soft[0, ..., :3] - gt_img_torch[0, ..., :3]) ** 2
                )
                loss += torch.mean(
                    (pred_soft[0, ..., 3] - gt_sil_torch[0, ..., 3]) ** 2
                )

            elif method == "video_nvdiffrast":
                ref = video_scheduler.get_ref(iteration, view=0)
                render_res = model._nv_renderer.render(scene, DcDt=True, view=0)
                # loss = gaussian_pyramid_loss(render_res["images"][0][..., :3], ref[..., :3])
                loss = torch.mean((render_res["images"][0][..., :3] - ref[..., :3]) ** 2)

                _mse_gt = torch.mean(
                    (render_res["images"][0][..., :3].detach() - gt_img_drot["images"][0][..., :3]) ** 2
                ).item()

            elif method == "video_pytorch3d":
                if iteration < video_scheduler.n_video_iters:
                    ref = video_scheduler._iter_to_frame(iteration)
                    render_res = model._nv_renderer.render(scene, DcDt=True, view=0)
                    loss = gaussian_pyramid_loss(render_res["images"][0][..., :3], ref[..., :3])
                    _mse_gt = torch.mean(
                        (render_res["images"][0][..., :3].detach() - gt_img_drot["images"][0][..., :3]) ** 2
                    ).item()
                else:
                    pred_hard = gt_renderer(
                        model.gen_mesh_pytorch3d(),
                        cameras=model._p3d_cameras,
                        lights=albedo_lights,
                        materials=albedo_material,
                    )
                    pred_hard = pred_hard[0, ..., :3].flip(dims=[0])
                    gt_torch_albedo_bottomup = gt_img_torch[0, ..., :3].flip(dims=[0])
                    loss = gaussian_pyramid_loss(pred_hard, gt_torch_albedo_bottomup)
                    _mse_gt = torch.mean(
                        (pred_hard.detach() - gt_torch_albedo_bottomup) ** 2
                    ).item()

            elif method == "prdpt":
                # Linear sigma annealing
                if args.prdpt_sigma_min > 0:
                    frac = iteration / max(args.n_iters - 1, 1)
                    prdpt_ctx["sigma"] = max(
                        args.prdpt_sigma_min,
                        args.prdpt_sigma * (1.0 - frac) + args.prdpt_sigma_min * frac,
                    )
                # Pack both axes into [1, 2] for PRDPT
                params = torch.stack([model.rotation_y, model.rotation_x]).reshape(1, 2)
                loss, _ = smooth_render(params, prdpt_ctx)

            else:
                loss = torch.tensor(0.0, device=device)

            loss_all = loss_all + loss

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            scheduler.step()

            if args.save_trajectory_video:
                with torch.no_grad():
                    traj_render = model.render_nvdiffrast(glctx, with_grad=False)
                trajectory_frames.append((_to_uint8_frame(traj_render), iteration))

                if method == "video_nvdiffrast":
                    ref = video_scheduler.get_ref(iteration, view=0)
                    ref_np = ref.cpu().numpy() if torch.is_tensor(ref) else ref
                    ref_np = _replace_white_bg(ref_np)
                    trajectory_ref_frames.append((_to_uint8_frame(ref_np), iteration))
                elif method == "video_pytorch3d":
                    if iteration < video_scheduler.n_video_iters:
                        ref = video_scheduler._iter_to_frame(iteration)
                    else:
                        ref = gt_img_torch[0, ..., :3].flip(dims=[0])
                    ref_np = ref.cpu().numpy() if torch.is_tensor(ref) else ref
                    ref_np = _replace_white_bg(ref_np)
                    trajectory_ref_frames.append((_to_uint8_frame(ref_np), iteration))

            deg_y = float(np.rad2deg(model.rotation_y.item()))
            deg_x = float(np.rad2deg(model.rotation_x.item()))
            dy    = abs(((deg_y - args.target_y) + 180.0) % 360.0 - 180.0)
            dx    = abs(((deg_x - args.target_x) + 180.0) % 360.0 - 180.0)
            combined_error = float((dy + dx) / 2.0)

            # Record step data with parameters
            step_data = {
                "iteration": iteration,
                "mse_loss": float(loss_all.item()),
                "mse_loss_gt": _mse_gt if _mse_gt is not None else float(loss_all.item()),
                "angle_error_deg": combined_error,
                "rotation_y_error_deg": dy,
                "rotation_x_error_deg": dx,
                "rotation_y_deg": deg_y,
                "rotation_x_deg": deg_x,
            }
            method_data.append(step_data)

            pbar.set_description(
                f"{method_tag}  loss={loss_all.item():.6f}  "
                f"y={deg_y:.1f}°  x={deg_x:.1f}°  err={combined_error:.2f}°"
            )

            if (iteration + 1) % args.log_every == 0:
                with torch.no_grad():
                    snap = model.render_nvdiffrast(glctx, with_grad=False).cpu().numpy()
                _save_png(
                    out_dir / method_tag / f"iter_{iteration + 1:04d}.png",
                    snap,
                )

                if method in {"video_nvdiffrast"}:
                    ref = video_scheduler.get_ref(iteration, view=0)
                    ref_np = ref.cpu().numpy() if torch.is_tensor(ref) else ref
                    ref_np = _replace_white_bg(ref_np)
                    _save_png(
                        out_dir / method_tag / f"iter_{iteration + 1:04d}_ref.png",
                        ref_np,
                    )
                elif method in {"video_pytorch3d"}:
                    if iteration < video_scheduler.n_video_iters:
                        ref = video_scheduler._iter_to_frame(iteration)
                    else:
                        ref = gt_img_torch[0, ..., :3].flip(dims=[0])
                    ref_np = ref.cpu().numpy() if torch.is_tensor(ref) else ref
                    ref_np = _replace_white_bg(ref_np)
                    _save_png(
                        out_dir / method_tag / f"iter_{iteration + 1:04d}_ref.png",
                        ref_np,
                    )
                elif method in {"pytorch3d_rgb", "pytorch3d_sil", "pytorch3d_sil_rgb", "rgbxy_pytorch3d"}:
                    torch_ref = gt_img_torch[0][..., :3].cpu().numpy()
                    torch_ref = _replace_white_bg(torch_ref)
                    _save_png(
                        out_dir / method_tag / f"iter_{iteration + 1:04d}_ref.png",
                        torch_ref,
                        flip=False,
                    )
                else:
                    drot_ref = gt_img_drot["images"][0][..., :3].cpu().numpy()
                    drot_ref = _replace_white_bg(drot_ref)
                    _save_png(
                        out_dir / method_tag / f"iter_{iteration + 1:04d}_ref.png",
                        drot_ref,
                    )

        # ------ final result ---------------------------------------------------
        elapsed = time.perf_counter() - t_start

        with torch.no_grad():
            final_img = model.render_nvdiffrast(glctx, with_grad=False)

        final_y = float(np.rad2deg(model.rotation_y.item()))
        final_x = float(np.rad2deg(model.rotation_x.item()))
        _save_png(out_dir / f"final_{method_tag}.png", final_img.cpu().numpy())
        print(
            f"  Final: y={final_y:.2f}° (target {args.target_y}°)  "
            f"x={final_x:.2f}° (target {args.target_x}°)"
        )

        # Save method's own JSON with metadata
        method_json = {
            "method": method_tag,
            "total_time_sec": elapsed,
            "final_rotation_y_deg": final_y,
            "final_rotation_x_deg": final_x,
            "target_y": args.target_y,
            "target_x": args.target_x,
            "steps": method_data,
        }
        json_path = out_dir / f"{method_tag}.json"
        with open(json_path, "w") as f:
            json.dump(method_json, f, indent=2)
        print(f"Saved {json_path}")
        summary_path = write_method_json_timing_summary(out_dir)
        print(f"Saved {summary_path}")

        if args.save_trajectory_video:
            _save_video(out_dir / method_tag / "trajectory.mp4", trajectory_frames, fps=args.trajectory_fps)
            if method in {"video_nvdiffrast", "video_pytorch3d"}:
                _save_video(
                    out_dir / method_tag / "trajectory_ref.mp4",
                    trajectory_ref_frames,
                    fps=args.trajectory_fps,
                )


if __name__ == "__main__":
    main()
