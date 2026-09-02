"""gugugaga.py – single-axis (X) differentiable translation optimisation for the gugugaga scene.

Differences from castle.py:
  - Optimised parameter: X-axis translation offset (metres / scene units) instead of Y rotation.
  - Camera: front view — placed on the +Z side, looking toward the scene centre along -Z.
  - Default scene: scenes/pose/gugugaga/gugugaga.glb
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
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

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
# Differentiable X-translation render helper
# ---------------------------------------------------------------------------
def _translation_x_torch(tx: torch.Tensor, device) -> torch.Tensor:
    """Build a 4×4 translation-along-X matrix from a differentiable offset (scene units)."""
    z = torch.zeros([], device=device, dtype=torch.float32)
    o = torch.ones([], device=device, dtype=torch.float32)
    return torch.stack([
        torch.stack([o, z, z, tx]),
        torch.stack([z, o, z,  z]),
        torch.stack([z, z, o,  z]),
        torch.stack([z, z, z,  o]),
    ])


def _render_translate_x(
    glctx,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    uvs: torch.Tensor,
    tex: torch.Tensor,
    proj_view: torch.Tensor,
    tx: torch.Tensor,
    resolution: int,
    device,
) -> torch.Tensor:
    """Differentiable render with X-axis translation applied to the mesh.

    tx – scalar tensor (may carry grad) representing the X-axis offset.
    """
    T = _translation_x_torch(tx, device)
    mvp = proj_view @ T

    pos_w = torch.cat([vertices, torch.ones([vertices.shape[0], 1], device=device)], dim=1)
    pos_clip = (pos_w @ mvp.t())[None]

    rast_out, _ = dr.rasterize(glctx, pos_clip, faces, resolution=[resolution, resolution])
    uv_interp, _ = dr.interpolate(uvs[None], rast_out, faces)
    color = dr.texture(tex[None], uv_interp, filter_mode="linear")

    mask = (rast_out[..., 3:4] > 0).to(color.dtype)
    bg = torch.tensor([0.96, 0.97, 0.98, 1.0], device=device, dtype=color.dtype).view(1, 1, 1, 4)
    color = color * mask + bg * (1.0 - mask)
    color = dr.antialias(color, rast_out, pos_clip, faces)
    return color


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
# Gugugaga model
# ---------------------------------------------------------------------------
class Gugugaga:
    """Wraps the gugugaga GLB with a differentiable X-translation parameter.

    Camera: front view — placed on the +Z axis, looking toward the scene centre along -Z.
    """

    def __init__(self) -> None:
        self._vertices: torch.Tensor | None = None
        self._faces: torch.Tensor | None = None
        self._uvs: torch.Tensor | None = None
        self._tex: torch.Tensor | None = None
        self._proj_view: torch.Tensor | None = None
        self._center_t: torch.Tensor | None = None
        self._resolution: int = 512

        self._p3d_mesh_template: Meshes | None = None
        self._p3d_cameras: OpenGLOrthographicCameras | None = None
        self._p3d_lights: PointLights | None = None

        self._sensors: list | None = None
        self._nv_renderer: NVDiffRastFullRenderer | None = None

        self.translation_x: torch.Tensor | None = None
        self.optim: list[torch.Tensor] = []

    def load_scene(
        self,
        glb_path: str,
        init_offset: float = 0.0,
        resolution: int = 512,
        camera_scale: float | None = None,
    ) -> None:
        mesh, texture = load_scene_mesh(glb_path)

        bounds = mesh.bounds.astype(np.float32)
        center_np = bounds.mean(axis=0)
        extents = bounds[1] - bounds[0]
        radius = 0.5 * float(np.linalg.norm(extents))

        self._vertices = torch.from_numpy(mesh.vertices.astype(np.float32)).to(device)
        self._faces = torch.from_numpy(mesh.faces.astype(np.int32)).to(device)
        self._uvs = torch.from_numpy(mesh.visual.uv.astype(np.float32)).to(device)
        self._tex = torch.from_numpy(texture.astype(np.float32)).to(device)
        self._center_t = torch.tensor(center_np, dtype=torch.float32, device=device)
        self._resolution = resolution

        if camera_scale is None:
            camera_scale = 1.15 * radius

        # Front view: camera on +Z axis looking toward scene centre along -Z.
        view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        up_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        eye_np = center_np + view_dir * (4.0 * radius)
        near = max(0.1 * radius, 4.0 * radius - radius)
        far = 4.0 * radius + radius

        view_np = look_at(eye_np, center_np, up_dir)
        proj_np = orthographic(camera_scale, near, far)
        vp_np = proj_np @ view_np

        self._proj_view = torch.tensor(vp_np, dtype=torch.float32, device=device)

        # PyTorch3D mesh template
        texture_p3d = np.flip(texture, axis=0).copy()
        tex_maps = torch.from_numpy(texture_p3d[..., :3].astype(np.float32)).unsqueeze(0).to(device)
        faces_uvs = torch.from_numpy(mesh.faces.astype(np.int64)).unsqueeze(0).to(device)
        verts_uvs = torch.from_numpy(mesh.visual.uv.astype(np.float32)).unsqueeze(0).to(device)
        textures_uv = TexturesUV(maps=tex_maps, faces_uvs=faces_uvs, verts_uvs=verts_uvs)

        verts_p3d = torch.from_numpy(mesh.vertices.astype(np.float32)).unsqueeze(0).to(device)
        faces_p3d = torch.from_numpy(mesh.faces.astype(np.int64)).unsqueeze(0).to(device)
        self._p3d_mesh_template = Meshes(verts=verts_p3d, faces=faces_p3d, textures=textures_uv)

        # PyTorch3D camera (front view)
        position = torch.tensor(eye_np, dtype=torch.float32).unsqueeze(0).to(device)
        at = torch.tensor(center_np, dtype=torch.float32).unsqueeze(0).to(device)
        R = look_at_rotation(position, at=at, device=device)
        R_p3d = R.clone()
        R_p3d[:, :, 0] *= -1
        R_p3d[:, :, 2] *= -1
        T_p3d = -torch.bmm(R_p3d.transpose(1, 2), position.unsqueeze(-1)).squeeze(-1)

        self._p3d_cameras = OpenGLOrthographicCameras(
            device=device,
            R=R_p3d,
            T=T_p3d,
            znear=near,
            zfar=far,
            top=camera_scale,
            bottom=-camera_scale,
            left=-camera_scale,
            right=camera_scale,
        )
        self._p3d_lights = PointLights(
            device=device,
            location=position.tolist(),
            ambient_color=[[1.0, 1.0, 1.0]],
            diffuse_color=[[1.0, 1.0, 1.0]],
            specular_color=[[1.0, 1.0, 1.0]],
        )

        view_t = torch.tensor(view_np, dtype=torch.float32).unsqueeze(0).to(device)
        proj_t = torch.tensor(proj_np, dtype=torch.float32).unsqueeze(0).to(device)
        vp_t = torch.tensor(vp_np, dtype=torch.float32).unsqueeze(0).to(device)
        self._sensors = [{
            "position": position,
            "resolution": (resolution, resolution),
            "matrix": vp_t,
            "camera_matrix": view_t,
            "perspective_matrix": proj_t,
            "center": at,
            "init_position": position.clone(),
            "init_center": at.clone(),
        }]

        self._nv_renderer = NVDiffRastFullRenderer(
            device=device,
            settings={"background": "white", "shading": False, "light_power": 1.0},
            resolution=[resolution, resolution],
        )

        self.translation_x = torch.tensor(
            float(init_offset), dtype=torch.float32, device=device, requires_grad=True
        )
        self.optim = [self.translation_x]

    def render_nvdiffrast(self, glctx: dr.RasterizeCudaContext, with_grad: bool = True) -> torch.Tensor:
        tx = self.translation_x if with_grad else self.translation_x.detach()
        color = _render_translate_x(
            glctx,
            self._vertices,
            self._faces,
            self._uvs,
            self._tex,
            self._proj_view,
            tx,
            self._resolution,
            device,
        )
        img = color[0]
        if img.shape[-1] == 4:
            img = torch.cat([img[..., :3], torch.ones_like(img[..., 3:4])], dim=-1)
        return img

    def gen_mesh_pytorch3d(self) -> Meshes:
        src_mesh = self._p3d_mesh_template.clone()
        n_verts = src_mesh.verts_packed().shape[0]
        offset = torch.zeros(n_verts, 3, device=device)
        offset[:, 0] = self.translation_x
        src_mesh.offset_verts_(offset)
        return src_mesh

    def scene_dict(self, material: Materials) -> dict:
        return {
            "meshes": [{"model": self.gen_mesh_pytorch3d()}],
            "sensors": self._sensors,
            "material": material,
        }

    def render_drot(self, material: Materials, DcDt: bool = False) -> dict:
        result = self._nv_renderer.render(self.scene_dict(material), DcDt=DcDt)
        imgs = result["images"]
        if torch.is_tensor(imgs) and imgs.shape[-1] == 4:
            result["images"] = torch.cat([imgs[..., :3], torch.ones_like(imgs[..., 3:4])], dim=-1)
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _composite_bg(img: torch.Tensor, bg_color: tuple = (0.961, 0.969, 0.980)) -> torch.Tensor:
    bg_tensor = torch.tensor(bg_color, device=img.device, dtype=img.dtype).view(1, 1, -1)
    if img.shape[-1] == 4:
        rgb = img[..., :3]
        alpha = img[..., 3:4]
        return rgb + (1.0 - alpha) * bg_tensor
    return img


def _replace_white_bg(img: np.ndarray, bg_color: tuple = (0.961, 0.969, 0.980)) -> np.ndarray:
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


def _save_png(path: pathlib.Path, img_hwc: np.ndarray, flip: bool = True) -> None:
    if torch.is_tensor(img_hwc):
        img_hwc = img_hwc.detach().cpu().numpy()
    if flip:
        img_hwc = img_hwc[::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.clip(np.rint(img_hwc * 255.0), 0, 255).astype(np.uint8))
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# PRDPT render function
# ---------------------------------------------------------------------------
def render_smooth_gugugaga(delta_p: torch.Tensor, update_fn, context_args: dict):
    """Batch render for PRDPT importance sampling (X-translation variant).

    Args:
        delta_p:      shape [N, 1] — N perturbed X-translation offsets (scene units)
        update_fn:    unused (kept for API compatibility with prdpt)
        context_args: must contain 'glctx', 'model', 'gt_image'

    Returns:
        losses:   shape [N] — per-sample MSE vs gt_image
        avg_img:  average rendered image across sampled parameters (for logging)
    """
    glctx = context_args["glctx"]
    model = context_args["model"]
    gt_img = context_args["gt_image"]  # [H, W, C]

    losses = []
    avg_img_sum = None
    with torch.no_grad():
        for i in range(delta_p.shape[0]):
            tx = delta_p[i, 0]
            color = _render_translate_x(
                glctx,
                model._vertices, model._faces, model._uvs, model._tex,
                model._proj_view,
                tx, model._resolution, device,
            )
            img = color[0]
            if img.shape[-1] == 4:
                img = torch.cat([img[..., :3], torch.ones_like(img[..., 3:4])], dim=-1)
            loss = torch.mean((img[..., :3] - gt_img[..., :3]) ** 2)
            losses.append(loss)
            img_detached = img.detach()
            if avg_img_sum is None:
                avg_img_sum = img_detached.clone()
            else:
                avg_img_sum = avg_img_sum + img_detached

    avg_img = avg_img_sum / max(delta_p.shape[0], 1)
    return torch.stack(losses), avg_img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Differentiable single-axis (X) translation optimisation for the gugugaga scene."
    )
    parser.add_argument(
        "--input",
        default=str(SCENES_ROOT / "gugugaga" / "gugugaga.glb"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUTS_ROOT / "gugugaga"),
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--camera-scale", type=float, default=None)
    parser.add_argument("--init-offset", type=float, default=0.0,
                        help="Initial X-axis translation offset (scene units).")
    parser.add_argument("--target-offset", type=float, default=0.5,
                        help="Target X-axis translation offset (scene units).")
    parser.add_argument("--n-iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--video-path", type=str, default=str(SCENES_ROOT / "gugugaga" / "videos" / "output.mp4"))
    parser.add_argument("--video-ratio", type=float, default=0.9)
    parser.add_argument(
        "--video-max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of video frames, sampled evenly across the source video.",
    )
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
    parser.add_argument("--matching-weight", type=float, default=1.0)
    parser.add_argument("--matching-interval", type=int, default=0)
    # PRDPT hyperparameters
    parser.add_argument("--prdpt-sigma", type=float, default=0.5,
                        help="Initial kernel bandwidth (scene units) for PRDPT.")
    parser.add_argument("--prdpt-sigma-min", type=float, default=0.01,
                        help="Minimum sigma after annealing (0 = no annealing).")
    parser.add_argument("--prdpt-nsamples", type=int, default=4,
                        help="Monte Carlo samples per PRDPT iteration (antithetic doubles this).")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = SimpleLogger(out_dir)

    resolution = [args.resolution, args.resolution]
    material = Materials(
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

    # 1) Ground-truth render at target state
    with torch.no_grad():
        gt_model = Gugugaga()
        gt_model.load_scene(
            args.input,
            init_offset=args.target_offset,
            resolution=args.resolution,
            camera_scale=args.camera_scale,
        )
        gt_img_nv = gt_model.render_nvdiffrast(glctx, with_grad=False)
        gt_img_drot = gt_model.render_drot(material, DcDt=False)
        gt_mesh_p3d = gt_model.gen_mesh_pytorch3d()
        gt_img_torch = gt_renderer(
            gt_mesh_p3d,
            cameras=gt_model._p3d_cameras,
            lights=albedo_lights,
            materials=albedo_material,
        )
        gt_img_torch = torch.cat([gt_img_torch[..., :3], torch.ones_like(gt_img_torch[..., 3:4])], dim=-1)
        gt_sil_torch = sil_renderer(
            gt_mesh_p3d,
            cameras=gt_model._p3d_cameras,
            lights=gt_model._p3d_lights,
            materials=material,
        )
        gt_sil_torch = torch.cat([gt_sil_torch[..., :3], torch.ones_like(gt_sil_torch[..., 3:4])], dim=-1)

    _save_png(out_dir / "gt_nv.png", _composite_bg(gt_img_nv).cpu().numpy())
    drot_img = _composite_bg(gt_img_drot["images"][0]).cpu().numpy()
    drot_img = _replace_white_bg(drot_img)
    _save_png(out_dir / "gt_drot.png", drot_img)
    torch_img = _composite_bg(gt_img_torch[0]).cpu().numpy()
    torch_img = _replace_white_bg(torch_img)
    _save_png(out_dir / "gt_torch.png", torch_img, flip=False)

    # 2) Initial render at init state
    with torch.no_grad():
        init_model = Gugugaga()
        init_model.load_scene(
            args.input,
            init_offset=args.init_offset,
            resolution=args.resolution,
            camera_scale=args.camera_scale,
        )
        init_img = init_model.render_nvdiffrast(glctx, with_grad=False)
    _save_png(out_dir / "init.png", init_img.cpu().numpy())

    # 3) Method loop
    matching_settings = {
        "matching_weight": args.matching_weight,
        "matching_interval": args.matching_interval,
        "matcher": "Sinkhorn",
    }

    for method in args.methods:
        method_tag = f"prdpt_s{args.prdpt_sigma:.3g}" if method == "prdpt" else method
        print(f"\n=== Method: {method_tag} ===")

        model = Gugugaga()
        model.load_scene(
            args.input,
            init_offset=args.init_offset,
            resolution=args.resolution,
            camera_scale=args.camera_scale,
        )

        optimizer = torch.optim.Adam(model.optim, lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

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
        prdpt_ctx = None
        smooth_render = None
        if method == "prdpt":
            prdpt_ctx = {
                "glctx": glctx,
                "model": model,
                "gt_image": gt_img_nv,
                "sigma": args.prdpt_sigma,
                "nsamples": args.prdpt_nsamples,
                "antithetic": True,
                "sampler": "importance",
                "update_fn": None,
                "device": device,
            }
            smooth_render = smoothFn(render_smooth_gugugaga, context_args=None, device=device)

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
                max_frames=args.video_max_frames,
            )

        method_data: list[dict] = []
        trajectory_frames: list[tuple[np.ndarray, int]] = []
        trajectory_ref_frames: list[tuple[np.ndarray, int]] = []
        t_start = time.perf_counter()

        pbar = tqdm(range(args.n_iters))
        for iteration in pbar:
            scene = model.scene_dict(material)
            prdpt_avg_img = None

            loss_all = torch.tensor(0.0, device=device)
            _mse_gt: float | None = None

            if method == "nvdiffrast":
                pred = model.render_nvdiffrast(glctx, with_grad=True)
                loss = torch.mean((pred - gt_img_nv) ** 2)

            elif method == "pytorch3d_rgb":
                pred_p3d = gt_renderer(
                    model.gen_mesh_pytorch3d(),
                    cameras=model._p3d_cameras,
                    lights=model._p3d_lights,
                    materials=material,
                )
                pred_p3d = torch.cat([pred_p3d[..., :3], torch.ones_like(pred_p3d[..., 3:4])], dim=-1)
                loss = torch.mean((pred_p3d[0, ..., :3] - gt_img_torch[0, ..., :3]) ** 2)

            elif method == "pytorch3d_sil":
                pred_sil = sil_renderer(
                    model.gen_mesh_pytorch3d(),
                    cameras=model._p3d_cameras,
                    lights=model._p3d_lights,
                    materials=material,
                )
                pred_sil = torch.cat([pred_sil[..., :3], torch.ones_like(pred_sil[..., 3:4])], dim=-1)
                loss = torch.mean((pred_sil[0, ..., 3] - gt_sil_torch[0, ..., 3]) ** 2)

            elif method == "pytorch3d_sil_rgb":
                pred_soft = gt_renderer(
                    model.gen_mesh_pytorch3d(),
                    cameras=model._p3d_cameras,
                    lights=model._p3d_lights,
                    materials=material,
                )
                pred_soft = torch.cat([pred_soft[..., :3], torch.ones_like(pred_soft[..., 3:4])], dim=-1)
                loss = torch.mean((pred_soft[0, ..., :3] - gt_img_torch[0, ..., :3]) ** 2)
                pred_sil = sil_renderer(
                    model.gen_mesh_pytorch3d(),
                    cameras=model._p3d_cameras,
                    lights=model._p3d_lights,
                    materials=material,
                )
                pred_sil = torch.cat([pred_sil[..., :3], torch.ones_like(pred_sil[..., 3:4])], dim=-1)
                loss += torch.mean((pred_sil[0, ..., 3] - gt_sil_torch[0, ..., 3]) ** 2)

            elif method == "random":
                if random.randint(0, 1) == 0:
                    loss, _ = loss_func(gt_img_drot, iteration=iteration, scene=scene, view=0)
                else:
                    render_res = model._nv_renderer.render(scene, DcDt=True, view=0)
                    loss = torch.mean((render_res["images"][0] - gt_img_drot["images"][0]) ** 2)

            elif method == "finetune":
                switch = int(args.n_iters * 0.75)
                if iteration < switch:
                    loss, _ = loss_func(gt_img_drot, iteration=iteration, scene=scene, view=0)
                else:
                    if iteration == switch:
                        optimizer = torch.optim.SGD(model.optim, lr=args.lr * (args.gamma ** iteration))
                    render_res = model._nv_renderer.render(scene, DcDt=True, view=0)
                    loss = torch.mean((render_res["images"][0] - gt_img_drot["images"][0]) ** 2)

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
                pred_soft = torch.cat([pred_soft[..., :3], torch.ones_like(pred_soft[..., 3:4])], dim=-1)
                loss += torch.mean((pred_soft[0, ..., :3] - gt_img_torch[0, ..., :3]) ** 2)
                loss += torch.mean((pred_soft[0, ..., 3] - gt_sil_torch[0, ..., 3]) ** 2)

            elif method == "video_nvdiffrast":
                ref = video_scheduler.get_ref(iteration, view=0)
                render_res = model._nv_renderer.render(scene, DcDt=True, view=0)
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
                    pred_hard = torch.cat([pred_hard[..., :3], torch.ones_like(pred_hard[..., 3:4])], dim=-1)
                    pred_hard = pred_hard[0, ..., :3].flip(dims=[0])
                    gt_torch_albedo_bottomup = gt_img_torch[0, ..., :3].flip(dims=[0])
                    loss = gaussian_pyramid_loss(pred_hard, gt_torch_albedo_bottomup)
                    _mse_gt = torch.mean((pred_hard.detach() - gt_torch_albedo_bottomup) ** 2).item()

            elif method == "prdpt":
                # Linear sigma annealing
                if args.prdpt_sigma_min > 0:
                    frac = iteration / max(args.n_iters - 1, 1)
                    prdpt_ctx["sigma"] = max(
                        args.prdpt_sigma_min,
                        args.prdpt_sigma * (1.0 - frac) + args.prdpt_sigma_min * frac,
                    )
                # Variational gradient estimate — no autodiff through renderer
                loss, prdpt_avg_img = smooth_render(model.translation_x.reshape(1, 1), prdpt_ctx)

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

                if method in {"video_nvdiffrast", "video_pytorch3d"}:
                    if method == "video_nvdiffrast":
                        ref = video_scheduler.get_ref(iteration, view=0)
                    elif iteration < video_scheduler.n_video_iters:
                        ref = video_scheduler._iter_to_frame(iteration)
                    else:
                        ref = gt_img_torch[0, ..., :3].flip(dims=[0])
                    ref_np = ref.cpu().numpy() if torch.is_tensor(ref) else ref
                    ref_np = _replace_white_bg(ref_np)
                    trajectory_ref_frames.append((_to_uint8_frame(ref_np), iteration))

            tx_val = float(model.translation_x.item())
            tx_err = abs(tx_val - args.target_offset)

            step_data = {
                "iteration": iteration,
                "mse_loss": float(loss_all.item()),
                "mse_loss_gt": _mse_gt if _mse_gt is not None else float(loss_all.item()),
                "translation_x_error": tx_err,
                "translation_x": tx_val,
            }
            method_data.append(step_data)

            pbar.set_description(
                f"{method_tag}  loss={loss_all.item():.6f}  tx={tx_val:.4f}  err={tx_err:.4f}"
            )

            if method == "prdpt" and prdpt_avg_img is not None:
                _save_png(
                    out_dir / method_tag / f"iter_{iteration + 1:04d}_avg.png",
                    prdpt_avg_img.detach().cpu().numpy(),
                )

            if (iteration + 1) % args.log_every == 0:
                with torch.no_grad():
                    snap = model.render_nvdiffrast(glctx, with_grad=False).cpu().numpy()
                _save_png(out_dir / method_tag / f"iter_{iteration + 1:04d}.png", snap)

                if method in {"video_nvdiffrast", "video_pytorch3d"}:
                    if method == "video_nvdiffrast":
                        ref = video_scheduler.get_ref(iteration, view=0)
                    elif iteration < video_scheduler.n_video_iters:
                        ref = video_scheduler._iter_to_frame(iteration)
                    else:
                        ref = gt_img_torch[0, ..., :3].flip(dims=[0])
                    ref_np = ref.cpu().numpy() if torch.is_tensor(ref) else ref
                    ref_np = _replace_white_bg(ref_np)
                    _save_png(out_dir / method_tag / f"iter_{iteration + 1:04d}_ref.png", ref_np)
                elif method in {"pytorch3d_rgb", "pytorch3d_sil", "pytorch3d_sil_rgb", "rgbxy_pytorch3d"}:
                    torch_ref = gt_img_torch[0][..., :3].cpu().numpy()
                    torch_ref = _replace_white_bg(torch_ref)
                    _save_png(out_dir / method_tag / f"iter_{iteration + 1:04d}_ref.png", torch_ref, flip=False)
                else:
                    drot_ref = gt_img_drot["images"][0][..., :3].cpu().numpy()
                    drot_ref = _replace_white_bg(drot_ref)
                    _save_png(out_dir / method_tag / f"iter_{iteration + 1:04d}_ref.png", drot_ref)

        elapsed = time.perf_counter() - t_start

        with torch.no_grad():
            final_img = model.render_nvdiffrast(glctx, with_grad=False)

        final_tx = float(model.translation_x.item())
        _save_png(out_dir / f"final_{method_tag}.png", final_img.cpu().numpy())
        print(f"  Final: tx={final_tx:.4f} (target {args.target_offset})")

        method_json = {
            "method": method_tag,
            "total_time_sec": elapsed,
            "final_translation_x": final_tx,
            "target_offset": args.target_offset,
            "steps": method_data,
        }
        json_path = out_dir / f"{method_tag}.json"
        with open(json_path, "w") as f:
            json.dump(method_json, f, indent=2)
        print(f"Saved {json_path}")
        summary_path = write_method_json_timing_summary(out_dir)
        print(f"Saved {summary_path}")

        if args.save_trajectory_video:
            method_dir = out_dir / method_tag
            _save_video(method_dir / "trajectory.mp4", trajectory_frames, fps=args.trajectory_fps)
            if method in {"video_nvdiffrast", "video_pytorch3d"}:
                _save_video(
                    method_dir / "trajectory_ref.mp4",
                    trajectory_ref_frames,
                    fps=args.trajectory_fps,
                )


if __name__ == "__main__":
    main()
