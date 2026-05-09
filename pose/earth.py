"""earth.py – full 6-DOF (rotation + translation) differentiable pose
optimisation for the Earth scene.

Extends the single-axis castle.py pattern to a full SE(3) parameterisation:
  • rotation_xyz  – [rx, ry, rz] Euler angles (intrinsic XYZ order, radians)
  • translation_xyz – [tx, ty, tz] world-space offset applied after rotation

Translation is optimized in normalized scene units internally, then scaled back
to world units for rendering. This keeps learning-rate magnitudes reasonable
when translations span tens of world units while rotations stay in radians.

Usage:
    python earth.py
    python earth.py --methods nvdiffrast prdpt loir video \\
        --target-rotation 0 90 0 --target-translation 0.3 0 0.3
"""

from __future__ import annotations

import json
import os
import sys
import pathlib
import argparse
import time
import random

import imageio
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import torch.nn.functional as F
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

from utils_fns import smoothFn                                   # noqa: E402
from renderer import _normalize, look_at, orthographic, load_scene_mesh  # noqa: E402
from core.NvDiffRastRenderer import NVDiffRastFullRenderer       # noqa: E402
from core.LossFunction import PointLossFunction                  # noqa: E402
from timing_summary import write_subdir_summary_timing_summary   # noqa: E402
from video_guided_methods import VideoFrameScheduler, resize_image_tensor  # noqa: E402

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

DEFAULT_GLB = str(SCENES_ROOT / "earth" / "planet_earth.glb")


class SimpleLogger:
    def __init__(self, out_dir: pathlib.Path) -> None:
        self.out_dir = out_dir

    def add_image(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Differentiable rotation helpers
# ---------------------------------------------------------------------------
def _rotation_x_torch(a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a), torch.sin(a)
    z = torch.zeros([], device=a.device, dtype=torch.float32)
    o = torch.ones([],  device=a.device, dtype=torch.float32)
    return torch.stack([
        torch.stack([ o,  z,  z, z]),
        torch.stack([ z,  c, -s, z]),
        torch.stack([ z,  s,  c, z]),
        torch.stack([ z,  z,  z, o]),
    ])


def _rotation_y_torch(a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a), torch.sin(a)
    z = torch.zeros([], device=a.device, dtype=torch.float32)
    o = torch.ones([],  device=a.device, dtype=torch.float32)
    return torch.stack([
        torch.stack([ c,  z,  s, z]),
        torch.stack([ z,  o,  z, z]),
        torch.stack([-s,  z,  c, z]),
        torch.stack([ z,  z,  z, o]),
    ])


def _rotation_z_torch(a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a), torch.sin(a)
    z = torch.zeros([], device=a.device, dtype=torch.float32)
    o = torch.ones([],  device=a.device, dtype=torch.float32)
    return torch.stack([
        torch.stack([ c, -s,  z, z]),
        torch.stack([ s,  c,  z, z]),
        torch.stack([ z,  z,  o, z]),
        torch.stack([ z,  z,  z, o]),
    ])


def _build_model_matrix(
    rotation_xyz: torch.Tensor,    # [3]  rx, ry, rz (radians)
    translation_xyz: torch.Tensor, # [3]  tx, ty, tz
    center_t: torch.Tensor,        # [3]  mesh centroid
) -> torch.Tensor:
    """SE(3) model matrix: rotate about mesh center, then translate globally.

    Transform order (right-to-left):
        1. T_origin  – shift centroid to origin
        2. R = Rz @ Ry @ Rx – intrinsic XYZ Euler rotation
        3. T_back    – shift centroid back
        4. T_world   – global translation
    """
    Rx = _rotation_x_torch(rotation_xyz[0])
    Ry = _rotation_y_torch(rotation_xyz[1])
    Rz = _rotation_z_torch(rotation_xyz[2])
    R  = Rz @ Ry @ Rx

    T_origin = torch.eye(4, device=device, dtype=torch.float32)
    T_origin[:3, 3] = -center_t
    T_back = torch.eye(4, device=device, dtype=torch.float32)
    T_back[:3, 3] = center_t
    T_world = torch.eye(4, device=device, dtype=torch.float32)
    T_world[:3, 3] = translation_xyz
    return T_world @ T_back @ R @ T_origin


def _rotation_mat_3x3(rotation_xyz: torch.Tensor) -> torch.Tensor:
    """Return the upper-left 3×3 of the composed rotation (for geodesic error)."""
    return (_rotation_z_torch(rotation_xyz[2])
            @ _rotation_y_torch(rotation_xyz[1])
            @ _rotation_x_torch(rotation_xyz[0]))[:3, :3]


# ---------------------------------------------------------------------------
# Vertex normals  (area-weighted, matches DROT's NVDiffRastFullRenderer)
# ---------------------------------------------------------------------------
def _compute_vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    idx  = faces.to(torch.int64)
    v0, v1, v2 = vertices[idx[:, 0]], vertices[idx[:, 1]], vertices[idx[:, 2]]
    face_nrm = torch.cross(v1 - v0, v2 - v0, dim=-1)
    vnrm = torch.zeros_like(vertices)
    vnrm.index_add_(0, idx[:, 0], face_nrm)
    vnrm.index_add_(0, idx[:, 1], face_nrm)
    vnrm.index_add_(0, idx[:, 2], face_nrm)
    return F.normalize(vnrm, p=2, dim=-1, eps=1e-6)


# ---------------------------------------------------------------------------
# Core 6-DOF nvdiffrast render  (Phong shading matching DROT)
# ---------------------------------------------------------------------------
def _render_6dof(
    glctx,
    vertices: torch.Tensor,     # [V, 3]
    faces: torch.Tensor,        # [F, 3]  int32
    uvs: torch.Tensor,          # [V, 2]
    tex: torch.Tensor,          # [H, W, 4]
    model_mat: torch.Tensor,    # [4, 4]
    proj_view: torch.Tensor,    # [4, 4]
    resolution: int,
    # shading (None = unlit / albedo-only)
    light_pos:      torch.Tensor | None = None,  # [3] world-space
    eye_pos:        torch.Tensor | None = None,  # [3] world-space
    ambient_color:  torch.Tensor | None = None,  # [1, 3]
    diffuse_color:  torch.Tensor | None = None,  # [1, 3]
    specular_color: torch.Tensor | None = None,  # [1, 3]
    shininess:      float = 64.0,
    light_power:    float = 1.0,
) -> torch.Tensor:
    """Rasterize with a full 4×4 model matrix and optional Phong shading.

    Normals are computed in world space by transforming vertex normals through
    the upper-left 3×3 of model_mat (rotation only; translation doesn't affect
    directions). Returns [1, H, W, 4].
    """
    shade = light_pos is not None

    # --- geometry ---
    pos_w    = torch.cat([vertices, torch.ones(vertices.shape[0], 1, device=device)], dim=1)
    # World-space positions after model transform (for lighting)
    pos_world = (pos_w @ model_mat.t())[..., :3].contiguous()  # [V, 3]
    pos_clip  = (pos_w @ (proj_view @ model_mat).t())[None]  # [1, V, 4]

    rast_out, _ = dr.rasterize(glctx, pos_clip, faces, resolution=[resolution, resolution])
    uv_px,   _  = dr.interpolate(uvs[None],       rast_out, faces)
    color        = dr.texture(tex[None], uv_px, filter_mode="linear")  # [1,H,W,C]

    if shade:
        # Transform vertex normals to world space
        vnrm_obj   = _compute_vertex_normals(vertices, faces)                      # [V, 3]
        R3         = model_mat[:3, :3].contiguous()
        vnrm_world = F.normalize(vnrm_obj @ R3.t(), p=2, dim=-1, eps=1e-6).contiguous()

        pos_px,  _ = dr.interpolate(pos_world[None].contiguous(),  rast_out, faces)  # [1,H,W,3]
        nrm_px,  _ = dr.interpolate(vnrm_world[None].contiguous(), rast_out, faces)  # [1,H,W,3]
        nrm_px     = F.normalize(nrm_px, p=2, dim=-1, eps=1e-6)

        lp  = light_pos.view(1, 1, 1, 3)
        vp  = eye_pos.view(1, 1, 1, 3)

        direction     = F.normalize(lp - pos_px, p=2, dim=-1, eps=1e-6)
        cos_angle     = (nrm_px * direction).sum(-1, keepdim=True)
        front_mask    = (cos_angle > 0).float()
        light_diffuse = light_power * cos_angle * front_mask

        view_dir      = F.normalize(vp - pos_px, p=2, dim=-1, eps=1e-6)
        reflect_dir   = -direction + 2.0 * cos_angle * nrm_px
        alpha         = F.relu((view_dir * reflect_dir).sum(-1, keepdim=True)) * front_mask
        light_specular = light_power * alpha.pow(shininess)

        rgb    = color[..., :3]
        shaded = (ambient_color + light_diffuse * diffuse_color) * rgb \
                 + light_specular * specular_color
        color  = torch.cat([shaded, color[..., 3:4]], dim=-1)

    mask  = (rast_out[..., 3:4] > 0).to(color.dtype)
    bg    = torch.tensor([0.03, 0.05, 0.18, 1.0], device=device, dtype=color.dtype).view(1, 1, 1, 4)
    color = color * mask + bg * (1.0 - mask)
    color = dr.antialias(color, rast_out, pos_clip, faces)
    return color   # [1, H, W, 4]


# ---------------------------------------------------------------------------
# Earth model
# ---------------------------------------------------------------------------
class Earth:
    """Wraps the Earth GLB with differentiable 6-DOF pose parameters."""

    def __init__(self) -> None:
        self._vertices:   torch.Tensor | None = None
        self._faces:      torch.Tensor | None = None
        self._uvs:        torch.Tensor | None = None
        self._tex:        torch.Tensor | None = None
        self._proj_view:  torch.Tensor | None = None
        self._center_t:   torch.Tensor | None = None
        self._radius:     float = 1.0
        self._resolution: int   = 512

        self._p3d_mesh_template: Meshes | None = None
        self._p3d_cameras: OpenGLOrthographicCameras | None = None
        self._p3d_lights:  PointLights | None = None
        self._sensors: list | None = None
        self._nv_renderer: NVDiffRastFullRenderer | None = None
        self._eye_t: torch.Tensor | None = None  # camera / headlamp position [3]

        # default Phong material (set by load_scene)
        self._ambient_color:  torch.Tensor | None = None
        self._diffuse_color:  torch.Tensor | None = None
        self._specular_color: torch.Tensor | None = None
        self._shininess:  float = 64.0
        self._light_power: float = 1.0

        self.rotation_xyz:    torch.Tensor | None = None   # [3] radians
        self._translation_param: torch.Tensor | None = None   # [3] normalized scene units
        self._translation_scale: float = 1.0
        self.optim: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    def load_scene(
        self,
        glb_path: str,
        init_rotation: tuple  = (0.0, 0.0, 0.0),   # degrees (rx, ry, rz)
        init_translation: tuple = (0.0, 0.0, 0.0), # world units
        resolution: int = 512,
        drot_resolution: int | None = None,
        camera_scale: float | None = None,
        translation_bound: float = 0.0,
        # Phong material
        ambient_color:  tuple = (0.08, 0.08, 0.08),
        diffuse_color:  tuple = (0.8,  0.8,  0.8 ),
        specular_color: tuple = (0.6,  0.6,  0.6 ),
        shininess: float = 64.0,
        light_power: float = 1.4,
    ) -> None:
        mesh, texture = load_scene_mesh(glb_path)
        # Recenter the Earth at its fitted sphere center so all transforms are
        # applied about the globe itself, not about an arbitrary scene origin.
        sphere_center = np.asarray(mesh.bounding_sphere.primitive.center, dtype=np.float32)
        mesh = mesh.copy()
        mesh.vertices = mesh.vertices - sphere_center[None, :]

        bounds     = mesh.bounds.astype(np.float32)
        center_np  = mesh.vertices.astype(np.float32).mean(axis=0)
        extents    = bounds[1] - bounds[0]
        radius     = 0.5 * float(np.linalg.norm(extents))
        max_radius = radius
        self._radius = radius
        default_camera_scale = 1.15 * max_radius + float(translation_bound)
        print(f"  Scene radius={radius:.4f}  frustum half-width={default_camera_scale:.4f}"
              f"  — keep translations well below {radius:.2f} to stay in frame")

        if camera_scale is None:
            camera_scale = default_camera_scale

        # Camera: 3/4 view (same as castle.py)
        view_dir = _normalize(np.asarray([1.0, 0.5, 1.0], dtype=np.float32))
        up_dir   = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        eye_dist = 4.0 * max_radius
        eye_np   = center_np + view_dir * eye_dist
        depth_margin = max_radius + float(translation_bound) + 0.25 * radius
        near     = max(0.05 * radius, eye_dist - depth_margin)
        far      = eye_dist + depth_margin

        view_np  = look_at(eye_np, center_np, up_dir)
        proj_np  = orthographic(camera_scale, near, far)
        vp_np    = proj_np @ view_np

        self._vertices  = torch.from_numpy(mesh.vertices.astype(np.float32)).to(device)
        self._faces     = torch.from_numpy(mesh.faces.astype(np.int32)).to(device)
        self._uvs       = torch.from_numpy(mesh.visual.uv.astype(np.float32)).to(device)
        self._tex       = torch.from_numpy(texture.astype(np.float32)).to(device)
        self._proj_view = torch.tensor(vp_np, dtype=torch.float32, device=device)
        self._center_t  = torch.tensor(center_np, dtype=torch.float32, device=device)
        self._eye_t     = torch.tensor(eye_np, dtype=torch.float32, device=device)
        self._resolution = resolution
        if drot_resolution is None:
            drot_resolution = resolution

        # Phong material tensors
        self._ambient_color  = torch.tensor([ambient_color],  dtype=torch.float32, device=device)
        self._diffuse_color  = torch.tensor([diffuse_color],  dtype=torch.float32, device=device)
        self._specular_color = torch.tensor([specular_color], dtype=torch.float32, device=device)
        self._shininess  = shininess
        self._light_power = light_power

        # pytorch3d assets for DROT
        tex_p3d    = np.flip(texture, axis=0).copy()
        tex_maps   = torch.from_numpy(tex_p3d[..., :3].astype(np.float32)).unsqueeze(0).to(device)
        faces_uvs  = torch.from_numpy(mesh.faces.astype(np.int64)).unsqueeze(0).to(device)
        verts_uvs  = torch.from_numpy(mesh.visual.uv.astype(np.float32)).unsqueeze(0).to(device)
        textures_uv = TexturesUV(maps=tex_maps, faces_uvs=faces_uvs, verts_uvs=verts_uvs)
        verts_p3d  = self._vertices.unsqueeze(0)
        faces_p3d  = self._faces.to(torch.int64).unsqueeze(0)
        self._p3d_mesh_template = Meshes(verts=verts_p3d, faces=faces_p3d, textures=textures_uv)

        position = torch.tensor(eye_np, dtype=torch.float32).unsqueeze(0).to(device)
        at       = torch.tensor(center_np, dtype=torch.float32).unsqueeze(0).to(device)
        R        = look_at_rotation(position, at=at, device=device)
        R_p3d    = R.clone()
        R_p3d[:, :, 0] *= -1
        R_p3d[:, :, 2] *= -1
        T_p3d = -torch.bmm(R_p3d.transpose(1, 2), position.unsqueeze(-1)).squeeze(-1)
        self._p3d_cameras = OpenGLOrthographicCameras(
            device=device, R=R_p3d, T=T_p3d,
            znear=near, zfar=far,
            top=camera_scale, bottom=-camera_scale,
            left=-camera_scale, right=camera_scale,
        )
        self._p3d_lights = PointLights(
            device=device, location=position.tolist(),
            ambient_color=[[1.0, 1.0, 1.0]],
            diffuse_color=[[1.0, 1.0, 1.0]],
            specular_color=[[1.0, 1.0, 1.0]],
        )
        view_t = torch.tensor(view_np, dtype=torch.float32).unsqueeze(0).to(device)
        proj_t = torch.tensor(proj_np, dtype=torch.float32).unsqueeze(0).to(device)
        vp_t   = torch.tensor(vp_np,  dtype=torch.float32).unsqueeze(0).to(device)
        self._sensors = [{
            "position":           position,
            "resolution":         (drot_resolution, drot_resolution),
            "matrix":             vp_t,
            "camera_matrix":      view_t,
            "perspective_matrix": proj_t,
            "center":             at,
            "init_position":      position.clone(),
            "init_center":        at.clone(),
        }]
        self._nv_renderer = NVDiffRastFullRenderer(
            device=device,
            settings={"background": "white", "shading": False, "light_power": 1.0},
            resolution=[drot_resolution, drot_resolution],
        )

        # Differentiable pose parameters
        rot_rad  = np.deg2rad(init_rotation).astype(np.float32)
        self.rotation_xyz = torch.tensor(
            rot_rad, dtype=torch.float32, device=device, requires_grad=True)
        self._translation_scale = max(radius, 1e-6)
        init_translation_np = np.asarray(init_translation, dtype=np.float32)
        self._translation_param = torch.tensor(
            init_translation_np / self._translation_scale,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.optim = [self.rotation_xyz, self._translation_param]

    @property
    def translation_xyz(self) -> torch.Tensor:
        return self._translation_param * self._translation_scale

    # ------------------------------------------------------------------
    def _model_mat(self) -> torch.Tensor:
        return _build_model_matrix(self.rotation_xyz, self.translation_xyz, self._center_t)

    def render_nvdiffrast(self, glctx, with_grad: bool = True) -> torch.Tensor:
        rot = self.rotation_xyz    if with_grad else self.rotation_xyz.detach()
        tra = self.translation_xyz if with_grad else self.translation_xyz.detach()
        model = _build_model_matrix(rot, tra, self._center_t)
        color = _render_6dof(
            glctx, self._vertices, self._faces, self._uvs, self._tex,
            model, self._proj_view, self._resolution,
        )
        img = color[0]
        if img.shape[-1] == 4:
            img = torch.cat([img[..., :3], torch.ones_like(img[..., 3:4])], dim=-1)
        return img   # [H, W, 4]

    def gen_mesh_pytorch3d(self) -> Meshes:
        """Apply current 6-DOF transform to the mesh template for DROT renders."""
        model = self._model_mat()
        R3   = model[:3, :3]    # [3, 3]
        t3   = model[:3,  3]    # [3]

        src  = self._p3d_mesh_template.clone()
        verts = src.verts_packed()            # [V, 3]
        verts = verts @ R3.t() + t3           # rotate + translate
        src   = src.update_padded(verts.unsqueeze(0))
        return src

    def scene_dict(self, material: Materials) -> dict:
        return {
            "meshes":   [{"model": self.gen_mesh_pytorch3d()}],
            "sensors":  self._sensors,
            "material": material,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _composite_bg(img: torch.Tensor, bg: tuple = (0.03, 0.05, 0.18)) -> torch.Tensor:
    bg_t = torch.tensor(bg, device=img.device, dtype=img.dtype).view(1, 1, -1)
    if img.shape[-1] == 4:
        return img[..., :3] + (1.0 - img[..., 3:4]) * bg_t
    return img


def _save_png(path: pathlib.Path, img: np.ndarray | torch.Tensor, flip: bool = True) -> None:
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    if flip:
        img = img[::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8))
    print(f"Saved {path}")


def _to_uint8_frame(img: np.ndarray | torch.Tensor, flip: bool = True) -> np.ndarray:
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    if flip:
        img = img[::-1]
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
    annotated_frames = [_annotate_iteration(frame, iteration) for frame, iteration in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, annotated_frames, fps=fps)
    print(f"Saved {path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def bilinear_downsample(t: torch.Tensor, size: int) -> torch.Tensor:
    """Bilinearly downsample [1,H,W,C] or [H,W,C] -> matching shape with size."""
    spatial = t.shape[-3:-1]
    if spatial[0] == size and spatial[1] == size:
        return t
    squeezed = False
    if t.ndim == 3:
        t = t.unsqueeze(0)
        squeezed = True
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


def _geodesic_error_deg(R_pred: torch.Tensor, R_target: torch.Tensor) -> float:
    """Geodesic angle (degrees) between two 3×3 rotation matrices."""
    with torch.no_grad():
        R_rel   = R_pred.t() @ R_target
        trace   = torch.clamp((torch.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        return float(torch.rad2deg(torch.acos(trace)).item())


class LOILoss(torch.nn.Module):
    def __init__(self, channels: int = 3, device=None):
        super().__init__()
        dev = device if device is not None else torch.device("cuda")
        n_bins, sigma, alpha = 8, (1.0, 15.0, 45.0), (1.0, 15.0, 45.0)
        beta = [1.0 / n_bins]
        bounds = [-4.0 * beta[0], 1.0 + 4.0 * beta[0]]
        kernel_size = 121
        self.n_bins, self.n_sigma, self.n_alpha = n_bins, len(sigma), len(alpha)
        self.n_beta, self.channels, self.kernel_size = len(beta), channels, kernel_size
        sigma_ks = max(3, int(sigma[-1] * 4))
        if sigma_ks % 2 == 0:
            sigma_ks += 1
        self.sigma_ks = sigma_ks

        def _g2d(sz, sig):
            x = torch.linspace(-sz // 2 + 1, sz // 2, sz, dtype=torch.float32)
            gx, gy = torch.meshgrid(x, x, indexing="ij")
            k = torch.exp(-(gx**2 + gy**2) / (2.0 * sig**2))
            return k / k.sum()

        sk = torch.stack([_g2d(sigma_ks, s) for s in sigma])
        self.register_buffer("sigma_kernel", sk.unsqueeze(1).repeat(channels, 1, 1, 1))
        bin_edges = torch.linspace(bounds[0], bounds[1], n_bins + 1)
        bin_val = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        self.register_buffer("bin_val", bin_val.view(1, n_bins, 1, 1, 1, 1))
        self.register_buffer("beta_val", torch.tensor(beta).view(-1, 1, 1, 1, 1, 1))
        n_groups = len(beta) * n_bins * channels * len(sigma)
        ak = torch.stack([_g2d(kernel_size, s) for s in alpha])
        self.register_buffer("alpha_kernel", ak.unsqueeze(1).repeat(n_groups, 1, 1, 1))
        self.to(dev)

    def forward(self, im: torch.Tensor) -> torch.Tensor:
        im = im.permute(2, 0, 1)
        with torch.no_grad():
            im.data = im.data.clamp(0.0, 1.0)
        c, h, w = im.shape
        cs = F.conv2d(im.unsqueeze(0), self.sigma_kernel, padding=self.sigma_ks // 2, groups=c)
        cs = cs.view(c, self.n_sigma, h, w)
        diff = cs.unsqueeze(0).unsqueeze(1) - self.bin_val
        iso = torch.exp(-0.5 * (diff / self.beta_val).pow(2))
        iso = iso / (np.sqrt(2.0 * np.pi) * self.beta_val)
        n_groups = self.n_beta * self.n_bins * c * self.n_sigma
        iso = iso.reshape(n_groups, h, w)
        loi = F.conv2d(iso.unsqueeze(0), self.alpha_kernel, padding=self.kernel_size // 2, groups=n_groups)
        loi = loi.reshape(self.n_beta, self.n_bins, c, self.n_sigma, self.n_alpha, h, w)
        return loi.permute(4, 0, 3, 2, 5, 6, 1)

    @staticmethod
    def emd(loi1: torch.Tensor, loi2: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(torch.cumsum(loi1, dim=-1) - torch.cumsum(loi2, dim=-1)))


# ---------------------------------------------------------------------------
# PRDPT batch render  (theta_p: [N, 6]  rx,ry,rz,tx,ty,tz)
# ---------------------------------------------------------------------------
def render_smooth_prince(theta_p: torch.Tensor, update_fn, context_args: dict):
    glctx    = context_args["glctx"]
    model    = context_args["model"]
    gt_img   = context_args["gt_image"]   # [H, W, 3] at loss resolution
    translation_scale = context_args["translation_scale"]
    loss_resolution = context_args["loss_resolution"]

    losses, avg_img = [], None
    with torch.no_grad():
        for i in range(theta_p.shape[0]):
            rot = theta_p[i, :3]
            tra = theta_p[i, 3:] * translation_scale
            mat = _build_model_matrix(rot, tra, model._center_t)
            color = _render_6dof(
                glctx, model._vertices, model._faces, model._uvs,
                model._tex, mat, model._proj_view, model._resolution,
            )
            img = color[0]
            img = bilinear_downsample(img[..., :3], loss_resolution)
            loss = torch.mean((img - gt_img) ** 2)
            losses.append(loss)
            if avg_img is None:
                avg_img = color[0].detach()

    return torch.stack(losses), avg_img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full 6-DOF pose optimisation for the Earth scene."
    )
    parser.add_argument("--input", default=DEFAULT_GLB)
    parser.add_argument("--output-dir",
                        default=str(OUTPUTS_ROOT / "earth"))
    parser.add_argument("--resolution",   type=int,   default=512)
    parser.add_argument(
        "--loss-resolution",
        type=int,
        default=256,
        help="Resolution used for optimization losses after bilinear downsampling from render resolution.",
    )
    parser.add_argument("--camera-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)

    # Pose init / target (degrees and world units)
    parser.add_argument("--init-rotation",    type=float, nargs=3,
                        default=[0.0, 0.0, 0.0],   metavar=("RX", "RY", "RZ"),
                        help="Initial Euler angles (degrees, XYZ intrinsic).")
    parser.add_argument("--init-translation", type=float, nargs=3,
                        default=[0.0, 0.0, 0.0],   metavar=("TX", "TY", "TZ"),
                        help="Initial translation (world units).")
    parser.add_argument("--target-rotation",    type=float, nargs=3,
                        default=[0.0, 90.0, 0.0],  metavar=("RX", "RY", "RZ"),
                        help="Target Euler angles (degrees, XYZ intrinsic).")
    parser.add_argument("--target-translation", type=float, nargs=3,
                        default=[0.0, 0.0, 0.0],   metavar=("TX", "TY", "TZ"),
                        help="Target translation (world units).")

    # Optimiser
    parser.add_argument("--n-iters",   type=int,   default=300)
    parser.add_argument("--lr",        type=float, default=0.05)
    parser.add_argument("--gamma",     type=float, default=1.0,
                        help="LR scheduler multiplicative factor per step.")
    parser.add_argument("--log-every", type=int,   default=50)

    # Methods
    parser.add_argument(
        "--methods", nargs="+", default=["nvdiffrast"],
        choices=["nvdiffrast", "prdpt", "loir", "video", "rgbxy"],
        help="Optimisation method(s) to run.",
    )
    parser.add_argument("--video-path", type=str, default=None)
    parser.add_argument("--video-ratio", type=float, default=0.8)
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
        "--trajectory-resolution",
        type=int,
        default=None,
        help="Resolution used for saved trajectory video frames. Defaults to --resolution.",
    )
    parser.add_argument(
        "--video-loss-resolution",
        type=int,
        default=None,
        help="Resolution used for video loss after bilinear downsampling from the render resolution. Defaults to --loss-resolution.",
    )
    parser.add_argument(
        "--video-loss",
        choices=["tone_mse", "gaussian_pyramid"],
        default="gaussian_pyramid",
        help="Image-space loss used by the video method.",
    )
    parser.add_argument("--pyramid-levels", type=int, default=4)

    # PRDPT
    parser.add_argument("--prdpt-sigma",     type=float, default=0.3,
                        help="Initial PRDPT kernel bandwidth.")
    parser.add_argument("--prdpt-sigma-min", type=float, default=0.01,
                        help="Minimum sigma after annealing (0 = no annealing).")
    parser.add_argument("--prdpt-nsamples",  type=int,   default=4,
                        help="Monte Carlo samples per PRDPT iteration.")

    # Phong shading
    parser.add_argument("--ambient",   type=float, nargs=3,
                        default=[0.08, 0.08, 0.08], metavar=("R", "G", "B"),
                        help="Ambient colour (RGB, 0-1).")
    parser.add_argument("--diffuse",   type=float, nargs=3,
                        default=[0.8,  0.8,  0.8 ], metavar=("R", "G", "B"),
                        help="Diffuse colour (RGB, 0-1).")
    parser.add_argument("--specular",  type=float, nargs=3,
                        default=[0.6,  0.6,  0.6 ], metavar=("R", "G", "B"),
                        help="Specular colour (RGB, 0-1).")
    parser.add_argument("--shininess", type=float, default=64.0,
                        help="Phong shininess exponent.")
    parser.add_argument("--light-power", type=float, default=2.0,
                        help="Light power multiplier.")

    args = parser.parse_args()
    set_seed(args.seed)

    shading_kwargs = dict(
        ambient_color=tuple(args.ambient),
        diffuse_color=tuple(args.diffuse),
        specular_color=tuple(args.specular),
        shininess=args.shininess,
        light_power=args.light_power,
    )
    loss_resolution = args.loss_resolution
    video_loss_resolution = args.video_loss_resolution or loss_resolution
    trajectory_resolution = args.trajectory_resolution or args.resolution
    translation_bound = max(
        float(np.linalg.norm(np.asarray(args.init_translation, dtype=np.float32))),
        float(np.linalg.norm(np.asarray(args.target_translation, dtype=np.float32))),
    )

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = SimpleLogger(out_dir)

    glctx = dr.RasterizeCudaContext()

    # ------------------------------------------------------------------
    # 1. Ground-truth render at target pose
    # ------------------------------------------------------------------
    print("Building ground-truth render …")
    with torch.no_grad():
        gt_model = Earth()
        gt_model.load_scene(
            args.input,
            init_rotation=args.target_rotation,
            init_translation=args.target_translation,
            resolution=args.resolution,
            drot_resolution=loss_resolution,
            camera_scale=args.camera_scale,
            translation_bound=translation_bound,
            **shading_kwargs,
        )
        gt_img = gt_model.render_nvdiffrast(glctx, with_grad=False)  # [H,W,4]
        gt_img_loss = bilinear_downsample(gt_img[..., :3], loss_resolution)
        gt_scene = gt_model.scene_dict(
            Materials(
                device=device,
                ambient_color=((0.5, 0.5, 0.5),),
                diffuse_color=((0.7, 0.7, 0.7),),
                specular_color=((0.2, 0.2, 0.2),),
            )
        )
        gt_img_drot = gt_model._nv_renderer.render(gt_scene, DcDt=False, view=0)

    _save_png(out_dir / "gt.png", _composite_bg(gt_img).cpu().numpy())
    print(f"  GT   mean pixel: {gt_img[..., :3].mean().item():.4f}")

    # Precompute target rotation matrix for geodesic error logging
    target_rot_rad = np.deg2rad(args.target_rotation).astype(np.float32)
    target_rot_t   = torch.tensor(target_rot_rad, device=device)
    R_target_3x3   = _rotation_mat_3x3(target_rot_t)[:3, :3].detach()
    target_tra_t   = torch.tensor(args.target_translation, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # 2. Init render
    # ------------------------------------------------------------------
    print("Building init render …")
    with torch.no_grad():
        init_model = Earth()
        init_model.load_scene(
            args.input,
            init_rotation=args.init_rotation,
            init_translation=args.init_translation,
            resolution=args.resolution,
            drot_resolution=loss_resolution,
            camera_scale=args.camera_scale,
            translation_bound=translation_bound,
            **shading_kwargs,
        )
        init_img = init_model.render_nvdiffrast(glctx, with_grad=False)

    _save_png(out_dir / "init.png", _composite_bg(init_img).cpu().numpy())
    print(f"  Init mean pixel: {init_img[..., :3].mean().item():.4f}")
    diff = (gt_img[..., :3] - init_img[..., :3]).abs().mean().item()
    print(f"  Mean absolute pixel difference (gt vs init): {diff:.4f}"
          + ("  ← renders are identical!" if diff < 1e-4 else "  ← renders differ (good)"))

    # ------------------------------------------------------------------
    # 3. Method loop
    # ------------------------------------------------------------------
    all_summaries: dict = {}

    for method in args.methods:
        print(f"\n=== {method.upper()} ===")

        model = Earth()
        model.load_scene(
            args.input,
            init_rotation=args.init_rotation,
            init_translation=args.init_translation,
            resolution=args.resolution,
            drot_resolution=loss_resolution,
            camera_scale=args.camera_scale,
            translation_bound=translation_bound,
            **shading_kwargs,
        )
        vis_model = None
        if args.save_trajectory_video:
            vis_model = Earth()
            vis_model.load_scene(
                args.input,
                init_rotation=args.init_rotation,
                init_translation=args.init_translation,
                resolution=trajectory_resolution,
                drot_resolution=loss_resolution,
                camera_scale=args.camera_scale,
                translation_bound=translation_bound,
                **shading_kwargs,
            )

        optimizer = torch.optim.Adam(model.optim, lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=args.gamma)
        material = Materials(
            device=device,
            ambient_color=((0.5, 0.5, 0.5),),
            diffuse_color=((0.7, 0.7, 0.7),),
            specular_color=((0.2, 0.2, 0.2),),
        )
        loss_func = PointLossFunction(
            debug=False,
            resolution=[loss_resolution, loss_resolution],
            settings={
                "matching_weight": 1.0,
                "matching_interval": 0,
                "matcher": "Sinkhorn",
            },
            device=device,
            renderer=model._nv_renderer,
            num_views=1,
            logger=logger,
        )

        # PRDPT setup – theta is 6-D: [rx, ry, rz, tx, ty, tz]
        prdpt_ctx   = None
        smooth_render = None
        loi_model = None
        target_loi = None
        video_scheduler = None
        if method == "prdpt":
            prdpt_ctx = {
                "glctx":      glctx,
                "model":      model,
                "gt_image":   gt_img_loss,
                "translation_scale": model._translation_scale,
                "loss_resolution": loss_resolution,
                "sigma":      args.prdpt_sigma,
                "nsamples":   args.prdpt_nsamples,
                "antithetic": True,
                "sampler":    "importance",
                "update_fn":  None,
                "device":     device,
            }
            smooth_render = smoothFn(render_smooth_prince, context_args=None, device=device)
        elif method == "loir":
            loi_model = LOILoss(channels=3, device=device)
            with torch.no_grad():
                target_loi = loi_model(gt_img_loss)
        elif method == "video":
            if args.video_path is None:
                print("  [WARN] --video-path not set for video; skipping.")
                continue
            video_scheduler = VideoFrameScheduler(
                video_path=args.video_path,
                n_iter=args.n_iters,
                fallback=gt_img[..., :3],
                resolution=args.resolution,
                device=device,
                video_ratio=args.video_ratio,
            )

        method_data: list[dict] = []
        t_start = time.perf_counter()
        method_dir = out_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        trajectory_frames: list[tuple[np.ndarray, int]] = []
        trajectory_ref_frames: list[tuple[np.ndarray, int]] = []

        pbar = tqdm(range(args.n_iters), desc=method)
        for iteration in pbar:
            current_ref = gt_img[..., :3]

            if method == "nvdiffrast":
                pred = model.render_nvdiffrast(glctx, with_grad=True)
                pred_loss = bilinear_downsample(pred[..., :3], loss_resolution)
                loss = torch.mean((pred_loss - gt_img_loss) ** 2)

            elif method == "loir":
                pred = model.render_nvdiffrast(glctx, with_grad=True)
                pred_loss = bilinear_downsample(pred[..., :3], loss_resolution)
                est_loi = loi_model(pred_loss)
                loss = LOILoss.emd(target_loi, est_loi)

            elif method == "video":
                pred = model.render_nvdiffrast(glctx, with_grad=True)
                current_ref = video_scheduler.get_ref(iteration)
                pred_loss = bilinear_downsample(pred[..., :3], video_loss_resolution)
                ref_loss = bilinear_downsample(current_ref, video_loss_resolution)
                if args.video_loss == "gaussian_pyramid":
                    loss = gaussian_pyramid_loss(
                        pred_loss, ref_loss, n_levels=args.pyramid_levels
                    )
                else:
                    pred_tm = pred_loss / (1.0 + pred_loss)
                    ref_tm = ref_loss / (1.0 + ref_loss)
                    loss = (pred_tm - ref_tm).pow(2).mean()

            elif method == "rgbxy":
                scene = model.scene_dict(material)
                loss, _ = loss_func(gt_img_drot, iteration=iteration, scene=scene, view=0)

            elif method == "prdpt":
                if args.prdpt_sigma_min > 0:
                    frac = iteration / max(args.n_iters - 1, 1)
                    prdpt_ctx["sigma"] = max(
                        args.prdpt_sigma_min,
                        args.prdpt_sigma * (1.0 - frac) + args.prdpt_sigma_min * frac,
                    )
                # theta: [1, 6]  = [rx, ry, rz, tx, ty, tz]
                theta = torch.cat([model.rotation_xyz, model._translation_param]).reshape(1, 6)
                loss, _ = smooth_render(theta, prdpt_ctx)

            else:
                loss = torch.tensor(0.0, device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if args.save_trajectory_video:
                with torch.no_grad():
                    assert vis_model is not None
                    vis_model.rotation_xyz = model.rotation_xyz.detach().clone()
                    vis_model._translation_param = model._translation_param.detach().clone()
                    traj_render = vis_model.render_nvdiffrast(glctx, with_grad=False)
                trajectory_frames.append(
                    (_to_uint8_frame(_composite_bg(traj_render).cpu().numpy()), iteration)
                )
                if method == "video":
                    ref_vis = resize_image_tensor(current_ref.detach(), trajectory_resolution)
                    trajectory_ref_frames.append(
                        (_to_uint8_frame(ref_vis.cpu().numpy(), flip=True), iteration)
                    )

            with torch.no_grad():
                R_pred_3x3 = _rotation_mat_3x3(model.rotation_xyz)[:3, :3]
                rot_err_deg  = _geodesic_error_deg(R_pred_3x3, R_target_3x3)
                trans_err    = float((model.translation_xyz - target_tra_t).norm().item())
                rot_deg      = np.rad2deg(model.rotation_xyz.detach().cpu().numpy()).tolist()
                tra_val      = model.translation_xyz.detach().cpu().numpy().tolist()

            step_data = {
                "iteration":          iteration,
                "loss":               float(loss.item()),
                "rot_error_deg":      rot_err_deg,
                "trans_error":        trans_err,
                "rotation_xyz_deg":   rot_deg,
                "translation_xyz":    tra_val,
            }
            method_data.append(step_data)

            pbar.set_description(
                f"{method}  loss={loss.item():.5f}"
                f"  rot_err={rot_err_deg:.1f}°  trans_err={trans_err:.4f}"
            )

            if (iteration + 1) % args.log_every == 0:
                with torch.no_grad():
                    snap = model.render_nvdiffrast(glctx, with_grad=False)
                _save_png(method_dir / f"iter_{iteration + 1:04d}.png",
                          _composite_bg(snap).cpu().numpy())
                if method == "video":
                    _save_png(
                        method_dir / f"iter_{iteration + 1:04d}_ref.png",
                        current_ref.detach().cpu().numpy(),
                    )

        elapsed = time.perf_counter() - t_start

        with torch.no_grad():
            final_img = model.render_nvdiffrast(glctx, with_grad=False)
        _save_png(out_dir / f"final_{method}.png", _composite_bg(final_img).cpu().numpy())

        summary = {
            "method":           method,
            "total_time_sec":   elapsed,
            "final_rotation_xyz_deg":  np.rad2deg(model.rotation_xyz.detach().cpu().numpy()).tolist(),
            "final_translation_xyz":   model.translation_xyz.detach().cpu().numpy().tolist(),
            "target_rotation_deg":     args.target_rotation,
            "target_translation":      args.target_translation,
            "final_rot_error_deg":     method_data[-1]["rot_error_deg"],
            "final_trans_error":       method_data[-1]["trans_error"],
            "steps":                   method_data,
        }
        (method_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        all_summaries[method] = {k: v for k, v in summary.items() if k != "steps"}

        print(f"  Final rot_err={summary['final_rot_error_deg']:.2f}°"
              f"  trans_err={summary['final_trans_error']:.4f}"
              f"  ({elapsed:.1f}s)")

        if args.save_trajectory_video:
            _save_video(method_dir / "trajectory.mp4", trajectory_frames, fps=args.trajectory_fps)
            if method == "video":
                _save_video(
                    method_dir / "trajectory_ref.mp4",
                    trajectory_ref_frames,
                    fps=args.trajectory_fps,
                )

    (out_dir / "all_summaries.json").write_text(json.dumps(all_summaries, indent=2))
    write_subdir_summary_timing_summary(out_dir)
    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
