"""floor.py – nvdiffrast + DROT renderer for the bathroom floor quad scene.

Loads bathroomFloor.fbx (a flat floor quad) and its PBR textures, positions
a camera looking down at the quad, and renders via nvdiffrast and DROT.

Usage:
    python floor.py
    python floor.py --resolution 1024 --output-dir results/floor
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import pathlib
import time

import imageio
import torch.nn.functional as F
import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    Materials,
    PointLights,
    TexturesUV,
    look_at_rotation,
    OpenGLOrthographicCameras,
)

import nvdiffrast.torch as dr

# ---------------------------------------------------------------------------
# Path setup – mirror the same layout used in codes/pose/castle.py
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VNC_DIR    = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
REPO_DIR   = os.path.normpath(os.path.join(VNC_DIR, ".."))
LEGACY_CODES_DIR = os.path.join(REPO_DIR, "codes")

DROT_ROOT  = os.path.join(LEGACY_CODES_DIR, "transformations", "DROT")
SRC_DIR    = os.path.join(LEGACY_CODES_DIR, "transformations", "src")
PRDPT_DIR  = os.path.join(VNC_DIR, "pose", "prdpt")
if not os.path.isdir(PRDPT_DIR):
    PRDPT_DIR = os.path.join(LEGACY_CODES_DIR, "pose", "prdpt")
for p in (DROT_ROOT, SRC_DIR, SCRIPT_DIR, PRDPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.NvDiffRastRenderer import NVDiffRastFullRenderer  # noqa: E402
from utils_fns import smoothFn                               # noqa: E402

# ---------------------------------------------------------------------------
# Paths to scene assets
# ---------------------------------------------------------------------------
SCENE_DIR = os.path.join(VNC_DIR, "scenes", "lights", "floor")
FBX_PATH  = os.path.join(SCENE_DIR, "fbx", "source", "bathroomFloor.fbx")
TEX_DIR   = os.path.join(SCENE_DIR, "fbx", "textures")

TEX_FILES = {
    "base_color": os.path.join(TEX_DIR, "bathroom_floor_Base_Color.png"),
    "roughness":  os.path.join(TEX_DIR, "bathroom_floor_Roughness.png"),
    "metallic":   os.path.join(TEX_DIR, "bathroom_floor_Metallic.png"),
    "normal":     os.path.join(TEX_DIR, "bathroom_floor_Normal_OpenGL.png"),
    "height":     os.path.join(TEX_DIR, "bathroom_floor_Height.png"),
}

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------

# Blender conversion script (written to a temp file and executed)
_BLENDER_CONVERT = """\
import bpy, sys, os
fbx_path = sys.argv[sys.argv.index("--") + 1]
obj_path = sys.argv[sys.argv.index("--") + 2]
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
bpy.ops.import_scene.fbx(filepath=fbx_path)
bpy.ops.object.select_all(action="SELECT")
try:
    bpy.ops.wm.obj_export(
        filepath=obj_path,
        export_selected_objects=True,
        export_uv=True,
        export_normals=True,
        export_materials=False,
        export_triangulated_mesh=True,
        forward_axis="NEGATIVE_Z",
        up_axis="Y",
    )
except AttributeError:
    bpy.ops.export_scene.obj(
        filepath=obj_path,
        use_selection=True,
        use_uvs=True,
        use_normals=True,
        use_materials=False,
        use_triangles=True,
        axis_forward="-Z",
        axis_up="Y",
    )
print("OBJ exported:", obj_path)
"""


def _fbx_to_obj(fbx_path: str, obj_path: str) -> None:
    """Convert FBX → OBJ via a headless Blender subprocess."""
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(_BLENDER_CONVERT)
        script_path = f.name
    try:
        result = subprocess.run(
            ["blender", "--background", "--python", script_path, "--", fbx_path, obj_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Blender conversion failed:\n{result.stderr[-2000:]}")
        print(f"  Blender converted FBX → OBJ: {obj_path}")
    finally:
        os.unlink(script_path)


def load_fbx(fbx_path: str):
    """Load FBX via Blender → OBJ → trimesh, return (Trimesh, uv: np.ndarray).

    The OBJ is cached next to the FBX source so conversion only runs once.
    """
    import trimesh

    obj_path = os.path.join(os.path.dirname(fbx_path), "bathroomFloor.obj")
    if not os.path.isfile(obj_path):
        print("  FBX not directly loadable by trimesh – converting via Blender …")
        _fbx_to_obj(fbx_path, obj_path)

    loaded = trimesh.load(obj_path, force="scene", process=False)

    if isinstance(loaded, trimesh.Scene):
        meshes = list(loaded.geometry.values())
        mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    else:
        mesh = loaded

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Could not load a Trimesh from {obj_path}")

    mesh = mesh.copy()

    if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        uv = mesh.visual.uv.astype(np.float32)
    else:
        # Fallback: planar-project UVs from XZ extents
        verts = mesh.vertices.astype(np.float32)
        mn, mx = verts.min(0), verts.max(0)
        ext = np.where(mx - mn > 1e-6, mx - mn, 1.0)
        uv = (verts[:, [0, 2]] - mn[[0, 2]]) / ext[[0, 2]]

    return mesh, uv


def load_texture(path: str) -> np.ndarray:
    """Load an image as float32 RGBA [H,W,4] in [0,1]."""
    img = imageio.v3.imread(path).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3 + [np.ones_like(img)], axis=-1)
    elif img.shape[-1] == 3:
        img = np.concatenate([img, np.ones((*img.shape[:2], 1), dtype=np.float32)], axis=-1)
    return img


# ---------------------------------------------------------------------------
# Camera helpers (same as codes/pose/renderer.py)
# ---------------------------------------------------------------------------
def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0.0 else v / n


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = _normalize(target - eye)
    right   = _normalize(np.cross(forward, up))
    cam_up  = np.cross(right, forward)
    return np.array([
        [right[0],   right[1],   right[2],   -np.dot(right,   eye)],
        [cam_up[0],  cam_up[1],  cam_up[2],  -np.dot(cam_up,  eye)],
        [-forward[0],-forward[1],-forward[2], np.dot(forward,  eye)],
        [0.0,         0.0,        0.0,         1.0               ],
    ], dtype=np.float32)


def orthographic(scale: float, near: float, far: float) -> np.ndarray:
    return np.array([
        [1.0 / scale, 0.0,  0.0,                     0.0                    ],
        [0.0,  1.0 / scale, 0.0,                     0.0                    ],
        [0.0,         0.0, -2.0 / (far - near), -(far + near) / (far - near)],
        [0.0,         0.0,  0.0,                     1.0                    ],
    ], dtype=np.float32)


def build_floor_camera(mesh, camera_scale=None, device=device):
    """Position the camera above and slightly to the side to view the floor quad."""
    bounds = mesh.bounds.astype(np.float32)
    center = bounds.mean(axis=0)
    extents = bounds[1] - bounds[0]
    radius  = 0.5 * float(np.linalg.norm(extents))

    if camera_scale is None:
        camera_scale = 0.8 * radius

    # Look straight down from above.
    view_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    up_dir   = np.array([0.0, 0.0, -1.0], dtype=np.float32)   # Z- as screen-up
    dist     = 4.0 * radius
    eye_np   = center + view_dir * dist
    near     = max(0.05 * radius, dist - radius)
    far      = dist + radius

    view_np = look_at(eye_np, center, up_dir)
    proj_np = orthographic(camera_scale, near, far)
    vp_np   = proj_np @ view_np

    proj_view = torch.tensor(vp_np,   dtype=torch.float32, device=device)
    center_t  = torch.tensor(center,  dtype=torch.float32, device=device)
    eye_t     = torch.tensor(eye_np,  dtype=torch.float32, device=device)

    # pytorch3d camera (for DROT path)
    eye_pt = eye_t.unsqueeze(0)
    at_pt  = center_t.unsqueeze(0)
    R      = look_at_rotation(eye_pt, at=at_pt, device=device)
    R_p3d  = R.clone()
    R_p3d[:, :, 0] *= -1
    R_p3d[:, :, 2] *= -1
    T_p3d  = -torch.bmm(R_p3d.transpose(1, 2), eye_pt.unsqueeze(-1)).squeeze(-1)

    p3d_cam = OpenGLOrthographicCameras(
        device=device, R=R_p3d, T=T_p3d,
        znear=near, zfar=far,
        top=camera_scale, bottom=-camera_scale,
        left=-camera_scale, right=camera_scale,
    )

    view_t = torch.tensor(view_np, dtype=torch.float32).unsqueeze(0).to(device)
    proj_t = torch.tensor(proj_np, dtype=torch.float32).unsqueeze(0).to(device)
    vp_t   = torch.tensor(vp_np,   dtype=torch.float32).unsqueeze(0).to(device)
    resolution = None  # set later
    sensor = {
        "position":            eye_pt,
        "matrix":              vp_t,
        "camera_matrix":       view_t,
        "perspective_matrix":  proj_t,
        "center":              at_pt,
        "init_position":       eye_pt.clone(),
        "init_center":         at_pt.clone(),
    }

    return proj_view, center_t, radius, p3d_cam, eye_t, sensor, near, far, camera_scale


# ---------------------------------------------------------------------------
# nvdiffrast render – matches DROT's Phong shading exactly
# ---------------------------------------------------------------------------
def _compute_vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted vertex normals, identical to NVDiffRastFullRenderer._compute_vertex_normals."""
    import torch.nn.functional as F
    pos_idx_long = faces.to(torch.int64)
    v0 = vertices[pos_idx_long[:, 0]]
    v1 = vertices[pos_idx_long[:, 1]]
    v2 = vertices[pos_idx_long[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    vertex_normals = torch.zeros_like(vertices)
    vertex_normals.index_add_(0, pos_idx_long[:, 0], face_normals)
    vertex_normals.index_add_(0, pos_idx_long[:, 1], face_normals)
    vertex_normals.index_add_(0, pos_idx_long[:, 2], face_normals)
    return F.normalize(vertex_normals, p=2, dim=-1, eps=1e-6)


def render_floor_shaded(
    glctx,
    vertices: torch.Tensor,  # [V, 3]
    faces: torch.Tensor,     # [F, 3]  int32
    uvs: torch.Tensor,       # [V, 2]
    tex: torch.Tensor,       # [H, W, C]  base-color (NOT yet flipped)
    proj_view: torch.Tensor, # [4, 4]
    light_pos: torch.Tensor, # [3]
    eye_pos:   torch.Tensor, # [3]
    resolution: int,
    ambient_color:  torch.Tensor,  # [1, 3]
    diffuse_color:  torch.Tensor,  # [1, 3]
    specular_color: torch.Tensor,  # [1, 3]
    shininess: float,
    light_power: float = 1.0,
    background: tuple = (0.05, 0.05, 0.05),
) -> torch.Tensor:
    """Phong shading that matches DROT's NVDiffRastFullRenderer formula exactly."""
    import torch.nn.functional as F

    vertex_normals = _compute_vertex_normals(vertices, faces)

    pos_w    = torch.cat([vertices, torch.ones(vertices.shape[0], 1, device=device)], dim=1)
    pos_clip = (pos_w @ proj_view.t())[None]

    rast_out, _ = dr.rasterize(glctx, pos_clip, faces, resolution=[resolution, resolution])

    # No flip: DROT stores the texture pre-flipped in the p3d mesh, then flipud()s it
    # back inside render_mesh → net zero flips. Match that here by using tex as-is.
    uv_px, _ = dr.interpolate(uvs[None], rast_out, faces)
    color     = dr.texture(tex[None], uv_px, filter_mode="linear")  # [1,H,W,C]

    # Interpolate normals and positions (same face indices as geometry)
    nrm_px, _ = dr.interpolate(vertex_normals[None], rast_out, faces)
    pos_px, _ = dr.interpolate(vertices[None],       rast_out, faces)
    nrm_px = F.normalize(nrm_px, p=2, dim=-1, eps=1e-6)

    # Light / view directions  (light_pos / eye_pos are [3]; broadcast to [1,1,1,3])
    lp = light_pos.view(1, 1, 1, 3)
    vp = eye_pos.view(1, 1, 1, 3)

    direction      = F.normalize(lp - pos_px, p=2, dim=-1, eps=1e-6)
    cos_angle      = (nrm_px * direction).sum(-1, keepdim=True)
    front_mask     = (cos_angle > 0).float()
    light_diffuse  = light_power * cos_angle * front_mask

    view_dir       = F.normalize(vp - pos_px, p=2, dim=-1, eps=1e-6)
    reflect_dir    = -direction + 2.0 * cos_angle * nrm_px
    alpha          = F.relu((view_dir * reflect_dir).sum(-1, keepdim=True)) * front_mask
    light_specular = light_power * alpha.pow(shininess)

    # DROT formula: (ambient + diffuse_weight) * texture + specular_weight  (RGB only)
    rgb    = color[..., :3]
    shaded = (ambient_color + light_diffuse * diffuse_color) * rgb + light_specular * specular_color

    # Background compositing (same as DROT)
    bg      = torch.tensor(background, device=device).view(1, 1, 1, 3)
    bg_mask = (rast_out[..., -1:] == 0).expand_as(shaded)
    shaded  = torch.where(bg_mask, bg.expand_as(shaded), shaded)

    shaded = dr.antialias(shaded.contiguous(), rast_out, pos_clip, faces)
    return shaded  # [1, H, W, C]


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------
def save_png(path: pathlib.Path, img: np.ndarray | torch.Tensor, flip: bool = True) -> None:
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    if flip:
        img = img[::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8))
    print(f"Saved {path}")


def _annotate_iteration(frame: np.ndarray, iteration: int) -> np.ndarray:
    import cv2
    img = frame.copy()
    label = f"iter {iteration}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
    x, y = 12, 28
    cv2.rectangle(img, (x - 6, y - text_h - 6), (x + text_w + 6, y + baseline + 6), (0, 0, 0), thickness=-1)
    cv2.putText(img, label, (x, y), font, scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    return img


def build_method_videos(method_dir: pathlib.Path, fps: int = 12) -> None:
    def _collect(pattern: str) -> list[tuple[np.ndarray, int]]:
        frames: list[tuple[np.ndarray, int]] = []
        for path in sorted(method_dir.glob(pattern)):
            try:
                iteration = int(path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            img = imageio.v3.imread(path)
            frames.append((_annotate_iteration(img, iteration), iteration))
        if not frames:
            return frames
        max_frames = max(1, int(round(fps * 3.0)))
        if len(frames) > max_frames:
            keep_idx = np.linspace(0, len(frames) - 1, num=max_frames, dtype=int)
            frames = [frames[idx] for idx in keep_idx]
        return frames

    render_frames = _collect("iter_*_render.png")
    ref_frames = _collect("iter_*_ref.png")
    if render_frames:
        imageio.mimwrite(method_dir / "trajectory.mp4", [frame for frame, _ in render_frames], fps=fps)
        print(f"Saved {method_dir / 'trajectory.mp4'}")
    if ref_frames:
        imageio.mimwrite(method_dir / "trajectory_ref.mp4", [frame for frame, _ in ref_frames], fps=fps)
        print(f"Saved {method_dir / 'trajectory_ref.mp4'}")


def bilinear_downsample(t: torch.Tensor, size: int) -> torch.Tensor:
    """Bilinearly downsample [1,H,W,C] → [1,size,size,C]."""
    return F.interpolate(
        t.permute(0, 3, 1, 2).contiguous(),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)


# ---------------------------------------------------------------------------
# Render + 2D position map  (needed for RGBXY loss)
# ---------------------------------------------------------------------------
def render_floor_with_pos(
    glctx,
    vertices, faces, uvs, tex,
    proj_view,               # [4,4]  view-projection (orthographic)
    light_pos, eye_pos,      # [3]
    resolution: int,
    ambient_color, diffuse_color, specular_color,
    shininess: float, light_power: float,
    background: tuple = (0.0, 0.0, 0.0),
):
    """Like render_floor_shaded, additionally returns pos_2d [1,H,W,2] in [-1,1] and mask."""
    import torch.nn.functional as F

    vertex_normals = _compute_vertex_normals(vertices, faces)

    pos_w    = torch.cat([vertices, torch.ones(vertices.shape[0], 1, device=device)], dim=1)
    pos_clip = (pos_w @ proj_view.t())[None]

    rast_out, _ = dr.rasterize(glctx, pos_clip, faces, resolution=[resolution, resolution])

    uv_px,  _ = dr.interpolate(uvs[None],             rast_out, faces)
    color      = dr.texture(tex[None],   uv_px,   filter_mode="linear")
    nrm_px, _ = dr.interpolate(vertex_normals[None],  rast_out, faces)
    pos_px, _ = dr.interpolate(vertices[None],         rast_out, faces)
    nrm_px     = F.normalize(nrm_px, p=2, dim=-1, eps=1e-6)

    lp = light_pos.view(1, 1, 1, 3)
    vp = eye_pos.view(1, 1, 1, 3)
    direction      = F.normalize(lp - pos_px, p=2, dim=-1, eps=1e-6)
    cos_angle      = (nrm_px * direction).sum(-1, keepdim=True)
    front_mask     = (cos_angle > 0).float()
    light_diffuse  = light_power * cos_angle * front_mask
    view_dir       = F.normalize(vp - pos_px, p=2, dim=-1, eps=1e-6)
    reflect_dir    = -direction + 2.0 * cos_angle * nrm_px
    alpha          = F.relu((view_dir * reflect_dir).sum(-1, keepdim=True)) * front_mask
    light_specular = light_power * alpha.pow(shininess)

    rgb    = color[..., :3]
    shaded = (ambient_color + light_diffuse * diffuse_color) * rgb + light_specular * specular_color

    bg      = torch.tensor(background, device=device).view(1, 1, 1, 3)
    vis_mask = (rast_out[..., -1:] > 0)
    shaded   = torch.where(~vis_mask.expand_as(shaded), bg.expand_as(shaded), shaded)
    shaded   = dr.antialias(shaded.contiguous(), rast_out, pos_clip, faces)

    # 2D projected position in NDC [-1,1]  (orthographic: clip XY already normalised)
    # Flatten to [N,3] → matmul → reshape back to avoid cuBLAS batched-stride issues
    B, H, W, _ = pos_px.shape
    pos_flat = pos_px.reshape(-1, 3)
    pos_hom_flat = torch.cat([pos_flat, torch.ones(pos_flat.shape[0], 1, device=device)], dim=-1)
    clip_flat = pos_hom_flat @ proj_view.t()          # [N, 4]
    clip_px   = clip_flat.reshape(B, H, W, 4)
    pos_2d    = clip_px[..., :2]                       # orthographic: w=1
    pos_2d    = torch.where(vis_mask.expand_as(pos_2d), pos_2d,
                            torch.zeros_like(pos_2d))

    return shaded, pos_2d, vis_mask[..., 0]   # [1,H,W,3], [1,H,W,2], [1,H,W]


# ---------------------------------------------------------------------------
# RGBXY loss  (replicates PointLossFunction.match_Sinkhorn)
# ---------------------------------------------------------------------------
_sinkhorn_loss = None

def rgbxy_loss(pred_rgb, pred_pos, gt_rgb, gt_pos, mask, resolution: int) -> torch.Tensor:
    """Sinkhorn OT on 5-D (R,G,B,x,y) clouds, then MSE of matched displacement."""
    global _sinkhorn_loss
    if _sinkhorn_loss is None:
        from geomloss import SamplesLoss
        _sinkhorn_loss = SamplesLoss("sinkhorn", blur=0.01)

    B, H, W, _ = pred_rgb.shape
    pred_5d = torch.cat([pred_rgb, pred_pos], dim=-1).reshape(B, H * W, 5)  # [1,N,5]
    gt_5d   = torch.cat([gt_rgb,   gt_pos],   dim=-1).reshape(B, H * W, 5)

    pred_5d_match = pred_5d.clone().clamp(0.0, 1.0)
    gt_5d_clamped = gt_5d.clone().clamp(0.0, 1.0)

    # Sinkhorn gradient gives optimal transport direction (no grad through matching)
    pl = _sinkhorn_loss(pred_5d_match, gt_5d_clamped) * (H * W)
    (g,) = torch.autograd.grad(torch.sum(pl), [pred_5d_match])

    matched = (pred_5d - g.reshape(B, H * W, 5)).detach()
    return torch.mean((matched - pred_5d) ** 2)

# ---------------------------------------------------------------------------
# LOI Loss  (Mehta et al., CVPR 2025) — inlined from optimize_skeleton_coeffs.py
# ---------------------------------------------------------------------------
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
        C, H, W = im.shape
        cs = F.conv2d(im.unsqueeze(0), self.sigma_kernel, padding=self.sigma_ks // 2, groups=C)
        cs = cs.view(C, self.n_sigma, H, W)
        diff = cs.unsqueeze(0).unsqueeze(1) - self.bin_val
        iso = torch.exp(-0.5 * (diff / self.beta_val).pow(2))
        iso = iso / (np.sqrt(2.0 * np.pi) * self.beta_val)
        n_groups = self.n_beta * self.n_bins * C * self.n_sigma
        iso = iso.reshape(n_groups, H, W)
        loi = F.conv2d(iso.unsqueeze(0), self.alpha_kernel, padding=self.kernel_size // 2, groups=n_groups)
        loi = loi.reshape(self.n_beta, self.n_bins, C, self.n_sigma, self.n_alpha, H, W)
        return loi.permute(4, 0, 3, 2, 5, 6, 1)

    @staticmethod
    def emd(loi1: torch.Tensor, loi2: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(torch.cumsum(loi1, dim=-1) - torch.cumsum(loi2, dim=-1)))


# ---------------------------------------------------------------------------
# Video frame scheduler  — inlined from optimize_skeleton_coeffs.py
# ---------------------------------------------------------------------------
class VideoFrameScheduler:
    def __init__(self, video_path: str, n_iter: int, fallback: torch.Tensor,
                 resolution: int, device: torch.device, video_ratio: float = 0.8):
        reader = imageio.get_reader(video_path)
        frames = []
        for frame in reader:
            t = torch.from_numpy(frame.astype(np.float32) / 255.0).to(device)[..., :3].flip(0)
            if t.shape[0] != resolution or t.shape[1] != resolution:
                t = bilinear_downsample(t.unsqueeze(0), resolution).squeeze(0)
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


# ---------------------------------------------------------------------------
# Batch render for PRDPT  (theta_p: [N, 2] XZ offsets)
# ---------------------------------------------------------------------------
def render_smooth_floor(theta_p: torch.Tensor, update_fn, context_args: dict):
    gt_rgb  = context_args["gt_rgb"]        # [1,H,W,3]
    center  = context_args["center"]
    height  = context_args["height"]
    rk      = context_args["render_kwargs"]
    losses, avg_img = [], None
    with torch.no_grad():
        for i in range(theta_p.shape[0]):
            xz = theta_p[i]
            lp = torch.stack([center[0] + xz[0],
                               torch.tensor(height, device=device),
                               center[2] + xz[1]])
            rgb, _, _ = render_floor_with_pos(light_pos=lp, **rk)
            loss = torch.mean((rgb - gt_rgb) ** 2)
            losses.append(loss)
            if avg_img is None:
                avg_img = rgb.detach()
    return torch.stack(losses), avg_img


# ---------------------------------------------------------------------------
# Optimization loop
# ---------------------------------------------------------------------------
def optimize_light(
    method: str,
    glctx,
    vertices, faces, uvs, tex,
    proj_view, eye_pos,
    center, radius,
    init_xz: tuple, target_xz: tuple,
    light_height: float,
    gt_rgb, gt_pos, gt_mask, gt_rgb_hi,
    out_dir: pathlib.Path,
    resolution: int = 128,
    save_resolution: int = 512,
    n_iters: int = 200,
    lr: float = 0.05,
    log_every: int = 20,
    trajectory_every: int | None = None,
    ambient_color=None, diffuse_color=None, specular_color=None,
    shininess: float = 128.0, light_power: float = 1.0,
    # video
    video_path: str | None = None,
    video_ratio: float = 0.8,
    # prdpt
    prdpt_sigma: float = 0.5,
    prdpt_sigma_min: float = 0.01,
    prdpt_nsamples: int = 4,
    trajectory_fps: int = 12,
):
    from tqdm import tqdm

    def make_light(xz):
        return torch.stack([
            center[0] + xz[0],
            torch.tensor(light_height, device=device),
            center[2] + xz[1],
        ])

    shared = dict(
        glctx=glctx, vertices=vertices, faces=faces, uvs=uvs, tex=tex,
        proj_view=proj_view, eye_pos=eye_pos,
        ambient_color=ambient_color, diffuse_color=diffuse_color,
        specular_color=specular_color, shininess=shininess, light_power=light_power,
    )
    # Render at save_resolution, then bilinearly downsample to resolution for loss –
    # the same pipeline the VideoFrameScheduler uses for video frames.
    save_kwargs   = {**shared, "resolution": save_resolution}
    render_kwargs = {**shared, "resolution": resolution}   # kept for PRDPT batch renders

    method_dir = out_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Per-method setup
    # ------------------------------------------------------------------
    loi_model = target_loi = video_sched = video_sched_vis = smooth_render_fn = prdpt_ctx = None

    if method == "loir":
        loi_model = LOILoss(channels=3, device=device)
        with torch.no_grad():
            target_loi = loi_model(gt_rgb[0])  # gt_rgb[0]: [H,W,3]

    elif method == "video":
        if video_path is None:
            raise ValueError("--video-path is required for method 'video'")
        video_sched = VideoFrameScheduler(
            video_path, n_iters, gt_rgb[0], resolution, device, video_ratio)
        video_sched_vis = VideoFrameScheduler(
            video_path, n_iters, gt_rgb_hi[0], save_resolution, device, video_ratio)

    elif method == "prdpt":
        prdpt_ctx = {
            "gt_rgb": gt_rgb,
            "center": center,
            "height": light_height,
            "render_kwargs": render_kwargs,
            "sigma": prdpt_sigma,
            "nsamples": prdpt_nsamples,
            "antithetic": True,
            "sampler": "importance",
            "update_fn": None,
            "device": device,
        }
        smooth_render_fn = smoothFn(render_smooth_floor, context_args=None, device=device)

    # ------------------------------------------------------------------
    # Optimisable parameter
    # ------------------------------------------------------------------
    light_xz  = torch.tensor(list(init_xz), dtype=torch.float32,
                              device=device, requires_grad=True)
    optimizer = torch.optim.Adam([light_xz], lr=lr)

    history = []
    t0 = time.perf_counter()
    pbar = tqdm(range(n_iters), desc=method)
    capture_every = trajectory_every if trajectory_every is not None else log_every

    for i in pbar:
        lp = make_light(light_xz)

        # current_ref tracks which reference image was used this iteration
        current_ref: torch.Tensor | None = None

        if method == "prdpt":
            # sigma annealing
            if prdpt_sigma_min > 0:
                frac = i / max(n_iters - 1, 1)
                prdpt_ctx["sigma"] = max(
                    prdpt_sigma_min,
                    prdpt_sigma * (1 - frac) + prdpt_sigma_min * frac,
                )
            loss, _ = smooth_render_fn(light_xz.reshape(1, 2), prdpt_ctx)
            current_ref = gt_rgb[0]
        else:
            pred_rgb_hi, pred_pos_hi, _ = render_floor_with_pos(
                light_pos=lp, **save_kwargs)
            pred_rgb = bilinear_downsample(pred_rgb_hi, resolution)
            pred_pos = bilinear_downsample(pred_pos_hi, resolution)

            if method == "mse":
                loss = torch.mean((pred_rgb - gt_rgb) ** 2)
                current_ref = gt_rgb[0]
            elif method == "rgbxy":
                loss = rgbxy_loss(pred_rgb, pred_pos, gt_rgb, gt_pos,
                                  None, resolution)
                current_ref = gt_rgb[0]
            elif method == "loir":
                est_loi = loi_model(pred_rgb[0])
                loss = LOILoss.emd(target_loi, est_loi)
                current_ref = gt_rgb[0]
            elif method == "video":
                ref = video_sched.get_ref(i)          # [H,W,3]
                # Reinhard tone-map both into [0,1): y = x/(1+x).
                # Gradient ∂y/∂x = 1/(1+x)² > 0 everywhere — overexposed pixels
                # still pull the optimiser, just with softer force than in-range ones.
                pred_tm = pred_rgb[0] / (1.0 + pred_rgb[0])
                ref_tm  = ref         / (1.0 + ref)
                loss = (pred_tm - ref_tm).pow(2).mean()
                current_ref = ref
            else:
                raise ValueError(method)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        xz = light_xz.detach().cpu().numpy()
        history.append({"iter": i, "loss": float(loss.item()),
                        "xz": xz.tolist()})
        pbar.set_description(
            f"{method}  loss={loss.item():.5f}  xz=[{xz[0]:.3f},{xz[1]:.3f}]")

        if (i + 1) % capture_every == 0 or i == 0:
            if method == "prdpt":
                with torch.no_grad():
                    snap_hi, _, _ = render_floor_with_pos(light_pos=lp.detach(), **save_kwargs)
                pred_vis_np = snap_hi[0].cpu().numpy()
            else:
                pred_vis_np = pred_rgb_hi[0].detach().cpu().numpy()

            if method == "video":
                ref_vis_np = video_sched_vis.get_ref(i).detach().cpu().numpy()
            else:
                ref_vis_np = gt_rgb_hi[0].detach().cpu().numpy()

            save_png(method_dir / f"iter_{i+1:04d}_render.png", pred_vis_np)
            save_png(method_dir / f"iter_{i+1:04d}_ref.png", ref_vis_np)

    # Final render + JSON logs
    with torch.no_grad():
        final_rgb, _, _ = render_floor_with_pos(
            light_pos=make_light(light_xz.detach()), **save_kwargs)
    save_png(out_dir / f"final_{method}.png", final_rgb[0].cpu().numpy())

    xz_final = light_xz.detach().cpu().numpy().tolist()
    elapsed = time.perf_counter() - t0

    summary = {
        "method": method,
        "final_xz": xz_final,
        "target_xz": list(target_xz),
        "final_loss": history[-1]["loss"],
        "n_iters": n_iters,
        "lr": lr,
        "total_time_sec": elapsed,
    }
    (method_dir / "history.json").write_text(json.dumps(history, indent=2))
    (method_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    build_method_videos(method_dir, fps=trajectory_fps)
    print(f"  {method}: final xz={xz_final}  target={list(target_xz)}"
          f"  ({elapsed:.1f}s)")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Render bathroomFloor FBX with nvdiffrast + DROT.")
    parser.add_argument("--fbx",         default=FBX_PATH)
    parser.add_argument("--output-dir",  default=os.path.join(VNC_DIR, "outputs", "lights", "floor"))
    parser.add_argument("--resolution",      type=int, default=128,
                        help="Resolution used during optimization.")
    parser.add_argument("--save-resolution", type=int, default=512,
                        help="Resolution used for saved debug images.")
    parser.add_argument("--camera-scale",type=float, default=None)
    parser.add_argument("--n-iters",     type=int,   default=200)
    parser.add_argument("--lr",          type=float, default=0.05)
    parser.add_argument("--log-every",   type=int,   default=20)
    parser.add_argument("--trajectory-every", type=int, default=None,
                        help="Save trajectory video frames every N iterations. Defaults to --log-every.")
    parser.add_argument("--methods", nargs="+",
                        default=["mse"],
                        choices=["mse", "rgbxy", "loir", "video", "prdpt"],
                        help="Optimisation method(s) to run.")
    parser.add_argument("--video-path",    type=str,   default=None,
                        help="Path to a video file (required for method 'video').")
    parser.add_argument("--video-ratio",   type=float, default=0.8,
                        help="Fraction of iterations that use video frames.")
    parser.add_argument("--prdpt-sigma",     type=float, default=0.5)
    parser.add_argument("--prdpt-sigma-min", type=float, default=0.01)
    parser.add_argument("--prdpt-nsamples",  type=int,   default=4)
    parser.add_argument("--trajectory-fps",  type=int,   default=12,
                        help="Frames per second for saved optimization trajectory videos.")
    args = parser.parse_args()
    
    SHININESS = 5000.0

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolution = args.resolution

    # ------------------------------------------------------------------
    # 1. Load geometry
    # ------------------------------------------------------------------
    print(f"Loading FBX: {args.fbx}")
    mesh, uv_np = load_fbx(args.fbx)
    print(f"  Vertices: {mesh.vertices.shape[0]}  Faces: {mesh.faces.shape[0]}")

    vertices = torch.from_numpy(mesh.vertices.astype(np.float32)).to(device)
    faces    = torch.from_numpy(mesh.faces.astype(np.int32)).to(device)
    uvs      = torch.from_numpy(uv_np).to(device)

    # ------------------------------------------------------------------
    # 2. Load base-color texture
    # ------------------------------------------------------------------
    print("Loading textures …")
    tex_base = torch.from_numpy(load_texture(TEX_FILES["base_color"])).to(device)  # [H,W,4]

    # ------------------------------------------------------------------
    # 3. Build camera looking straight down at the floor quad
    # ------------------------------------------------------------------
    (proj_view, center_t, radius, p3d_cam, eye_t,
     sensor, near, far, cam_scale) = build_floor_camera(
        mesh, camera_scale=args.camera_scale, device=device
    )
    sensor["resolution"] = (resolution, resolution)

    # ------------------------------------------------------------------
    # 4. nvdiffrast shaded render  (same material as DROT path)
    # ------------------------------------------------------------------
    # Use GL context to match DROT (RasterizeGLContext internally)
    glctx = dr.RasterizeGLContext()

    amb_t = torch.tensor([[0.08, 0.08, 0.08]], device=device)
    dif_t = torch.tensor([[0.8,  0.8,  0.8 ]], device=device)
    spe_t = torch.tensor([[0.8,  0.8,  0.8 ]], device=device)

    with torch.no_grad():
        nv_img = render_floor_shaded(
            glctx, vertices, faces, uvs, tex_base,
            proj_view,
            light_pos=eye_t,   # DROT uses sensor["position"] for both view_pos and light_pos
            eye_pos=eye_t,
            resolution=resolution,
            ambient_color=amb_t, diffuse_color=dif_t, specular_color=spe_t,
            shininess=SHININESS, light_power=1.0,
            background=(0.0, 0.0, 0.0),
        )

    save_png(out_dir / "floor_nvdiffrast.png", nv_img[0].cpu().numpy())

    # ------------------------------------------------------------------
    # 5. DROT render  (NVDiffRastFullRenderer via pytorch3d Meshes)
    # ------------------------------------------------------------------
    # Build pytorch3d mesh with UV texture
    tex_np  = load_texture(TEX_FILES["base_color"])
    tex_p3d = torch.from_numpy(np.flip(tex_np, axis=0).copy()[..., :3]).unsqueeze(0).to(device)
    faces_long = faces.to(torch.int64)
    uvs_p3d    = uvs.unsqueeze(0)
    faces_uvs  = faces_long.unsqueeze(0)
    textures   = TexturesUV(maps=tex_p3d, faces_uvs=faces_uvs, verts_uvs=uvs_p3d)

    verts_p3d  = vertices.unsqueeze(0)
    faces_p3d  = faces_long.unsqueeze(0)
    p3d_mesh   = Meshes(verts=verts_p3d, faces=faces_p3d, textures=textures)

    p3d_lights = PointLights(
        device=device,
        location=eye_t.unsqueeze(0).tolist(),
        ambient_color=[[0.08, 0.08, 0.08]],
        diffuse_color=[[1.0,  1.0,  1.0 ]],
        specular_color=[[1.0, 1.0,  1.0 ]],
    )
    material = Materials(
        device=device,
        ambient_color=((0.08, 0.08, 0.08),),
        diffuse_color=((0.8,  0.8,  0.8 ),),
        specular_color=((0.8, 0.8,  0.8 ),),
        shininess=SHININESS,
    )

    drot_renderer = NVDiffRastFullRenderer(
        device=device,
        settings={"background": "black", "shading": True, "light_power": 1.0},
        resolution=[resolution, resolution],
    )

    scene = {
        "meshes":  [{"model": p3d_mesh}],
        "sensors": [sensor],
        "material": material,
    }

    with torch.no_grad():
        drot_result = drot_renderer.render(scene, DcDt=False)

    drot_img = drot_result["images"][0].cpu().numpy()   # [H, W, 3]
    save_png(out_dir / "floor_drot.png", drot_img)

    print("\nDone with static renders.")

    # ------------------------------------------------------------------
    # 6. Light-position optimisation
    #    Image space:  right = world +X,  up = world −Z  (camera up=[0,0,-1])
    #    "top-left"  in image → world (−offset, h, +offset)
    #    "bottom-right" in image → world (+offset, h, −offset)
    # ------------------------------------------------------------------
    opt_offset     = 1.0 * radius          # how far from centre to place the light
    light_height   = float(eye_t[1].item())  # same height as camera (matches DROT)
    init_xz        = (-opt_offset,  opt_offset)   # top-left in image
    target_xz      = ( opt_offset, -opt_offset)   # bottom-right in image
    

    save_resolution = args.save_resolution

    shared_kw = dict(
        glctx=glctx, vertices=vertices, faces=faces, uvs=uvs, tex=tex_base,
        proj_view=proj_view, eye_pos=eye_t,
        ambient_color=amb_t, diffuse_color=dif_t, specular_color=spe_t,
        shininess=SHININESS, light_power=1.0,
    )

    def make_light(xz_tuple):
        x, z = xz_tuple
        return torch.tensor([float(center_t[0]) + x,
                              float(eye_t[1]),
                              float(center_t[2]) + z], device=device)

    # Save init and target renders at save_resolution before any optimisation
    print("\nSaving init and target renders …")
    with torch.no_grad():
        init_rgb, init_pos, init_mask = render_floor_with_pos(
            light_pos=make_light(init_xz),   **shared_kw, resolution=save_resolution)
        tgt_rgb,  _,        _          = render_floor_with_pos(
            light_pos=make_light(target_xz), **shared_kw, resolution=save_resolution)
    save_png(out_dir / "init.png",   init_rgb[0].cpu().numpy())
    save_png(out_dir / "target.png", tgt_rgb[0].cpu().numpy())

    # GT tensors: render at save_resolution then bilinearly downsample to resolution –
    # same pipeline as VideoFrameScheduler uses for video frames.
    with torch.no_grad():
        gt_hi, gt_pos_hi, _ = render_floor_with_pos(
            light_pos=make_light(target_xz), **shared_kw, resolution=save_resolution)
    gt_rgb  = bilinear_downsample(gt_hi,      resolution)
    gt_pos  = bilinear_downsample(gt_pos_hi,  resolution)

    opt_kwargs = dict(
        **shared_kw,
        center=center_t, radius=radius,
        init_xz=init_xz, target_xz=target_xz,
        light_height=float(eye_t[1]),
        gt_rgb=gt_rgb, gt_pos=gt_pos, gt_mask=None, gt_rgb_hi=gt_hi,
        out_dir=out_dir,
        resolution=resolution,
        save_resolution=save_resolution,
        n_iters=args.n_iters,
        lr=args.lr,
        log_every=args.log_every,
        trajectory_every=args.trajectory_every,
        video_path=args.video_path,
        video_ratio=args.video_ratio,
        prdpt_sigma=args.prdpt_sigma,
        prdpt_sigma_min=args.prdpt_sigma_min,
        prdpt_nsamples=args.prdpt_nsamples,
        trajectory_fps=args.trajectory_fps,
    )

    import json
    all_summaries = {}
    for method in args.methods:
        print(f"\n=== {method.upper()} ===")
        summary = optimize_light(method=method, **opt_kwargs)
        all_summaries[method] = summary

    (out_dir / "all_summaries.json").write_text(json.dumps(all_summaries, indent=2))
    print(f"\nSaved all_summaries.json  ({', '.join(args.methods)})")


if __name__ == "__main__":
    main()
