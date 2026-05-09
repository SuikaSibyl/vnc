import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def import_nvdiffrast(repo_root: Path):
    try:
        import nvdiffrast.torch as dr  # type: ignore

        return dr
    except Exception:
        local_root = repo_root / "codes" / "geometry" / "largesteps" / "ext" / "nvdiffrast"
        major, minor = sys.version_info[:2]
        build_dir = local_root / "build" / f"lib.linux-x86_64-cpython-{major}{minor}"
        if build_dir.exists():
            sys.path.insert(0, str(build_dir))
        sys.path.insert(0, str(local_root))
        import nvdiffrast.torch as dr  # type: ignore

        return dr


def import_pytorch3d():
    try:
        from pytorch3d.renderer import TexturesVertex  # type: ignore
        from pytorch3d.structures import Meshes  # type: ignore
    except Exception as error:
        raise RuntimeError(
            "PyTorch3D is required for the DROT renderer path. "
            "Install it in the active environment before using method='rgbxy'."
        ) from error
    return Meshes, TexturesVertex


def save_image(path: Path, image: torch.Tensor) -> None:
    array = image.detach().clamp(0.0, 1.0).flip(0).cpu().numpy()
    Image.fromarray((array * 255.0).round().astype(np.uint8)).save(path)


def load_reference_image(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).flip(0)


def perspective(fovy_deg: float, aspect: float, znear: float, zfar: float, device: torch.device) -> torch.Tensor:
    f = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    matrix = torch.zeros((4, 4), dtype=torch.float32, device=device)
    matrix[0, 0] = f / aspect
    matrix[1, 1] = f
    matrix[2, 2] = (zfar + znear) / (znear - zfar)
    matrix[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
    matrix[3, 2] = -1.0
    return matrix


def look_at(eye, at, up, device: torch.device) -> torch.Tensor:
    # Build the camera matrix on CPU first. On some CUDA/PyTorch setups,
    # the tiny matrix-vector multiply used for the translation term can trip
    # CUBLAS_STATUS_INVALID_VALUE during initialization.
    cpu = torch.device("cpu")
    eye = torch.tensor(eye, dtype=torch.float32, device=cpu)
    at = torch.tensor(at, dtype=torch.float32, device=cpu)
    up = torch.tensor(up, dtype=torch.float32, device=cpu)

    z_axis = F.normalize(eye - at, dim=0)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=0), dim=0)
    y_axis = torch.cross(z_axis, x_axis, dim=0)

    matrix = torch.eye(4, dtype=torch.float32, device=cpu)
    matrix[0, :3] = x_axis
    matrix[1, :3] = y_axis
    matrix[2, :3] = z_axis
    matrix[:3, 3] = -matrix[:3, :3] @ eye
    return matrix.to(device=device)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True).clamp_min(1e-9)
    axis = axis_angle / theta
    x, y, z = axis.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        [
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ],
        dim=-1,
    ).reshape(axis_angle.shape[:-1] + (3, 3))
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).expand(axis_angle.shape[:-1] + (3, 3))
    outer = axis.unsqueeze(-1) * axis.unsqueeze(-2)
    theta = theta.unsqueeze(-1)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    return cos_theta * eye + (1.0 - cos_theta) * outer + sin_theta * skew


def matmul_4x4(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sum(a.unsqueeze(-1) * b.unsqueeze(-3), dim=-2)


def transform_homogeneous(mats: torch.Tensor, vecs: torch.Tensor) -> torch.Tensor:
    return torch.sum(mats * vecs.unsqueeze(-2), dim=-1)


def compute_vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    normals = torch.zeros_like(verts)
    normals.index_add_(0, faces[:, 0], face_normals)
    normals.index_add_(0, faces[:, 1], face_normals)
    normals.index_add_(0, faces[:, 2], face_normals)
    return F.normalize(normals, dim=-1, eps=1e-6)


def overlay_skeleton_on_image(
    image: torch.Tensor,
    joint_pixels: torch.Tensor,
    bone_parents: torch.Tensor,
    valid: torch.Tensor | None = None,
    joint_radius: int = 4,
    line_width: int = 2,
) -> torch.Tensor:
    canvas = image.detach().clamp(0.0, 1.0).flip(0).cpu().numpy()
    canvas_img = Image.fromarray((canvas * 255.0).round().astype(np.uint8))
    draw = ImageDraw.Draw(canvas_img, "RGBA")

    if valid is None:
        valid = torch.ones(joint_pixels.shape[0], dtype=torch.bool)
    joint_pixels_cpu = joint_pixels.detach().cpu()
    valid_cpu = valid.detach().cpu()
    parents_cpu = bone_parents.detach().cpu()

    bone_color = (255, 120, 60, 220)
    joint_color = (45, 180, 255, 255)
    joint_outline = (255, 255, 255, 255)

    for child_idx, parent_idx in enumerate(parents_cpu.tolist()):
        if parent_idx < 0 or not bool(valid_cpu[child_idx]) or not bool(valid_cpu[parent_idx]):
            continue
        x0, y0 = joint_pixels_cpu[parent_idx].tolist()
        x1, y1 = joint_pixels_cpu[child_idx].tolist()
        draw.line((x0, y0, x1, y1), fill=bone_color, width=line_width)

    for joint_idx in range(joint_pixels_cpu.shape[0]):
        if not bool(valid_cpu[joint_idx]):
            continue
        x, y = joint_pixels_cpu[joint_idx].tolist()
        draw.ellipse(
            (x - joint_radius - 1, y - joint_radius - 1, x + joint_radius + 1, y + joint_radius + 1),
            fill=joint_outline,
        )
        draw.ellipse(
            (x - joint_radius, y - joint_radius, x + joint_radius, y + joint_radius),
            fill=joint_color,
        )

    overlay = torch.from_numpy(np.asarray(canvas_img, dtype=np.float32) / 255.0)
    return overlay.flip(0).to(device=image.device, dtype=image.dtype)


def require_working_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the nvdiffrast renderer, but torch.cuda.is_available() is False.")
    try:
        _ = torch.zeros(1, device="cuda") + 1.0
    except Exception as error:
        raise RuntimeError(
            "CUDA is visible, but the active PyTorch build cannot execute kernels on this GPU. "
            "Install a PyTorch build that supports the local CUDA architecture before running optimization."
        ) from error


@dataclass
class SkeletonAsset:
    vertices: torch.Tensor
    faces: torch.Tensor
    skin_indices: torch.Tensor
    skin_weights: torch.Tensor
    mesh_vertex_ranges: torch.Tensor
    mesh_face_ranges: torch.Tensor
    mesh_albedos: torch.Tensor
    bone_parents: torch.Tensor
    bone_rest_mats: torch.Tensor
    bone_local_rest_mats: torch.Tensor
    bone_inv_bind_mats: torch.Tensor
    albedo: torch.Tensor
    bone_names: list[str]
    mesh_names: list[str]
    mesh_name: str
    reference_rotvecs: torch.Tensor | None = None
    reference_translations: torch.Tensor | None = None
    reference_root_translation: torch.Tensor | None = None
    reference_frame: int | None = None

    @classmethod
    def from_npz(cls, path: Path, device: torch.device):
        data = np.load(path, allow_pickle=False)
        bone_names = [str(name) for name in data["bone_names"].tolist()]
        mesh_names = [str(name) for name in data["mesh_names"].tolist()] if "mesh_names" in data else [str(data["mesh_name"].tolist())]
        reference_rotvecs = None
        reference_translations = None
        reference_root_translation = None
        reference_frame = None
        if "reference_rotvecs" in data:
            reference_rotvecs = torch.tensor(data["reference_rotvecs"], dtype=torch.float32, device=device)
        if "reference_translations" in data:
            reference_translations = torch.tensor(data["reference_translations"], dtype=torch.float32, device=device)
        if "reference_root_translation" in data:
            reference_root_translation = torch.tensor(
                data["reference_root_translation"], dtype=torch.float32, device=device
            )
        if reference_translations is None and reference_root_translation is not None and reference_rotvecs is not None:
            reference_translations = torch.zeros_like(reference_rotvecs)
            if reference_translations.shape[0] > 0:
                reference_translations[0] = reference_root_translation
        if "reference_frame" in data:
            reference_frame = int(data["reference_frame"])
        mesh_albedos_np = data["mesh_albedos"] if "mesh_albedos" in data else data["albedo"][None]
        return cls(
            vertices=torch.tensor(data["vertices"], dtype=torch.float32, device=device),
            faces=torch.tensor(data["faces"], dtype=torch.int32, device=device),
            skin_indices=torch.tensor(data["skin_indices"], dtype=torch.long, device=device),
            skin_weights=torch.tensor(data["skin_weights"], dtype=torch.float32, device=device),
            mesh_vertex_ranges=torch.tensor(
                data["mesh_vertex_ranges"] if "mesh_vertex_ranges" in data else [[0, data["vertices"].shape[0]]],
                dtype=torch.long,
                device=device,
            ),
            mesh_face_ranges=torch.tensor(
                data["mesh_face_ranges"] if "mesh_face_ranges" in data else [[0, data["faces"].shape[0]]],
                dtype=torch.long,
                device=device,
            ),
            mesh_albedos=torch.tensor(
                mesh_albedos_np,
                dtype=torch.float32,
                device=device,
            ),
            bone_parents=torch.tensor(data["bone_parents"], dtype=torch.long, device=device),
            bone_rest_mats=torch.tensor(data["bone_rest_mats"], dtype=torch.float32, device=device),
            bone_local_rest_mats=torch.tensor(data["bone_local_rest_mats"], dtype=torch.float32, device=device),
            bone_inv_bind_mats=torch.tensor(data["bone_inv_bind_mats"], dtype=torch.float32, device=device),
            albedo=torch.tensor(mesh_albedos_np[0], dtype=torch.float32, device=device),
            bone_names=bone_names,
            mesh_names=mesh_names,
            mesh_name=str(data["mesh_name"].tolist()),
            reference_rotvecs=reference_rotvecs,
            reference_translations=reference_translations,
            reference_root_translation=reference_root_translation,
            reference_frame=reference_frame,
        )


class DifferentiableSkeletonRenderer:
    def __init__(
        self,
        asset: SkeletonAsset,
        image_size: int = 512,
        device: str = "cuda",
        fovy_deg: float = 40.0,
        azimuth_deg: float = 15.0,
        elevation_deg: float = 12.0,
        distance_scale: float = 1.8,
        background=(1.0, 1.0, 1.0),
        ambient: float = 0.35,
    ):
        self.asset = asset
        self.device = torch.device(device)
        self.image_size = image_size
        repo_root = Path(__file__).resolve().parents[2]
        self.dr = import_nvdiffrast(repo_root)
        self.glctx = None
        self.background = torch.tensor(background, dtype=torch.float32, device=self.device)
        self.ambient = ambient
        self.light_dir = F.normalize(torch.tensor([0.3, 0.6, 1.0], dtype=torch.float32, device=self.device), dim=0)
        self.projection_matrix, self.view_matrix, self.eye_position, self.center_position = self._build_default_camera(
            fovy_deg, azimuth_deg, elevation_deg, distance_scale
        )
        self.mvp = self.projection_matrix @ self.view_matrix
        self.skinning_device = self.device
        self.vertices_skin = self.asset.vertices.detach().to(self.skinning_device)
        self.skin_indices_skin = self.asset.skin_indices.detach().to(self.skinning_device)
        self.skin_weights_skin = self.asset.skin_weights.detach().to(self.skinning_device)
        self.bone_parents_skin = self.asset.bone_parents.detach().to(self.skinning_device)
        self.bone_local_rest_mats_skin = self.asset.bone_local_rest_mats.detach().to(self.skinning_device)
        self.bone_inv_bind_mats_skin = self.asset.bone_inv_bind_mats.detach().to(self.skinning_device)

    def vertex_albedos(self) -> torch.Tensor:
        colors = torch.empty((self.asset.vertices.shape[0], 3), dtype=torch.float32, device=self.device)
        for mesh_idx in range(self.asset.mesh_vertex_ranges.shape[0]):
            start, end = self.asset.mesh_vertex_ranges[mesh_idx].tolist()
            colors[start:end] = self.asset.mesh_albedos[mesh_idx]
        return colors

    def _ensure_glctx(self):
        if self.glctx is None:
            try:
                self.glctx = self.dr.RasterizeGLContext()
            except Exception as error:
                raise RuntimeError(
                    "Unable to initialize the nvdiffrast OpenGL rasterizer context. "
                    "This usually means the local nvdiffrast extension could not be built or loaded "
                    "for the active PyTorch/C++ toolchain."
                ) from error
        return self.glctx

    def _build_default_camera(self, fovy_deg, azimuth_deg, elevation_deg, distance_scale):
        verts = self.asset.vertices
        center = verts.mean(dim=0)
        extent = (verts.max(dim=0).values - verts.min(dim=0).values).amax().item()
        extent = max(extent, 1e-3)
        azimuth = math.radians(azimuth_deg)
        elevation = math.radians(elevation_deg)
        radius = extent * distance_scale
        eye = [
            center[0].item() + radius * math.cos(elevation) * math.sin(azimuth),
            center[1].item() + radius * math.sin(elevation),
            center[2].item() + radius * math.cos(elevation) * math.cos(azimuth),
        ]
        view = look_at(eye=eye, at=center.tolist(), up=[0.0, 1.0, 0.0], device=self.device)
        proj = perspective(fovy_deg, 1.0, 0.01, 1000.0, device=self.device)
        return (
            proj,
            view,
            torch.tensor(eye, dtype=torch.float32, device=self.device).view(1, 3),
            center.view(1, 3),
        )

    def forward_kinematics(self, rotvecs: torch.Tensor, translations: torch.Tensor | None = None) -> torch.Tensor:
        rotvecs = rotvecs.to(self.skinning_device)
        batch_size, num_bones = rotvecs.shape[:2]
        if translations is None:
            translations = torch.zeros((batch_size, num_bones, 3), dtype=torch.float32, device=self.skinning_device)
        else:
            translations = translations.to(self.skinning_device)
        delta_rot = axis_angle_to_matrix(rotvecs)
        delta = torch.eye(4, dtype=torch.float32, device=self.skinning_device).view(1, 1, 4, 4).repeat(batch_size, num_bones, 1, 1)
        delta[:, :, :3, :3] = delta_rot
        delta[:, :, :3, 3] = translations

        globals_per_bone = []
        for bone_idx in range(num_bones):
            local = matmul_4x4(self.bone_local_rest_mats_skin[bone_idx].view(1, 4, 4), delta[:, bone_idx])
            parent_idx = int(self.bone_parents_skin[bone_idx].item())
            if parent_idx < 0:
                global_transform = local
            else:
                global_transform = matmul_4x4(globals_per_bone[parent_idx], local)
            globals_per_bone.append(global_transform)
        return torch.stack(globals_per_bone, dim=1)

    def skin_vertices(self, rotvecs: torch.Tensor, translations: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        globals_per_bone = self.forward_kinematics(rotvecs, translations)
        skin_mats = matmul_4x4(globals_per_bone, self.bone_inv_bind_mats_skin.view(1, -1, 4, 4))

        per_vertex_mats = skin_mats[:, self.skin_indices_skin]
        verts_h = torch.cat([self.vertices_skin, torch.ones_like(self.vertices_skin[:, :1])], dim=-1)
        transformed = transform_homogeneous(per_vertex_mats, verts_h.view(1, -1, 1, 4))
        verts = (
            transformed[:, :, :, :3]
            * self.skin_weights_skin.view(1, -1, self.skin_weights_skin.shape[1], 1)
        ).sum(dim=2)
        return verts.to(self.device), globals_per_bone.to(self.device)

    def build_pytorch3d_mesh(self, rotvecs: torch.Tensor, translations: torch.Tensor | None = None):
        Meshes, TexturesVertex = import_pytorch3d()
        verts_batch, _ = self.skin_vertices(rotvecs, translations)
        verts = verts_batch[0].contiguous()
        faces = self.asset.faces.to(dtype=torch.int64, device=self.device).contiguous()
        colors = self.vertex_albedos().unsqueeze(0).contiguous()
        textures = TexturesVertex(verts_features=colors)
        return Meshes(verts=[verts], faces=[faces], textures=textures)

    def build_drot_sensors(self):
        return [
            {
                "position": self.eye_position,
                "resolution": (self.image_size, self.image_size),
                "matrix": self.mvp.unsqueeze(0),
                "camera_matrix": self.view_matrix.unsqueeze(0),
                "perspective_matrix": self.projection_matrix.unsqueeze(0),
                "center": self.center_position,
                "init_position": self.eye_position.clone(),
                "init_center": self.center_position.clone(),
            }
        ]

    def build_drot_scene(self, rotvecs: torch.Tensor, translations: torch.Tensor | None = None):
        mesh = self.build_pytorch3d_mesh(rotvecs, translations)
        diffuse_scale = 1.0 - self.ambient
        material = SimpleNamespace(
            ambient_color=torch.tensor(
                [[self.ambient, self.ambient, self.ambient]], dtype=torch.float32, device=self.device
            ),
            diffuse_color=torch.tensor(
                [[diffuse_scale, diffuse_scale, diffuse_scale]], dtype=torch.float32, device=self.device
            ),
            specular_color=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=self.device),
            shininess=torch.tensor([1.0], dtype=torch.float32, device=self.device),
        )
        return {
            "meshes": [{"model": mesh}],
            "sensors": self.build_drot_sensors(),
            "material": material,
        }

    def project_joints(self, rotvecs: torch.Tensor, translations: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        globals_per_bone = self.forward_kinematics(rotvecs, translations).to(self.device)
        joint_world = globals_per_bone[:, :, :3, 3]
        joint_h = torch.cat([joint_world, torch.ones_like(joint_world[..., :1])], dim=-1)
        clip = torch.matmul(joint_h, self.mvp.t())
        w = clip[..., 3].abs().clamp_min(1e-6)
        ndc = clip[..., :3] / w.unsqueeze(-1)
        x = ((ndc[..., 0] + 1.0) * 0.5) * (self.image_size - 1)
        y = (1.0 - (ndc[..., 1] + 1.0) * 0.5) * (self.image_size - 1)
        pixels = torch.stack([x, y], dim=-1)
        valid = (clip[..., 3] > 0.0) & (ndc[..., 2] >= -1.5) & (ndc[..., 2] <= 1.5)
        return pixels, valid

    def render_joint_overlay(
        self,
        image: torch.Tensor,
        rotvecs: torch.Tensor,
        translations: torch.Tensor | None = None,
        joint_radius: int = 4,
        line_width: int = 2,
    ) -> torch.Tensor:
        joint_pixels, valid = self.project_joints(rotvecs, translations)
        return overlay_skeleton_on_image(
            image=image,
            joint_pixels=joint_pixels[0],
            bone_parents=self.asset.bone_parents,
            valid=valid[0],
            joint_radius=joint_radius,
            line_width=line_width,
        )

    def render_pose(self, rotvecs: torch.Tensor, translations: torch.Tensor | None = None, global_trans: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        glctx = self._ensure_glctx()
        verts_batch, globals_per_bone = self.skin_vertices(rotvecs, translations)
        verts = verts_batch[0]
        if global_trans is not None:
            verts = verts + global_trans.view(1, 3)

        normals = compute_vertex_normals(verts, self.asset.faces.long())
        light_pos = self.eye_position[0]
        light_dir = F.normalize(light_pos.view(1, 3) - verts, dim=-1, eps=1e-6)
        diffuse = torch.sum(normals * light_dir, dim=-1, keepdim=True).clamp_min(0.0)
        colors = self.vertex_albedos() * (self.ambient + (1.0 - self.ambient) * diffuse)

        verts_h = torch.cat([verts, torch.ones_like(verts[:, :1])], dim=-1)
        clip = (verts_h @ self.mvp.t()).unsqueeze(0).contiguous()
        rast, _ = self.dr.rasterize(glctx, clip, self.asset.faces, resolution=[self.image_size, self.image_size])
        color, _ = self.dr.interpolate(colors.unsqueeze(0), rast, self.asset.faces)

        mask = (rast[..., 3:] > 0).float()
        image = color * mask + self.background.view(1, 1, 1, 3) * (1.0 - mask)
        image = self.dr.antialias(image, rast, clip, self.asset.faces)
        alpha = self.dr.antialias(mask, rast, clip, self.asset.faces)
        return {
            "image": image[0],
            "mask": alpha[0, :, :, 0],
            "verts": verts,
            "joint_mats": globals_per_bone[0],
        }
