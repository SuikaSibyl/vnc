import argparse
import pathlib

import imageio
import numpy as np
import torch
import trimesh

import nvdiffrast.torch as dr

import util


def _normalize(v: np.ndarray) -> np.ndarray:
	length = np.linalg.norm(v)
	if length == 0.0:
		return v
	return v / length


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
	forward = _normalize(target - eye)
	right = _normalize(np.cross(forward, up))
	camera_up = np.cross(right, forward)

	return np.asarray(
		[
			[right[0], right[1], right[2], -np.dot(right, eye)],
			[camera_up[0], camera_up[1], camera_up[2], -np.dot(camera_up, eye)],
			[-forward[0], -forward[1], -forward[2], np.dot(forward, eye)],
			[0.0, 0.0, 0.0, 1.0],
		],
		dtype=np.float32,
	)


def orthographic(scale: float, near: float, far: float) -> np.ndarray:
	return np.asarray(
		[
			[1.0 / scale, 0.0, 0.0, 0.0],
			[0.0, 1.0 / scale, 0.0, 0.0],
			[0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near)],
			[0.0, 0.0, 0.0, 1.0],
		],
		dtype=np.float32,
	)


def load_scene_mesh(glb_path: str):
	scene = trimesh.load(glb_path, force='scene')
	# Apply the scene transform to get the properly-oriented mesh
	# The GLB root node has a transform that defines the canonical orientation
	mesh = scene.to_geometry()

	uv = getattr(mesh.visual, 'uv', None)
	if uv is None:
		uv = np.full((mesh.vertices.shape[0], 2), 0.5, dtype=np.float32)
		mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv)

	material = getattr(mesh.visual, 'material', None)
	if material is None or getattr(material, 'baseColorTexture', None) is None:
		texture = np.ones((1, 1, 4), dtype=np.float32)
	else:
		texture = np.asarray(material.baseColorTexture.convert('RGBA'), dtype=np.float32) / 255.0
	texture = np.flip(texture, axis=0).copy()
	return mesh, texture


def rotation_y(degrees: float) -> np.ndarray:
	rad = np.deg2rad(degrees)
	c, s = np.cos(rad), np.sin(rad)
	return np.asarray(
		[
			[ c, 0.0,  s, 0.0],
			[0.0, 1.0, 0.0, 0.0],
			[-s, 0.0,  c, 0.0],
			[0.0, 0.0, 0.0, 1.0],
		],
		dtype=np.float32,
	)


def rotation_x(degrees: float) -> np.ndarray:
	rad = np.deg2rad(degrees)
	c, s = np.cos(rad), np.sin(rad)
	return np.asarray(
		[
			[1.0, 0.0, 0.0, 0.0],
			[0.0,  c, -s, 0.0],
			[0.0,  s,  c, 0.0],
			[0.0, 0.0, 0.0, 1.0],
		],
		dtype=np.float32,
	)


def _build_camera(mesh, camera_scale, device):
	"""Compute projection@view matrix and scene params shared across renderers."""
	bounds = mesh.bounds.astype(np.float32)
	center = bounds.mean(axis=0)
	extents = bounds[1] - bounds[0]
	radius = 0.5 * np.linalg.norm(extents)
	if camera_scale is None:
		camera_scale = 1.15 * radius

	view_dir = _normalize(np.asarray([1.0, 0.5, 1.0], dtype=np.float32))
	up_dir = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
	eye = center + view_dir * (4.0 * radius)
	view = look_at(eye, center, up_dir)
	near = max(0.1 * radius, 4.0 * radius - radius)
	far = 4.0 * radius + radius
	projection = orthographic(camera_scale, near, far)

	proj_view = torch.tensor(projection @ view, dtype=torch.float32, device=device)
	center_t = torch.tensor(center, dtype=torch.float32, device=device)
	return proj_view, center_t, radius


def _rotation_y_torch(angle_rad: torch.Tensor, device) -> torch.Tensor:
	"""Build a 4x4 Y-axis rotation matrix from a differentiable angle (radians)."""
	c = torch.cos(angle_rad)
	s = torch.sin(angle_rad)
	z = torch.zeros([], device=device, dtype=torch.float32)
	o = torch.ones([], device=device, dtype=torch.float32)
	return torch.stack([
		torch.stack([ c,  z,  s, z]),
		torch.stack([ z,  o,  z, z]),
		torch.stack([-s,  z,  c, z]),
		torch.stack([ z,  z,  z, o]),
	])


def _rotation_x_torch(angle_rad: torch.Tensor, device) -> torch.Tensor:
	"""Build a 4x4 X-axis rotation matrix from a differentiable angle (radians)."""
	c = torch.cos(angle_rad)
	s = torch.sin(angle_rad)
	z = torch.zeros([], device=device, dtype=torch.float32)
	o = torch.ones([], device=device, dtype=torch.float32)
	return torch.stack([
		torch.stack([ o,  z,  z, z]),
		torch.stack([ z,  c, -s, z]),
		torch.stack([ z,  s,  c, z]),
		torch.stack([ z,  z,  z, o]),
	])


def _render(glctx, vertices, faces, uvs, tex, proj_view, center_t, angle_rad, resolution, device,
            angle_x_rad=None):
	"""Core differentiable render.

	angle_rad   – Y-axis rotation (radians, may have grad).
	angle_x_rad – optional X-axis rotation (radians, may have grad).
	              Applied before Y: model = T_back @ R_y @ R_x @ T_origin.
	"""
	rot = _rotation_y_torch(angle_rad, device)
	if angle_x_rad is not None:
		rot = rot @ _rotation_x_torch(angle_x_rad, device)

	# model = T_back @ R @ T_origin  (rotate around mesh center)
	T_origin = torch.eye(4, device=device)
	T_origin[:3, 3] = -center_t
	T_back = torch.eye(4, device=device)
	T_back[:3, 3] = center_t
	model = T_back @ rot @ T_origin

	mvp = proj_view @ model

	pos_w = torch.cat([vertices, torch.ones([vertices.shape[0], 1], device=device)], dim=1)
	pos_clip = (pos_w @ mvp.t())[None]

	rast_out, _ = dr.rasterize(glctx, pos_clip, faces, resolution=[resolution, resolution])
	uv_interp, _ = dr.interpolate(uvs[None], rast_out, faces)
	color = dr.texture(tex[None], uv_interp, filter_mode='linear')

	mask = (rast_out[..., 3:4] > 0).to(color.dtype)
	bg = torch.tensor([0.96, 0.97, 0.98, 1.0], device=device, dtype=color.dtype).view(1, 1, 1, 4)
	color = color * mask + bg * (1.0 - mask)
	color = dr.antialias(color, rast_out, pos_clip, faces)
	return color


def render_mesh(mesh: trimesh.Trimesh, texture: np.ndarray, resolution: int = 512,
                camera_scale: float = None, rotate_y: float = 0.0, rotate_x: float = 0.0):
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required for nvdiffrast rendering.')

	device = torch.device('cuda')
	glctx = dr.RasterizeCudaContext()

	vertices = torch.from_numpy(mesh.vertices.astype(np.float32)).to(device)
	faces = torch.from_numpy(mesh.faces.astype(np.int32)).to(device)
	uvs = torch.from_numpy(mesh.visual.uv.astype(np.float32)).to(device)
	tex = torch.from_numpy(texture.astype(np.float32)).to(device)

	proj_view, center_t, _ = _build_camera(mesh, camera_scale, device)
	angle_y = torch.tensor(np.deg2rad(rotate_y), dtype=torch.float32, device=device)
	angle_x = torch.tensor(np.deg2rad(rotate_x), dtype=torch.float32, device=device) if rotate_x != 0.0 else None

	with torch.no_grad():
		color = _render(glctx, vertices, faces, uvs, tex, proj_view, center_t, angle_y, resolution, device,
		                angle_x_rad=angle_x)

	return color[0].detach().cpu().numpy()


def optimize_rotation(
	mesh: trimesh.Trimesh,
	texture: np.ndarray,
	init_deg: float = 0.0,
	target_deg: float = 180.0,
	resolution: int = 512,
	camera_scale: float = None,
	n_iters: int = 300,
	lr: float = 0.05,
):
	"""Differentiably optimize Y-axis rotation to match a target-angle render.

	Returns (optimized_degrees, target_image_np, final_image_np).
	"""
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required for nvdiffrast rendering.')

	device = torch.device('cuda')
	glctx = dr.RasterizeCudaContext()

	vertices = torch.from_numpy(mesh.vertices.astype(np.float32)).to(device)
	faces = torch.from_numpy(mesh.faces.astype(np.int32)).to(device)
	uvs = torch.from_numpy(mesh.visual.uv.astype(np.float32)).to(device)
	tex = torch.from_numpy(texture.astype(np.float32)).to(device)

	proj_view, center_t, _ = _build_camera(mesh, camera_scale, device)

	# Render fixed target image
	target_rad = torch.tensor(np.deg2rad(target_deg), dtype=torch.float32, device=device)
	with torch.no_grad():
		target_img = _render(glctx, vertices, faces, uvs, tex, proj_view, center_t, target_rad, resolution, device)

	# Differentiable angle parameter starting at init_deg
	angle_rad = torch.tensor(np.deg2rad(init_deg), dtype=torch.float32, device=device, requires_grad=True)
	optimizer = torch.optim.Adam([angle_rad], lr=lr)

	print(f'Optimizing rotation: init={init_deg}°  target={target_deg}°')
	for i in range(n_iters):
		optimizer.zero_grad()
		pred_img = _render(glctx, vertices, faces, uvs, tex, proj_view, center_t, angle_rad, resolution, device)
		loss = torch.mean((pred_img - target_img) ** 2)
		loss.backward()
		optimizer.step()

		if (i + 1) % 50 == 0 or i == 0:
			deg = np.rad2deg(angle_rad.item())
			print(f'  iter {i+1:4d}  loss={loss.item():.6f}  angle={deg:.2f}°')

	final_deg = np.rad2deg(angle_rad.item())
	print(f'Done. Final angle={final_deg:.2f}° (target={target_deg}°)')

	with torch.no_grad():
		final_img = _render(glctx, vertices, faces, uvs, tex, proj_view, center_t, angle_rad.detach(), resolution, device)

	return final_deg, target_img[0].cpu().numpy(), final_img[0].cpu().numpy()


def main():
	parser = argparse.ArgumentParser(description='Render fantasy_house.glb to a PNG image.')
	parser.add_argument('--input', default=str(pathlib.Path(__file__).resolve().parent / 'scenes' / 'fantasy_house.glb'))
	parser.add_argument('--output', default=str(pathlib.Path(__file__).resolve().parent / 'fantasy_house.png'))
	parser.add_argument('--resolution', type=int, default=512)
	parser.add_argument('--camera-scale', type=float, default=None)
	parser.add_argument('--rotate-y', type=float, default=None,
		help='Y-axis rotation in degrees. If omitted, renders both 0° and 180°.')
	parser.add_argument('--rotate-x', type=float, default=0.0,
		help='X-axis rotation in degrees (pitch). Applied before Y rotation.')
	parser.add_argument('--optimize', action='store_true',
		help='Run differentiable optimizer from 0° toward 180° and save results.')
	parser.add_argument('--n-iters', type=int, default=300, help='Optimizer iterations (used with --optimize).')
	parser.add_argument('--lr', type=float, default=0.05, help='Optimizer learning rate (used with --optimize).')
	args = parser.parse_args()

	mesh, texture = load_scene_mesh(args.input)
	output_path = pathlib.Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	if args.optimize:
		_, target_np, final_np = optimize_rotation(
			mesh, texture,
			init_deg=0.0, target_deg=180.0,
			resolution=args.resolution, camera_scale=args.camera_scale,
			n_iters=args.n_iters, lr=args.lr,
		)
		for img_np, suffix in [(target_np, 'target'), (final_np, 'optimized')]:
			img_np = img_np[::-1]
			path = output_path.with_stem(f'{output_path.stem}_{suffix}')
			imageio.imwrite(path, np.clip(np.rint(img_np * 255.0), 0, 255).astype(np.uint8))
			print(f'Saved {path}')
	else:
		rotations = [args.rotate_y] if args.rotate_y is not None else [0.0, 180.0]
		for deg in rotations:
			image = render_mesh(mesh, texture, resolution=args.resolution, camera_scale=args.camera_scale, rotate_y=deg, rotate_x=args.rotate_x)
			image = image[::-1]
			if len(rotations) > 1:
				path = output_path.with_stem(f'{output_path.stem}_rot{int(deg):03d}')
			else:
				path = output_path
			imageio.imwrite(path, np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8))
			print(f'Saved {path}')


if __name__ == '__main__':
	main()
