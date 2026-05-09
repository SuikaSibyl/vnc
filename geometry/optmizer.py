from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parent
LARGESTEPS_DIR = ROOT_DIR / "largesteps"

if str(LARGESTEPS_DIR) not in sys.path:
	sys.path.insert(0, str(LARGESTEPS_DIR))

_constants = importlib.import_module("scripts.constants")
_geometry = importlib.import_module("scripts.geometry")
_load_xml = importlib.import_module("scripts.load_xml")
_render = importlib.import_module("scripts.render")
_main = importlib.import_module("scripts.main")
_io_ply = importlib.import_module("scripts.io_ply")

SCENES_DIR = _constants.SCENES_DIR
OUTPUT_DIR = _constants.OUTPUT_DIR
compute_face_normals = _geometry.compute_face_normals
compute_vertex_normals = _geometry.compute_vertex_normals
load_scene = _load_xml.load_scene
NVDRenderer = _render.NVDRenderer
optimize_shape = _main.optimize_shape
write_ply = _io_ply.write_ply


def _make_voxel_remesh_fn(resolution: int):
	"""Return a remesh callable that uses trimesh voxelization + marching cubes.

	Behaves similarly to Blender's Voxel Remesh modifier: the mesh is voxelized
	at `resolution` voxels along its longest bounding-box axis and the surface is
	reconstructed via marching cubes.  The result is a uniformly-distributed,
	topology-agnostic mesh with no dependency on the original connectivity.

	Parameters
	----------
	resolution:
		Number of voxels along the longest bounding-box dimension.
	"""
	import trimesh

	def _fn(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
		longest = mesh.bounding_box.extents.max()
		if longest < 1e-8:
			raise RuntimeError("Voxel remesh: mesh bounding box is degenerate.")
		pitch = longest / resolution
		vox = mesh.voxelized(pitch=pitch).fill()
		new_mesh = vox.marching_cubes
		# marching_cubes returns vertices in voxel-index space; apply the
		# VoxelGrid transform to convert back to the original world-space frame.
		v_new = trimesh.transformations.transform_points(
			np.asarray(new_mesh.vertices, dtype=np.float64), vox.transform
		)
		return v_new, np.asarray(new_mesh.faces, dtype=np.int32)

	return _fn


def _save_image(path: Path, image, gamma: float = 2.2) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	img = (image[..., :3].clamp(0, 1).pow(1 / gamma).detach().cpu().numpy() * 255).astype(np.uint8)
	img = img[::-1, ..., ::-1]
	cv2.imwrite(str(path), img)


def _annotate_iteration(frame_bgr: np.ndarray, iteration: int) -> np.ndarray:
	img = frame_bgr.copy()
	label = f"iter {iteration + 1}"
	font = cv2.FONT_HERSHEY_SIMPLEX
	scale = 0.8
	thickness = 2
	(text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
	x, y = 12, 28
	cv2.rectangle(img, (x - 6, y - text_h - 6), (x + text_w + 6, y + baseline + 6), (0, 0, 0), thickness=-1)
	cv2.putText(img, label, (x, y), font, scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
	return img


def _save_video(path: Path, frames: list[tuple[np.ndarray, int]], fps: int) -> None:
	if not frames:
		return
	max_frames = max(1, int(round(fps * 3.0)))
	if len(frames) > max_frames:
		keep_idx = np.linspace(0, len(frames) - 1, num=max_frames, dtype=int)
		frames = [frames[idx] for idx in keep_idx]
	annotated_rgb = [cv2.cvtColor(_annotate_iteration(frame, iteration), cv2.COLOR_BGR2RGB) for frame, iteration in frames]
	path.parent.mkdir(parents=True, exist_ok=True)
	imageio.mimwrite(path, annotated_rgb, fps=fps)
	print(f"Saved: {path}")


def _save_trajectory_videos(out_dir: Path, view_indices: list[int], fps: int) -> None:
	for view_idx in view_indices:
		render_frames: list[tuple[np.ndarray, int]] = []
		ref_frames: list[tuple[np.ndarray, int]] = []
		pair_frames: list[tuple[np.ndarray, int]] = []

		for render_path in sorted(out_dir.glob(f"render_step_*_view_{view_idx:03d}.png")):
			stem_parts = render_path.stem.split("_")
			try:
				step = int(stem_parts[2])
			except (IndexError, ValueError):
				continue

			render_img = cv2.imread(str(render_path), cv2.IMREAD_COLOR)
			if render_img is not None:
				render_frames.append((render_img, step))

			ref_matches = sorted(out_dir.glob(f"ref_step_{step:06d}_*_view_{view_idx:03d}.png"))
			if ref_matches:
				ref_img = cv2.imread(str(ref_matches[-1]), cv2.IMREAD_COLOR)
				if ref_img is not None:
					ref_frames.append((ref_img, step))

			pair_matches = sorted(out_dir.glob(f"pair_step_{step:06d}_*_view_{view_idx:03d}.png"))
			if pair_matches:
				pair_img = cv2.imread(str(pair_matches[-1]), cv2.IMREAD_COLOR)
				if pair_img is not None:
					pair_frames.append((pair_img, step))

		_save_video(out_dir / f"trajectory_view_{view_idx:03d}.mp4", render_frames, fps=fps)
		_save_video(out_dir / f"trajectory_ref_view_{view_idx:03d}.mp4", ref_frames, fps=fps)
		_save_video(out_dir / f"trajectory_pair_view_{view_idx:03d}.mp4", pair_frames, fps=fps)


def _load_video_frames_ldr_to_hdr(
	video_path: str,
	resolution: tuple[int, int],
	device: torch.device,
	gamma: float,
	anchor_image: torch.Tensor | None = None,
	apply_first_frame_remap: bool = True,
	video_limited_range: bool = False,
) -> list[torch.Tensor]:
	def _fit_global_linear_map(first_frame: torch.Tensor, anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""Fit per-channel affine map y ~= a*x + b from frame to anchor.

		The fit uses background pixels if an alpha channel is present in anchor,
		otherwise it falls back to all pixels.
		"""
		anchor_rgb = anchor[..., :3]
		if anchor_rgb.shape[:2] != first_frame.shape[:2]:
			anchor_chw = anchor_rgb.permute(2, 0, 1)[None, ...]
			anchor_rgb = F.interpolate(
				anchor_chw,
				size=first_frame.shape[:2],
				mode="bilinear",
				align_corners=False,
			)[0].permute(1, 2, 0)

		if anchor.shape[-1] >= 4:
			alpha = anchor[..., 3]
			if alpha.shape != first_frame.shape[:2]:
				alpha = F.interpolate(alpha[None, None, ...], size=first_frame.shape[:2], mode="nearest")[0, 0]
			bg_mask = alpha < 0.5
		else:
			bg_mask = torch.ones(first_frame.shape[:2], dtype=torch.bool, device=first_frame.device)

		if int(bg_mask.sum().item()) < 128:
			bg_mask = torch.ones(first_frame.shape[:2], dtype=torch.bool, device=first_frame.device)

		x = first_frame[bg_mask].reshape(-1, 3)
		y = anchor_rgb[bg_mask].reshape(-1, 3)

		x_mean = x.mean(dim=0)
		y_mean = y.mean(dim=0)
		x_centered = x - x_mean
		y_centered = y - y_mean
		var = (x_centered * x_centered).mean(dim=0)
		cov = (x_centered * y_centered).mean(dim=0)
		a = cov / torch.clamp(var, min=1e-8)
		b = y_mean - a * x_mean
		return a, b

	def _apply_global_linear_map(frame: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		return (frame * a.view(1, 1, 3) + b.view(1, 1, 3)).clamp(0, 1)

	path = Path(video_path)
	if not path.exists():
		raise FileNotFoundError(f"Video not found: {path}")
	
	width, height = resolution

	# If a directory is given, load image frames directly to avoid video codec/color conversions.
	if path.is_dir():
		patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
		frame_paths: list[Path] = []
		for pattern in patterns:
			frame_paths.extend(sorted(path.glob(pattern)))
		if not frame_paths:
			raise RuntimeError(f"No image frames found in directory: {path}")

		frames: list[torch.Tensor] = []

		for frame_path in frame_paths:
			frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
			if frame is None:
				continue
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			if frame.shape[1] != width or frame.shape[0] != height:
				frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
			frame = frame[::-1, ...]
			frame = torch.from_numpy(frame.astype(np.float32) / 255.0).pow(gamma)
			frames.append(frame.to(device))

		if not frames:
			raise RuntimeError(f"Failed to decode any valid image frames from: {path}")

		# if apply_first_frame_remap and anchor_image is not None:
		# 	a, b = _fit_global_linear_map(frames[0], anchor_image.to(device))
		# 	# frames = [_apply_global_linear_map(frame, a, b) for frame in frames]
		# 	print(f"Applied first-frame global remap (a={a.detach().cpu().numpy()}, b={b.detach().cpu().numpy()})")
		return frames

	capture = cv2.VideoCapture(str(path))
	if not capture.isOpened():
		raise RuntimeError(f"Cannot open video: {path}")

	frames: list[torch.Tensor] = []
	while True:
		ok, frame = capture.read()
		if not ok:
			break
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		if frame.shape[1] != width or frame.shape[0] != height:
			frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
		frame = frame[::-1, ...]
		# Convert LDR sRGB-ish frame to linear/HDR-like space used by optimization targets.
		f = frame.astype(np.float32)
		frame = torch.from_numpy(f / 255.0).clip(0, 1.0).pow(gamma)
		frames.append(frame.to(device))
	capture.release()

	if not frames:
		raise RuntimeError(f"No frames decoded from: {path}")

	return frames


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run a single-view large-steps optimization for a chosen scene.")
	parser.add_argument("--scene-name", default="dragon", help="Scene name inside largesteps/scenes, e.g. dragon or bunny.")
	parser.add_argument(
		"--scene-dir",
		default=None,
		help="Optional explicit scene directory. Overrides --scene-name if provided.",
	)
	parser.add_argument("--view-idx", type=int, default=1, help="View index to render and optimize against, default: 1 (view 001).")
	parser.add_argument(
		"--view-idxs",
		type=int,
		nargs="+",
		default=None,
		help="Optional list of view indices to optimize jointly. Overrides --view-idx when provided.",
	)
	parser.add_argument("--gamma", type=float, default=2.2, help="Gamma used when saving PNGs.")
	parser.add_argument("--boost", type=float, default=3.0, help="Renderer gradient boost parameter.")
	parser.add_argument("--black-background", action="store_true", help="Render with black background while still using envmap for lighting.")
	parser.add_argument(
		"--optimize-translation",
		action="store_true",
		help="Jointly optimize a global translation of the mesh. Off by default.",
	)
	parser.add_argument("--steps", type=int, default=16000, help="Number of optimization iterations.")
	parser.add_argument(
		"--step-size",
		type=float,
		default=1e-1,
		help="Optimization learning rate (step size).",
	)
	parser.add_argument(
		"--translation-step-size",
		type=float,
		default=None,
		help="Optional separate learning rate for global translation when --optimize-translation is enabled. "
		"Defaults to --step-size.",
	)
	parser.add_argument(
		"--lambda",
		dest="lambda_",
		type=float,
		default=19.0,
		help="Large-steps lambda regularization/parameterization weight.",
	)
	parser.add_argument(
		"--no-remesh",
		action="store_true",
		help="Disable remeshing during optimization.",
	)
	parser.add_argument(
		"--remesh-interval",
		type=int,
		default=0,
		help="Remesh every N iterations. 0 keeps the default fixed remesh schedule.",
	)
	parser.add_argument(
		"--video-path",
		default=None,
		help="Optional guide target source. Can be a video file or a directory of image frames.",
	)
	parser.add_argument(
		"--video-paths",
		nargs="+",
		default=None,
		help="Optional per-view guide target sources for multi-view optimization. "
		"Provide one path per --view-idxs entry, in the same order.",
	)
	parser.add_argument("--video-ratio", type=float, default=0.9, help="Fraction of iterations guided by video targets.")
	parser.add_argument(
		"--disable-video-remap",
		action="store_true",
		help="Disable first-frame global remap for video targets.",
	)
	parser.add_argument(
		"--video-limited-range",
		action="store_true",
		help="Treat video frames as limited range (Y: 16-235) and expand to full range before processing. "
		"Use when colors appear washed out or shifted with video input.",
	)
	parser.add_argument("--video-width", type=int, default=None, help="Optional video target width; defaults to scene resolution.")
	parser.add_argument("--video-height", type=int, default=None, help="Optional video target height; defaults to scene resolution.")
	parser.add_argument(
		"--save-trajectory-video",
		action="store_true",
		help="If set, build short MP4 trajectory videos from saved optimization renders.",
	)
	parser.add_argument(
		"--trajectory-fps",
		type=int,
		default=12,
		help="Frames per second for saved optimization trajectory videos.",
	)
	parser.add_argument(
		"--remesh-method",
		choices=["botsch", "voxel"],
		default="botsch",
		help="Remeshing algorithm to use at each remesh step. "
		"'botsch' (default) runs 5 iterations of Botsch-Kobbelt isotropic remeshing. "
		"'voxel' voxelizes the mesh and reconstructs via marching cubes, similar to Blender's Voxel Remesh modifier.",
	)
	parser.add_argument(
		"--voxel-resolution",
		type=int,
		default=128,
		help="Voxel resolution for --remesh-method=voxel: number of voxels along the longest bounding-box axis (default: 128).",
	)
	parser.add_argument(
		"--render-interval",
		type=int,
		default=1000,
		help="Save an optimization PNG every N iterations.",
	)
	parser.add_argument(
		"--output-dir",
		default=None,
		help="Optional output directory. Default: <largesteps output>/<scene-name>_view_<view-idx>",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()

	scene_dir = Path(args.scene_dir).resolve() if args.scene_dir else Path(SCENES_DIR) / args.scene_name
	scene_name = scene_dir.name
	scene_xml = scene_dir / f"{scene_name}.xml"
	if not scene_xml.exists():
		raise FileNotFoundError(f"Scene XML not found: {scene_xml}")

	scene_params = load_scene(str(scene_xml))
	renderer = NVDRenderer(scene_params, shading=True, boost=args.boost, black_background=args.black_background)

	v_src = scene_params["mesh-source"]["vertices"]
	f_src = scene_params["mesh-source"]["faces"]
	src_face_normals = compute_face_normals(v_src, f_src)
	n_src = compute_vertex_normals(v_src, f_src, src_face_normals)

	v_ref = scene_params["mesh-target"]["vertices"]
	f_ref = scene_params["mesh-target"]["faces"]
	if "normals" in scene_params["mesh-target"]:
		n_ref = scene_params["mesh-target"]["normals"]
	else:
		ref_face_normals = compute_face_normals(v_ref, f_ref)
		n_ref = compute_vertex_normals(v_ref, f_ref, ref_face_normals)

	init_imgs = renderer.render(v_src, n_src, f_src)
	ref_imgs = renderer.render(v_ref, n_ref, f_ref)

	num_views = int(ref_imgs.shape[0])
	view_indices = list(args.view_idxs) if args.view_idxs else [args.view_idx]
	if len(view_indices) == 0:
		raise ValueError("At least one reference view must be provided.")
	for view_idx in view_indices:
		if view_idx < 0 or view_idx >= num_views:
			raise ValueError(f"view-idx {view_idx} out of range [0, {num_views - 1}]")

	video_frames = None
	video_frames_per_view = None
	if args.video_paths:
		if len(args.video_paths) != len(view_indices):
			raise ValueError(
				f"--video-paths expects one path per selected view: got {len(args.video_paths)} paths for {len(view_indices)} views."
			)
		if len(view_indices) == 1:
			video_width = args.video_width if args.video_width is not None else int(scene_params["res_x"])
			video_height = args.video_height if args.video_height is not None else int(scene_params["res_y"])
			video_frames = _load_video_frames_ldr_to_hdr(
				video_path=args.video_paths[0],
				resolution=(video_width, video_height),
				device=v_src.device,
				gamma=args.gamma,
				anchor_image=init_imgs[view_indices[0]],
				apply_first_frame_remap=not args.disable_video_remap,
				video_limited_range=args.video_limited_range,
			)
		else:
			video_frames_per_view = []
			for video_path, view_idx in zip(args.video_paths, view_indices):
				video_width = args.video_width if args.video_width is not None else int(scene_params["res_x"])
				video_height = args.video_height if args.video_height is not None else int(scene_params["res_y"])
				video_frames_per_view.append(
					_load_video_frames_ldr_to_hdr(
						video_path=video_path,
						resolution=(video_width, video_height),
						device=v_src.device,
						gamma=args.gamma,
						anchor_image=init_imgs[view_idx],
						apply_first_frame_remap=not args.disable_video_remap,
						video_limited_range=args.video_limited_range,
					)
				)
	elif args.video_path:
		video_width = args.video_width if args.video_width is not None else int(scene_params["res_x"])
		video_height = args.video_height if args.video_height is not None else int(scene_params["res_y"])
		video_frames = _load_video_frames_ldr_to_hdr(
			video_path=args.video_path,
			resolution=(video_width, video_height),
			device=v_src.device,
			gamma=args.gamma,
			anchor_image=init_imgs[view_indices[0]],
			apply_first_frame_remap=not args.disable_video_remap,
			video_limited_range=args.video_limited_range,
		)

	view_tag = "_".join(f"{view_idx:03d}" for view_idx in view_indices)
	default_name = f"{scene_name}_view_{view_tag}" if len(view_indices) == 1 else f"{scene_name}_views_{view_tag}"
	out_dir = Path(args.output_dir).resolve() if args.output_dir else Path(OUTPUT_DIR) / default_name
	for view_idx in view_indices:
		init_path = out_dir / f"init_view_{view_idx:03d}.png"
		ref_path = out_dir / f"ref_view_{view_idx:03d}.png"
		pair_path = out_dir / f"pair_view_{view_idx:03d}.png"
		_save_image(init_path, init_imgs[view_idx], gamma=args.gamma)
		_save_image(ref_path, ref_imgs[view_idx], gamma=args.gamma)
		_save_image(pair_path, torch.cat([ref_imgs[view_idx], init_imgs[view_idx]], dim=1), gamma=args.gamma)

	if args.no_remesh:
		remesh_steps = []
	elif args.remesh_interval > 0:
		remesh_steps = list(range(args.remesh_interval, args.steps, args.remesh_interval))
	else:
		remesh_steps = [500, 1500, 3000, 4500, 7000, 10000, 12000, 14000]
	params = {
		"steps": args.steps,
		"step_size": args.step_size,
		"translation_step_size": args.translation_step_size if args.translation_step_size is not None else args.step_size,
		"loss": "l1",
		"boost": 3,
		"lambda": args.lambda_,
		"remesh": remesh_steps.copy(),
		"loss_view_idx": view_indices[0] if len(view_indices) == 1 else None,
		"loss_view_indices": view_indices if len(view_indices) > 1 else None,
		"save_render_interval": args.render_interval,
		"save_render_dir": str(out_dir),
		"save_render_view_idx": view_indices[0],
		"save_render_view_indices": view_indices,
		"save_render_gamma": args.gamma,
		"black_background": args.black_background,
		"use_tr": args.optimize_translation,
	}
	if video_frames is not None:
		params["video_frames"] = video_frames
		params["video_ratio"] = args.video_ratio
	if video_frames_per_view is not None:
		params["video_frames_per_view"] = video_frames_per_view
		params["video_ratio"] = args.video_ratio
	if args.remesh_method == "voxel" and remesh_steps:
		params["remesh_fn"] = _make_voxel_remesh_fn(args.voxel_resolution)

	t_start = time.perf_counter()
	out = optimize_shape(str(scene_xml), params)
	elapsed = time.perf_counter() - t_start

	max_step = len(out["vert_steps"]) - 1
	valid_remesh_steps = [step for step in remesh_steps if step <= max_step]
	all_steps = [0, *valid_remesh_steps, -1]
	num_faces = len(all_steps) - 2
	with (out_dir / "remesh_steps.csv").open("w", newline="") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(["", "0"])
		for index, step in enumerate(all_steps):
			writer.writerow([index, step])
	for i, step in enumerate(all_steps):
		verts = out["vert_steps"][step] + out["tr_steps"][step]
		faces = out["f"][min(i, num_faces)]
		write_ply(out_dir / f"res_{i:02d}.ply", verts, faces)

	verts = out["vert_steps"][-1] + out["tr_steps"][-1]
	faces = out["f"][-1]
	write_ply(out_dir / f"res_{i + 1:02d}.ply", verts, faces)

	method_tag = "video" if (args.video_path or args.video_paths) else "static"
	summary = {
		"method": method_tag,
		"scene_name": scene_name,
		"scene_dir": str(scene_dir),
		"output_dir": str(out_dir),
		"total_time_sec": elapsed,
		"steps": args.steps,
		"render_interval": args.render_interval,
		"remesh_interval": args.remesh_interval,
		"video_ratio": args.video_ratio if (args.video_path or args.video_paths) else None,
		"view_indices": view_indices,
		"final_num_faces": int(len(faces)),
	}
	summary_path = out_dir / "summary.json"
	summary_path.write_text(json.dumps(summary, indent=2))
	print(f"Saved: {summary_path}")

	if args.save_trajectory_video:
		_save_trajectory_videos(out_dir, view_indices, fps=args.trajectory_fps)

	for view_idx in view_indices:
		print(f"Saved: {out_dir / f'init_view_{view_idx:03d}.png'}")
		print(f"Saved: {out_dir / f'ref_view_{view_idx:03d}.png'}")
		print(f"Saved: {out_dir / f'pair_view_{view_idx:03d}.png'}")
	print(f"Saved optimization outputs to: {out_dir}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
