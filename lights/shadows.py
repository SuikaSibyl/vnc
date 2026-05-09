"""Render a simple Mitsuba shadow scene with the Stanford bunny.

The setup is intentionally arranged so the camera sees only a wall and the
shadow projected onto it:
1. A diffuse wall sits in front of the camera.
2. The bunny and point light are placed behind the camera along the view axis.
3. The point light shines toward the wall so the bunny casts a visible shadow.

Usage:
    /home/haolin/.conda/envs/mitsuba/bin/python codes/lights/shadows.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import math
import time

import drjit as dr
import imageio.v2 as imageio
import mitsuba as mi
import numpy as np
from PIL import Image, ImageFilter


SCRIPT_DIR = Path(__file__).resolve().parent
VNC_DIR = SCRIPT_DIR.parent
SCENE_DIR = VNC_DIR / "scenes" / "lights" / "shadows"
BUNNY_PLY = VNC_DIR / "scenes" / "lights" / "bunny" / "bunny.ply"
WALL_DIR = VNC_DIR / "scenes" / "lights" / "wall"
WALL_FBX = WALL_DIR / "source" / "wall_03 model.fbx.fbx"
WALL_OBJ = WALL_DIR / "source" / "wall_03_model.obj"
WALL_COLOR = WALL_DIR / "textures" / "wall_03_model_COLOR.png"
DEFAULT_LIGHT_Y = 1.95
DEFAULT_LIGHT_Z = 4.7

_BLENDER_CONVERT = """\
import bpy, sys
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


def configure_variant(preferred: str) -> str:
    variants = [preferred]
    if preferred != "llvm_ad_rgb":
        variants.append("llvm_ad_rgb")

    errors: list[str] = []
    for variant in variants:
        try:
            mi.set_variant(variant)
            return variant
        except Exception as exc:
            errors.append(f"{variant}: {exc}")

    raise RuntimeError("Could not initialize a Mitsuba variant:\n" + "\n".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a wall-shadow scene with the bunny hidden behind the camera."
    )
    parser.add_argument(
        "--mode",
        choices=["render", "optimize"],
        default="render",
        help="Render a single image or optimize the light x-position against a target render.",
    )
    parser.add_argument("--width", type=int, default=768, help="Output image width.")
    parser.add_argument("--height", type=int, default=768, help="Output image height.")
    parser.add_argument("--spp", type=int, default=512, help="Samples per pixel.")
    parser.add_argument(
        "--opt-width",
        type=int,
        default=256,
        help="Optimization render width. Lower than --width to reduce AD memory use.",
    )
    parser.add_argument(
        "--opt-height",
        type=int,
        default=256,
        help="Optimization render height. Lower than --height to reduce AD memory use.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="cuda_ad_rgb",
        help="Preferred Mitsuba variant. Falls back to llvm_ad_rgb.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCENE_DIR / "shadow_wall.png",
        help="Output PNG path.",
    )
    parser.add_argument("--light-x", type=float, default=0.0, help="Point light x-position for render mode.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=VNC_DIR / "outputs" / "lights" / "shadows_opt",
        help="Output directory for optimization artifacts.",
    )
    parser.add_argument(
        "--method",
        choices=["mse", "loir", "prdpt", "video"],
        default="mse",
        help="Optimization method to use in --mode optimize.",
    )
    parser.add_argument("--init-light-x", type=float, default=0.25, help="Initial light x-position for optimization.")
    parser.add_argument("--target-light-x", type=float, default=-0.25, help="Target light x-position for optimization.")
    parser.add_argument("--n-iter", type=int, default=120, help="Number of optimization iterations.")
    parser.add_argument("--lr", type=float, default=0.01, help="Adam learning rate for light x optimization.")
    parser.add_argument("--opt-spp", type=int, default=64, help="Samples per pixel during optimization.")
    parser.add_argument("--final-spp", type=int, default=256, help="Samples per pixel for final saved renders.")
    parser.add_argument("--fd-eps", type=float, default=0.03, help="Finite-difference step for non-AD methods.")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N iterations.")
    parser.add_argument(
        "--trajectory-every",
        type=int,
        default=None,
        help="Save trajectory video frames every N iterations. Defaults to --log-every.",
    )
    parser.add_argument("--video-path", type=Path, default=None, help="Optional external video reference for method=video.")
    parser.add_argument("--video-ratio", type=float, default=0.8, help="Fraction of iterations driven by video frames.")
    parser.add_argument("--prdpt-sigma", type=float, default=0.08, help="Initial PRDPT smoothing sigma for light_x.")
    parser.add_argument("--prdpt-sigma-min", type=float, default=0.01, help="Final PRDPT smoothing sigma floor.")
    parser.add_argument("--prdpt-nsamples", type=int, default=8, help="Antithetic sample count for PRDPT.")
    parser.add_argument("--trajectory-fps", type=int, default=12, help="Frames per second for saved optimization trajectory videos.")
    return parser.parse_args()


def ensure_wall_obj() -> Path:
    if WALL_OBJ.exists():
        return WALL_OBJ

    if not WALL_FBX.exists():
        raise FileNotFoundError(f"Wall FBX not found: {WALL_FBX}")

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as handle:
        handle.write(_BLENDER_CONVERT)
        script_path = handle.name

    try:
        result = subprocess.run(
            ["blender", "--background", "--python", script_path, "--", str(WALL_FBX), str(WALL_OBJ)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Blender FBX conversion failed:\n{result.stderr[-4000:]}")
    finally:
        os.unlink(script_path)

    if not WALL_OBJ.exists():
        raise FileNotFoundError(f"Expected converted OBJ was not created: {WALL_OBJ}")
    return WALL_OBJ


def obj_bounds(obj_path: Path) -> tuple[list[float], list[float]]:
    min_xyz = [float("inf"), float("inf"), float("inf")]
    max_xyz = [float("-inf"), float("-inf"), float("-inf")]

    with open(obj_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
            for i in range(3):
                min_xyz[i] = min(min_xyz[i], xyz[i])
                max_xyz[i] = max(max_xyz[i], xyz[i])

    if min_xyz[0] == float("inf"):
        raise RuntimeError(f"No vertices found in OBJ: {obj_path}")
    return min_xyz, max_xyz


def build_wall_transform() -> mi.ScalarTransform4f:
    wall_obj = ensure_wall_obj()
    min_xyz, max_xyz = obj_bounds(wall_obj)
    center = [(lo + hi) * 0.5 for lo, hi in zip(min_xyz, max_xyz)]
    extents = [hi - lo for lo, hi in zip(min_xyz, max_xyz)]
    max_extent = max(extents)
    scale = 3.1 / max_extent if max_extent > 1e-6 else 1.0

    return (
        mi.ScalarTransform4f()
        .translate(mi.ScalarPoint3f(0.0, 0.12, 0.0))
        .rotate(mi.ScalarVector3f(0.0, 1.0, 0.0), 90.0)
        .scale(mi.ScalarVector3f(scale, scale / 1.8, scale / 1.8))
        .translate(mi.ScalarPoint3f(-center[0], -center[1], -center[2]))
    )


def build_scene(width: int, height: int, light_x: float = 0.0):
    sensor_to_world = mi.ScalarTransform4f().look_at(
        origin=mi.ScalarPoint3f(0.0, 0.10, 2.2),
        target=mi.ScalarPoint3f(0.0, 0.10, 0.0),
        up=mi.ScalarVector3f(0.0, 1.0, 0.0),
    )

    bunny_to_world = (
        mi.ScalarTransform4f()
        .translate(mi.ScalarPoint3f(0.0, 1.10, 2.45))
        .rotate(mi.ScalarVector3f(0.0, 1.0, 0.0), 180.0)
        .scale(mi.ScalarVector3f(0.62, 0.62, 0.62))
    )

    wall_to_world = build_wall_transform()

    return mi.load_dict(
        {
            "type": "scene",
            "integrator": {"type": "path", "max_depth": 8},
            "sensor": {
                "type": "perspective",
                "fov": 24.0,
                "to_world": sensor_to_world,
                "sampler": {"type": "independent", "sample_count": 64},
                "film": {
                    "type": "hdrfilm",
                    "width": width,
                    "height": height,
                    "pixel_format": "rgb",
                    "component_format": "float32",
                },
            },
            "wall": {
                "type": "obj",
                "filename": str(WALL_OBJ),
                "to_world": wall_to_world,
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "bitmap",
                        "filename": str(WALL_COLOR),
                    },
                },
            },
            "bunny": {
                "type": "ply",
                "filename": str(BUNNY_PLY),
                "face_normals": True,
                "to_world": bunny_to_world,
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {"type": "rgb", "value": [0.15, 0.15, 0.15]},
                },
            },
            "light": {
                "type": "point",
                "position": [light_x, DEFAULT_LIGHT_Y, DEFAULT_LIGHT_Z],
                "intensity": {"type": "rgb", "value": [105.0, 105.0, 105.0]},
            },
        }
    )


def set_light_x(params: mi.SceneParameters, light_x) -> None:
    params["light.position"] = mi.Point3f(light_x, DEFAULT_LIGHT_Y, DEFAULT_LIGHT_Z)
    params.update()


def save_image(path: Path, image: mi.TensorXf) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mi.util.write_bitmap(str(path), image)


def save_png_np(path: Path, image_np: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(np.rint(np.clip(image_np, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    imageio.imwrite(path, img)


def srgb_to_linear_np(image_np: np.ndarray) -> np.ndarray:
    image_np = np.clip(image_np, 0.0, 1.0).astype(np.float32)
    return np.where(
        image_np <= 0.04045,
        image_np / 12.92,
        ((image_np + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def annotate_iteration(frame: np.ndarray, iteration: int) -> np.ndarray:
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


def build_trajectory_videos(debug_dir: Path, fps: int = 12) -> None:
    render_frames: list[tuple[np.ndarray, int]] = []
    ref_frames: list[tuple[np.ndarray, int]] = []
    pair_frames: list[tuple[np.ndarray, int]] = []
    for render_path in sorted(debug_dir.glob("iter_*_render.png")):
        try:
            iteration = int(render_path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        ref_path = debug_dir / f"iter_{iteration:04d}_ref.png"
        pair_path = debug_dir / f"iter_{iteration:04d}_pair.png"
        if render_path.exists():
            render_frames.append((annotate_iteration(imageio.imread(render_path), iteration), iteration))
        if ref_path.exists():
            ref_frames.append((annotate_iteration(imageio.imread(ref_path), iteration), iteration))
        if pair_path.exists():
            pair_frames.append((annotate_iteration(imageio.imread(pair_path), iteration), iteration))
    for name, frames in (
        ("trajectory_render.mp4", render_frames),
        ("trajectory_ref.mp4", ref_frames),
        ("trajectory_pair.mp4", pair_frames),
    ):
        if not frames:
            continue
        max_frames = max(1, int(round(fps * 3.0)))
        if len(frames) > max_frames:
            keep_idx = np.linspace(0, len(frames) - 1, num=max_frames, dtype=int)
            frames = [frames[idx] for idx in keep_idx]
        imageio.mimwrite(debug_dir.parent / name, [frame for frame, _ in frames], fps=fps)
        print(f"Saved {debug_dir.parent / name}")


def scalar_to_float(value) -> float:
    arr = np.asarray(value.numpy())
    return float(arr.reshape(-1)[0])


def image_to_numpy(image: mi.TensorXf) -> np.ndarray:
    return np.array(image, copy=True)


def render_with_light_x(scene: mi.Scene, params: mi.SceneParameters, light_x: float, spp: int, seed: int) -> mi.TensorXf:
    set_light_x(params, mi.Float(light_x))
    return mi.render(scene, params=params, spp=spp, seed=seed)


def mse_loss_np(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def grayscale_np(image: np.ndarray) -> np.ndarray:
    return (
        0.2126 * image[..., 0]
        + 0.7152 * image[..., 1]
        + 0.0722 * image[..., 2]
    ).astype(np.float32)


def loir_loss_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred_gray = grayscale_np(pred)
    target_gray = grayscale_np(target)

    loss = 0.0
    bins = np.linspace(0.0, 1.0, 17)
    for sigma in (0.0, 2.0, 5.0):
        if sigma > 0.0:
            pred_blur = np.asarray(Image.fromarray(np.uint8(np.clip(pred_gray * 255.0, 0, 255))).filter(ImageFilter.GaussianBlur(radius=sigma)), dtype=np.float32) / 255.0
            target_blur = np.asarray(Image.fromarray(np.uint8(np.clip(target_gray * 255.0, 0, 255))).filter(ImageFilter.GaussianBlur(radius=sigma)), dtype=np.float32) / 255.0
        else:
            pred_blur = pred_gray
            target_blur = target_gray

        pred_hist, _ = np.histogram(pred_blur, bins=bins, density=True)
        target_hist, _ = np.histogram(target_blur, bins=bins, density=True)
        pred_cdf = np.cumsum(pred_hist)
        target_cdf = np.cumsum(target_hist)
        loss += float(np.mean(np.abs(pred_cdf - target_cdf)))

    return loss / 3.0


class VideoReferenceScheduler:
    def __init__(
        self,
        scene: mi.Scene,
        params: mi.SceneParameters,
        init_light_x: float,
        target_light_x: float,
        n_iter: int,
        spp: int,
        video_ratio: float,
        video_path: Path | None = None,
    ) -> None:
        self.scene = scene
        self.params = params
        self.init_light_x = init_light_x
        self.target_light_x = target_light_x
        self.n_iter = n_iter
        self.spp = spp
        self.n_video_iters = int(max(1, n_iter * video_ratio))
        self.external_frames: list[np.ndarray] | None = None
        if video_path is not None:
            self.external_frames = self._load_video_frames(video_path)

    def _load_video_frames(self, video_path: Path) -> list[np.ndarray]:
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        ffmpeg_cmd = [
            "ffmpeg", "-v", "error", "-i", str(video_path),
            "-vf", f"scale={self.scene.sensors()[0].film().crop_size()[0]}:{self.scene.sensors()[0].film().crop_size()[1]}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        proc = subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        width = int(self.scene.sensors()[0].film().crop_size()[0])
        height = int(self.scene.sensors()[0].film().crop_size()[1])
        frame_size = width * height * 3
        raw = proc.stdout
        frames = []
        for i in range(len(raw) // frame_size):
            start = i * frame_size
            end = start + frame_size
            frame = np.frombuffer(raw[start:end], dtype=np.uint8).reshape(height, width, 3).astype(np.float32) / 255.0
            frame = srgb_to_linear_np(frame)
            frames.append(frame)
        if not frames:
            raise RuntimeError(f"No frames decoded from video: {video_path}")
        return frames

    def get_ref(self, iteration: int) -> np.ndarray:
        if self.external_frames is not None and iteration < self.n_video_iters:
            idx = min(int(round(iteration / max(self.n_video_iters - 1, 1) * (len(self.external_frames) - 1))), len(self.external_frames) - 1)
            return self.external_frames[idx]

        if iteration >= self.n_video_iters:
            light_x = self.target_light_x
        else:
            frac = iteration / max(self.n_video_iters - 1, 1)
            light_x = (1.0 - frac) * self.init_light_x + frac * self.target_light_x
        image = render_with_light_x(self.scene, self.params, light_x, self.spp, seed=1000 + iteration)
        return image_to_numpy(image)


def adam_update(x: float, grad: float, state: dict, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> float:
    state["t"] += 1
    state["m"] = beta1 * state["m"] + (1.0 - beta1) * grad
    state["v"] = beta2 * state["v"] + (1.0 - beta2) * (grad * grad)
    m_hat = state["m"] / (1.0 - beta1 ** state["t"])
    v_hat = state["v"] / (1.0 - beta2 ** state["t"])
    return x - lr * m_hat / (math.sqrt(v_hat) + eps)


def optimize_light_x(args: argparse.Namespace) -> None:
    method_dirname = args.method
    if args.method == "prdpt":
        sigma_str = f"{args.prdpt_sigma:g}".replace(".", "p")
        method_dirname = f"{args.method}_sigma{sigma_str}"
    out_dir = (args.out_dir.resolve() / method_dirname)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    capture_every = args.trajectory_every if args.trajectory_every is not None else args.log_every

    train_scene = build_scene(args.opt_width, args.opt_height, light_x=args.init_light_x)
    train_params = mi.traverse(train_scene)
    light_key = "light.position"
    target_train_scene = build_scene(args.opt_width, args.opt_height, light_x=args.target_light_x)
    target_train_params = mi.traverse(target_train_scene)

    vis_scene = build_scene(args.width, args.height, light_x=args.init_light_x)
    vis_params = mi.traverse(vis_scene)
    target_vis_scene = build_scene(args.width, args.height, light_x=args.target_light_x)
    target_vis_params = mi.traverse(target_vis_scene)

    init_image = mi.render(vis_scene, params=vis_params, spp=args.final_spp, seed=0)
    save_image(out_dir / "init.png", init_image)
    save_png_np(out_dir / "init_trajectory_tone.png", image_to_numpy(init_image))

    target_train_image = mi.render(target_train_scene, params=target_train_params, spp=args.final_spp, seed=1)
    target_vis_image = mi.render(target_vis_scene, params=target_vis_params, spp=args.final_spp, seed=2)
    save_image(out_dir / "target.png", target_vis_image)
    save_png_np(out_dir / "target_trajectory_tone.png", image_to_numpy(target_vis_image))
    target_train_np = image_to_numpy(target_train_image)

    losses: list[float] = []
    history: list[dict] = []

    if args.method == "mse":
        opt = mi.ad.Adam(lr=args.lr)
        opt["light_x"] = mi.Float(args.init_light_x)

        for step in range(args.n_iter):
            set_light_x(train_params, opt["light_x"])
            image = mi.render(train_scene, params=train_params, spp=args.opt_spp, seed=step + 7)
            loss = dr.mean(dr.square(image - target_train_image))
            dr.backward(loss)
            loss_value = float(loss.numpy())
            losses.append(loss_value)
            opt.step()
            opt["light_x"] = dr.clip(opt["light_x"], -1.5, 1.5)
            current_x = scalar_to_float(opt["light_x"])
            history.append({"iter": step, "loss": loss_value, "light_x": current_x})

            if step % capture_every == 0 or step == args.n_iter - 1:
                print(f"iter={step:03d} loss={loss_value:.6f} light_x={current_x:.4f}")
                debug_image = mi.render(vis_scene, params=vis_params, spp=args.final_spp, seed=6000 + step)
                debug_np = image_to_numpy(debug_image)
                ref_np = image_to_numpy(target_vis_image)
                save_png_np(debug_dir / f"iter_{step+1:04d}_render.png", debug_np)
                save_png_np(debug_dir / f"iter_{step+1:04d}_ref.png", ref_np)
                save_png_np(debug_dir / f"iter_{step+1:04d}_pair.png", np.concatenate([ref_np, debug_np], axis=1))

        final_light_x = scalar_to_float(opt["light_x"])
    else:
        light_x = float(args.init_light_x)
        adam_state = {"m": 0.0, "v": 0.0, "t": 0}
        video_sched = None
        vis_video_sched = None
        if args.method == "video":
            video_sched = VideoReferenceScheduler(
                train_scene,
                train_params,
                args.init_light_x,
                args.target_light_x,
                args.n_iter,
                args.opt_spp,
                args.video_ratio,
                args.video_path,
            )
            vis_video_sched = VideoReferenceScheduler(
                vis_scene,
                vis_params,
                args.init_light_x,
                args.target_light_x,
                args.n_iter,
                args.final_spp,
                args.video_ratio,
                args.video_path,
            )

        def objective(x: float, iteration: int, seed: int) -> float:
            pred_np = image_to_numpy(render_with_light_x(train_scene, train_params, x, args.opt_spp, seed=seed))
            if args.method == "mse":
                return mse_loss_np(pred_np, target_train_np)
            if args.method == "loir":
                return loir_loss_np(pred_np, target_train_np)
            if args.method == "video":
                assert video_sched is not None
                ref_np = video_sched.get_ref(iteration)
                return mse_loss_np(pred_np, ref_np)
            if args.method == "prdpt":
                return mse_loss_np(pred_np, target_train_np)
            raise ValueError(args.method)

        for step in range(args.n_iter):
            if args.method == "prdpt":
                frac = step / max(args.n_iter - 1, 1)
                sigma = max(args.prdpt_sigma_min, args.prdpt_sigma * (1.0 - frac) + args.prdpt_sigma_min * frac)
                grad_acc = 0.0
                loss_acc = 0.0
                half = max(1, args.prdpt_nsamples // 2)
                for k in range(half):
                    eps = sigma * np.random.randn()
                    lp = objective(light_x + eps, step, seed=2000 + 2 * k)
                    lm = objective(light_x - eps, step, seed=2000 + 2 * k + 1)
                    grad_acc += ((lp - lm) / (2.0 * sigma)) * (eps / sigma)
                    loss_acc += 0.5 * (lp + lm)
                grad = grad_acc / half
                loss_value = loss_acc / half
            else:
                loss_value = objective(light_x, step, seed=3000 + step)
                lp = objective(light_x + args.fd_eps, step, seed=4000 + 2 * step)
                lm = objective(light_x - args.fd_eps, step, seed=4000 + 2 * step + 1)
                grad = (lp - lm) / (2.0 * args.fd_eps)

            light_x = adam_update(light_x, grad, adam_state, args.lr)
            light_x = float(np.clip(light_x, -1.5, 1.5))
            losses.append(loss_value)
            history.append({"iter": step, "loss": loss_value, "light_x": light_x})

            if step % capture_every == 0 or step == args.n_iter - 1:
                print(f"iter={step:03d} loss={loss_value:.6f} light_x={light_x:.4f}")
                set_light_x(vis_params, mi.Float(light_x))
                debug_image = mi.render(vis_scene, params=vis_params, spp=args.final_spp, seed=7000 + step)
                debug_np = image_to_numpy(debug_image)
                if args.method == "video" and vis_video_sched is not None:
                    ref_np = vis_video_sched.get_ref(step)
                else:
                    ref_np = image_to_numpy(target_vis_image)
                save_png_np(debug_dir / f"iter_{step+1:04d}_render.png", debug_np)
                save_png_np(debug_dir / f"iter_{step+1:04d}_ref.png", ref_np)
                save_png_np(debug_dir / f"iter_{step+1:04d}_pair.png", np.concatenate([ref_np, debug_np], axis=1))

        final_light_x = light_x

    set_light_x(vis_params, mi.Float(final_light_x))
    final_image = mi.render(vis_scene, params=vis_params, spp=args.final_spp, seed=args.n_iter + 99)
    save_image(out_dir / "final.png", final_image)
    save_png_np(out_dir / "final_trajectory_tone.png", image_to_numpy(final_image))
    elapsed = time.perf_counter() - t0
    build_trajectory_videos(debug_dir, fps=args.trajectory_fps)

    with open(out_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": args.method,
                "init_light_x": args.init_light_x,
                "target_light_x": args.target_light_x,
                "final_light_x": final_light_x,
                "n_iter": args.n_iter,
                "lr": args.lr,
                "losses": losses,
                "total_time_sec": elapsed,
            },
            handle,
            indent=2,
        )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": args.method,
                "init_light_x": args.init_light_x,
                "target_light_x": args.target_light_x,
                "final_light_x": final_light_x,
                "final_loss": losses[-1] if losses else None,
                "n_iter": args.n_iter,
                "lr": args.lr,
                "total_time_sec": elapsed,
            },
            handle,
            indent=2,
        )

    print(f"Saved optimization outputs to: {out_dir}")
    print(f"Final optimized light x: {final_light_x:.4f}")


def main() -> None:
    args = parse_args()
    active_variant = configure_variant(args.variant)

    if not BUNNY_PLY.exists():
        raise FileNotFoundError(f"Bunny mesh not found: {BUNNY_PLY}")
    if not WALL_FBX.exists():
        raise FileNotFoundError(f"Wall mesh not found: {WALL_FBX}")

    ensure_wall_obj()

    print(f"Using Mitsuba variant: {active_variant}")
    if args.mode == "optimize":
        optimize_light_x(args)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene = build_scene(args.width, args.height, light_x=args.light_x)
    image = mi.render(scene, spp=args.spp)
    mi.util.write_bitmap(str(args.output), image)
    print(f"Wrote render to: {args.output}")


if __name__ == "__main__":
    main()
