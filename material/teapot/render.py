from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import drjit as dr
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent.parent
    default_scene = root_dir / "scenes" / "material" / "teapot" / "scene.xml"
    default_out_dir = root_dir / "outputs" / "material" / "teapot"

    parser = argparse.ArgumentParser(
        description="Optimize the alpha texture of the teapot scene to match a reference render."
    )
    parser.add_argument("--scene", type=Path, default=default_scene, help="Path to the Mitsuba scene XML.")
    parser.add_argument("--out-dir", type=Path, default=default_out_dir, help="Directory for all generated outputs.")
    parser.add_argument("--width", type=int, default=256, help="Render width for optimization and outputs.")
    parser.add_argument("--height", type=int, default=256, help="Render height for optimization and outputs.")
    parser.add_argument("--vis-width", type=int, default=512, help="Render width for saved visualization outputs.")
    parser.add_argument("--vis-height", type=int, default=512, help="Render height for saved visualization outputs.")
    parser.add_argument("--tex-res", type=int, default=64, help="Square alpha texture resolution.")
    parser.add_argument("--init-alpha", type=float, default=0.3, help="Initial alpha value for the optimized texture.")
    parser.add_argument("--ref-alpha", type=float, default=0.8, help="Reference alpha value for the optimized texture.")
    parser.add_argument("--n-iter", type=int, default=100, help="Number of optimization iterations.")
    parser.add_argument("--lr", type=float, default=0.02, help="Adam learning rate.")
    parser.add_argument("--opt-spp", type=int, default=64, help="Samples per pixel used during optimization.")
    parser.add_argument("--ref-spp", type=int, default=512, help="Samples per pixel for reference and init renders.")
    parser.add_argument("--final-spp", type=int, default=512, help="Samples per pixel for the final render.")
    parser.add_argument(
        "--debug-spp",
        type=int,
        default=512,
        help="Samples per pixel used when re-rendering debug image pairs.",
    )
    parser.add_argument(
        "--debug-every",
        type=int,
        default=10,
        help="Save the current render/reference pair every N iterations into a debug folder. Set <=0 to disable.",
    )
    parser.add_argument(
        "--render-interval",
        type=int,
        default=None,
        help="Alias for --debug-every. Set to 1 to save every iteration for trajectory videos.",
    )
    parser.add_argument(
        "--reference-mode",
        type=str,
        choices=["static", "video"],
        default="static",
        help="Use a static rendered target or a video-guided schedule that falls back to the static target.",
    )
    parser.add_argument("--video-path", type=Path, default=None, help="Optional video used as the training reference.")
    parser.add_argument(
        "--video-ratio",
        type=float,
        default=0.9,
        help="Fraction of optimization iterations that should use video frames before falling back to the static target.",
    )
    parser.add_argument(
        "--video-dedup-threshold",
        type=float,
        default=0.01,
        help="Mean absolute difference threshold for dropping near-duplicate video frames.",
    )
    parser.add_argument(
        "--video-exposure",
        type=float,
        default=0.0,
        help="Exposure used when reconstructing linear HDR values from tone-mapped video frames.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="cuda_ad_rgb",
        help="Preferred Mitsuba variant. Falls back to llvm_ad_rgb if unavailable.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Optional output path for the summary figure. Defaults to <script dir>/teapot_summary.png.",
    )
    parser.add_argument(
        "--save-trajectory-video",
        action="store_true",
        help="If set, build short MP4 trajectory videos from saved debug renders.",
    )
    parser.add_argument(
        "--trajectory-fps",
        type=int,
        default=12,
        help="Frames per second for saved optimization trajectory videos.",
    )
    return parser.parse_args()


def configure_variant(preferred_variant: str) -> str:
    variants = [preferred_variant]
    if preferred_variant != "llvm_ad_rgb":
        variants.append("llvm_ad_rgb")

    errors: list[str] = []
    for variant in variants:
        try:
            mi.set_variant(variant)
            return variant
        except Exception as exc:
            errors.append(f"{variant}: {exc}")

    joined = "\n".join(errors)
    raise RuntimeError(f"Could not initialize any Mitsuba variant.\n{joined}")


def prepend_scene_dir(scene_dir: Path) -> None:
    resolver = mi.Thread.thread().file_resolver()
    resolver.prepend(str(scene_dir))


def build_scene_xml(scene_path: Path, width: int, height: int) -> tuple[str, Path]:
    scene_dir = scene_path.parent
    alpha_tex_path = scene_dir / "alpha_init.exr"

    with open(scene_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    xml_mod = xml_content
    xml_mod = re.sub(r'(<default name="resx" value=")[^"]*(")', rf"\g<1>{width}\2", xml_mod)
    xml_mod = re.sub(r'(<default name="resy" value=")[^"]*(")', rf"\g<1>{height}\2", xml_mod)

    mesh001_replacement = (
        '<shape type="obj" id="Material_0001">\n'
        '    <string name="filename" value="models/Mesh001.obj" />\n'
        '    <transform name="to_world">\n'
        '        <matrix value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1" />\n'
        '    </transform>\n'
        '    <bsdf type="twosided">\n'
        '        <bsdf type="roughconductor">\n'
        '            <string name="distribution" value="ggx" />\n'
        '            <texture name="alpha" type="bitmap">\n'
        '                <string name="filename" value="alpha_init.exr" />\n'
        '            </texture>\n'
        '        </bsdf>\n'
        '    </bsdf>\n'
        '</shape>'
    )

    xml_mod, n_sub = re.subn(
        r'<shape type="obj" id="Material_0001">.*?</shape>',
        mesh001_replacement,
        xml_mod,
        flags=re.DOTALL,
    )
    if n_sub != 1:
        raise RuntimeError("Failed to patch Material_0001 in XML.")

    return xml_mod, alpha_tex_path


def find_alpha_key(params: mi.SceneParameters) -> str:
    alpha_keys = [k for k in params.keys() if "alpha" in k.lower()]

    alpha_key = next(
        (k for k in alpha_keys if hasattr(params[k], "shape") and "Material_0001" in k),
        None,
    )
    if alpha_key is None:
        alpha_key = next((k for k in alpha_keys if hasattr(params[k], "shape")), None)
    if alpha_key is None:
        raise RuntimeError("Could not find alpha texture tensor key.")

    return alpha_key


def load_scene_with_params(scene_path: Path, width: int, height: int) -> tuple[mi.Scene, mi.SceneParameters, str]:
    xml_mod, _ = build_scene_xml(scene_path, width, height)
    scene = mi.load_string(xml_mod)
    params = mi.traverse(scene)
    alpha_key = find_alpha_key(params)
    return scene, params, alpha_key


def set_alpha_texture(params: mi.SceneParameters, alpha_key: str, alpha_tex: mi.TensorXf) -> None:
    params[alpha_key] = alpha_tex
    params.update()


def inverse_tone_map_aces(mapped: np.ndarray, exposure: float = 0.0) -> np.ndarray:
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14

    y = np.clip(mapped.astype(np.float32), 0.0, 1.0)
    qa = y * c - a
    qb = y * d - b
    qc = y * e

    linear = np.empty_like(y, dtype=np.float32)
    linear_mask = np.abs(qa) < 1e-8
    linear[linear_mask] = np.divide(
        -qc[linear_mask],
        qb[linear_mask],
        out=np.zeros_like(qc[linear_mask]),
        where=np.abs(qb[linear_mask]) > 1e-8,
    )

    quad_mask = ~linear_mask
    disc = np.maximum(qb[quad_mask] * qb[quad_mask] - 4.0 * qa[quad_mask] * qc[quad_mask], 0.0)
    sqrt_disc = np.sqrt(disc)
    denom = 2.0 * qa[quad_mask]
    root1 = (-qb[quad_mask] + sqrt_disc) / denom
    root2 = (-qb[quad_mask] - sqrt_disc) / denom
    roots = np.stack([root1, root2], axis=0)
    roots = np.where(roots >= 0.0, roots, -np.inf)
    linear[quad_mask] = np.max(roots, axis=0)
    linear = np.where(np.isfinite(linear), linear, 0.0)
    linear = np.clip(linear, 0.0, None)
    return linear / (2.0 ** exposure)


def ldr_to_hdr(frame_ldr: np.ndarray, exposure: float = 0.0) -> np.ndarray:
    mapped = np.clip(frame_ldr.astype(np.float32), 0.0, 1.0) ** 2.2
    return inverse_tone_map_aces(mapped, exposure=exposure)


def tone_map_aces(image: mi.TensorXf, exposure: float = 0.0) -> np.ndarray:
    rgb = np.array(image, copy=False)
    rgb = np.clip(rgb * (2.0 ** exposure), 0.0, None)

    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    mapped = (rgb * (a * rgb + b)) / (rgb * (c * rgb + d) + e)
    mapped = np.clip(mapped, 0.0, 1.0)
    return np.uint8(np.round(mapped ** (1.0 / 2.2) * 255.0))


def save_hdr_and_ldr(image: mi.TensorXf, hdr_path: Path, ldr_path: Path, exposure: float = 0.0) -> None:
    mi.util.write_bitmap(str(hdr_path), image)
    mi.util.write_bitmap(str(ldr_path), tone_map_aces(image, exposure=exposure))


def save_debug_pair(
    render_img: mi.TensorXf,
    ref_img: mi.TensorXf,
    debug_dir: Path,
    iteration: int,
    exposure: float = 0.0,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = f"iter_{iteration + 1:04d}"
    mi.util.write_bitmap(str(debug_dir / f"{stem}_render.png"), tone_map_aces(render_img, exposure=exposure))
    mi.util.write_bitmap(str(debug_dir / f"{stem}_ref.png"), tone_map_aces(ref_img, exposure=exposure))
    mi.util.write_bitmap(str(debug_dir / f"{stem}_render.exr"), render_img)
    mi.util.write_bitmap(str(debug_dir / f"{stem}_ref.exr"), ref_img)


class StaticReferenceProvider:
    def __init__(self, ref_image: mi.TensorXf) -> None:
        self.ref_image = ref_image

    def get_ref(self, iteration: int) -> mi.TensorXf:
        del iteration
        return self.ref_image


class VideoReferenceScheduler:
    def __init__(
        self,
        video_path: Path,
        n_iter: int,
        gt_img: mi.TensorXf,
        resolution: tuple[int, int],
        video_ratio: float,
        dedup_threshold: float,
        exposure: float,
    ) -> None:
        self.gt_img = gt_img
        self.n_iter = n_iter
        self.n_video_iters = int(n_iter * video_ratio)
        self.frames_ldr, self.frames_hdr = self._load_video(
            video_path=video_path,
            resolution=resolution,
            dedup_threshold=dedup_threshold,
            exposure=exposure,
        )
        self.n_frames = len(self.frames_hdr)

        print(
            f"[VideoReferenceScheduler] loaded {self.n_frames} frames from {video_path}\n"
            f"  video iters: 0 - {max(self.n_video_iters - 1, -1)} | "
            f"static target iters: {self.n_video_iters} - {n_iter - 1}"
        )

    def _load_video(
        self,
        video_path: Path,
        resolution: tuple[int, int],
        dedup_threshold: float,
        exposure: float,
    ) -> tuple[list[np.ndarray], list[mi.TensorXf]]:
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        width, height = resolution
        frame_size = width * height * 3
        ffmpeg_cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"scale={width}:{height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        try:
            proc = subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required to decode video references.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(f"Could not decode video {video_path} with ffmpeg:\n{stderr}") from exc

        all_frames_ldr: list[np.ndarray] = []
        raw = proc.stdout
        if len(raw) % frame_size != 0:
            raise RuntimeError(
                f"Decoded video byte count {len(raw)} is not divisible by expected frame size {frame_size}."
            )
        n_frames = len(raw) // frame_size
        for idx in range(n_frames):
            start = idx * frame_size
            end = start + frame_size
            frame = np.frombuffer(raw[start:end], dtype=np.uint8).reshape(height, width, 3)
            all_frames_ldr.append(frame.astype(np.float32) / 255.0)

        if not all_frames_ldr:
            raise RuntimeError(f"No frames decoded from: {video_path}")

        kept_ldr = [all_frames_ldr[0]]
        for frame in all_frames_ldr[1:]:
            if np.mean(np.abs(frame - kept_ldr[-1])) > dedup_threshold:
                kept_ldr.append(frame)
        if not np.array_equal(all_frames_ldr[-1], kept_ldr[-1]):
            kept_ldr.append(all_frames_ldr[-1])

        print(
            f"[VideoReferenceScheduler] {len(all_frames_ldr)} raw frames -> {len(kept_ldr)} kept "
            f"(dedup_threshold={dedup_threshold})"
        )

        kept_hdr = [mi.TensorXf(ldr_to_hdr(frame, exposure=exposure)) for frame in kept_ldr]
        return kept_ldr, kept_hdr

    def _iter_to_frame(self, iteration: int) -> mi.TensorXf:
        t = iteration / max(self.n_video_iters - 1, 1)
        frame_idx = int(round(t * (self.n_frames - 1)))
        frame_idx = max(0, min(frame_idx, self.n_frames - 1))
        return self.frames_hdr[frame_idx]

    def get_ref(self, iteration: int) -> mi.TensorXf:
        if iteration < self.n_video_iters:
            return self._iter_to_frame(iteration)
        return self.gt_img


def render_bitmap(image: mi.TensorXf) -> np.ndarray:
    return mi.util.convert_to_bitmap(image, uint8_srgb=True)


def save_summary_plot(
    ref_img: mi.TensorXf,
    opt_img: mi.TensorXf,
    alpha_tex: np.ndarray,
    losses: list[float],
    ref_alpha: float,
    plot_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(render_bitmap(ref_img))
    axes[0].set_title(f"Reference (alpha = {ref_alpha})")
    axes[0].axis("off")

    axes[1].imshow(render_bitmap(opt_img))
    axes[1].set_title("Optimized (texture)")
    axes[1].axis("off")

    im = axes[2].imshow(alpha_tex, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title(f"Optimized alpha texture\n(mean = {alpha_tex.mean():.3f})")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    loss_path = plot_path.with_name(f"{plot_path.stem}_loss{plot_path.suffix}")
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(losses)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("L2 Loss")
    ax2.set_title("Optimization Loss")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(loss_path, dpi=180, bbox_inches="tight")
    plt.close(fig2)


def annotate_iteration(frame: np.ndarray, iteration: int) -> np.ndarray:
    import cv2

    img = frame.copy()
    label = f"iter {iteration + 1}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
    x, y = 12, 28
    cv2.rectangle(img, (x - 6, y - text_h - 6), (x + text_w + 6, y + baseline + 6), (0, 0, 0), thickness=-1)
    cv2.putText(img, label, (x, y), font, scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    return img


def save_trajectory_video(path: Path, frames: list[tuple[np.ndarray, int]], fps: int) -> None:
    if not frames:
        return
    max_frames = max(1, int(round(fps * 3.0)))
    if len(frames) > max_frames:
        keep_idx = np.linspace(0, len(frames) - 1, num=max_frames, dtype=int)
        frames = [frames[idx] for idx in keep_idx]
    annotated_frames = [annotate_iteration(frame, iteration) for frame, iteration in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, annotated_frames, fps=fps)


def build_trajectory_videos(debug_dir: Path, out_dir: Path, fps: int) -> None:
    render_frames: list[tuple[np.ndarray, int]] = []
    ref_frames: list[tuple[np.ndarray, int]] = []
    pair_frames: list[tuple[np.ndarray, int]] = []

    for render_path in sorted(debug_dir.glob("iter_*_render.png")):
        try:
            iteration = int(render_path.stem.split("_")[1]) - 1
        except (IndexError, ValueError):
            continue
        ref_path = debug_dir / f"iter_{iteration + 1:04d}_ref.png"
        if not ref_path.exists():
            continue
        render = imageio.imread(render_path)
        ref = imageio.imread(ref_path)
        render_frames.append((render, iteration))
        ref_frames.append((ref, iteration))
        pair_frames.append((np.concatenate([ref, render], axis=1), iteration))

    save_trajectory_video(out_dir / "trajectory_render.mp4", render_frames, fps=fps)
    save_trajectory_video(out_dir / "trajectory_ref.mp4", ref_frames, fps=fps)
    save_trajectory_video(out_dir / "trajectory_pair.mp4", pair_frames, fps=fps)


def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()

    if args.render_interval is not None:
        args.debug_every = args.render_interval

    active_variant = configure_variant(args.variant)

    scene_path = args.scene.resolve()
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene XML does not exist: {scene_path}")

    scene_dir = scene_path.parent
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prepend_scene_dir(scene_dir)

    alpha_init_np = np.full((args.tex_res, args.tex_res, 1), args.init_alpha, dtype=np.float32)
    _, alpha_tex_path = build_scene_xml(scene_path, args.width, args.height)
    mi.Bitmap(alpha_init_np).write(str(alpha_tex_path))

    scene, params, alpha_key = load_scene_with_params(scene_path, args.width, args.height)
    vis_scene, vis_params, vis_alpha_key = load_scene_with_params(scene_path, args.vis_width, args.vis_height)

    print(f"Using Mitsuba variant: {active_variant}")
    print(f"Using alpha key: {alpha_key} shape={params[alpha_key].shape}")

    ref_tex_train = dr.full(mi.TensorXf, args.ref_alpha, shape=params[alpha_key].shape)
    ref_tex_vis = dr.full(mi.TensorXf, args.ref_alpha, shape=vis_params[vis_alpha_key].shape)
    set_alpha_texture(params, alpha_key, ref_tex_train)
    set_alpha_texture(vis_params, vis_alpha_key, ref_tex_vis)

    ref_img = mi.render(scene, params=params, spp=args.ref_spp)
    vis_ref_img = mi.render(vis_scene, params=vis_params, spp=args.ref_spp)
    ref_exr_path = out_dir / "ref.exr"
    ref_png_path = out_dir / "ref_ldr.png"
    save_hdr_and_ldr(vis_ref_img, ref_exr_path, ref_png_path)
    print(f"Saved reference render to {ref_exr_path}")
    print(f"Saved reference LDR render to {ref_png_path}")

    reference_provider: StaticReferenceProvider | VideoReferenceScheduler = StaticReferenceProvider(ref_img)
    vis_reference_provider: StaticReferenceProvider | VideoReferenceScheduler = StaticReferenceProvider(vis_ref_img)
    if args.reference_mode == "video":
        if args.video_path is None:
            raise ValueError("--video-path is required when --reference-mode=video")
        reference_provider = VideoReferenceScheduler(
            video_path=args.video_path.resolve(),
            n_iter=args.n_iter,
            gt_img=ref_img,
            resolution=(args.width, args.height),
            video_ratio=args.video_ratio,
            dedup_threshold=args.video_dedup_threshold,
            exposure=args.video_exposure,
        )
        vis_reference_provider = VideoReferenceScheduler(
            video_path=args.video_path.resolve(),
            n_iter=args.n_iter,
            gt_img=vis_ref_img,
            resolution=(args.vis_width, args.vis_height),
            video_ratio=args.video_ratio,
            dedup_threshold=args.video_dedup_threshold,
            exposure=args.video_exposure,
        )

    init_tex_train = dr.full(mi.TensorXf, args.init_alpha, shape=params[alpha_key].shape)
    init_tex_vis = dr.full(mi.TensorXf, args.init_alpha, shape=vis_params[vis_alpha_key].shape)
    set_alpha_texture(params, alpha_key, init_tex_train)
    set_alpha_texture(vis_params, vis_alpha_key, init_tex_vis)

    init_img = mi.render(scene, params=params, spp=args.ref_spp)
    vis_init_img = mi.render(vis_scene, params=vis_params, spp=args.ref_spp)
    init_exr_path = out_dir / "init.exr"
    init_png_path = out_dir / "init_ldr.png"
    save_hdr_and_ldr(vis_init_img, init_exr_path, init_png_path)
    print(f"Saved initialization render to {init_exr_path}")
    print(f"Saved initialization LDR render to {init_png_path}")

    debug_dir = out_dir / "debug"
    opt = mi.ad.Adam(lr=args.lr)
    opt[alpha_key] = params[alpha_key]
    losses: list[float] = []

    for i in range(args.n_iter):
        params[alpha_key] = opt[alpha_key]
        params.update()

        img = mi.render(scene, params=params, spp=args.opt_spp, seed=i)
        ref_current = reference_provider.get_ref(i)
        loss = dr.mean(dr.square(img - ref_current))

        dr.backward(loss)
        losses.append(float(loss.numpy()))

        if args.debug_every > 0 and ((i + 1) % args.debug_every == 0):
            vis_alpha_tex = dr.full(mi.TensorXf, float(dr.mean(opt[alpha_key]).numpy()), shape=vis_params[vis_alpha_key].shape)
            if hasattr(opt[alpha_key], "numpy"):
                vis_alpha_np = np.array(opt[alpha_key].numpy(), copy=True).astype(np.float32)
                vis_alpha_tex = mi.TensorXf(vis_alpha_np)
            set_alpha_texture(vis_params, vis_alpha_key, vis_alpha_tex)
            debug_img = mi.render(vis_scene, params=vis_params, spp=args.debug_spp, seed=args.n_iter + i)
            debug_ref = vis_reference_provider.get_ref(i)
            save_debug_pair(debug_img, debug_ref, debug_dir, i, exposure=args.video_exposure)

        opt.step()
        opt[alpha_key] = dr.clamp(opt[alpha_key], 0.01, 1.0)

        if i % 10 == 0:
            mean_alpha = float(dr.mean(opt[alpha_key]).numpy())
            print(f"Iter {i:3d} | loss = {losses[-1]:.6f} | mean alpha = {mean_alpha:.4f}")

    set_alpha_texture(params, alpha_key, opt[alpha_key])
    final_alpha_np = np.array(opt[alpha_key].numpy(), copy=True).astype(np.float32)
    set_alpha_texture(vis_params, vis_alpha_key, mi.TensorXf(final_alpha_np))

    final_img = mi.render(scene, params=params, spp=args.final_spp)
    vis_final_img = mi.render(vis_scene, params=vis_params, spp=args.final_spp)
    final_exr_path = out_dir / "final.exr"
    mi.util.write_bitmap(str(final_exr_path), vis_final_img)
    print(f"Saved final render to {final_exr_path}")

    alpha_np = opt[alpha_key].numpy().squeeze()
    print(f"Optimization complete. Final alpha mean: {alpha_np.mean():.6f}")

    teapot_reference_path = out_dir / "teapot_reference.exr"
    teapot_optimized_path = out_dir / "teapot_optimized.exr"
    mi.util.write_bitmap(str(teapot_reference_path), vis_ref_img)
    mi.util.write_bitmap(str(teapot_optimized_path), vis_final_img)

    plot_path = args.plot_path.resolve() if args.plot_path is not None else out_dir / "teapot_summary.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    save_summary_plot(vis_ref_img, vis_final_img, alpha_np, losses, args.ref_alpha, plot_path)

    elapsed = time.perf_counter() - t_start
    summary = {
        "method": args.reference_mode,
        "scene": str(scene_path),
        "out_dir": str(out_dir),
        "variant": active_variant,
        "total_time_sec": elapsed,
        "n_iter": args.n_iter,
        "debug_every": args.debug_every,
        "video_ratio": args.video_ratio if args.reference_mode == "video" else None,
        "final_alpha_mean": float(alpha_np.mean()),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    if args.save_trajectory_video and args.debug_every > 0:
        build_trajectory_videos(debug_dir, out_dir, fps=args.trajectory_fps)

    print(f"Saved {teapot_reference_path}")
    print(f"Saved {teapot_optimized_path}")
    print(f"Saved summary plot to {plot_path}")
    print(f"Saved loss plot to {plot_path.with_name(f'{plot_path.stem}_loss{plot_path.suffix}')}")
    print(f"Saved summary JSON to {summary_path}")


if __name__ == "__main__":
    main()
