import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

DROT_ROOT = Path(__file__).resolve().parents[2] / "codes" / "transformations" / "DROT"
if str(DROT_ROOT) not in sys.path:
    sys.path.insert(0, str(DROT_ROOT))

from core.LossFunction import PointLossFunction
from core.NvDiffRastRenderer import NVDiffRastFullRenderer

from diff_skinning_renderer import (
    DifferentiableSkeletonRenderer,
    SkeletonAsset,
    load_reference_image,
    require_working_cuda,
    save_image,
)


def rgbxy_chamfer_loss(
    pred_img: torch.Tensor,
    target_img: torch.Tensor,
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    n_samples: int = 2048,
    xy_weight: float = 2.0,
    return_debug: bool = False,
):
    h, w = pred_img.shape[:2]
    device = pred_img.device

    ys = torch.linspace(-1.0, 1.0, h, device=device)
    xs = torch.linspace(-1.0, 1.0, w, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    xy = torch.stack([grid_x, grid_y], dim=-1) * xy_weight
    pixel_xy = torch.stack(
        [
            ((grid_x + 1.0) * 0.5) * (w - 1),
            ((1.0 - (grid_y + 1.0) * 0.5)) * (h - 1),
        ],
        dim=-1,
    )

    def fg_points(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        fg = (mask > 0.5).reshape(-1)
        pts = torch.cat([img.reshape(-1, 3), xy.reshape(-1, 2)], dim=-1)[fg]
        coords = pixel_xy.reshape(-1, 2)[fg]
        if pts.shape[0] > n_samples:
            idx = torch.randperm(pts.shape[0], device=device)[:n_samples]
            pts = pts[idx]
            coords = coords[idx]
        return pts, coords

    pred_pts, pred_coords = fg_points(pred_img, pred_mask)
    tgt_pts, tgt_coords = fg_points(target_img, target_mask)
    if pred_pts.shape[0] == 0 or tgt_pts.shape[0] == 0:
        loss = F.mse_loss(pred_img, target_img)
        if return_debug:
            return loss, None
        return loss

    dists = torch.cdist(pred_pts.unsqueeze(0), tgt_pts.unsqueeze(0))[0]
    pred_nn = dists.argmin(dim=1)
    tgt_nn = dists.argmin(dim=0)
    loss = 0.5 * (dists.min(dim=1)[0].mean() + dists.min(dim=0)[0].mean())
    if not return_debug:
        return loss

    max_draw = min(96, pred_coords.shape[0], tgt_coords.shape[0])
    pred_draw_idx = torch.linspace(0, pred_coords.shape[0] - 1, steps=max_draw, device=device).round().long()
    tgt_draw_idx = torch.linspace(0, tgt_coords.shape[0] - 1, steps=max_draw, device=device).round().long()
    debug = {
        "pred_coords": pred_coords[pred_draw_idx].detach().cpu(),
        "pred_match_coords": tgt_coords[pred_nn[pred_draw_idx]].detach().cpu(),
        "tgt_coords": tgt_coords[tgt_draw_idx].detach().cpu(),
        "tgt_match_coords": pred_coords[tgt_nn[tgt_draw_idx]].detach().cpu(),
    }
    return loss, debug


def save_rgbxy_correspondence_debug(
    path: Path,
    pred_img: torch.Tensor,
    target_img: torch.Tensor,
    debug_data,
) -> None:
    if debug_data is None:
        return

    pred = (pred_img.detach().clamp(0.0, 1.0).flip(0).cpu().numpy() * 255.0).round().astype(np.uint8)
    target = (target_img.detach().clamp(0.0, 1.0).flip(0).cpu().numpy() * 255.0).round().astype(np.uint8)
    h, w = pred.shape[:2]
    gap = 24
    canvas = Image.new("RGB", (w * 2 + gap, h), color=(255, 255, 255))
    canvas.paste(Image.fromarray(pred), (0, 0))
    canvas.paste(Image.fromarray(target), (w + gap, 0))

    draw = ImageDraw.Draw(canvas, "RGBA")

    if "colors" in debug_data:
        for pred_xy, match_xy, color in zip(
            debug_data["pred_coords"].tolist(),
            debug_data["pred_match_coords"].tolist(),
            debug_data["colors"].tolist(),
        ):
            x0, y0 = pred_xy
            x1, y1 = match_xy
            rgb = tuple(int(max(0, min(255, round(c * 255.0)))) for c in color)
            draw.line((x0, y0, x1 + w + gap, y1), fill=(*rgb, 110), width=1)
            draw.ellipse((x0 - 2, y0 - 2, x0 + 2, y0 + 2), fill=(*rgb, 255))
            draw.ellipse((x1 + w + gap - 2, y1 - 2, x1 + w + gap + 2, y1 + 2), fill=(*rgb, 255))
    else:
        pred_color = (255, 80, 80, 255)
        tgt_color = (80, 120, 255, 255)
        line_color_a = (255, 120, 0, 110)
        line_color_b = (0, 160, 255, 110)

        for pred_xy, match_xy in zip(debug_data["pred_coords"].tolist(), debug_data["pred_match_coords"].tolist()):
            x0, y0 = pred_xy
            x1, y1 = match_xy
            draw.line((x0, y0, x1 + w + gap, y1), fill=line_color_a, width=1)
            draw.ellipse((x0 - 2, y0 - 2, x0 + 2, y0 + 2), fill=pred_color)
            draw.ellipse((x1 + w + gap - 2, y1 - 2, x1 + w + gap + 2, y1 + 2), fill=pred_color)

        for tgt_xy, match_xy in zip(debug_data["tgt_coords"].tolist(), debug_data["tgt_match_coords"].tolist()):
            x0, y0 = tgt_xy
            x1, y1 = match_xy
            draw.line((x1, y1, x0 + w + gap, y0), fill=line_color_b, width=1)
            draw.ellipse((x1 - 2, y1 - 2, x1 + 2, y1 + 2), fill=tgt_color)
            draw.ellipse((x0 + w + gap - 2, y0 - 2, x0 + w + gap + 2, y0 + 2), fill=tgt_color)

    canvas.save(path)


def _annotate_iteration(frame: np.ndarray, step: int) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    label = f"iter {step}"
    x, y = 12, 10
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((x, y), label, font=font)
    draw.rectangle((bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4), fill=(0, 0, 0))
    draw.text((x, y), label, fill=(255, 255, 255), font=font)
    return np.array(img)


def build_method_videos(method_dir: Path, fps: int = 12) -> None:
    trajectory_dir = method_dir / "_trajectory_frames"

    def _collect(prefix: str) -> list[tuple[np.ndarray, int]]:
        frames: list[tuple[np.ndarray, int]] = []
        for image_path in sorted(trajectory_dir.glob(f"{prefix}_*.png")):
            try:
                step = int(image_path.stem.split("_")[-1])
            except (IndexError, ValueError):
                continue
            frames.append((imageio.v3.imread(image_path), step))
        if not frames:
            return frames
        max_frames = max(1, int(round(fps * 3.0)))
        if len(frames) > max_frames:
            keep_idx = np.linspace(0, len(frames) - 1, num=max_frames, dtype=int)
            frames = [frames[idx] for idx in keep_idx]
        return [(_annotate_iteration(frame, step), step) for frame, step in frames]

    render_frames = _collect("render")
    ref_frames = _collect("ref")
    if render_frames:
        imageio.mimwrite(method_dir / "trajectory.mp4", [frame for frame, _ in render_frames], fps=fps)
    if ref_frames:
        imageio.mimwrite(method_dir / "trajectory_ref.mp4", [frame for frame, _ in ref_frames], fps=fps)


def write_timing_summary(output_dir: Path, summaries: dict) -> Path:
    methods = {}
    for method, summary in summaries.items():
        total_time_sec = summary.get("total_time_sec")
        if isinstance(total_time_sec, (int, float)):
            methods[method] = float(total_time_sec)
    summary_path = output_dir / "optimization_time_summary.json"
    summary_path.write_text(json.dumps({"methods": dict(sorted(methods.items()))}, indent=2))
    return summary_path


def build_drot_correspondence_debug(render_res, matched_point_5d, max_draw: int = 96):
    if matched_point_5d is None or not torch.is_tensor(matched_point_5d):
        return None
    render_rgb = render_res["images"][0]
    render_pos = render_res["pos"][0]
    render_mask = render_res["msk"][0]
    match_pos = matched_point_5d[0, ..., 3:]

    valid = render_mask.reshape(-1)
    if valid.sum().item() == 0:
        return None

    render_xy = ((render_pos.reshape(-1, 2)[valid] + 1.0) * 0.5).clamp(0.0, 1.0)
    match_xy = match_pos.reshape(-1, 2)[valid].clamp(0.0, 1.0)
    colors = render_rgb.reshape(-1, render_rgb.shape[-1])[valid][..., :3]

    n = render_xy.shape[0]
    take = min(max_draw, n)
    idx = torch.linspace(0, n - 1, steps=take, device=render_xy.device).round().long()
    h, w = render_rgb.shape[:2]
    return {
        "pred_coords": torch.stack([render_xy[idx, 0] * (w - 1), (1.0 - render_xy[idx, 1]) * (h - 1)], dim=-1).cpu(),
        "pred_match_coords": torch.stack([match_xy[idx, 0] * (w - 1), (1.0 - match_xy[idx, 1]) * (h - 1)], dim=-1).cpu(),
        "colors": colors[idx].cpu(),
    }


class NullLogger:
    def add_image(self, name, img, step):
        pass


def gaussian_pyramid_loss(pred: torch.Tensor, target: torch.Tensor, n_levels: int = 4) -> torch.Tensor:
    k1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=pred.device, dtype=pred.dtype)
    k1d = k1d / k1d.sum()
    kernel = (k1d[:, None] * k1d[None, :]).unsqueeze(0).unsqueeze(0)
    channels = pred.shape[-1]

    loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
    p, t = pred, target
    for level in range(n_levels):
        loss = loss + torch.mean((p - t) ** 2)
        if level < n_levels - 1:
            k = kernel.expand(channels, 1, 5, 5)
            xp = F.conv2d(p.permute(2, 0, 1).unsqueeze(0), k, padding=2, groups=channels)[:, :, ::2, ::2]
            xt = F.conv2d(t.permute(2, 0, 1).unsqueeze(0), k, padding=2, groups=channels)[:, :, ::2, ::2]
            p = xp.squeeze(0).permute(1, 2, 0)
            t = xt.squeeze(0).permute(1, 2, 0)
    return loss


class LOILoss(torch.nn.Module):
    """LOIR: 1-Wasserstein on Locally Orderless Images (Mehta et al., CVPR 2025).

    Inlined from codes/pose/loir/model.py — pure PyTorch, no drjit dependency.
    Default hyperparameters match the paper's config.
    """

    def __init__(self, channels: int = 3, device=None):
        super().__init__()
        dev = device if device is not None else torch.device("cuda")
        n_bins = 8
        sigma = (1.0, 15.0, 45.0)
        alpha = (1.0, 15.0, 45.0)
        beta = [1.0 / n_bins]
        bounds = [-4.0 * beta[0], 1.0 + 4.0 * beta[0]]
        kernel_size = 121

        self.n_bins = n_bins
        self.n_sigma = len(sigma)
        self.n_alpha = len(alpha)
        self.n_beta = len(beta)
        self.channels = channels
        self.kernel_size = kernel_size

        sigma_ks = max(3, int(sigma[-1] * 4))
        if sigma_ks % 2 == 0:
            sigma_ks += 1
        self.sigma_ks = sigma_ks

        def _gauss2d(size: int, sig: float) -> torch.Tensor:
            x = torch.linspace(-size // 2 + 1, size // 2, size, dtype=torch.float32)
            gx, gy = torch.meshgrid(x, x, indexing="ij")
            k = torch.exp(-(gx**2 + gy**2) / (2.0 * sig**2))
            return k / k.sum()

        # sigma_kernel: [n_sigma*C, 1, sigma_ks, sigma_ks] — depthwise conv per channel
        sigma_k = torch.stack([_gauss2d(sigma_ks, s) for s in sigma])
        self.register_buffer("sigma_kernel", sigma_k.unsqueeze(1).repeat(channels, 1, 1, 1))

        bin_edges = torch.linspace(bounds[0], bounds[1], n_bins + 1, dtype=torch.float32)
        bin_val = (bin_edges[1:] + bin_edges[:-1]) / 2.0  # [n_bins]
        self.register_buffer("bin_val", bin_val.view(1, n_bins, 1, 1, 1, 1))
        self.register_buffer("beta_val", torch.tensor(beta, dtype=torch.float32).view(-1, 1, 1, 1, 1, 1))

        # alpha_kernel: [n_alpha * n_groups, 1, kernel_size, kernel_size] — depthwise conv
        n_groups = self.n_beta * n_bins * channels * self.n_sigma
        alpha_k = torch.stack([_gauss2d(kernel_size, s) for s in alpha])
        self.register_buffer("alpha_kernel", alpha_k.unsqueeze(1).repeat(n_groups, 1, 1, 1))

        self.to(dev)

    def forward(self, im: torch.Tensor) -> torch.Tensor:
        """Compute LOI histogram tensor.

        Args:
            im: [H, W, C] float tensor in [0, 1].
        Returns:
            [n_alpha, n_beta, n_sigma, C, H, W, n_bins] histogram tensor.
        """
        im = im.permute(2, 0, 1)  # [C, H, W]
        with torch.no_grad():
            im.data = im.data.clamp(0.0, 1.0)
        C, H, W = im.shape

        conv_sigma = F.conv2d(im.unsqueeze(0), self.sigma_kernel, padding=self.sigma_ks // 2, groups=C)
        conv_sigma = conv_sigma.view(C, self.n_sigma, H, W)  # [C, n_sigma, H, W]

        im_diff = conv_sigma.unsqueeze(0).unsqueeze(1) - self.bin_val  # [n_beta, n_bins, C, n_sigma, H, W]
        im_iso = torch.exp(-0.5 * (im_diff / self.beta_val).pow(2))
        im_iso = im_iso / (np.sqrt(2.0 * np.pi) * self.beta_val)
        im_iso = im_iso.reshape(self.n_beta * self.n_bins * C * self.n_sigma, H, W)

        n_groups = self.n_beta * self.n_bins * C * self.n_sigma
        loi = F.conv2d(im_iso.unsqueeze(0), self.alpha_kernel, padding=self.kernel_size // 2, groups=n_groups)
        loi = loi.reshape(self.n_beta, self.n_bins, C, self.n_sigma, self.n_alpha, H, W)
        loi = loi.permute(4, 0, 3, 2, 5, 6, 1)  # [n_alpha, n_beta, n_sigma, C, H, W, n_bins]
        return loi

    @staticmethod
    def emd(loi1: torch.Tensor, loi2: torch.Tensor) -> torch.Tensor:
        """L1 Earth Mover's Distance between two LOI tensors (bins on last dim)."""
        return torch.mean(torch.abs(torch.cumsum(loi1, dim=-1) - torch.cumsum(loi2, dim=-1)))


class VideoFrameScheduler:
    def __init__(
        self,
        video_path: str,
        n_iter: int,
        fallback_image: torch.Tensor,
        fallback_export_image: torch.Tensor | None,
        resolution: int,
        export_resolution: int | None,
        device: torch.device,
        video_ratio: float = 0.8,
        video_trim: float = 1.0,
    ):
        try:
            import imageio.v2 as imageio  # type: ignore
        except ImportError as error:
            raise RuntimeError("imageio is required for video target mode. Install it with: pip install imageio") from error

        reader = None
        decode_errors = []
        for plugin_kwargs in ({"format": "FFMPEG"}, {"format": "pyav"}, {}):
            try:
                reader = imageio.get_reader(video_path, **plugin_kwargs)
                break
            except Exception as error:
                decode_errors.append(f"{plugin_kwargs or {'format': 'default'}} -> {error}")
        if reader is None:
            raise RuntimeError(
                "Unable to decode the video file for video fitting. "
                + " | ".join(decode_errors)
            )
        all_frames = []
        export_frames = []
        for frame in reader:
            frame = np.asarray(frame, dtype=np.uint8)
            tensor = torch.from_numpy(frame.astype(np.float32) / 255.0).to(device)
            tensor = tensor[..., :3]
            tensor = tensor.flip(0)
            export_tensor = tensor
            if export_resolution is not None and (export_tensor.shape[0] != export_resolution or export_tensor.shape[1] != export_resolution):
                export_tensor = (
                    F.interpolate(
                        export_tensor.permute(2, 0, 1).unsqueeze(0),
                        size=(export_resolution, export_resolution),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                )
            if tensor.shape[0] != resolution or tensor.shape[1] != resolution:
                tensor = (
                    F.interpolate(
                        tensor.permute(2, 0, 1).unsqueeze(0),
                        size=(resolution, resolution),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                )
            all_frames.append(tensor)
            export_frames.append(export_tensor)
        reader.close()
        if not all_frames:
            raise RuntimeError(
                f"Video target mode could not decode any frames from {video_path!r}. "
                "Install a working video backend such as imageio-ffmpeg or pyav."
            )
        keep = max(1, int(len(all_frames) * min(max(video_trim, 0.0), 1.0)))
        self.frames = all_frames[:keep]
        self.export_frames = export_frames[:keep]
        self.fallback_image = fallback_image
        self.fallback_export_image = fallback_export_image if fallback_export_image is not None else fallback_image
        if export_resolution is not None and (
            self.fallback_export_image.shape[0] != export_resolution
            or self.fallback_export_image.shape[1] != export_resolution
        ):
            self.fallback_export_image = (
                F.interpolate(
                    self.fallback_export_image.permute(2, 0, 1).unsqueeze(0),
                    size=(export_resolution, export_resolution),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .permute(1, 2, 0)
            )
        self.n_video_iters = int(n_iter * video_ratio)

    def get_ref(self, iteration: int) -> torch.Tensor:
        if iteration >= self.n_video_iters or not self.frames:
            return self.fallback_image
        frac = iteration / max(self.n_video_iters - 1, 1)
        frame_idx = min(int(round(frac * (len(self.frames) - 1))), len(self.frames) - 1)
        return self.frames[frame_idx]

    def get_export_ref(self, iteration: int) -> torch.Tensor:
        if iteration >= self.n_video_iters or not self.export_frames:
            return self.fallback_export_image
        frac = iteration / max(self.n_video_iters - 1, 1)
        frame_idx = min(int(round(frac * (len(self.export_frames) - 1))), len(self.export_frames) - 1)
        return self.export_frames[frame_idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize skeleton coefficients to match a reference render.")
    parser.add_argument("--fbx", type=Path, default=None)
    parser.add_argument("--asset", type=Path, default=None)
    parser.add_argument("--mesh-name", default=None)
    parser.add_argument("--reference-image", type=Path, default=None)
    parser.add_argument("--reference-frame", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--render-size", type=int, default=None, help="Internal render resolution (supersampling). Defaults to --image-size.")
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--lr-root", type=float, default=None, help="LR for root bone (defaults to --lr).")
    parser.add_argument("--lr-trans", type=float, default=None, help="LR for global translation (defaults to --lr).")
    parser.add_argument("--lr-decay", type=float, default=0.999, help="Multiplicative LR decay per step (StepLR gamma).")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--azimuth", type=float, default=15.0)
    parser.add_argument("--elevation", type=float, default=12.0)
    parser.add_argument("--distance-scale", type=float, default=1.8)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--blender-bin", default="blender")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["mse"],
        choices=["mse", "rgbxy", "video", "video+rgbxy", "loir", "video+loir"],
        help=(
            "Optimization methods to run.\n"
            "  mse          - Gaussian pyramid MSE against the final reference render.\n"
            "  rgbxy        - RGBXY plus Gaussian pyramid MSE against the final reference render.\n"
            "  video        - coarse-to-fine video-guided fitting, then final reference render.\n"
            "  video+rgbxy  - video guidance plus RGBXY on the current frame.\n"
            "  loir         - LOIR EMD loss (1-Wasserstein on LOI histograms) against the final reference.\n"
            "  video+loir   - video guidance with LOIR EMD loss on each video frame."
        ),
    )
    parser.add_argument("--video-path", type=str, default=None, help="Path to a video file for video-based fitting.")
    parser.add_argument("--video-ratio", type=float, default=0.8, help="Fraction of steps guided by video frames.")
    parser.add_argument(
        "--video-trim",
        type=float,
        default=1.0,
        help="Only use the first fraction of the video, e.g. 0.5 keeps the first half of frames.",
    )
    parser.add_argument("--pyramid-levels", type=int, default=4, help="Gaussian pyramid levels for video fitting.")
    parser.add_argument("--rgbxy-samples", type=int, default=2048, help="Foreground sample budget for RGBXY.")
    parser.add_argument("--rgbxy-xy-weight", type=float, default=2.0, help="XY scaling factor for RGBXY.")
    parser.add_argument("--render-joints", action="store_true", help="Overlay joints and bones on saved debug renders.")
    parser.add_argument(
        "--trajectory-every",
        type=int,
        default=None,
        help="Save trajectory video frames every N iterations. Defaults to --save-every when omitted.",
    )
    parser.add_argument("--trajectory-fps", type=int, default=12, help="Frames per second for saved optimization trajectory videos.")
    return parser.parse_args()


def export_asset_if_needed(args) -> Path:
    if args.asset is not None:
        return args.asset
    if args.fbx is None:
        raise ValueError("Provide either --asset or --fbx.")

    asset_path = args.output_dir / "exported_skinning_asset.npz"
    exporter = Path(__file__).with_name("export_fbx_skinning.py")
    cmd = [
        sys.executable,
        str(exporter),
        "--fbx",
        str(args.fbx),
        "--output",
        str(asset_path),
        "--blender-bin",
        args.blender_bin,
    ]
    if args.mesh_name:
        cmd.extend(["--mesh-name", args.mesh_name])
    if args.reference_frame is not None:
        cmd.extend(["--reference-frame", str(args.reference_frame)])
    subprocess.run(cmd, check=True)
    return asset_path


def build_reference(renderer: DifferentiableSkeletonRenderer, asset: SkeletonAsset, args, drot_renderer=None):
    if args.reference_image is not None:
        ref = load_reference_image(args.reference_image, args.image_size).to(renderer.device)
        mask = (ref.amax(dim=-1) < 0.95).float()
        gt_drot = {"images": ref.unsqueeze(0)}
        return ref, mask, gt_drot, None, None

    if asset.reference_rotvecs is None:
        raise ValueError(
            "No reference image was provided and the asset does not contain an exported reference pose. "
            "Pass --reference-image or export the FBX with --reference-frame."
        )
    reference_rotvecs = asset.reference_rotvecs.clone()
    reference_translations = torch.zeros_like(reference_rotvecs)
    rendered = renderer.render_pose(reference_rotvecs.unsqueeze(0), reference_translations.unsqueeze(0))
    gt_drot = {"images": rendered["image"].detach().unsqueeze(0)}
    return (
        rendered["image"].detach(),
        rendered["mask"].detach(),
        gt_drot,
        reference_rotvecs.unsqueeze(0),
        reference_translations.unsqueeze(0),
    )


def run_method(
    method: str,
    method_dir: Path,
    renderer: DifferentiableSkeletonRenderer,
    trajectory_renderer: DifferentiableSkeletonRenderer,
    drot_renderer: NVDiffRastFullRenderer,
    asset: SkeletonAsset,
    target_image: torch.Tensor,
    target_mask: torch.Tensor,
    gt_drot,
    args,
):
    method_start_time = time.perf_counter()
    num_bones = len(asset.bone_names)
    root_rotvecs = torch.nn.Parameter(torch.zeros((1, 1, 3), dtype=torch.float32, device=renderer.device))
    joint_rotvecs = torch.nn.Parameter(torch.zeros((1, num_bones - 1, 3), dtype=torch.float32, device=renderer.device))
    global_trans = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32, device=renderer.device))
    fixed_translations = torch.zeros((1, num_bones, 3), dtype=torch.float32, device=renderer.device)
    lr_root = args.lr_root if args.lr_root is not None else args.lr
    lr_trans = args.lr_trans if args.lr_trans is not None else args.lr
    optimizer = torch.optim.Adam([
        {"params": [root_rotvecs], "lr": lr_root},
        {"params": [joint_rotvecs], "lr": args.lr},
        {"params": [global_trans], "lr": lr_trans},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    loss_func = None
    if method in {"rgbxy", "video+rgbxy"}:
        loss_func = PointLossFunction(
            debug=False,
            resolution=[args.image_size, args.image_size],
            settings={"matching_weight": 1.0, "matching_interval": 0, "matcher": "Sinkhorn"},
            device=renderer.device,
            renderer=drot_renderer,
            num_views=1,
            logger=NullLogger(),
        )

    loi_model = None
    target_loi = None
    if method in {"loir", "video+loir"}:
        loi_model = LOILoss(channels=3, device=renderer.device)
        with torch.no_grad():
            target_loi = loi_model(target_image)

    video_scheduler = None
    if method in {"video", "video+rgbxy", "video+loir"}:
        if args.video_path is None:
            raise ValueError(f"Method '{method}' requires --video-path.")
        video_scheduler = VideoFrameScheduler(
            video_path=args.video_path,
            n_iter=args.steps,
            fallback_image=target_image,
            fallback_export_image=args.export_target_image,
            resolution=args.image_size,
            export_resolution=trajectory_renderer.image_size,
            device=renderer.device,
            video_ratio=args.video_ratio,
            video_trim=args.video_trim,
        )

    trajectory_every = args.trajectory_every if args.trajectory_every is not None else args.save_every
    trajectory_dir = method_dir / "_trajectory_frames"
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    save_image(method_dir / "target_final.png", target_image)
    history = []

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        full_rotvecs = torch.cat([root_rotvecs, joint_rotvecs], dim=1)
        full_translations = fixed_translations
        rendered = renderer.render_pose(full_rotvecs, full_translations, global_trans)
        if renderer.image_size != args.image_size:
            _img = F.interpolate(
                rendered["image"].permute(2, 0, 1).unsqueeze(0),
                size=(args.image_size, args.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
            rendered = {**rendered, "image": _img}
        ref = target_image
        rgbxy_debug = None
        scene = None

        if method == "rgbxy":
            scene = renderer.build_drot_scene(full_rotvecs, full_translations)
            rgbxy_target = gt_drot
            loss_main, render_res = loss_func(rgbxy_target, iteration=step - 1, scene=scene, view=0)
            rendered = {"image": render_res["images"][0], "mask": render_res["msk"][0].float()}
            if torch.is_tensor(loss_func.matchings[0]):
                rgbxy_debug = build_drot_correspondence_debug(render_res, loss_func.matchings[0])
            loss_main = loss_main + gaussian_pyramid_loss(rendered["image"], target_image, n_levels=args.pyramid_levels)
        elif method == "video":
            ref = video_scheduler.get_ref(step - 1)
            loss_main = gaussian_pyramid_loss(rendered["image"], ref, n_levels=args.pyramid_levels)
        elif method == "video+rgbxy":
            ref = video_scheduler.get_ref(step - 1)
            scene = renderer.build_drot_scene(full_rotvecs, full_translations)
            rgbxy_target = {"images": ref.unsqueeze(0)}
            rgbxy_loss, render_res = loss_func(rgbxy_target, iteration=step - 1, scene=scene, view=0)
            rendered = {"image": render_res["images"][0], "mask": render_res["msk"][0].float()}
            if torch.is_tensor(loss_func.matchings[0]):
                rgbxy_debug = build_drot_correspondence_debug(render_res, loss_func.matchings[0])
            loss_main = gaussian_pyramid_loss(rendered["image"], ref, n_levels=args.pyramid_levels) + 0.25 * rgbxy_loss
        elif method == "loir":
            est_loi = loi_model(rendered["image"])
            loss_main = LOILoss.emd(target_loi, est_loi)
        elif method == "video+loir":
            ref = video_scheduler.get_ref(step - 1)
            with torch.no_grad():
                ref_loi = loi_model(ref)
            est_loi = loi_model(rendered["image"])
            loss_main = LOILoss.emd(ref_loi, est_loi)
        else:
            loss_main = gaussian_pyramid_loss(rendered["image"], target_image, n_levels=args.pyramid_levels)

        reg_loss = 1e-5 * torch.mean(full_rotvecs**2)  # rotation-only; global_trans not regularized
        loss = loss_main + reg_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        history.append(
            {
                "step": step,
                "loss": float(loss.item()),
                "main_loss": float(loss_main.item()),
                "reg_loss": float(reg_loss.item()),
            }
        )

        should_save_trajectory = step % trajectory_every == 0 or step == 1 or step == args.steps
        should_save_checkpoint = step % args.save_every == 0 or step == 1 or step == args.steps

        if should_save_checkpoint:
            snap_dir = method_dir / f"step_{step:04d}"
            snap_dir.mkdir(parents=True, exist_ok=True)
            snap_img = rendered["image"]
            if snap_img.shape[0] != 512 or snap_img.shape[1] != 512:
                with torch.no_grad():
                    snap_img = F.interpolate(
                        snap_img.permute(2, 0, 1).unsqueeze(0),
                        size=(512, 512),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).permute(1, 2, 0)
            save_image(snap_dir / "render.png", snap_img)
            if args.render_joints:
                render_with_joints = renderer.render_joint_overlay(snap_img, full_rotvecs, full_translations)
                save_image(snap_dir / "render_with_joints.png", render_with_joints)
            save_image(snap_dir / "target_step.png", ref)
            save_image(snap_dir / "target_final.png", target_image)
            if rgbxy_debug is not None:
                save_rgbxy_correspondence_debug(
                    snap_dir / "rgbxy_correspondence.png",
                    snap_img,
                    ref if method in {"video+rgbxy", "video+loir"} else target_image,
                    rgbxy_debug,
                )
            _rv = full_rotvecs.detach().cpu()
            _tr = full_translations.detach().cpu()
            _gt = global_trans.detach().cpu()
            torch.save(
                {
                    "rotvecs": _rv,
                    "translations": _tr,
                    "global_trans": _gt,
                    "step": step,
                    "loss": float(loss.item()),
                    "render_512": snap_img.detach().cpu(),
                    "bone_names": asset.bone_names,
                },
                snap_dir / "coefficients.pt",
            )
            np.savez(
                snap_dir / "coefficients.npz",
                rotvecs=_rv.numpy(),
                translations=_tr.numpy(),
                global_trans=_gt.numpy(),
                bone_names=np.array(asset.bone_names),
            )
        if should_save_trajectory:
            with torch.no_grad():
                traj_render = trajectory_renderer.render_pose(full_rotvecs, full_translations, global_trans)["image"]
            save_image(trajectory_dir / f"render_{step:04d}.png", traj_render)
            if video_scheduler is not None:
                ref_img = video_scheduler.get_export_ref(step - 1)
            else:
                ref_img = ref
                if ref_img.shape[0] != trajectory_renderer.image_size or ref_img.shape[1] != trajectory_renderer.image_size:
                    with torch.no_grad():
                        ref_img = F.interpolate(
                            ref_img.permute(2, 0, 1).unsqueeze(0),
                            size=(trajectory_renderer.image_size, trajectory_renderer.image_size),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0).permute(1, 2, 0)
            save_image(trajectory_dir / f"ref_{step:04d}.png", ref_img)
        if should_save_checkpoint or should_save_trajectory:
            print(
                f"[{method}] step={step:04d} loss={loss.item():.6f} "
                f"main={loss_main.item():.6f} reg={reg_loss.item():.6f}"
            )

    final_rotvecs = torch.cat([root_rotvecs, joint_rotvecs], dim=1)
    final_translations = fixed_translations
    with torch.no_grad():
        final_render = renderer.render_pose(final_rotvecs, final_translations, global_trans)
        final_img = final_render["image"]
        if final_img.shape[0] != 512 or final_img.shape[1] != 512:
            final_img = F.interpolate(
                final_img.permute(2, 0, 1).unsqueeze(0),
                size=(512, 512),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
        final_render = {**final_render, "image": final_img}
    save_image(method_dir / "final.png", final_render["image"])
    if args.render_joints:
        final_with_joints = renderer.render_joint_overlay(final_render["image"], final_rotvecs, final_translations)
        save_image(method_dir / "final_with_joints.png", final_with_joints)
    _rv = final_rotvecs.detach().cpu()
    _tr = final_translations.detach().cpu()
    _gt = global_trans.detach().cpu()
    torch.save(
        {
            "rotvecs": _rv,
            "translations": _tr,
            "global_trans": _gt,
            "history": history,
            "render_512": final_render["image"].detach().cpu(),
            "bone_names": asset.bone_names,
        },
        method_dir / "optimized_coefficients.pt",
    )
    np.savez(
        method_dir / "optimized_coefficients.npz",
        rotvecs=_rv.numpy(),
        translations=_tr.numpy(),
        global_trans=_gt.numpy(),
        bone_names=np.array(asset.bone_names),
    )
    elapsed_seconds = time.perf_counter() - method_start_time
    summary = {
        "method": method,
        "final_loss": history[-1]["loss"],
        "steps": args.steps,
        "lr": args.lr,
        "image_size": args.image_size,
        "reference_frame": asset.reference_frame,
        "mesh_name": asset.mesh_name,
        "num_bones": num_bones,
        "total_time_sec": elapsed_seconds,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_step": elapsed_seconds / max(args.steps, 1),
    }
    (method_dir / "history.json").write_text(json.dumps(history, indent=2))
    (method_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    build_method_videos(method_dir, fps=args.trajectory_fps)
    return summary


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    require_working_cuda()
    asset_path = export_asset_if_needed(args)
    asset = SkeletonAsset.from_npz(asset_path, device=torch.device("cuda"))
    render_size = args.render_size if args.render_size is not None else args.image_size
    renderer = DifferentiableSkeletonRenderer(
        asset=asset,
        image_size=render_size,
        device="cuda",
        azimuth_deg=args.azimuth,
        elevation_deg=args.elevation,
        distance_scale=args.distance_scale,
    )
    trajectory_renderer = renderer
    if renderer.image_size != 512:
        trajectory_renderer = DifferentiableSkeletonRenderer(
            asset=asset,
            image_size=512,
            device="cuda",
            azimuth_deg=args.azimuth,
            elevation_deg=args.elevation,
            distance_scale=args.distance_scale,
        )
    drot_renderer = NVDiffRastFullRenderer(
        device=renderer.device,
        settings={"background": "white", "shading": True, "light_power": 1.0},
        resolution=[args.image_size, args.image_size],
    )

    target_image, target_mask, gt_drot, reference_rotvecs, reference_root_translation = build_reference(
        renderer, asset, args, drot_renderer=drot_renderer
    )
    if args.reference_image is not None:
        export_target_image = load_reference_image(args.reference_image, trajectory_renderer.image_size).to(renderer.device)
    elif reference_rotvecs is not None and reference_root_translation is not None:
        with torch.no_grad():
            export_target_image = trajectory_renderer.render_pose(reference_rotvecs, reference_root_translation)["image"].detach()
    else:
        export_target_image = target_image
    args.export_target_image = export_target_image
    save_image(args.output_dir / "reference.png", target_image)
    if args.render_joints and reference_rotvecs is not None and reference_root_translation is not None:
        reference_with_joints = renderer.render_joint_overlay(target_image, reference_rotvecs, reference_root_translation)
        save_image(args.output_dir / "reference_with_joints.png", reference_with_joints)

    summaries = {}
    for method in args.methods:
        method_dir = args.output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        summaries[method] = run_method(
            method,
            method_dir,
            renderer,
            trajectory_renderer,
            drot_renderer,
            asset,
            target_image,
            target_mask,
            gt_drot,
            args,
        )

    (args.output_dir / "all_summaries.json").write_text(json.dumps(summaries, indent=2))
    write_timing_summary(args.output_dir, summaries)


if __name__ == "__main__":
    main()
