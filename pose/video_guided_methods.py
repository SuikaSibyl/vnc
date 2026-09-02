from __future__ import annotations

import imageio
import numpy as np
import torch
import torch.nn.functional as F


def _normalize_hw(resolution) -> tuple[int, int]:
    if isinstance(resolution, int):
        return resolution, resolution
    if isinstance(resolution, (list, tuple)):
        if len(resolution) == 2:
            return int(resolution[0]), int(resolution[1])
        if len(resolution) == 1:
            side = int(resolution[0])
            return side, side
    raise TypeError(f"Unsupported resolution format: {resolution!r}")


def resize_image_tensor(image: torch.Tensor, resolution) -> torch.Tensor:
    out_h, out_w = _normalize_hw(resolution)
    if image.shape[0] == out_h and image.shape[1] == out_w:
        return image
    resized = F.interpolate(
        image.permute(2, 0, 1).unsqueeze(0),
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0].permute(1, 2, 0)


class VideoFrameScheduler:
    def __init__(
        self,
        video_path: str,
        n_iter: int,
        resolution: int,
        device: torch.device,
        fallback: torch.Tensor | None = None,
        video_ratio: float = 0.8,
        gt_img: torch.Tensor | dict | None = None,
        max_frames: int | None = None,
    ) -> None:
        reader = imageio.get_reader(video_path)
        frames = []
        for frame in reader:
            image = torch.from_numpy(frame.astype("float32") / 255.0).to(device)[..., :3].flip(0)
            frames.append(resize_image_tensor(image, resolution))
        reader.close()

        if max_frames is not None and max_frames > 0 and len(frames) > max_frames:
            keep_idx = np.linspace(0, len(frames) - 1, num=max_frames, dtype=int)
            frames = [frames[idx] for idx in keep_idx]

        self.frames = frames
        if fallback is None:
            if isinstance(gt_img, dict):
                fallback = gt_img["images"][0]
            else:
                fallback = gt_img
        if fallback is None:
            raise ValueError("VideoFrameScheduler requires either fallback or gt_img.")
        self.fallback = fallback
        self.n_video_iters = int(n_iter * video_ratio)

    def get_ref(self, iteration: int, view: int | None = None) -> torch.Tensor:
        if iteration >= self.n_video_iters or not self.frames:
            return self.fallback
        frac = iteration / max(self.n_video_iters - 1, 1)
        idx = min(int(round(frac * (len(self.frames) - 1))), len(self.frames) - 1)
        return self.frames[idx]
