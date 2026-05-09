import random
import argparse
from tqdm.std import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import imageio
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from pytorch3d.renderer import Materials
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import join_meshes_as_scene
from pytorch3d.transforms import euler_angles_to_matrix

import sys, os, json, time, pathlib

from repo_paths import DROT_ROOT, OUTPUTS_ROOT, PRDPT_ROOT, SCENES_ROOT, setup_python_paths

# ---------------------------------------------------------------------
# Robust path handling
# ---------------------------------------------------------------------
SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
PROJECT_ROOT = str(DROT_ROOT)

setup_python_paths()

from experiments.common import *
from experiments.logger import Logger
from core.LossFunction import PointLossFunction
from core.NvDiffRastRenderer import NVDiffRastFullRenderer

from video_guided_methods import VideoFrameScheduler
from timing_summary import write_explicit_timing_summary

PRDPT_DIR = str(PRDPT_ROOT)
if PRDPT_DIR not in sys.path:
    sys.path.insert(0, PRDPT_DIR)
from utils_fns import smoothFn


def resolve_path(path_str, base_dir=PROJECT_ROOT):
    """Resolve config/data paths robustly."""
    if path_str is None:
        return None
    if os.path.isabs(path_str):
        return path_str

    # Try relative to project root first
    cand1 = os.path.normpath(os.path.join(base_dir, path_str))
    if os.path.exists(cand1):
        return cand1

    # Try relative to the script directory
    cand2 = os.path.normpath(os.path.join(SCRIPT_DIR, path_str))
    if os.path.exists(cand2):
        return cand2

    # Fallback: still return project-root-based path
    return cand1


def resolve_paths_in_obj(obj, base_dir=PROJECT_ROOT):
    """Recursively fix likely file/dir paths inside config dict/list."""
    if isinstance(obj, dict):
        return {k: resolve_paths_in_obj(v, base_dir) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_paths_in_obj(v, base_dir) for v in obj]
    if isinstance(obj, str):
        # Heuristic: resolve strings that look like file/dir paths
        if "/" in obj or "\\" in obj or obj.endswith((".json", ".obj", ".txt", ".png", ".jpg", ".jpeg", ".exr")):
            return resolve_path(obj, base_dir)
    return obj


LOCAL_FURNITURE_ROOT = os.path.join(str(SCENES_ROOT), "furniture")


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


class Furniture:
    def __init__(self) -> None:
        self.meshes = []
        self.origin_trans = []

        self.optim_mesh = []
        self.optim_trans = []

    def load_furniture_scene(self, device, task, filepath):
        if filepath is None:
            raise FileNotFoundError("Scene directory path is missing.")

        # Furniture configs often use short local scene names like "init" / "gt".
        # Resolve those against the local furniture scene root first.
        if not os.path.isabs(filepath):
            local_candidate = os.path.join(LOCAL_FURNITURE_ROOT, filepath)
            if os.path.isdir(local_candidate):
                filepath = local_candidate
            else:
                filepath = resolve_path(filepath)
        else:
            filepath = resolve_path(filepath)

        if not os.path.isdir(filepath):
            raise FileNotFoundError(f"Scene directory not found: {filepath}")

        files = os.listdir(filepath)
        scene = {"origin_meshes": [], "optim": [], "sensors": [], "meshes": [], "origin_sensors": [], "material": []}

        for file in files:
            if not file.endswith(".obj"):
                continue
            fullpath = os.path.join(filepath, file)
            mesh = load_objs_as_meshes([fullpath], device=device)
            trans_x = torch.zeros((), device=device, requires_grad=True)
            trans_z = torch.zeros((), device=device, requires_grad=True)
            rot_y = torch.zeros((), device=device, requires_grad=True)
            optim_trans = {
                "translation_x": trans_x,
                "translation_z": trans_z,
                "rotation_y": rot_y,
            }
            scene["optim"] += [trans_x, trans_z, rot_y]
            self.meshes.append({"model": mesh, "optim": True, "trans": {}, "optim_trans": optim_trans})

        cam_path = os.path.join(filepath, "camera.txt")
        if not os.path.isfile(cam_path):
            raise FileNotFoundError(f"camera.txt not found: {cam_path}")

        cam = np.loadtxt(cam_path).tolist()
        poss = []
        cens = []
        for i in range(0, len(cam), 2):
            pos = [cam[i + 0][0], cam[i + 0][2], -cam[i + 0][1]]
            rot = [cam[i + 1][0], cam[i + 1][2], cam[i + 1][1]]
            v = torch.tensor([0, -1, 0]).float()
            rot = torch.tensor(rot).float()
            dir = torch.matmul(v, euler_angles_to_matrix(rot, convention="XYZ"))
            dir[2] *= -1
            center = torch.tensor(pos).float() + dir
            poss.append(pos)
            cens.append(center.tolist())

        task["view"].update({"num": len(poss), "direction": "manual", "position": poss, "center": cens})
        return scene, task

    def gen_mesh(self, optim, optim_values=None):
        tmp = []
        value_idx = 0
        for mesh_info in self.meshes:
            src_mesh = mesh_info["model"].clone()
            if optim:
                if optim_values is None:
                    trans_x = mesh_info["optim_trans"]["translation_x"]
                    trans_z = mesh_info["optim_trans"]["translation_z"]
                    rot_y = mesh_info["optim_trans"]["rotation_y"]
                else:
                    trans_x = optim_values[value_idx + 0]
                    trans_z = optim_values[value_idx + 1]
                    rot_y = optim_values[value_idx + 2]
                    value_idx += 3
                zero = torch.zeros((), device=device)
                rot = torch.stack([zero, rot_y, zero])
                optim_trans = torch.stack([trans_x, zero, trans_z])
                optim_rot = euler_angles_to_matrix(rot, convention="XYZ")
                with torch.no_grad():
                    rot_center = torch.mean(src_mesh.verts_list()[0], axis=0)
                src_mesh.offset_verts_(-rot_center)
                src_mesh.transform_verts_(optim_rot)
                src_mesh.offset_verts_(rot_center)
                src_mesh.offset_verts_(optim_trans)
            tmp.append(src_mesh)
        return join_meshes_as_scene(tmp)


METHOD_ALIASES = {
    "ours": "our",
    "rgbxy": "our",
    "rgbxy_pytorch3d": "our_pytorch3d",
    "video+nvdiffrast": "video_nvdiffrast",
    "video+rgbxy_pytorch3d": "video_our_pytorch3d",
    "video+our_pytorch3d": "video_our_pytorch3d",
}


def canonical_method_name(method):
    return METHOD_ALIASES.get(method, method)


def pack_optim_values(scene):
    return torch.stack([param.detach().clone() for param in scene["optim"]])


def render_scene_with_values(model, scene, renderer, optim_values, view=None, dcdt=False):
    tmp_scene = dict(scene)
    tmp_scene["meshes"] = [{"model": model.gen_mesh(True, optim_values=optim_values)}]
    return renderer.render(tmp_scene, DcDt=dcdt, view=view)


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
            pred_blur = np.asarray(
                Image.fromarray(np.uint8(np.clip(pred_gray * 255.0, 0, 255))).filter(
                    ImageFilter.GaussianBlur(radius=sigma)
                ),
                dtype=np.float32,
            ) / 255.0
            target_blur = np.asarray(
                Image.fromarray(np.uint8(np.clip(target_gray * 255.0, 0, 255))).filter(
                    ImageFilter.GaussianBlur(radius=sigma)
                ),
                dtype=np.float32,
            ) / 255.0
        else:
            pred_blur = pred_gray
            target_blur = target_gray

        pred_hist, _ = np.histogram(pred_blur, bins=bins, density=True)
        target_hist, _ = np.histogram(target_blur, bins=bins, density=True)
        pred_cdf = np.cumsum(pred_hist)
        target_cdf = np.cumsum(target_hist)
        loss += float(np.mean(np.abs(pred_cdf - target_cdf)))

    return loss / 3.0


def render_smooth_furniture(theta_p: torch.Tensor, update_fn, context_args: dict):
    renderer = context_args["renderer"]
    model = context_args["model"]
    scene = context_args["scene"]
    gt_image = context_args["gt_image"]
    view = context_args["view"]

    losses = []
    avg_img = None
    with torch.no_grad():
        for i in range(theta_p.shape[0]):
            render_res = render_scene_with_values(
                model,
                scene,
                renderer,
                theta_p[i],
                view=view,
                dcdt=False,
            )
            img = render_res["images"][0]
            loss = torch.mean((img[..., :3] - gt_image[..., :3]) ** 2)
            losses.append(loss)
            if avg_img is None:
                avg_img = img.detach()

    return torch.stack(losses), avg_img


def keep_only_view_zero(sensors, torch_cameras, torch_lights):
    if len(torch_cameras) == 0:
        raise ValueError("No camera views found for furniture scene.")
    return [sensors[0]], [torch_cameras[0]], [torch_lights[0]]


def parse_args():
    parser = argparse.ArgumentParser(description="Run furniture optimization experiments.")
    parser.add_argument("config_id", help="Config number, e.g. 2")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override optimizer learning rate.")
    parser.add_argument("--niter", type=int, default=None, help="Override number of optimization iterations.")
    parser.add_argument("--video-ratio", type=float, default=None, help="Override fraction of iterations that use video guidance.")
    parser.add_argument("--tag", default=None, help="Optional suffix for the logger experiment name.")
    parser.add_argument(
        "--summary-name",
        default=None,
        help="Optional explicit filename for the timing JSON written under outputs/furniture.",
    )
    return parser.parse_args()


def get_video_ratio(task):
    return task["setting"].get("video_ratio", task.get("video_ratio", 0.9))


def resize_hwc_bilinear(img: torch.Tensor, out_hw) -> torch.Tensor:
    if tuple(img.shape[:2]) == tuple(out_hw):
        return img
    chw = img.permute(2, 0, 1).unsqueeze(0)
    chw = F.interpolate(chw, size=out_hw, mode="bilinear", align_corners=False)
    return chw.squeeze(0).permute(1, 2, 0)


def make_debug_triptych(render_img: torch.Tensor, ref_img: torch.Tensor, target_img: torch.Tensor) -> np.ndarray:
    panels = []
    for img in (render_img, ref_img, target_img):
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
        panels.append(np.clip(img[..., :3], 0.0, 1.0))
    return np.concatenate(panels, axis=1)


def gaussian_pyramid_downsample(x: torch.Tensor) -> torch.Tensor:
    kernel_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=x.device, dtype=x.dtype)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.view(1, 1, 5, 5).repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, kernel_2d, stride=2, padding=2, groups=x.shape[1])


def gaussian_pyramid_loss(pred: torch.Tensor, target: torch.Tensor, num_levels: int = 4) -> torch.Tensor:
    pred_level = pred.permute(2, 0, 1).unsqueeze(0)
    target_level = target.permute(2, 0, 1).unsqueeze(0)
    loss = torch.zeros((), device=pred.device, dtype=pred.dtype)

    for level in range(num_levels):
        weight = 1.0 / (2 ** level)
        loss = loss + weight * torch.mean(torch.abs(pred_level - target_level))
        if level + 1 == num_levels:
            break
        pred_level = gaussian_pyramid_downsample(pred_level)
        target_level = gaussian_pyramid_downsample(target_level)

    return loss


def _to_uint8_frame(img_hwc: np.ndarray | torch.Tensor, flip: bool = True) -> np.ndarray:
    if torch.is_tensor(img_hwc):
        img_hwc = img_hwc.detach().cpu().numpy()
    if flip:
        img_hwc = img_hwc[::-1]
    return np.clip(np.rint(img_hwc[..., :3] * 255.0), 0, 255).astype(np.uint8)


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


class LOILoss(torch.nn.Module):
    """Differentiable Locally Orderless Image loss (Mehta et al., CVPR 2025)."""
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
        loi = F.conv2d(iso.unsqueeze(0), self.alpha_kernel,
                       padding=self.kernel_size // 2, groups=n_groups)
        loi = loi.reshape(self.n_beta, self.n_bins, C, self.n_sigma, self.n_alpha, H, W)
        return loi.permute(4, 0, 3, 2, 5, 6, 1)

    @staticmethod
    def emd(loi1: torch.Tensor, loi2: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(torch.cumsum(loi1, dim=-1) - torch.cumsum(loi2, dim=-1)))


if __name__ == "__main__":
    args = parse_args()
    task_name = "furniture"

    # Prefer a local override next to this script, then fall back to vendored DROT configs.
    config_id = args.config_id
    config_candidates = [
        os.path.join(LOCAL_FURNITURE_ROOT, f"{config_id}.json"),
        os.path.join(SCRIPT_DIR, task_name, f"{config_id}.json"),
        os.path.join(PROJECT_ROOT, "experiments", task_name, f"{config_id}.json"),
    ]

    config_path = None
    for p in config_candidates:
        if os.path.isfile(p):
            config_path = p
            break
    if config_path is None:
        raise FileNotFoundError(
            "Could not find config file. Tried:\n" + "\n".join(config_candidates)
        )

    with open(config_path, "r") as f:
        task = json.load(f)

    # Normalize all likely paths from config so moved script still works.
    # Prefer resolving relative paths from the config file's own directory first.
    config_base_dir = os.path.dirname(config_path)
    task = resolve_paths_in_obj(task, config_base_dir)

    if args.learning_rate is not None:
        task["setting"]["learning_rate"] = args.learning_rate
    if args.niter is not None:
        task["setting"]["Niter"] = args.niter
    if args.video_ratio is not None:
        task["setting"]["video_ratio"] = args.video_ratio
        task["video_ratio"] = args.video_ratio

    bbox = task["bbox"]
    show = task.get("show", True)
    debug = task.get("debug", False)
    seed = task.get("seed", 0) if args.seed is None else args.seed
    methods = [canonical_method_name(m) for m in task.get(
        "method",
        ["our", "nvdiffrast", "video_nvdiffrast", "our_pytorch3d", "video_our_pytorch3d", "loir", "prdpt"],
    )]
    resolution = task["resolution"]
    highres_resolution = [512, 512]
    settings = task["setting"]

    # Make logger output independent of cwd
    results_dir = os.path.join(str(OUTPUTS_ROOT), "furniture")
    os.makedirs(results_dir, exist_ok=True)
    exp_name = task_name if args.tag is None else f"{task_name}_{args.tag}"
    Logger.init(exp_name=exp_name, show=show, debug=debug, path=results_dir)

    material = Materials(
        device=device,
        ambient_color=((0.5, 0.5, 0.5),),
        specular_color=((0.2, 0.2, 0.2),),
        diffuse_color=((0.7, 0.7, 0.7),),
    )

    testnum = 0
    with torch.no_grad():
        model = Furniture()
        gt_scene, task = model.load_furniture_scene(device, task, task["gt_scene_file"])
        sensors, torch_cameras, torch_lights = config_view_light(task, device)
        sensors, torch_cameras, torch_lights = keep_only_view_zero(sensors, torch_cameras, torch_lights)
        num_views = len(torch_cameras)
        gt_mesh = model.gen_mesh(optim=False)
        setup_seed(seed)
        gt_scene["material"] = material
        gt_scene["meshes"] = [{"model": gt_mesh}]
        gt_scene["sensors"] = sensors

        renderer = NVDiffRastFullRenderer(
            device=device,
            settings=task.get("renderer"),
            resolution=resolution,
        )
        highres_renderer = NVDiffRastFullRenderer(
            device=device,
            settings=task.get("renderer"),
            resolution=highres_resolution,
        )

        torch_gt_renderer, torch_sil_renderer, torch_soft_renderer, torch_point_renderer = get_pytorch3d_renderer(
            task.get("renderer").get("background", True),
            faces_per_pixel=4,
            resolution=resolution[0],
            persp=True,
        )

        gt_img = renderer.render(gt_scene, DcDt=False)
        gt_img_512 = highres_renderer.render(gt_scene, DcDt=False)
        if show:
            for i in range(num_views):
                show_img(gt_img["images"][i], title=str(i), flip=True)

        Logger.save_scene(gt_scene, name="%03d/scene_gt" % (testnum))
        gt_img_torch = [
            torch_gt_renderer(gt_scene["meshes"][0]["model"], cameras=torch_cameras[i], lights=torch_lights[i], materials=material)[0]
            for i in range(num_views)
        ]
        gt_sil_torch = [
            torch_sil_renderer(gt_scene["meshes"][0]["model"], cameras=torch_cameras[i], lights=torch_lights[i], materials=material)[0]
            for i in range(num_views)
        ]

        for i in range(num_views):
            Logger.save_img("%03d_gtimg_torch_%d.png" % (testnum, i), gt_img_torch[i].cpu().numpy(), flip=False)
            Logger.save_img("%03d_gtimg_nv_%d.png" % (testnum, i), gt_img["images"][i].cpu().numpy(), flip=True)

        Logger.save_img("ref.png", gt_img["images"][0].cpu().numpy(), flip=True)

    # Load initial (unoptimized) scene once and save init.png before any optimization
    with torch.no_grad():
        init_model = Furniture()
        init_scene, _ = init_model.load_furniture_scene(device, task, task["optim_scene_file"])
        init_scene["material"] = material
        init_scene["sensors"] = sensors
        init_mesh = init_model.gen_mesh(optim=True)
        init_scene["meshes"] = [{"model": init_mesh}]
        init_render = renderer.render(init_scene, DcDt=False)
    Logger.save_img("init.png", init_render["images"][0].cpu().numpy(), flip=True)

    method_times = {}
    for method in methods:
        setup_seed(seed)
        model = Furniture()
        scene, task = model.load_furniture_scene(device, task, task["optim_scene_file"])
        sensors, torch_cameras, torch_lights = config_view_light(task, device)
        sensors, torch_cameras, torch_lights = keep_only_view_zero(sensors, torch_cameras, torch_lights)
        num_views = len(torch_cameras)
        setup_seed(seed)
        scene["material"] = material
        scene["sensors"] = sensors
        optimizer_type = settings.get("optimizer", "Adam")

        gamma = settings["decay"]
        lr = settings["learning_rate"]
        fd_eps = settings.get("fd_eps", 0.03)
        prdpt_sigma = settings.get("prdpt_sigma", 0.08)
        prdpt_sigma_min = settings.get("prdpt_sigma_min", 0.01)
        prdpt_nsamples = settings.get("prdpt_nsamples", 8)
        if settings["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(scene["optim"], lr=lr)
        else:
            optimizer = torch.optim.SGD(scene["optim"], lr=lr, momentum=0.9)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=gamma, last_epoch=-1
        )

        view_per_iter = 1

        pbar = tqdm(range(settings["Niter"]))
        loss_func = PointLossFunction(
            debug=debug,
            resolution=task["resolution"],
            settings=task["matching"],
            device=device,
            renderer=renderer,
            num_views=num_views,
            logger=Logger,
        )

        # Build video scheduler for all video-guided methods
        video_scheduler = None
        if method in {"video_nvdiffrast", "video_our_pytorch3d"}:
            video_path = resolve_path(task.get("video_guide_path", ""))
            video_ratio = get_video_ratio(task)
            video_scheduler = VideoFrameScheduler(
                video_path=video_path,
                n_iter=settings["Niter"],
                resolution=highres_resolution if method == "video_nvdiffrast" else resolution,
                device=device,
                fallback=(gt_img_512 if method == "video_nvdiffrast" else gt_img)["images"][0],
                video_ratio=video_ratio,
            )

        loi_model = target_loi = None
        if method == "loir":
            loi_model = LOILoss(channels=3, device=device)
            with torch.no_grad():
                target_loi = loi_model(gt_img["images"][0][..., :3])

        smooth_render = None
        prdpt_ctx = None
        if method == "prdpt":
            prdpt_ctx = {
                "renderer": renderer,
                "model": model,
                "scene": scene,
                "gt_image": None,
                "sigma": prdpt_sigma,
                "nsamples": prdpt_nsamples,
                "antithetic": True,
                "sampler": "importance",
                "update_fn": None,
                "device": device,
                "view": 0,
            }
            smooth_render = smoothFn(render_smooth_furniture, context_args=None, device=device)

        start_time = time.time()
        trajectory_frames: list[tuple[np.ndarray, int]] = []
        trajectory_ref_frames: list[tuple[np.ndarray, int]] = []
        for iter in pbar:
            optim_mesh = model.gen_mesh(True)
            scene["meshes"] = [{"model": optim_mesh}]
            if method == methods[0] and iter == 0:
                Logger.save_scene(scene, name="%03d/scene_init" % (testnum))
                with torch.no_grad():
                    render_res = renderer.render(scene, DcDt=False)
                    for i in range(num_views):
                        Logger.save_img("%03d/init_img_nv_%d.png" % (testnum, i), render_res["images"][i].cpu().numpy(), flip=True)

            loss_all = torch.tensor(0.0, device=device)
            manual_grads = None
            debug_triptych = None
            for j in range(view_per_iter):
                select_view = 0
                if method == "our_torch":
                    render_res = torch_point_renderer.render(
                        scene["meshes"][0]["model"],
                        cameras=torch_cameras[select_view],
                        lights=torch_lights[select_view],
                        materials=material,
                    )
                    loss = loss_func.get_loss(render_res, gt_img_torch[select_view][..., :3], view=0)
                if method == "our":
                    loss, render_res = loss_func(gt_img, iteration=iter, scene=scene, view=select_view)
                elif method == "nvdiffrast":
                    render_res = renderer.render(scene, DcDt=True, view=select_view)
                    loss = torch.mean((render_res["images"][0] - gt_img["images"][select_view]) ** 2)
                elif method == "random":
                    x = random.randint(0, 1)
                    if x == 0:
                        loss, render_res = loss_func(gt_img, iteration=iter, scene=scene, view=select_view)
                    else:
                        render_res = renderer.render(scene, DcDt=True, view=select_view)
                        loss = torch.mean((render_res["images"][0] - gt_img["images"][select_view]) ** 2)
                elif method == "finetune":
                    if iter < settings["Niter"] * 0.75:
                        loss, render_res = loss_func(gt_img, iteration=iter, scene=scene, view=select_view)
                    else:
                        if iter == int(settings["Niter"] * 0.75):
                            optimizer = torch.optim.SGD(scene["optim"], lr=lr * (gamma ** iter))
                        render_res = renderer.render(scene, DcDt=True, view=select_view)
                        loss = torch.mean((render_res["images"][0] - gt_img["images"][select_view]) ** 2)
                elif method == "pytorch3d_rgb":
                    render_res = torch_soft_renderer(
                        scene["meshes"][0]["model"],
                        cameras=torch_cameras[select_view],
                        lights=torch_lights[select_view],
                        materials=material,
                    )
                    loss = torch.mean((render_res[0, ..., :3] - gt_img_torch[select_view][..., :3]) ** 2)
                elif method == "pytorch3d_sil_rgb":
                    render_res = torch_soft_renderer(
                        scene["meshes"][0]["model"],
                        cameras=torch_cameras[select_view],
                        lights=torch_lights[select_view],
                        materials=material,
                    )
                    loss = torch.mean((render_res[0, ..., :3] - gt_img_torch[select_view][..., :3]) ** 2)
                    loss += torch.mean((render_res[0, ..., 3] - gt_sil_torch[select_view][..., 3]) ** 2)
                elif method == "pytorch3d_sil":
                    render_res = torch_soft_renderer(
                        scene["meshes"][0]["model"],
                        cameras=torch_cameras[select_view],
                        lights=torch_lights[select_view],
                        materials=material,
                    )
                    loss = torch.mean((render_res[0, ..., 3] - gt_sil_torch[select_view][..., 3]) ** 2)
                elif method == "video_nvdiffrast":
                    ref = video_scheduler.get_ref(iter, view=select_view)
                    render_res = highres_renderer.render(scene, DcDt=True, view=select_view)
                    pred_ds = resize_hwc_bilinear(render_res["images"][0], tuple(resolution))
                    ref_ds = resize_hwc_bilinear(ref, tuple(resolution))
                    loss = gaussian_pyramid_loss(pred_ds[..., :3], ref_ds[..., :3])
                    debug_triptych = make_debug_triptych(
                        pred_ds,
                        ref_ds,
                        gt_img["images"][select_view],
                    )
                elif method == "our_pytorch3d":
                    loss, render_res = loss_func(gt_img, iteration=iter, scene=scene, view=select_view)
                    soft_res = torch_soft_renderer(
                        scene["meshes"][0]["model"],
                        cameras=torch_cameras[select_view],
                        lights=torch_lights[select_view],
                        materials=material,
                    )
                    loss += torch.mean((soft_res[0, ..., :3] - gt_img_torch[select_view][..., :3]) ** 2)
                    loss += torch.mean((soft_res[0, ..., 3]  - gt_sil_torch[select_view][..., 3])  ** 2)
                elif method == "video_our_pytorch3d":
                    ref = video_scheduler.get_ref(iter, view=select_view)
                    render_res = renderer.render(scene, DcDt=True, view=select_view)
                    loss = torch.mean((render_res["images"][0] - ref) ** 2)
                    soft_res = torch_soft_renderer(
                        scene["meshes"][0]["model"],
                        cameras=torch_cameras[select_view],
                        lights=torch_lights[select_view],
                        materials=material,
                    )
                    loss += torch.mean((soft_res[0, ..., :3] - ref[..., :3]) ** 2)
                    # silhouette only available from GT render
                    if iter >= video_scheduler.n_video_iters:
                        loss += torch.mean((soft_res[0, ..., 3] - gt_sil_torch[select_view][..., 3]) ** 2)
                elif method == "loir":
                    render_res = renderer.render(scene, DcDt=True, view=select_view)
                    est_loi = loi_model(render_res["images"][0][..., :3])
                    loss = LOILoss.emd(target_loi, est_loi)
                elif method == "prdpt":
                    base_values = pack_optim_values(scene).reshape(1, -1).requires_grad_(True)
                    prdpt_ctx["gt_image"] = gt_img["images"][select_view]
                    prdpt_ctx["view"] = select_view
                    if prdpt_sigma_min > 0:
                        frac = iter / max(settings["Niter"] - 1, 1)
                        prdpt_ctx["sigma"] = max(
                            prdpt_sigma_min,
                            prdpt_sigma * (1.0 - frac) + prdpt_sigma_min * frac,
                        )
                    loss, _ = smooth_render(base_values, prdpt_ctx)
                    loss.backward()
                    grad = base_values.grad.detach().reshape(-1)
                    if manual_grads is None:
                        manual_grads = torch.zeros_like(grad)
                    manual_grads += grad
                loss_all += loss
                with torch.no_grad():
                    render_res = renderer.render(scene, DcDt=False)
                    render_res_high = highres_renderer.render(scene, DcDt=False)
                    for i in range(num_views):
                        Logger.add_image("%03d_render_%s_%d" % (testnum, method, i), render_res["images"][i], flip=True)
                    trajectory_frames.append((_to_uint8_frame(render_res_high["images"][0], flip=True), iter))
                    if method in {"video_nvdiffrast", "video_our_pytorch3d"} and video_scheduler is not None:
                        ref_frame = video_scheduler.get_ref(iter, view=select_view)
                        trajectory_ref_frames.append((_to_uint8_frame(ref_frame, flip=True), iter))
                    if debug_triptych is not None:
                        Logger.add_image(
                            "%03d_debug_%s_triptych_%d" % (testnum, method, select_view),
                            debug_triptych,
                            flip=True,
                            resize=None,
                        )

            optimizer.zero_grad()
            loss_all /= view_per_iter
            if method in {"prdpt"}:
                assert manual_grads is not None
                manual_grads /= view_per_iter
                for idx, param in enumerate(scene["optim"]):
                    param.grad = manual_grads[idx].reshape_as(param)
            else:
                loss_all.backward()
            optimizer.step()
            if iter < settings["Niter"]:
                scheduler.step()

            MAE = torch.mean(torch.abs(gt_mesh.verts_list()[0] - optim_mesh.verts_list()[0])).item()
            pbar.set_description("%d %s MAE:%.4f" % (testnum, method, MAE))
            Logger.add_scalar("%03d_render_%s_mae" % (testnum, method), MAE, iter)

        end_time = time.time()
        method_times[method] = float(end_time - start_time)
        render_res = renderer.render(scene, DcDt=False)
        log_final(scene, gt_img, render_res, testnum, method, Logger, num_views, gt_mesh)
        method_dir = pathlib.Path(Logger.path) / method
        _save_video(method_dir / "trajectory.mp4", trajectory_frames, fps=12)
        if method in {"video_nvdiffrast", "video_our_pytorch3d"}:
            _save_video(method_dir / "trajectory_ref.mp4", trajectory_ref_frames, fps=12)
        Logger.clean()
        Logger.show_metric()
    results_path = pathlib.Path(results_dir)
    summary_path = write_explicit_timing_summary(results_path, method_times)
    print(f"Saved {summary_path}")
    if args.summary_name:
        summary_copy_path = results_path / args.summary_name
        summary_copy_path.write_text(summary_path.read_text())
        print(f"Saved {summary_copy_path}")
    Logger.exit()
