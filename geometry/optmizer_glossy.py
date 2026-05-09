from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import time
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent
LARGESTEPS_DIR = ROOT_DIR / "largesteps"

for _p in (str(LARGESTEPS_DIR), str(ROOT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_constants = importlib.import_module("scripts.constants")
_geometry  = importlib.import_module("scripts.geometry")
_load_xml  = importlib.import_module("scripts.load_xml")
_render_gl = importlib.import_module("scripts.render_glossy")
_io_ply    = importlib.import_module("scripts.io_ply")

from largesteps.optimize     import AdamUniform
from largesteps.geometry     import compute_matrix, laplacian_uniform
from largesteps.parameterize import to_differential, from_differential

# Reuse utilities from optmizer.py (video loading, remesh helpers, save_image)
import optmizer as _base
_load_video_frames_ldr_to_hdr = _base._load_video_frames_ldr_to_hdr
_make_voxel_remesh_fn         = _base._make_voxel_remesh_fn
_save_image                   = _base._save_image
_save_trajectory_videos       = _base._save_trajectory_videos

sys.path.append(_constants.REMESH_DIR)
from pyremesh import remesh_botsch

SCENES_DIR             = _constants.SCENES_DIR
OUTPUT_DIR             = _constants.OUTPUT_DIR
compute_face_normals   = _geometry.compute_face_normals
compute_vertex_normals = _geometry.compute_vertex_normals
remove_duplicates      = _geometry.remove_duplicates
average_edge_length    = _geometry.average_edge_length
load_scene             = _load_xml.load_scene
NVDRendererGlossy      = _render_gl.NVDRendererGlossy
write_ply              = _io_ply.write_ply


def optimize_shape_glossy(filepath, params):
    """
    Joint optimization of geometry and a uniform roughness parameter.

    Mirrors optimize_shape() in scripts/main.py but uses NVDRendererGlossy
    and adds roughness as an optimizable nn.Parameter.

    Extra params keys vs the original:
        roughness           float   initial roughness in [0, 1]  (default 0.5)
        roughness_step_size float   learning rate for roughness  (default 0.01)
        sh_order            int     SH order for glossy lobe     (default 4)
        diffuse_weight      float   blend weight for diffuse SH  (default 0.5)
        specular_weight     float   blend weight for glossy SH   (default 0.5)

    Extra result_dict keys vs the original:
        roughness_steps     list[float]  roughness value at each iteration
        final_roughness     float        roughness at end of optimization
    """
    # --- unpack params (mirrors main.py nomenclature) ---
    opt_time              = params.get("time", -1)
    steps                 = params.get("steps", 100)
    step_size             = params.get("step_size", 0.01)
    translation_step_size = params.get("translation_step_size", step_size)
    boost                 = params.get("boost", 1)
    smooth                = params.get("smooth", True)
    reg                   = params.get("reg", 0.0)
    solver                = params.get("solver", "Cholesky")
    lambda_               = params.get("lambda", 1.0)
    alpha                 = params.get("alpha", None)
    remesh                = params.get("remesh", -1)
    remesh_fn             = params.get("remesh_fn", None)
    optimizer_cls         = params.get("optimizer", AdamUniform)
    use_tr                = params.get("use_tr", False)
    loss_function         = params.get("loss", "l2")
    bilaplacian           = params.get("bilaplacian", True)
    loss_view_idx         = params.get("loss_view_idx", None)
    loss_view_indices     = params.get("loss_view_indices", None)
    video_frames          = params.get("video_frames", None)
    video_frames_per_view = params.get("video_frames_per_view", None)
    video_ratio           = params.get("video_ratio", 1.0)
    save_render_interval  = params.get("save_render_interval", 0)
    save_render_dir       = params.get("save_render_dir", None)
    save_render_gamma     = params.get("save_render_gamma", 2.2)
    black_background      = params.get("black_background", False)
    # glossy-specific
    init_roughness   = params.get("roughness", 0.5)
    roughness_lr     = params.get("roughness_step_size", 0.01)
    sh_order         = params.get("sh_order", 4)
    diffuse_weight   = params.get("diffuse_weight", 0.5)
    specular_weight  = params.get("specular_weight", 0.5)
    roughness_target = params.get("roughness_target", init_roughness)

    if loss_view_indices is None and loss_view_idx is not None:
        loss_view_indices = [loss_view_idx]
    if loss_view_indices is not None and len(loss_view_indices) == 0:
        loss_view_indices = None

    save_render_view_idx = params.get(
        "save_render_view_idx",
        loss_view_indices[0] if loss_view_indices is not None
        else (loss_view_idx if loss_view_idx is not None else 0),
    )
    save_render_view_indices = params.get(
        "save_render_view_indices",
        loss_view_indices if loss_view_indices is not None else [save_render_view_idx],
    )

    # --- load scene and meshes ---
    scene_params = load_scene(filepath)

    v_ref = scene_params["mesh-target"]["vertices"]
    f_ref = scene_params["mesh-target"]["faces"]
    if "normals" in scene_params["mesh-target"]:
        n_ref = scene_params["mesh-target"]["normals"]
    else:
        n_ref = compute_vertex_normals(v_ref, f_ref, compute_face_normals(v_ref, f_ref))

    v_src = scene_params["mesh-source"]["vertices"]
    f_src = scene_params["mesh-source"]["faces"]
    v_unique, f_unique, duplicate_idx = remove_duplicates(v_src, f_src)

    # --- reference renders at target roughness (fixed, not optimized) ---
    renderer_ref = NVDRendererGlossy(
        scene_params,
        shading=True,
        boost=boost,
        black_background=black_background,
        roughness=float(roughness_target),
        sh_order=sh_order,
        diffuse_weight=diffuse_weight,
        specular_weight=specular_weight,
    )
    ref_imgs_full = renderer_ref.render(v_ref, n_ref, f_ref).detach()

    # --- roughness parameter (passed by reference into the optimization renderer) ---
    roughness = torch.nn.Parameter(
        torch.tensor(float(init_roughness), dtype=torch.float32, device="cuda")
    )

    renderer = NVDRendererGlossy(
        scene_params,
        shading=True,
        boost=boost,
        black_background=black_background,
        roughness=roughness,
        sh_order=sh_order,
        diffuse_weight=diffuse_weight,
        specular_weight=specular_weight,
    )
    if save_render_interval > 0 and save_render_dir is not None:
        if save_render_view_idx < 0 or save_render_view_idx >= ref_imgs_full.shape[0]:
            raise ValueError(f"save_render_view_idx {save_render_view_idx} out of range")
        Path(save_render_dir).mkdir(parents=True, exist_ok=True)

    ref_imgs = ref_imgs_full[loss_view_indices] if loss_view_indices is not None else ref_imgs_full

    if loss_view_indices is not None:
        save_render_local_indices = [loss_view_indices.index(v) for v in save_render_view_indices]
    else:
        save_render_local_indices = list(save_render_view_indices)

    # --- video frames ---
    if video_frames is not None and len(video_frames) == 0:
        video_frames = None
    if video_frames_per_view is not None:
        video_frames_per_view = [f for f in video_frames_per_view if len(f) > 0] or None

    active_video = video_frames_per_view if video_frames_per_view is not None else video_frames
    n_video_iters = int(max(0, min(steps, round(steps * video_ratio)))) if active_video else 0

    def _select_video_ref(it):
        if active_video is None or n_video_iters <= 0 or it >= n_video_iters:
            return None, -1
        t_frac = it / max(n_video_iters - 1, 1)
        if video_frames_per_view is not None:
            refs, idxs = [], []
            for frames in video_frames_per_view:
                i = 0 if len(frames) == 1 else int(round(t_frac * (len(frames) - 1)))
                refs.append(frames[i]); idxs.append(i)
            return torch.stack(refs), idxs
        i = 0 if len(video_frames) == 1 else int(round(t_frac * (len(video_frames) - 1)))
        return video_frames[i], i

    def _image_loss(pred, target):
        c = min(pred.shape[-1], target.shape[-1])
        pred, target = pred[..., :c], target[..., :c]
        if loss_function == "l1":
            return (pred - target).abs().mean()
        return (pred - target).square().mean()

    def _save_png(path, img, gamma):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        out = (img[..., :3].clamp(0, 1).pow(1 / gamma).detach().cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(path), out[::-1, ..., ::-1])

    # --- laplacian + differential parameterization ---
    L = laplacian_uniform(v_unique, f_unique)
    tr = torch.zeros((1, 3), device="cuda", dtype=torch.float32)

    if smooth:
        M = compute_matrix(v_unique, f_unique, lambda_=lambda_, alpha=alpha)
        u_unique = to_differential(M, v_unique)

    def _init_opt(u, v, tr, roughness, lr, tr_lr, r_lr):
        groups = []
        if tr is not None:
            tr.requires_grad_(True)
            groups.append({"params": [tr], "lr": tr_lr})
        if u is not None:
            u.requires_grad_(True)
            groups.append({"params": [u], "lr": lr})
        else:
            v.requires_grad_(True)
            groups.append({"params": [v], "lr": lr})
        roughness.requires_grad_(True)
        groups.append({"params": [roughness], "lr": r_lr})
        return optimizer_cls(groups, lr=lr)

    opt = _init_opt(
        u_unique if smooth else None, v_unique,
        tr if use_tr else None, roughness,
        step_size, translation_step_size, roughness_lr,
    )

    if opt_time > 0:
        steps = -1
    it = 0
    t0 = time.perf_counter()
    t  = t0
    opt_time *= 60

    if isinstance(remesh, list):
        remesh_it = remesh.pop(0) if remesh else -1
    elif remesh is None:
        remesh_it = -1
    else:
        remesh_it = remesh

    result_dict = {
        "vert_steps":     [],
        "tr_steps":       [],
        "roughness_steps": [],
        "f":              [f_src.cpu().numpy().copy()],
        "losses":         [],
        "im_ref":         ref_imgs.detach().cpu().numpy().copy(),
        "v_ref":          v_ref.detach().cpu().numpy().copy(),
        "f_ref":          f_ref.detach().cpu().numpy().copy(),
    }

    with tqdm(total=max(steps, opt_time), ncols=120,
              bar_format="{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as pbar:
        while it < steps or (t - t0) < opt_time:

            # --- remesh ---
            if it == remesh_it:
                with torch.no_grad():
                    if smooth:
                        v_unique = from_differential(M, u_unique, solver)
                    v_cpu = v_unique.cpu().numpy()
                    f_cpu = f_unique.cpu().numpy()
                    if remesh_fn is not None:
                        v_new, f_new = remesh_fn(v_cpu, f_cpu)
                    else:
                        h = average_edge_length(v_unique, f_unique).cpu().numpy() * 0.5
                        v_new, f_new = remesh_botsch(
                            v_cpu.astype(np.double), f_cpu.astype(np.int32), 5, h, True
                        )
                    v_src = torch.from_numpy(v_new).cuda().float().contiguous()
                    f_src = torch.from_numpy(f_new).cuda().contiguous()
                    v_unique, f_unique, duplicate_idx = remove_duplicates(v_src, f_src)
                    result_dict["f"].append(f_new)
                    L = laplacian_uniform(v_unique, f_unique)
                    if smooth:
                        M = compute_matrix(v_unique, f_unique, lambda_=lambda_, alpha=alpha)
                        u_unique = to_differential(M, v_unique)
                    step_size *= 0.8
                    # roughness survives remesh — only geometry optimizer is rebuilt
                    opt = _init_opt(
                        u_unique if smooth else None, v_unique,
                        tr if use_tr else None, roughness,
                        step_size, translation_step_size, roughness_lr,
                    )
                if isinstance(remesh, list) and remesh:
                    remesh_it = remesh.pop(0)

            # --- forward ---
            if smooth:
                v_unique = from_differential(M, u_unique, solver)

            v_opt = v_unique[duplicate_idx]
            face_normals = compute_face_normals(v_unique, f_unique)
            n_unique     = compute_vertex_normals(v_unique, f_unique, face_normals)
            n_opt        = n_unique[duplicate_idx]

            opt_imgs_full = renderer.render(tr + v_opt, n_opt, f_src)

            video_ref, video_frame_idx = _select_video_ref(it)
            if video_ref is not None:
                ref_target = video_ref
                opt_target = opt_imgs_full[loss_view_indices] if loss_view_indices else opt_imgs_full[0]
                if isinstance(video_frame_idx, list):
                    frame_suffix = "-".join(f"{i:06d}" for i in video_frame_idx)
                else:
                    frame_suffix = f"{video_frame_idx:06d}"
                target_label = f"video_{frame_suffix}"
            else:
                ref_target   = ref_imgs
                opt_target   = opt_imgs_full[loss_view_indices] if loss_view_indices else opt_imgs_full
                target_label = "gt"

            # --- save intermediate renders ---
            if save_render_interval > 0 and save_render_dir and it % save_render_interval == 0:
                p = Path(save_render_dir)
                for sv_idx, sl_idx in zip(save_render_view_indices, save_render_local_indices):
                    if video_ref is not None:
                        save_ref_img = ref_target if ref_target.ndim == 3 else ref_target[sl_idx]
                    else:
                        save_ref_img = ref_imgs_full[sv_idx]
                    save_pred = opt_target if opt_target.ndim == 3 else opt_target[sl_idx]
                    _save_png(p / f"render_step_{it:06d}_view_{sv_idx:03d}.png",
                              opt_imgs_full[sv_idx], save_render_gamma)
                    _save_png(p / f"ref_step_{it:06d}_{target_label}_view_{sv_idx:03d}.png",
                              save_ref_img, save_render_gamma)
                    _save_png(p / f"pair_step_{it:06d}_{target_label}_view_{sv_idx:03d}.png",
                              torch.cat([save_ref_img[..., :3], save_pred[..., :3]], dim=1),
                              save_render_gamma)

            # --- loss + backward ---
            im_loss  = _image_loss(opt_target, ref_target)
            reg_loss = (L @ v_unique).square().mean() if bilaplacian else (v_unique * (L @ v_unique)).mean()
            loss     = im_loss + reg * reg_loss

            result_dict["losses"].append((
                im_loss.detach().cpu().item(),
                (L @ v_unique.detach()).square().mean().cpu().item(),
            ))
            result_dict["vert_steps"].append(v_opt.detach().cpu().numpy().copy())
            result_dict["tr_steps"].append(tr.detach().cpu().numpy().copy())
            result_dict["roughness_steps"].append(roughness.detach().cpu().item())

            opt.zero_grad()
            loss.backward()
            opt.step()

            # keep roughness in a valid range
            with torch.no_grad():
                roughness.clamp_(0.01, 0.99)

            pbar.set_postfix(loss=f"{im_loss.item():.4f}", roughness=f"{roughness.item():.3f}")
            it += 1
            t = time.perf_counter()
            pbar.update(1 if steps > -1 else min(opt_time, t - t0) - pbar.n)

    # --- final render save if last iter wasn't already captured ---
    if save_render_interval > 0 and save_render_dir and (it - 1) % save_render_interval != 0:
        with torch.no_grad():
            if smooth:
                v_unique = from_differential(M, u_unique, solver)
            v_f  = v_unique[duplicate_idx]
            n_f  = compute_vertex_normals(v_unique, f_unique, compute_face_normals(v_unique, f_unique))[duplicate_idx]
            final_imgs = renderer.render(tr + v_f, n_f, f_src)
        fs = it - 1
        vref_fin, vf_idx_fin = _select_video_ref(fs)
        tlabel_fin = f"video_{vf_idx_fin:06d}" if vref_fin is not None else "gt"
        p = Path(save_render_dir)
        for sv_idx, sl_idx in zip(save_render_view_indices, save_render_local_indices):
            save_ref_fin = (vref_fin if vref_fin.ndim == 3 else vref_fin[sl_idx]) \
                if vref_fin is not None else ref_imgs_full[sv_idx]
            _save_png(p / f"render_step_{fs:06d}_view_{sv_idx:03d}.png",
                      final_imgs[sv_idx], save_render_gamma)
            _save_png(p / f"ref_step_{fs:06d}_{tlabel_fin}_view_{sv_idx:03d}.png",
                      save_ref_fin, save_render_gamma)
            _save_png(p / f"pair_step_{fs:06d}_{tlabel_fin}_view_{sv_idx:03d}.png",
                      torch.cat([save_ref_fin[..., :3], final_imgs[sv_idx][..., :3]], dim=1),
                      save_render_gamma)

    result_dict["losses"]         = np.array(result_dict["losses"])
    result_dict["final_roughness"] = roughness.detach().cpu().item()
    return result_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jointly optimize geometry and a uniform roughness using glossy SH rendering."
    )
    parser.add_argument("--scene-name", default="dragon")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--view-idx", type=int, default=1)
    parser.add_argument("--view-idxs", type=int, nargs="+", default=None)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--boost", type=float, default=3.0)
    parser.add_argument("--black-background", action="store_true")
    parser.add_argument("--optimize-translation", action="store_true")
    parser.add_argument("--steps", type=int, default=16000)
    parser.add_argument("--step-size", type=float, default=1e-1)
    parser.add_argument("--translation-step-size", type=float, default=None)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=19.0)
    parser.add_argument("--no-remesh", action="store_true")
    parser.add_argument("--remesh-interval", type=int, default=0)
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--video-paths", nargs="+", default=None)
    parser.add_argument("--video-ratio", type=float, default=0.9)
    parser.add_argument("--disable-video-remap", action="store_true")
    parser.add_argument("--video-limited-range", action="store_true")
    parser.add_argument("--video-width", type=int, default=None)
    parser.add_argument("--video-height", type=int, default=None)
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
    parser.add_argument("--remesh-method", choices=["botsch", "voxel"], default="botsch")
    parser.add_argument("--voxel-resolution", type=int, default=128)
    parser.add_argument("--render-interval", type=int, default=1000)
    parser.add_argument("--output-dir", default=None)
    # glossy-specific
    parser.add_argument("--roughness", type=float, default=0.5,
                        help="Initial roughness in [0, 1]. 0=mirror, 1=diffuse-like. Will be optimized.")
    parser.add_argument("--roughness-target", type=float, default=None,
                        help="Roughness used to render the reference images. Defaults to --roughness if not set.")
    parser.add_argument("--roughness-step-size", type=float, default=0.01,
                        help="Learning rate for the roughness parameter.")
    parser.add_argument("--sh-order", type=int, default=4,
                        help="SH order for the glossy lobe (max 4 with built-in basis).")
    parser.add_argument("--diffuse-weight", type=float, default=0.5,
                        help="Blend weight for the diffuse (low-order SH) term.")
    parser.add_argument("--specular-weight", type=float, default=0.5,
                        help="Blend weight for the glossy (high-order SH) term.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    scene_dir  = Path(args.scene_dir).resolve() if args.scene_dir else Path(SCENES_DIR) / args.scene_name
    scene_name = scene_dir.name
    scene_xml  = scene_dir / f"{scene_name}.xml"
    if not scene_xml.exists():
        raise FileNotFoundError(f"Scene XML not found: {scene_xml}")

    scene_params = load_scene(str(scene_xml))

    roughness_target = args.roughness_target if args.roughness_target is not None else args.roughness

    v_src = scene_params["mesh-source"]["vertices"]
    f_src = scene_params["mesh-source"]["faces"]
    n_src = compute_vertex_normals(v_src, f_src, compute_face_normals(v_src, f_src))
    v_ref = scene_params["mesh-target"]["vertices"]
    f_ref = scene_params["mesh-target"]["faces"]
    n_ref = compute_vertex_normals(v_ref, f_ref, compute_face_normals(v_ref, f_ref))

    renderer_init = NVDRendererGlossy(
        scene_params, shading=True, boost=args.boost,
        black_background=args.black_background,
        roughness=float(args.roughness),
        sh_order=args.sh_order,
        diffuse_weight=args.diffuse_weight,
        specular_weight=args.specular_weight,
    )
    renderer_tgt = NVDRendererGlossy(
        scene_params, shading=True, boost=args.boost,
        black_background=args.black_background,
        roughness=float(roughness_target),
        sh_order=args.sh_order,
        diffuse_weight=args.diffuse_weight,
        specular_weight=args.specular_weight,
    )
    init_imgs = renderer_init.render(v_src, n_src, f_src)
    ref_imgs  = renderer_tgt.render(v_ref, n_ref, f_ref)

    num_views    = int(ref_imgs.shape[0])
    view_indices = list(args.view_idxs) if args.view_idxs else [args.view_idx]
    for vi in view_indices:
        if vi < 0 or vi >= num_views:
            raise ValueError(f"view-idx {vi} out of range [0, {num_views - 1}]")

    # Video frames
    video_frames          = None
    video_frames_per_view = None
    if args.video_paths:
        if len(args.video_paths) != len(view_indices):
            raise ValueError("--video-paths must have one entry per --view-idxs entry.")
        if len(view_indices) == 1:
            video_width  = args.video_width  or int(scene_params["res_x"])
            video_height = args.video_height or int(scene_params["res_y"])
            video_frames = _load_video_frames_ldr_to_hdr(
                args.video_paths[0], (video_width, video_height),
                v_src.device, args.gamma,
                anchor_image=init_imgs[view_indices[0]],
                apply_first_frame_remap=not args.disable_video_remap,
                video_limited_range=args.video_limited_range,
            )
        else:
            video_frames_per_view = []
            for vp, vi in zip(args.video_paths, view_indices):
                video_width  = args.video_width  or int(scene_params["res_x"])
                video_height = args.video_height or int(scene_params["res_y"])
                video_frames_per_view.append(_load_video_frames_ldr_to_hdr(
                    vp, (video_width, video_height),
                    v_src.device, args.gamma,
                    anchor_image=init_imgs[vi],
                    apply_first_frame_remap=not args.disable_video_remap,
                    video_limited_range=args.video_limited_range,
                ))
    elif args.video_path:
        video_width  = args.video_width  or int(scene_params["res_x"])
        video_height = args.video_height or int(scene_params["res_y"])
        video_frames = _load_video_frames_ldr_to_hdr(
            args.video_path, (video_width, video_height),
            v_src.device, args.gamma,
            anchor_image=init_imgs[view_indices[0]],
            apply_first_frame_remap=not args.disable_video_remap,
            video_limited_range=args.video_limited_range,
        )

    view_tag    = "_".join(f"{vi:03d}" for vi in view_indices)
    default_tag = f"{scene_name}_view_{view_tag}" if len(view_indices) == 1 else f"{scene_name}_views_{view_tag}"
    out_dir = Path(args.output_dir).resolve() if args.output_dir else Path(OUTPUT_DIR) / (default_tag + "_glossy")

    for vi in view_indices:
        _save_image(out_dir / f"init_view_{vi:03d}.png", init_imgs[vi], gamma=args.gamma)
        _save_image(out_dir / f"ref_view_{vi:03d}.png",  ref_imgs[vi],  gamma=args.gamma)
        _save_image(out_dir / f"pair_view_{vi:03d}.png",
                    torch.cat([ref_imgs[vi], init_imgs[vi]], dim=1), gamma=args.gamma)

    if args.no_remesh:
        remesh_steps = []
    elif args.remesh_interval > 0:
        remesh_steps = list(range(args.remesh_interval, args.steps, args.remesh_interval))
    else:
        remesh_steps = [500, 1500, 3000, 4500, 7000, 10000, 12000, 14000]

    params = {
        "steps":               args.steps,
        "step_size":           args.step_size,
        "translation_step_size": args.translation_step_size or args.step_size,
        "loss":                "l1",
        "boost":               args.boost,
        "lambda":              args.lambda_,
        "remesh":              remesh_steps.copy(),
        "loss_view_idx":       view_indices[0] if len(view_indices) == 1 else None,
        "loss_view_indices":   view_indices if len(view_indices) > 1 else None,
        "save_render_interval": args.render_interval,
        "save_render_dir":     str(out_dir),
        "save_render_view_idx": view_indices[0],
        "save_render_view_indices": view_indices,
        "save_render_gamma":   args.gamma,
        "black_background":    args.black_background,
        "use_tr":              args.optimize_translation,
        # glossy
        "roughness":           args.roughness,
        "roughness_target":    roughness_target,
        "roughness_step_size": args.roughness_step_size,
        "sh_order":            args.sh_order,
        "diffuse_weight":      args.diffuse_weight,
        "specular_weight":     args.specular_weight,
    }
    if video_frames is not None:
        params["video_frames"] = video_frames
        params["video_ratio"]  = args.video_ratio
    if video_frames_per_view is not None:
        params["video_frames_per_view"] = video_frames_per_view
        params["video_ratio"]           = args.video_ratio
    if args.remesh_method == "voxel" and remesh_steps:
        params["remesh_fn"] = _make_voxel_remesh_fn(args.voxel_resolution)

    t_start = time.perf_counter()
    out = optimize_shape_glossy(str(scene_xml), params)
    elapsed = time.perf_counter() - t_start

    print(f"Final roughness: {out['final_roughness']:.4f}")

    max_step = len(out["vert_steps"]) - 1
    valid_remesh_steps = [s for s in remesh_steps if s <= max_step]
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
        "final_roughness": float(out["final_roughness"]),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved: {summary_path}")

    if args.save_trajectory_video:
        _save_trajectory_videos(out_dir, view_indices, fps=args.trajectory_fps)

    for vi in view_indices:
        print(f"Saved: {out_dir / f'init_view_{vi:03d}.png'}")
        print(f"Saved: {out_dir / f'ref_view_{vi:03d}.png'}")
        print(f"Saved: {out_dir / f'pair_view_{vi:03d}.png'}")
    print(f"Saved optimization outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
