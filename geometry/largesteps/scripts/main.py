import torch
import time
import os
import cv2
from tqdm import tqdm
import numpy as np
import sys
from pathlib import Path

from largesteps.optimize import AdamUniform
from largesteps.geometry import compute_matrix, laplacian_uniform
from largesteps.parameterize import to_differential, from_differential
from scripts.render import NVDRenderer
from scripts.load_xml import load_scene
from scripts.constants import REMESH_DIR
from scripts.geometry import remove_duplicates, compute_face_normals, compute_vertex_normals, average_edge_length
sys.path.append(REMESH_DIR)
from pyremesh import remesh_botsch

def optimize_shape(filepath, params):
    """
    Optimize a shape given a scene.

    This will expect a Mitsuba scene as input containing the cameras, envmap and
    source and target models.

    Parameters
    ----------
    filepath : str Path to the XML file of the scene to optimize. params : dict
        Dictionary containing all optimization parameters.
    """
    opt_time = params.get("time", -1) # Optimization time (in minutes)
    steps = params.get("steps", 100) # Number of optimization steps (ignored if time > 0)
    step_size = params.get("step_size", 0.01) # Step size
    translation_step_size = params.get("translation_step_size", step_size) # Step size for global translation when enabled
    boost = params.get("boost", 1) # Gradient boost used in nvdiffrast
    smooth = params.get("smooth", True) # Use our method or not
    shading = params.get("shading", True) # Use shading, otherwise render silhouettes
    reg = params.get("reg", 0.0) # Regularization weight
    solver = params.get("solver", 'Cholesky') # Solver to use
    lambda_ = params.get("lambda", 1.0) # Hyperparameter lambda of our method, used to compute the parameterization matrix as (I + lambda_ * L)
    alpha = params.get("alpha", None) # Alternative hyperparameter, used to compute the parameterization matrix as ((1-alpha) * I + alpha * L)
    remesh = params.get("remesh", -1) # Time step(s) at which to remesh
    remesh_fn = params.get("remesh_fn", None) # Optional callable(v_cpu, f_cpu) -> (v_new, f_new); defaults to Botsch-Kobbelt
    optimizer = params.get("optimizer", AdamUniform) # Which optimizer to use
    use_tr = params.get("use_tr", False) # Optimize a global translation at the same time
    loss_function = params.get("loss", "l2") # Which loss to use
    bilaplacian = params.get("bilaplacian", True) # Use the bilaplacian or the laplacian regularization loss
    loss_view_idx = params.get("loss_view_idx", None) # Optional single-view loss index
    loss_view_indices = params.get("loss_view_indices", None) # Optional list of view indices to optimize jointly
    video_frames = params.get("video_frames", None) # Optional list of video target frames in linear/HDR space
    video_frames_per_view = params.get("video_frames_per_view", None) # Optional list of per-view frame lists
    video_ratio = params.get("video_ratio", 1.0) # Fraction of iterations using video targets when video_frames is provided
    save_render_interval = params.get("save_render_interval", 0) # Save intermediate renders every N iterations if > 0
    save_render_dir = params.get("save_render_dir", None) # Output directory for intermediate renders
    save_render_gamma = params.get("save_render_gamma", 2.2) # Gamma used when saving PNG renders
    if loss_view_indices is None and loss_view_idx is not None:
        loss_view_indices = [loss_view_idx]
    if loss_view_indices is not None and len(loss_view_indices) == 0:
        loss_view_indices = None
    save_render_view_idx = params.get(
        "save_render_view_idx",
        loss_view_indices[0] if loss_view_indices is not None else (loss_view_idx if loss_view_idx is not None else 0),
    ) # Which primary view to save
    save_render_view_indices = params.get(
        "save_render_view_indices",
        loss_view_indices if loss_view_indices is not None else [save_render_view_idx],
    ) # Which views to save during optimization
    black_background = params.get("black_background", False) # Render with black background

    # Load the scene
    scene_params = load_scene(filepath)

    # Load reference shape
    v_ref = scene_params["mesh-target"]["vertices"]
    f_ref = scene_params["mesh-target"]["faces"]
    if "normals" in scene_params["mesh-target"].keys():
        n_ref = scene_params["mesh-target"]["normals"]
    else:
        face_normals = compute_face_normals(v_ref, f_ref)
        n_ref = compute_vertex_normals(v_ref, f_ref, face_normals)

    # Load source shape
    v_src = scene_params["mesh-source"]["vertices"]
    f_src = scene_params["mesh-source"]["faces"]
    # Remove duplicates. This is necessary to avoid seams of meshes to rip apart during the optimization
    v_unique, f_unique, duplicate_idx = remove_duplicates(v_src, f_src)

    # Initialize the renderer
    renderer = NVDRenderer(scene_params, shading=shading, boost=boost, black_background=black_background)

    # Render the static reference images
    ref_imgs_full = renderer.render(v_ref, n_ref, f_ref)
    if save_render_interval > 0 and save_render_dir is not None:
        if save_render_view_idx < 0 or save_render_view_idx >= ref_imgs_full.shape[0]:
            raise ValueError(f"save_render_view_idx {save_render_view_idx} out of range [0, {ref_imgs_full.shape[0] - 1}]")
        Path(save_render_dir).mkdir(parents=True, exist_ok=True)
    if loss_view_indices is not None:
        ref_imgs = ref_imgs_full[loss_view_indices]
    else:
        ref_imgs = ref_imgs_full

    if loss_view_indices is not None:
        if save_render_view_idx not in loss_view_indices:
            raise ValueError(f"save_render_view_idx {save_render_view_idx} must be one of loss_view_indices {loss_view_indices}")
        save_render_local_indices = []
        for view_idx in save_render_view_indices:
            if view_idx < 0 or view_idx >= ref_imgs_full.shape[0]:
                raise ValueError(f"save_render_view_idx {view_idx} out of range [0, {ref_imgs_full.shape[0] - 1}]")
            if view_idx not in loss_view_indices:
                raise ValueError(f"save_render_view_idx {view_idx} must be one of loss_view_indices {loss_view_indices}")
            save_render_local_indices.append(loss_view_indices.index(view_idx))
    else:
        save_render_local_indices = []
        for view_idx in save_render_view_indices:
            if view_idx < 0 or view_idx >= ref_imgs_full.shape[0]:
                raise ValueError(f"save_render_view_idx {view_idx} out of range [0, {ref_imgs_full.shape[0] - 1}]")
            save_render_local_indices.append(view_idx)

    if video_frames is not None and len(video_frames) == 0:
        video_frames = None
    if video_frames_per_view is not None:
        video_frames_per_view = [frames for frames in video_frames_per_view if len(frames) > 0]
        if len(video_frames_per_view) == 0:
            video_frames_per_view = None

    active_video_source = video_frames_per_view if video_frames_per_view is not None else video_frames
    if active_video_source is not None:
        n_video_iters = int(max(0, min(steps, round(steps * float(video_ratio)))))
    else:
        n_video_iters = 0

    def _select_video_ref(iteration):
        if active_video_source is None or n_video_iters <= 0 or iteration >= n_video_iters:
            return None, -1
        t = iteration / max(n_video_iters - 1, 1)
        if video_frames_per_view is not None:
            refs = []
            frame_indices = []
            for frames in video_frames_per_view:
                if len(frames) == 1:
                    refs.append(frames[0])
                    frame_indices.append(0)
                else:
                    frame_idx = int(round(t * (len(frames) - 1)))
                    frame_idx = max(0, min(frame_idx, len(frames) - 1))
                    refs.append(frames[frame_idx])
                    frame_indices.append(frame_idx)
            return torch.stack(refs, dim=0), frame_indices
        if len(video_frames) == 1:
            return video_frames[0], 0
        frame_idx = int(round(t * (len(video_frames) - 1)))
        frame_idx = max(0, min(frame_idx, len(video_frames) - 1))
        return video_frames[frame_idx], frame_idx

    def _image_loss(pred, target, loss_kind):
        if pred.shape[-1] != target.shape[-1]:
            c = min(pred.shape[-1], target.shape[-1])
            pred = pred[..., :c]
            target = target[..., :c]
        if loss_kind == "l1":
            return (pred - target).abs().mean()
        if loss_kind == "l2":
            return (pred - target).square().mean()
        raise ValueError(f"Unsupported loss function: {loss_kind}")

    def _save_linear_png(path, image, gamma):
        save_img = (image[..., :3].clamp(0, 1).pow(1 / gamma).detach().cpu().numpy() * 255).astype(np.uint8)
        save_img = save_img[::-1, ..., ::-1]
        cv2.imwrite(str(path), save_img)

    # Compute the laplacian for the regularization term
    L = laplacian_uniform(v_unique, f_unique)

    # Initialize the optimized variables and the optimizer
    tr = torch.zeros((1,3), device='cuda', dtype=torch.float32)

    if smooth:
        # Compute the system matrix and parameterize
        M = compute_matrix(v_unique, f_unique, lambda_=lambda_, alpha=alpha)
        u_unique = to_differential(M, v_unique)

    def initialize_optimizer(u, v, tr, step_size, translation_step_size):
        """
        Initialize the optimizer

        Parameters
        ----------
        - u : torch.Tensor or None
            Parameterized coordinates to optimize if not None
        - v : torch.Tensor
            Cartesian coordinates to optimize if u is None
        - tr : torch.Tensor
            Global translation to optimize if not None
        - step_size : float
            Step size

        Returns
        -------
        a torch.optim.Optimizer containing the tensors to optimize.
        """
        opt_params = []
        if tr is not None:
            tr.requires_grad = True
            opt_params.append({"params": [tr], "lr": translation_step_size})
        if u is not None:
            u.requires_grad = True
            opt_params.append({"params": [u], "lr": step_size})
        else:
            v.requires_grad = True
            opt_params.append({"params": [v], "lr": step_size})

        return optimizer(opt_params, lr=step_size)

    opt = initialize_optimizer(
        u_unique if smooth else None,
        v_unique,
        tr if use_tr else None,
        step_size,
        translation_step_size,
    )

    # Set values for time and step count
    if opt_time > 0:
        steps = -1
    it = 0
    t0 = time.perf_counter()
    t = t0
    opt_time *= 60

    # Dictionary that is returned in the end, contains useful information for debug/analysis
    result_dict = {"vert_steps": [], "tr_steps": [], "f": [f_src.cpu().numpy().copy()],
                "losses": [], "im_ref": ref_imgs.cpu().numpy().copy(), "im":[],
                "v_ref": v_ref.cpu().numpy().copy(), "f_ref": f_ref.cpu().numpy().copy()}

    if isinstance(remesh, list):
        remesh_it = remesh.pop(0) if len(remesh) > 0 else -1
    elif remesh is None:
        remesh_it = -1
    else:
        remesh_it = remesh


    # Optimization loop
    with tqdm(total=max(steps, opt_time), ncols=100, bar_format="{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as pbar:
        while it < steps or (t-t0) < opt_time:
            if it == remesh_it:
                # Remesh
                with torch.no_grad():
                    if smooth:
                        v_unique = from_differential(M, u_unique, solver)

                    v_cpu = v_unique.cpu().numpy()
                    f_cpu = f_unique.cpu().numpy()
                    if remesh_fn is not None:
                        v_new, f_new = remesh_fn(v_cpu, f_cpu)
                    else:
                        # Target edge length
                        h = (average_edge_length(v_unique, f_unique)).cpu().numpy()*0.5
                        # Run 5 iterations of the Botsch-Kobbelt remeshing algorithm
                        v_new, f_new = remesh_botsch(v_cpu.astype(np.double), f_cpu.astype(np.int32), 5, h, True)

                    v_src = torch.from_numpy(v_new).cuda().float().contiguous()
                    f_src = torch.from_numpy(f_new).cuda().contiguous()

                    v_unique, f_unique, duplicate_idx = remove_duplicates(v_src, f_src)
                    result_dict["f"].append(f_new)
                    # Recompute laplacian
                    L = laplacian_uniform(v_unique, f_unique)

                    if smooth:
                        # Compute the system matrix and parameterize
                        M = compute_matrix(v_unique, f_unique, lambda_=lambda_, alpha=alpha)
                        u_unique = to_differential(M, v_unique)

                    step_size *= 0.8
                    opt = initialize_optimizer(
                        u_unique if smooth else None,
                        v_unique,
                        tr if use_tr else None,
                        step_size,
                        translation_step_size,
                    )

                # Get next remesh iteration if any
                if isinstance(remesh, list) and len(remesh) > 0:
                    remesh_it = remesh.pop(0)

            # Get cartesian coordinates
            if smooth:
                v_unique = from_differential(M, u_unique, solver)

            # Get the version of the mesh with the duplicates
            v_opt = v_unique[duplicate_idx]
            # Recompute vertex normals
            face_normals = compute_face_normals(v_unique, f_unique)
            n_unique = compute_vertex_normals(v_unique, f_unique, face_normals)
            n_opt = n_unique[duplicate_idx]

            # Render images
            opt_imgs_full = renderer.render(tr + v_opt, n_opt, f_src)

            video_ref, video_frame_idx = _select_video_ref(it)
            if video_ref is not None:
                ref_target = video_ref
                if loss_view_indices is not None:
                    opt_target = opt_imgs_full[loss_view_indices]
                    if isinstance(video_frame_idx, list):
                        frame_suffix = "-".join(f"{idx:06d}" for idx in video_frame_idx)
                    else:
                        frame_suffix = f"{video_frame_idx:06d}"
                    target_label = f"video_{frame_suffix}"
                else:
                    opt_target = opt_imgs_full[0]
                    target_label = f"video_{video_frame_idx:06d}"
            else:
                if loss_view_indices is not None:
                    ref_target = ref_imgs
                    opt_target = opt_imgs_full[loss_view_indices]
                else:
                    ref_target = ref_imgs
                    opt_target = opt_imgs_full
                target_label = "gt"

            if save_render_interval > 0 and save_render_dir is not None and (it % save_render_interval == 0):
                for save_view_idx, save_render_local_idx in zip(save_render_view_indices, save_render_local_indices):
                    render_path = Path(save_render_dir) / f"render_step_{it:06d}_view_{save_view_idx:03d}.png"
                    ref_path = Path(save_render_dir) / f"ref_step_{it:06d}_{target_label}_view_{save_view_idx:03d}.png"
                    pair_path = Path(save_render_dir) / f"pair_step_{it:06d}_{target_label}_view_{save_view_idx:03d}.png"

                    save_render = opt_imgs_full[save_view_idx]
                    if video_ref is not None:
                        save_ref = ref_target if ref_target.ndim == 3 else ref_target[save_render_local_idx]
                    else:
                        save_ref = ref_imgs_full[save_view_idx]
                    save_pred = opt_target if opt_target.ndim == 3 else opt_target[save_render_local_idx]
                    _save_linear_png(render_path, save_render, save_render_gamma)
                    _save_linear_png(ref_path, save_ref, save_render_gamma)
                    pair = torch.cat([save_ref[..., :3], save_pred[..., :3]], dim=1)
                    _save_linear_png(pair_path, pair, save_render_gamma)

            # Compute image loss
            im_loss = _image_loss(opt_target, ref_target, loss_function)

            # Add regularization
            if bilaplacian:
                reg_loss = (L@v_unique).square().mean()
            else:
                reg_loss = (v_unique * (L @v_unique)).mean()

            loss = im_loss + reg * reg_loss

            # Record optimization state for later processing
            result_dict["losses"].append((im_loss.detach().cpu().numpy().copy(), (L@v_unique.detach()).square().mean().cpu().numpy().copy()))
            result_dict["vert_steps"].append(v_opt.detach().cpu().numpy().copy())
            result_dict["tr_steps"].append(tr.detach().cpu().numpy().copy())

            # Backpropagate
            opt.zero_grad()
            loss.backward()
            # Update parameters
            opt.step()

            it += 1
            t = time.perf_counter()
            if steps > -1:
                pbar.update(1)
            else:
                pbar.update(min(opt_time, (t-t0)) - pbar.n)

    # Save a final render if the last iteration wasn't already captured by the interval.
    if save_render_interval > 0 and save_render_dir is not None and (it - 1) % save_render_interval != 0:
        with torch.no_grad():
            if smooth:
                v_unique = from_differential(M, u_unique, solver)
            v_opt_final = v_unique[duplicate_idx]
            face_normals_final = compute_face_normals(v_unique, f_unique)
            n_opt_final = compute_vertex_normals(v_unique, f_unique, face_normals_final)[duplicate_idx]
            final_imgs = renderer.render(tr + v_opt_final, n_opt_final, f_src)
        final_step = it - 1
        video_ref_final, video_frame_idx_final = _select_video_ref(final_step)
        if video_ref_final is not None:
            target_label_final = f"video_{video_frame_idx_final:06d}"
            ref_final = video_ref_final
        else:
            target_label_final = "gt"
            ref_final = ref_imgs_full[save_render_view_idx]
        for save_view_idx, save_render_local_idx in zip(save_render_view_indices, save_render_local_indices):
            render_path = Path(save_render_dir) / f"render_step_{final_step:06d}_view_{save_view_idx:03d}.png"
            ref_path = Path(save_render_dir) / f"ref_step_{final_step:06d}_{target_label_final}_view_{save_view_idx:03d}.png"
            pair_path = Path(save_render_dir) / f"pair_step_{final_step:06d}_{target_label_final}_view_{save_view_idx:03d}.png"
            _save_linear_png(render_path, final_imgs[save_view_idx], save_render_gamma)
            if video_ref_final is not None:
                save_ref = ref_final if ref_final.ndim == 3 else ref_final[save_render_local_idx]
            else:
                save_ref = ref_imgs_full[save_view_idx]
            _save_linear_png(ref_path, save_ref, save_render_gamma)
            pair = torch.cat([
                save_ref[..., :3],
                final_imgs[save_view_idx][..., :3],
            ], dim=1)
            _save_linear_png(pair_path, pair, save_render_gamma)

    result_dict["losses"] = np.array(result_dict["losses"])
    return result_dict
