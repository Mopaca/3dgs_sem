#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.dino_utils import DinoSimilarity ###
from utils.percentile_utils import get_priority_top_percent, get_dynamic_priority_top_percent ###
from utils.signal_maps import GuidanceMaps, SIGNALS ###
from utils.sem_logger import SemLogger ###
from utils.scale_weight_stats import scale_weight_analysis ###
import json ###
import time ###
import random ###
import subprocess ###
import numpy as np ###
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torchvision.utils as vutils
from utils.sh_utils import eval_sh

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, run_info=None):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    sem_cfg = resolve_sem_config(opt) ###

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    # Created at the same point as the previous DinoSimilarity, so a dino run consumes the RNG identically.
    guidance = GuidanceMaps([sem_cfg["signal"], sem_cfg["shadow"]], device="cuda", blur_sigma=opt.sem_blur_sigma, dino_chunk=opt.sem_dino_chunk) ###
    gaussians.training_setup(opt)
    sem_logger = SemLogger(scene.model_path, enabled=opt.sem_log) ###
    save_run_config(scene.model_path, dataset, opt, pipe, sem_cfg, run_info) ###
    priority_cutoff = None
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    white_background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # background = torch.tensor(white_background, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    torch.cuda.reset_peak_memory_stats() ###
    t_train_start = time.time() ###
    t_densify_end = None ###

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, similarity_aux=gaussians.get_similarity_aux)
        image, scalar_map, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["scalar_map"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"] ###

        gt_image = viewpoint_cam.original_image.cuda()

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

#####
        sim_map = None
        # do_similarity = iteration >= opt.sim_start_iter and iteration % opt.sim_interval == 0 and iteration < opt.densify_until_iter
        # do_similarity = iteration > 2900 and iteration < opt.densify_until_iter
        do_similarity = iteration > opt.densify_from_iter and iteration < opt.densify_until_iter
        if do_similarity:
            with torch.no_grad():
                gt_image = viewpoint_cam.original_image.cuda()
                # [1, H, W] goodness map (dino: same as the previous cosine_map call)
                sim_map = guidance.goodness(sem_cfg["signal"], image, gt_image)
                shadow_map = None
                if sem_cfg["shadow"] != "none":
                    shadow_map = guidance.goodness(sem_cfg["shadow"], image, gt_image)
                if sem_cfg["mode"] == "r9":
                    # [9]-style: back-project an error map instead of a goodness map
                    sim_map = 1.0 - sim_map
                    if shadow_map is not None:
                        shadow_map = 1.0 - shadow_map

            if scalar_map.dim() == 4:
                scalar_map = scalar_map.squeeze(0)
            if sim_map.dim() == 4:
                sim_map = sim_map.squeeze(0)

            aux_loss_num = (sim_map * scalar_map[0:1]).sum()
            aux_den = (torch.ones_like(scalar_map[1:2]) * scalar_map[1:2]).sum()
            aux_loss = aux_loss_num + aux_den
            if shadow_map is not None:
                # 3rd aux channel (otherwise unused): numerator of the shadow signal, same backward pass
                aux_loss = aux_loss + (shadow_map * scalar_map[2:3]).sum()

            aux_grad = torch.autograd.grad(outputs=aux_loss, inputs=gaussians.get_similarity_aux, retain_graph=True, create_graph=False, allow_unused=False)[0]
            num_grad = aux_grad[:, 0:1].detach()
            den_grad = aux_grad[:, 1:2].detach()

            if sem_cfg["view_norm"] == "mean":
                view_score = num_grad / (den_grad + 1e-8)     # contribution-weighted mean (proposed)
            else:
                view_score = num_grad                          # contribution-weighted sum ([9])

            if sem_cfg["mode"] == "ours":
                normalized_radius = normalize_projected_radius(radii=radii, min_radius=opt.sem_radius_min, max_radius=opt.sem_radius_max)
                lambda_scale = opt.sem_lambda_scale
                if opt.sem_scale_mode == "power":
                    # S~ = S^(1 + lambda_s * R): R=0 -> S, endpoints 0 and 1 fixed for every size
                    base = view_score.clamp(0.0, 1.0)
                    weighted_view_score = base.pow(1.0 + lambda_scale * normalized_radius)
                    # effective multiplier S~/S, used only for the diagnostics (sim_diag_accum)
                    scale_weight = torch.where(base > 1e-8, weighted_view_score / base.clamp_min(1e-8), torch.ones_like(base))
                else:
                    scale_weight = (1.0 - lambda_scale * normalized_radius)
                    weighted_view_score = (view_score * scale_weight)
                    weighted_view_score = weighted_view_score.clamp(0.0, 1.0)
            else:
                scale_weight = None
                weighted_view_score = view_score.clamp_min(0.0)  # [9] has no scale weight

            step = (iteration -1) % 100
            progress = step / 99.0

            wma_weight = 1.0 + progress

            view_visible_mask = visibility_filter.float()
            if view_visible_mask.dim() == 1:
                view_visible_mask = view_visible_mask.unsqueeze(1)

            visible_mask = view_visible_mask * (den_grad > 1e-6)
            # visible_mask = visible_mask * wma_weight

            gaussians.accumulate_similarity(weighted_view_score, visible_mask, agg=sem_cfg["view_agg"],
                                            radius=radii.detach().float().unsqueeze(1),
                                            scale_weight=scale_weight,
                                            raw_score=(view_score.clamp(0.0, 1.0) if sem_cfg["mode"] == "ours" else None))

            if sem_logger.enabled:
                shadow_view_score = None
                if shadow_map is not None:
                    shadow_num = aux_grad[:, 2:3].detach()
                    shadow_view_score = shadow_num / (den_grad + 1e-8) if sem_cfg["view_norm"] == "mean" else shadow_num
                sem_logger.observe_view(iteration, view_score, visible_mask, shadow_view_score)

        if iteration == opt.densify_until_iter:
            guidance.release()
            t_densify_end = time.time()
#####

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}", "Gaussians": f"{gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)
                    
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                ###
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.finalize_similarity_score(agg=sem_cfg["view_agg"]) ###
                    step_cfg = dict(sem_cfg["densify"]) ###
                    step_cfg["want_snapshot"] = sem_logger.wants_snapshot(iteration) ###
                    sw_summary = None ###
                    if sem_logger.enabled and sem_cfg["mode"] == "ours": ### scale-weight diagnostics (review item 4), before densification
                        sw_summary, sw_points = scale_weight_analysis(gaussians, radius_ref=opt.sem_radius_max)
                        if sw_points is not None and step_cfg["want_snapshot"]:
                            sem_logger.save_arrays("scale_weight", iteration, sw_points)
                    step_stats = gaussians.densify_and_prune_by_similarity(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii, iteration, model_path=scene.model_path, sem_cfg=step_cfg)
                    # opacity reset runs after densification in the same iteration, so the last reset before this step is:
                    last_reset = ((iteration - 1) // opt.opacity_reset_interval) * opt.opacity_reset_interval ###
                    sem_logger.log_step(iteration, step_stats, extra={ ###
                        "iters_since_opacity_reset": (iteration - last_reset) if last_reset > 0 else None, ###
                        "scale_weight": sw_summary}) ###
                    if sem_logger.enabled and iteration % 1000 == 0: ### one console line per 1000 iterations
                        progress_bar.write(sem_console_line(iteration, step_stats, sw_summary))
                    gaussians.reset_similarity_accum() ###
                    gaussians.reset_similarity_aux() ###
                ###

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        
        if do_similarity: ###
            gaussians.reset_similarity_aux() ###

    write_train_stats(scene.model_path, gaussians, opt, t_train_start, t_densify_end) ###

###
SEM_MODES = ("ours", "r9")
SEM_SELECTS = ("stratified", "random_matched", "radius_matched")
SEM_SPLITS = ("adc", "mask2")


def resolve_sem_config(opt):
    """Validate the sem_* options and resolve the 'auto' aggregation settings."""
    mode = opt.sem_mode
    if mode not in SEM_MODES:
        raise ValueError(f"--sem_mode must be one of {SEM_MODES}, got '{mode}'")
    if opt.sem_signal not in SIGNALS:
        raise ValueError(f"--sem_signal must be one of {SIGNALS}, got '{opt.sem_signal}'")
    if opt.sem_shadow != "none" and opt.sem_shadow not in SIGNALS:
        raise ValueError(f"--sem_shadow must be 'none' or one of {SIGNALS}, got '{opt.sem_shadow}'")
    if opt.sem_select not in SEM_SELECTS:
        raise ValueError(f"--sem_select must be one of {SEM_SELECTS}, got '{opt.sem_select}'")
    if opt.sem_split not in SEM_SPLITS:
        raise ValueError(f"--sem_split must be one of {SEM_SPLITS}, got '{opt.sem_split}'")

    view_norm = opt.sem_view_norm if opt.sem_view_norm != "auto" else ("mean" if mode == "ours" else "sum")
    view_agg = opt.sem_view_agg if opt.sem_view_agg != "auto" else ("mean" if mode == "ours" else "max")
    if view_norm not in ("mean", "sum"):
        raise ValueError(f"--sem_view_norm must be auto|mean|sum, got '{opt.sem_view_norm}'")
    if opt.sem_scale_mode not in ("linear", "power"):
        raise ValueError(f"--sem_scale_mode must be linear|power, got '{opt.sem_scale_mode}'")

    if mode == "ours":
        # A summed goodness score grows with coverage, i.e. large Gaussians would look "well reconstructed".
        if view_norm != "mean":
            raise ValueError("ours mode uses a goodness score; within-view 'sum' is only meaningful for an error score (use --sem_mode r9)")
        if view_agg not in ("mean", "min"):
            raise ValueError("ours mode supports --sem_view_agg mean|min (min = worst view of a goodness score)")
    else:
        if view_agg not in ("mean", "max"):
            raise ValueError("r9 mode supports --sem_view_agg mean|max (max = worst view of an error score)")
        if opt.sem_select != "stratified":
            raise ValueError("--sem_select applies to ours mode only; r9 mode uses the threshold + cap rule")
        if opt.sem_lambda_scale != 0.0:
            print(f"[sem] r9 mode: scale weight is not used (sem_lambda_scale={opt.sem_lambda_scale} ignored)")

    densify_cfg = {
        "mode": mode,
        "select": opt.sem_select,
        "sample_ratio": opt.sem_sample_ratio,
        "r9_threshold": opt.sem_r9_threshold,
        "r9_cap": opt.sem_r9_cap,
        "split": opt.sem_split,
        "radius_ref": opt.sem_radius_max,
        "collect_stats": bool(opt.sem_log),
    }
    cfg = {"mode": mode, "signal": opt.sem_signal, "shadow": opt.sem_shadow,
           "view_norm": view_norm, "view_agg": view_agg, "scale_mode": opt.sem_scale_mode, "densify": densify_cfg}
    print(f"[sem] config: {json.dumps(cfg)}")
    return cfg


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def save_run_config(model_path, dataset, opt, pipe, sem_cfg, run_info):
    """cfg_args only stores ModelParams, so the full option set is written here."""
    record = {
        "dataset": vars(dataset),
        "opt": vars(opt),
        "pipe": vars(pipe),
        "sem_resolved": sem_cfg,
        "run_info": run_info or {},
        "git_commit": _git_commit(),
    }
    with open(os.path.join(model_path, "sem_run_config.json"), "w") as f:
        json.dump(record, f, indent=2, default=str)


def write_train_stats(model_path, gaussians, opt, t_train_start, t_densify_end):
    t_end = time.time()
    stats = {
        "wall_time_min": (t_end - t_train_start) / 60.0,
        "densify_phase_min": ((t_densify_end - t_train_start) / 60.0) if t_densify_end else None,
        "peak_vram_allocated_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "peak_vram_reserved_gb": torch.cuda.max_memory_reserved() / 1024 ** 3,
        "num_gaussians": int(gaussians.get_xyz.shape[0]),
        "iterations": int(opt.iterations),
        "note": "wall time includes any test evaluation and saving done during training",
    }
    with open(os.path.join(model_path, "train_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[sem] train stats: {json.dumps(stats)}")


def sem_console_line(iteration, stats, sw_summary):
    """Short progress line of the semantic selection (printed through tqdm, does not break the bar)."""
    if not stats:
        return f"[sem {iteration}] no statistics"
    parts = [f"N={stats['n_total'] / 1e6:.2f}M", f"grad={stats['n_grad']}",
             f"sem={stats['n_sem']} (sem-only {stats['n_sem_only']})"]
    if "interval_counts" in stats:
        parts.append("G1/G2/G3=" + "/".join(str(c) for c in stats["interval_counts"]) +
                     " K=" + "/".join(str(k) for k in stats["interval_samples"]))
    if "r9_n_candidates" in stats:
        parts.append(f"r9 candidates={stats['r9_n_candidates']} cap={'hit' if stats.get('r9_cap_bound') else 'no'}")
    frac = (stats.get("radius_sem") or {}).get("frac_ge_ref")
    if frac is not None:
        parts.append(f"M_sem r>=R_max {100 * frac:.0f}%")
    if sw_summary and sw_summary.get("g3_only_by_scale_weight") is not None:
        parts.append(f"G3 only by scale weight {100 * sw_summary['g3_only_by_scale_weight']:.0f}%")
    return f"[sem {iteration}] " + " | ".join(parts)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

###
def normalize_projected_radius(
    radii,
    min_radius=1.0,
    max_radius=20.0,
    eps=1e-8
):
    """
    radii: [N], current-view projected Gaussian radius

    Returns:
        normalized_radius: [N, 1], range [0, 1]
    """
    normalized_radius = (
        (radii.detach().float() - min_radius)
        / (max_radius - min_radius + eps)
    )

    normalized_radius = normalized_radius.clamp(0.0, 1.0)

    return normalized_radius.unsqueeze(1)
###

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--seed", type=int, default=0) ### seed 0 reproduces the previous fixed-seed behaviour
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    set_seed(args.seed) ### safe_state always seeds 0; re-seed here (nothing consumes the RNG in between)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    run_info = {"seed": args.seed, "argv": sys.argv} ###
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, run_info=run_info)

    # All done
    print("\nTraining complete.")
