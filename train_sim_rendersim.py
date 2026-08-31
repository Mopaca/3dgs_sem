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
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    dino_sim = DinoSimilarity(model_name="dinov2_vits14", device="cuda")
    gaussians.training_setup(opt)
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
                img_bchw = image.unsqueeze(0)
                gt_bchw = gt_image.unsqueeze(0)
                sim_map = dino_sim.cosine_map(img_bchw, gt_bchw).squeeze(0).detach()

                # sim_map, view_sim_mean, view_sim_lower = stretch_similarity_map_below_mean(sim_map, lower_quantile=0.05)

            if scalar_map.dim() == 4:
                scalar_map = scalar_map.squeeze(0)
            if sim_map.dim() == 4:
                sim_map = sim_map.squeeze(0)

            # view_sim_root = os.path.join(scene.model_path, "similarity heatmap")
            # os.makedirs(view_sim_root, exist_ok=True)
            # view_sim_heatmap = similarity_map_to_heatmap(sim_map, value_min=0.0, value_max=1.0)
            # vutils.save_image(view_sim_heatmap, os.path.join(view_sim_root, f"iter_{iteration:06d}.png"))

            aux_loss_num = (sim_map * scalar_map[0:1]).sum()
            aux_den = (torch.ones_like(scalar_map[1:2]) * scalar_map[1:2]).sum()
            aux_loss = aux_loss_num + aux_den

            aux_grad = torch.autograd.grad(outputs=aux_loss, inputs=gaussians.get_similarity_aux, retain_graph=True, create_graph=False, allow_unused=False)[0]
            num_grad = aux_grad[:, 0:1].detach()
            den_grad = aux_grad[:, 1:2].detach()

            view_score = num_grad / (den_grad + 1e-8)
            normalized_radius = normalize_projected_radius(radii=radii, min_radius=1.0, max_radius=70.0)
            lambda_scale = 0.1
            scale_weight = (1.0 - lambda_scale * normalized_radius)
            weighted_view_score = (view_score * scale_weight)
            weighted_view_score = weighted_view_score.clamp(0.0, 1.0)

            view_visible_mask = visibility_filter.float()
            if view_visible_mask.dim() == 1:
                view_visible_mask = view_visible_mask.unsqueeze(1)
            
            visible_mask = view_visible_mask
            visible_mask = view_visible_mask * (den_grad > 1e-6)

            gaussians.accumulate_similarity(weighted_view_score, visible_mask)
            # save_similarity_statistics_json(scene, gaussians, iteration, weighted_view_score, visible_mask)

        if iteration == opt.densify_until_iter:
            del dino_sim
            torch.cuda.empty_cache()
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

        scaling = gaussians.get_scaling
        min_scale = scaling.min(dim=1).values
        max_scale = scaling.max(dim=1).values
        scale_ratio = min_scale / (max_scale + 1e-8)
        needle_threshold = 0.1
        needle_mask = scale_ratio < needle_threshold

        if needle_mask.any():
            needle_loss = (1 - scale_ratio[needle_mask]).mean()
        else:
            needle_loss = scaling.sum() * 0.0

        lambda_needle = 0.1
        # loss += lambda_needle * needle_loss

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
                    progress = (iteration - 6000) / float(15000 - 6000)
                    progress = max(0.0, min(1.0, progress))

                    current_grad_threshold = opt.densify_grad_threshold * (1.0 + progress)
                    size_threshold = 100 if iteration > opt.opacity_reset_interval else None
                    gaussians.finalize_similarity_score() ###
                    # save_final_similarity_comparison_json(scene, gaussians, iteration)
                    viewpoint_stack1 = scene.getTrainCameras().copy()
                    similarity_mask, densify_mask, mask3 = get_scale_mask(gaussians=gaussians, iteration=iteration)
                    similarity_colors = selected_gaussians_to_color(similarity_mask, densify_mask, mask3, valid_mask=((gaussians.sim_view_count > 0) & (gaussians.sim_score_accum > 0)))
                    similarity_heat_color  = similarity_to_heat_colors(similarity_score=gaussians.get_similarity_score, valid_mask=((gaussians.sim_view_count > 0) & (gaussians.sim_score_accum > 0)))
                    # grad_color = grad_to_heat_colors(grad_accum=gaussians.xyz_gradient_accum / (gaussians.denom + 1e-8))
                    grad_color = selected_gaussians_to_color((gaussians.max_radii2D > 50), (gaussians.max_radii2D > 70), (gaussians.max_radii2D > 80), valid_mask=((gaussians.sim_view_count > 0) & (gaussians.sim_score_accum > 0)))
                    # similarity_heat_color = priority_to_heat_colors(gaussians.sim_view_count, (gaussians.sim_view_count.squeeze(1) > 0))
                    # similarity_heat_color = priority_to_heat_colors(gaussians.get_similarity_score, ((gaussians.sim_view_count.squeeze(1) > 0) & (gaussians.get_similarity_score > 0.0)))
                    os.makedirs(os.path.join(scene.model_path, "test render", "original render"), exist_ok=True)
                    os.makedirs(os.path.join(scene.model_path, "test render", "similarity part render"), exist_ok=True)
                    os.makedirs(os.path.join(scene.model_path, "test render", "similarity render"), exist_ok=True)
                    os.makedirs(os.path.join(scene.model_path, "test render", "grad render"), exist_ok=True)
                    render_pkg = render(viewpoint_stack1[47], gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, similarity_aux=gaussians.get_similarity_aux)
                    vutils.save_image(render_pkg["render"], os.path.join(scene.model_path, "test render", "original render", f"iter_{iteration:06d}.png"))
                    render_pkg = render(viewpoint_stack1[47], gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, similarity_aux=gaussians.get_similarity_aux, override_color=similarity_colors)
                    vutils.save_image(render_pkg["render"], os.path.join(scene.model_path, "test render", "similarity part render", f"iter_{iteration:06d}.png"))
                    render_pkg = render(viewpoint_stack1[47], gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, similarity_aux=gaussians.get_similarity_aux, override_color=similarity_heat_color)
                    vutils.save_image(render_pkg["render"], os.path.join(scene.model_path, "test render", "similarity render", f"iter_{iteration:06d}.png"))
                    render_pkg = render(viewpoint_stack1[47], gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, similarity_aux=gaussians.get_similarity_aux, override_color=grad_color)
                    vutils.save_image(render_pkg["render"], os.path.join(scene.model_path, "test render", "grad render", f"iter_{iteration:06d}.png"))
                    save_similarity_range_statistics(gaussians=gaussians, iteration=iteration, range_mask_1=similarity_mask, range_mask_2=densify_mask, range_mask_3=mask3, save_dir=scene.model_path)
                    if (iteration in testing_iterations):
                        save_all_views_priority_similarity_renders(scene=scene, gaussians=gaussians, pipe=pipe, background=background, dataset=dataset, opt=opt, iteration=iteration, current_top_percent=0.0)
                        # save_similarity_scale_statistics_json(scene=scene, gaussians=gaussians, iteration=iteration)
                    # gaussians.densify_and_prune_by_similarity(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii, iteration, model_path=scene.model_path)
                    gaussians.densify_and_prune_by_similarity(current_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii, iteration, model_path=scene.model_path)
                    # gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                    gaussians.reset_similarity_accum() ###
                    gaussians.reset_similarity_aux() ###
                ###
                
                # ###
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                #     size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                #     gaussians.finalize_similarity_score() ###
                #     gaussians.finalize_norm_ratio_score() ###
                #     print("similarity mean:", gaussians.get_similarity_score.mean(), "similarity std:", gaussians.get_similarity_score.std(unbiased=False))
                #     if (iteration in saving_iterations):
                #         save_all_views_priority_similarity_renders(scene=scene, gaussians=gaussians, pipe=pipe, background=background, dataset=dataset, opt=opt, iteration=iteration, dino_sim=dino_sim, current_top_percent=0.0)
                #         save_score_statistics_json(scene=scene, gaussians=gaussians, opt=opt, iteration=iteration, priority_cutoff=priority_cutoff)
                #     gaussians.reset_similarity_accum() ###
                #     gaussians.reset_similarity_aux() ###

                #     gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                # ###

                # ###
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                #     size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                #     if iteration > opt.sim_start_iter: ###
                #         gaussians.finalize_similarity_score() ###
                #         gaussians.finalize_norm_ratio_score() ###
                #         current_top_percent = gaussians.get_dynamic_similarity_threshold(current_iteration=iteration, densify_until_iter=opt.densify_until_iter, start_threshold=0.4, end_threshold=0.4)
                #         if (iteration in testing_iterations):
                #             save_all_views_priority_similarity_renders(scene=scene, gaussians=gaussians, pipe=pipe, background=background, dataset=dataset, opt=opt, iteration=iteration, dino_sim=dino_sim, current_top_percent=current_top_percent)
                #             save_score_statistics_json(scene=scene, gaussians=gaussians, opt=opt, iteration=iteration, priority_cutoff=priority_cutoff)
                #         # current_percentile, priority_cutoff, reset_opacity_bool = gaussians.densify_and_prune_by_priority(iteration=iteration, densify_until_iter=opt.densify_until_iter, adc_grad_threshold=opt.densify_grad_threshold, min_opacity=0.005, extent=scene.cameras_extent, max_screen_size=size_threshold, radii=radii, top_percent=0.03, alpha=0.8, beta=0.2, grad_top_percent=0.02, reset_opacity=reset_opacity, split_N=2) ###
                        
                #         gaussians.densify_and_prune_by_similarity(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii, iteration)
                #         gaussians.reset_similarity_accum() #

                #     else: ###
                #         if (iteration in testing_iterations):
                #             # gaussians.finalize_norm_ratio_score() ###
                #             gaussians.finalize_similarity_score() ###
                #             save_all_views_priority_similarity_renders(scene=scene, gaussians=gaussians, pipe=pipe, background=background, dataset=dataset, opt=opt, iteration=iteration, dino_sim=dino_sim, current_top_percent=0.0)
                #             # save_score_statistics_json(scene=scene, gaussians=gaussians, opt=opt, iteration=iteration, priority_cutoff=priority_cutoff)
                #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii) ###
                #         gaussians.reset_similarity_accum() ###
                # ###

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

####
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

def stretch_similarity_map_below_mean(
    sim_map,
    lower_quantile=0.05,
    eps=1e-8
):
    """
    View-level similarity map을 상대적 점수로 변환.

    - view 평균 이상: 1
    - view 평균 미만: lower_quantile 값부터 평균까지 0~1로 stretch
    - lower reference 이하: 0

    Args:
        sim_map:
            [H, W], [1, H, W], 또는 유사한 형태의 Tensor

        lower_quantile:
            하위 기준값을 정하는 분위수.
            0.05이면 하위 5% 값을 0 기준으로 사용.

    Returns:
        transformed_map:
            sim_map과 같은 shape, 값 범위 [0, 1]

        sim_mean:
            원본 map 평균

        lower_ref:
            stretch의 하한 기준값
    """

    sim = sim_map.detach().float()

    sim_mean = sim.mean()
    lower_ref = torch.quantile(
        sim.reshape(-1),
        lower_quantile
    )

    below_mean_score = (
        (sim - lower_ref)
        / (sim_mean - lower_ref + eps)
    ).clamp(0.0, 1.0)

    transformed_map = torch.where(
        sim >= sim_mean,
        torch.ones_like(sim),
        below_mean_score
    )

    return transformed_map, sim_mean, lower_ref

def sort_cameras_stably(cameras):
    return sorted(cameras, key=lambda c: c.image_name)

def get_consistent_view_configs(scene, mode="render_py"):
    """
    mode:
      - "render_py": render.py와 동일한 view ordering
      - "training_report": training_report와 동일한 view ordering
    """
    if mode == "render_py":
        return (
            {"name": "train", "views": sort_cameras_stably(scene.getTrainCameras())},
            {"name": "test", "views": sort_cameras_stably(scene.getTestCameras())},
        )

    elif mode == "training_report":
        return (
            {"name": "test", "views": scene.getTestCameras()},
            {
                "name": "train",
                "views": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

    else:
        raise ValueError(f"Unsupported mode: {mode}")

def get_scale_mask(gaussians, iteration):
    sim_score = gaussians.get_similarity_score.detach().squeeze(1)
    sim_valid_mask = ((gaussians.sim_view_count.detach().squeeze(1) > 0) & (sim_score > 0.0))

    if not sim_valid_mask.any():
        empty_mask = torch.zeros_like(
            sim_score,
            dtype=torch.bool
        )
        return empty_mask, empty_mask.clone()

    sim_valid_score = sim_score[sim_valid_mask]
    sim_mean = sim_valid_score.mean()
    sim_std = sim_valid_score.std(unbiased=False)

    densify_lower = sim_mean - 3.0 * sim_std
    densify_upper = sim_mean - sim_std

    progress = (iteration - 6000) / float(15000 - 6000)
    progress = max(0.0, min(1.0, progress))

    kappa = 1.0 + 2.0 * progress

    similarity_threshold = (
        sim_mean - kappa * sim_std
    )

    low_boundary  = max(0.0,(sim_mean - 3.0 * sim_std).item())
    prune_threshold = low_boundary * progress

    # semantic_candidate_mask = (sim_valid_mask & (sim_score >= similarity_threshold2) & (sim_score < similarity_threshold))
    semantic_candidate_mask2 = (sim_valid_mask & (sim_score >= densify_lower) & (sim_score < sim_mean - 2.0 * sim_std))
    semantic_candidate_mask = (sim_valid_mask & (sim_score >= sim_mean - 2.0 * sim_std) & (sim_score < densify_upper))
    semantic_candidate_mask3 = (sim_valid_mask & (sim_score > 0) & (sim_score < densify_lower) & (sim_score > prune_threshold))

    candidate_indices = torch.nonzero(semantic_candidate_mask, as_tuple=False).squeeze(1)

    semantic_densify_mask = torch.zeros_like(sim_score, dtype=torch.bool)
    candidate_xyz = gaussians.get_xyz.detach()[candidate_indices]
    num_candidates = candidate_xyz.shape[0] ###

    sample_ratio = 0.01   # 10%
    num_sample = max(1, int(num_candidates * sample_ratio))
    num_sample = min(num_sample, num_candidates)

    # gaussian_scale = torch.max(gaussians.get_scaling.detach(), dim=1).values
    gaussian_scale = torch.prod(gaussians.get_scaling.detach(), dim=1)
    candidate_scales = gaussian_scale[candidate_indices]
    _, top_idx_in_candidates = torch.topk(candidate_scales, k=num_sample, largest=True, sorted=False)
    selected_indices = candidate_indices[top_idx_in_candidates]
    semantic_densify_mask[selected_indices] = True

    # return semantic_candidate_mask, semantic_densify_mask
    return semantic_candidate_mask, semantic_candidate_mask2, semantic_candidate_mask3

def selected_gaussians_to_color(candidate_mask, selected_mask, candidate_mask2, valid_mask=None, device="cuda"):
    mask1 = candidate_mask.detach().reshape(-1).bool()
    mask2 = selected_mask.detach().reshape(-1).bool()
    mask3 = candidate_mask2.detach().reshape(-1).bool()

    if valid_mask.dim() == 2:
        valid = valid_mask.squeeze(1).bool()
    if valid_mask is None:
        valid = torch.ones_like(mask2)
    else:
        valid = valid_mask.bool()


    colors = torch.zeros((mask2.shape[0], 3), dtype=torch.float32, device=device)
    colors[mask1] = torch.tensor([1.0, 1.0, 0.0],  dtype=torch.float32, device=device) # yellow
    colors[mask2] = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32, device=device) # orange
    colors[mask3] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device) # red
    colors[(~(mask1 | mask2 | mask3))] = 0.5 # gray

    return colors

def save_similarity_range_statistics(
    gaussians,
    iteration,
    range_mask_1,
    range_mask_2,
    range_mask_3,
    save_dir
):

    os.makedirs(save_dir, exist_ok=True)

    json_path = os.path.join(
        save_dir,
        "similarity_range_statistics.json"
    )

    sim_score = (
        gaussians.get_similarity_score
        .detach()
        .squeeze(1)
    )

    sim_valid_mask = (
        (gaussians.sim_view_count.detach().squeeze(1) > 0)
        & (sim_score > 0.0)
    )

    total_count = int(sim_score.numel())
    valid_count = int(sim_valid_mask.sum().item())

    range_mask_1 = range_mask_1.detach().reshape(-1).bool()
    range_mask_2 = range_mask_2.detach().reshape(-1).bool()
    range_mask_3 = range_mask_3.detach().reshape(-1).bool()

    range_1_count = int(range_mask_1.sum().item())
    range_2_count = int(range_mask_2.sum().item())
    range_3_count = int(range_mask_3.sum().item())

    if valid_count > 0:
        valid_scores = sim_score[sim_valid_mask]

        sim_mean = float(valid_scores.mean().item())
        sim_std = float(
            valid_scores.std(unbiased=False).item()
        )

        range_1_lower = sim_mean - 2.0 * sim_std
        range_1_upper = sim_mean - 1.0 * sim_std

        range_2_lower = sim_mean - 3.0 * sim_std
        range_2_upper = sim_mean - 2.0 * sim_std

        range_3_lower = 0.0
        range_3_upper = sim_mean - 3.0 * sim_std
    else:
        sim_mean = None
        sim_std = None

        range_1_lower = None
        range_1_upper = None
        range_2_lower = None
        range_2_upper = None
        range_3_lower = None
        range_3_upper = None

    def percentage(count, denominator):
        if denominator <= 0:
            return 0.0

        return 100.0 * count / denominator

    record = {
        "iteration": int(iteration),

        "gaussian_counts": {
            "total": total_count,
            "similarity_valid": valid_count
        },

        "similarity_distribution": {
            "mean": sim_mean,
            "std": sim_std
        },

        "ranges": {
            "mean_minus_2std_to_mean_minus_1std": {
                "lower": range_1_lower,
                "upper": range_1_upper,
                "count": range_1_count,
                "percent_of_total": percentage(
                    range_1_count,
                    total_count
                ),
                "percent_of_valid": percentage(
                    range_1_count,
                    valid_count
                )
            },

            "mean_minus_3std_to_mean_minus_2std": {
                "lower": range_2_lower,
                "upper": range_2_upper,
                "count": range_2_count,
                "percent_of_total": percentage(
                    range_2_count,
                    total_count
                ),
                "percent_of_valid": percentage(
                    range_2_count,
                    valid_count
                )
            },

            "below_mean_minus_3std": {
                "lower": range_3_lower,
                "upper": range_3_upper,
                "count": range_3_count,
                "percent_of_total": percentage(
                    range_3_count,
                    total_count
                ),
                "percent_of_valid": percentage(
                    range_3_count,
                    valid_count
                )
            }
        }
    }

    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            if not isinstance(data, list):
                data = []
        except (json.JSONDecodeError, OSError):
            data = []
    else:
        data = []

    # 동일 iteration이 이미 있으면 덮어쓰기
    data = [
        item
        for item in data
        if item.get("iteration") != int(iteration)
    ]

    data.append(record)

    data.sort(
        key=lambda item: item.get("iteration", 0)
    )

    with open(json_path, "w") as f:
        json.dump(
            data,
            f,
            indent=2
        )

    if (iteration % 3000 == 0):
        print(
            f"[ITER {iteration}] "
            f"range1={range_1_count} "
            f"({percentage(range_1_count, total_count):.2f}% total), "
            f"range2={range_2_count} "
            f"({percentage(range_2_count, total_count):.2f}% total), "
            f"range3={range_3_count} "
            f"({percentage(range_3_count, total_count):.2f}% total)"
        )

def similarity_map_to_heatmap(
    sim_map,
    value_min=-1.0,
    value_max=1.0
):
    """
    sim_map:
        [H, W], [1, H, W], 또는 [1, 1, H, W]

    Returns:
        heatmap: [3, H, W], CPU float tensor in [0, 1]

    low similarity  -> blue
    high similarity -> red
    """

    sim = sim_map.detach().float()

    if sim.dim() == 4:
        sim = sim.squeeze(0)

    if sim.dim() == 3 and sim.shape[0] == 1:
        sim = sim.squeeze(0)

    if sim.dim() != 2:
        raise ValueError(
            f"sim_map must become [H, W], but got {sim.shape}"
        )

    sim = sim.cpu()

    normalized = (
        (sim - value_min)
        / (value_max - value_min + 1e-8)
    )

    normalized = normalized.clamp(0.0, 1.0)

    cmap = cm.get_cmap("jet")
    rgb_np = cmap(normalized.numpy())[:, :, :3]

    heatmap = torch.from_numpy(rgb_np).float().permute(2, 0, 1)

    return heatmap

def make_selected_highlight_colors(
    view,
    gaussians,
    selected_mask,
    highlight_color=(1.0, 0.0, 0.0),
):
    """
    선택되지 않은 Gaussian: 현재 view 기준 기존 SH 색상
    선택된 Gaussian: highlight_color

    Returns:
        mixed_colors: [N, 3]
    """
    selected = selected_mask.detach().reshape(-1).bool()

    # [N, coeff, 3] -> [N, 3, coeff]
    shs_view = (
        gaussians.get_features
        .transpose(1, 2)
        .contiguous()
    )

    # Gaussian 위치에서 카메라 방향
    directions = (
        gaussians.get_xyz
        - view.camera_center.unsqueeze(0)
    )

    directions = directions / (
        directions.norm(dim=1, keepdim=True) + 1e-8
    )

    # 현재 view에서의 원래 Gaussian RGB
    original_colors = eval_sh(
        gaussians.active_sh_degree,
        shs_view,
        directions
    )

    # 원본 renderer와 동일한 SH -> RGB 처리
    original_colors = torch.clamp_min(
        original_colors + 0.5,
        0.0
    )

    mixed_colors = original_colors.clone()

    mixed_colors[selected] = torch.tensor(
        highlight_color,
        dtype=mixed_colors.dtype,
        device=mixed_colors.device
    )

    return mixed_colors

def similarity_to_heat_colors(similarity_score, valid_mask, device="cuda"):
    """
    similarity_score: [K,1]
    sim_view_count: [K,1]
    return: [K,3] RGB in [0,1]
    """
    score = similarity_score.detach().squeeze(1)   # [K]
    if valid_mask.dim() == 2:
        valid = valid_mask.squeeze(1).bool()
    else:
        valid = valid_mask.bool()

    colors = torch.zeros((score.shape[0], 3), device=device)

    if valid.any():
        valid_scores = score[valid]
        smin = valid_scores.min()
        smax = valid_scores.max()

        norm = torch.zeros_like(score)
        norm[valid] = (valid_scores - smin) / (smax - smin + 1e-8)

        # matplotlib colormap: returns RGBA
        cmap = cm.get_cmap("jet")
        norm_np = norm.detach().cpu().numpy()
        rgb_np = cmap(norm_np)[:, :3]
        rgb = torch.tensor(rgb_np, dtype=torch.float32, device=device)

        colors[valid] = rgb[valid]

        colors = rgb

    # invalid Gaussian은 회색
    colors[~valid] = 0.5

    return colors

def similarity_to_bucket_colors_by_distribution(similarity_score, valid_mask, device="cuda"):
    """
    similarity_score: [K,1]
    valid_mask: [K,1] or [K] bool
    return: [K,3] RGB in [0,1]

    Buckets based on valid-score distribution:
      1) score >= mean + std                  -> red
      2) mean <= score < mean + std          -> orange
      3) mean - std <= score < mean          -> yellow
      4) mean - 2*std <= score < mean - std  -> green
      5) score < mean - 2*std                -> blue
      invalid                                -> gray
    """
    score = similarity_score.detach().squeeze(1)

    if valid_mask.dim() == 2:
        valid = valid_mask.squeeze(1).bool()
    else:
        valid = valid_mask.bool()

    colors = torch.zeros((score.shape[0], 3), device=device)

    # invalid Gaussian
    colors[~valid] = torch.tensor([0.5, 0.5, 0.5], device=device)

    if not valid.any():
        return colors

    valid_scores = score[valid]
    mean_val = valid_scores.mean()
    std_val = valid_scores.std(unbiased=False)

    upper_1 = mean_val + std_val
    lower_1 = mean_val - std_val
    lower_2 = mean_val - 2.0 * std_val
    lower_3 = mean_val - 3.0 * std_val

    # bucket masks
    mask_1 = valid & (score >= mean_val)
    mask_2 = valid & (score >= lower_1) & (score < mean_val)
    mask_3 = valid & (score >= lower_2) & (score < lower_1)
    mask_4 = valid & (score >= lower_3) & (score < lower_2)
    mask_5 = valid & (score < lower_3)
    mask_6 = valid & (score == 0)

    # colors
    colors[mask_1] = torch.tensor([1.0, 0.0, 0.0], device=device)   # red
    colors[mask_2] = torch.tensor([1.0, 0.5, 0.0], device=device)   # orange
    colors[mask_3] = torch.tensor([1.0, 1.0, 0.0], device=device)   # yellow
    colors[mask_4] = torch.tensor([0.0, 1.0, 0.0], device=device)   # green
    colors[mask_5] = torch.tensor([0.0, 0.0, 1.0], device=device)   # blue
    colors[mask_6] = torch.tensor([1.0, 0.0, 1.0], device=device)

    return colors

def similarity_to_bucket_colors(similarity_score, valid_mask, current_top_percent, device="cuda"):
    """
    similarity_score: [K,1]
    valid_mask: [K,1] or [K] bool
    return: [K,3] RGB in [0,1]

    Buckets based on valid-score distribution:
      1) score >= mean + std                  -> red
      2) mean <= score < mean + std          -> orange
      3) mean - std <= score < mean          -> yellow
      4) mean - 2*std <= score < mean - std  -> green
      5) score < mean - 2*std                -> blue
      invalid                                -> gray
    """
    score = similarity_score.detach().squeeze(1)

    if valid_mask.dim() == 2:
        valid = valid_mask.squeeze(1).bool()
    else:
        valid = valid_mask.bool()

    colors = torch.zeros((score.shape[0], 3), device=device)

    # invalid Gaussian
    # colors[~valid] = torch.tensor([0.5, 0.5, 0.5], device=device)

    if not valid.any():
        return colors

    valid_scores = score[valid]
    mean_val = valid_scores.mean()
    std_val = valid_scores.std(unbiased=False)

    upper_1 = mean_val + std_val
    lower_1 = mean_val - std_val
    lower_2 = mean_val - 2.0 * std_val

    # bucket masks
    mask_1 = (score >= 0.5)                              # very high
    mask_3 = (score >= current_top_percent) & (score < 0.5)         # slightly below mean
    mask_4 = (score > 0.0) & (score < current_top_percent)          # low
    mask_5 = (score == 0.0)                               # very low

    # colors
    colors[mask_1] = torch.tensor([1.0, 0.0, 0.0], device=device)   # red
    colors[mask_3] = torch.tensor([1.0, 1.0, 0.0], device=device)   # yellow
    colors[mask_4] = torch.tensor([0.0, 1.0, 0.0], device=device)   # green
    colors[mask_5] = torch.tensor([0.0, 0.0, 1.0], device=device)   # blue

    return colors

def scale_grad_to_colors(scale_grad, device="cuda", eps=0.0):
    grad = scale_grad.detach()

    colors = torch.ones((grad.shape[0], 3), dtype=torch.float32, device=device)
    increase_mask = grad > eps
    decrease_mask = grad < eps

    colors[increase_mask] = torch.tensor([1.0, 0.0, 0.0], device=device)
    colors[decrease_mask] = torch.tensor([0.0, 0.0, 1.0], device=device)

    return colors

def priority_to_heat_colors(priority_score, valid_mask, device="cuda"):
    # score = priority_score.detach().squeeze(1)
    score = priority_score.detach()

    if score.dim() == 2 and score.shape[1] == 1:
        score = score.squeeze(1)
    
    if valid_mask.dim() == 2 and valid_mask.shape[1] == 1:
        valid = valid_mask.squeeze(1).bool()
    else:
        valid = valid_mask.reshape(-1).bool()

    colors = torch.zeros((priority_score.shape[0], 3), device=device)

    if valid.any():
        valid_scores = score[valid]
        # valid_scores = score
        smin = valid_scores.min()
        smax = valid_scores.max()

        # norm = torch.zeros_like(score)
        # norm[valid] = (valid_scores - smin) / (smax - smin + 1e-8)

        cmap = cm.get_cmap("jet")
        # rgb_np = cmap(norm.detach().cpu().numpy())[:, :3]
        # rgb = torch.tensor(rgb_np, dtype=torch.float32, device=device)
        # colors[valid] = rgb[valid]

        abs_max = valid_scores.abs().max()
        norm = torch.zeros_like(score)
        norm[valid] = (valid_scores + abs_max) / (2*abs_max + 13-8)
        rgb_np = cmap(norm.detach().cpu().numpy())[:,:3]
        rgb = torch.tensor(rgb_np, dtype=torch.float32, device=device)
        colors[valid] = rgb[valid]
        colors[~valid] = 0.5

    return colors

def count_score_buckets(score_1d, valid_mask):
    """
    score_1d: [K]
    valid_mask: [K] bool

    return: dict
    """
    valid_scores = score_1d[valid_mask]
    invalid_count = int((~valid_mask).sum().item())

    if valid_scores.numel() == 0:
        return {
            "count_0p8_1p0": 0,
            "count_0p6_0p8": 0,
            "count_0p4_0p6": 0,
            "count_0p2_0p4": 0,
            "count_0p0_0p2": 0,
            "count_below_0p0": 0,
            "invalid_count": invalid_count
        }

    bucket_dict = {
        "count_0p8_1p0": int(((score_1d >= 0.8) & valid_mask).sum().item()),
        "count_0p6_0p8": int(((score_1d >= 0.6) & (score_1d < 0.8) & valid_mask).sum().item()),
        "count_0p4_0p6": int(((score_1d >= 0.4) & (score_1d < 0.6) & valid_mask).sum().item()),
        "count_0p2_0p4": int(((score_1d >= 0.2) & (score_1d < 0.4) & valid_mask).sum().item()),
        "count_0p0_0p2": int(((score_1d >= 0.0) & (score_1d < 0.2) & valid_mask).sum().item()),
        "count_below_0p0": int(((score_1d < 0.0) & valid_mask).sum().item()),
        "invalid_count": invalid_count
    }

    return bucket_dict

def count_score_buckets_by_distribution(score_1d, valid_mask):
    """
    score_1d: [K]
    valid_mask: [K] bool

    Buckets:
      1) score >= mean + std
      2) mean <= score < mean + std
      3) mean - std <= score < mean
      4) mean - 2*std <= score < mean - std
      5) score < mean - 2*std
    """
    invalid_count = int((~valid_mask).sum().item())

    if not valid_mask.any():
        return {
            "mean": None,
            "std": None,
            "count_ge_mean_plus_1std": 0,
            "count_mean_to_mean_plus_1std": 0,
            "count_mean_minus_1std_to_mean": 0,
            "count_mean_minus_2std_to_mean_minus_1std": 0,
            "count_below_mean_minus_2std": 0,
            "invalid_count": invalid_count
        }

    valid_scores = score_1d[valid_mask]
    mean_val = valid_scores.mean()
    std_val = valid_scores.std(unbiased=False)

    lower_1 = mean_val - std_val
    lower_2 = mean_val - 2.0 * std_val

    bucket_dict = {
        "mean": float(mean_val.item()),
        "std": float(std_val.item()),
        "count_ge_mean": int(((score_1d >= mean_val) & valid_mask).sum().item()),
        "count_mean_minus_1std_to_mean": int(((score_1d >= lower_1) & (score_1d < mean_val) & valid_mask).sum().item()),
        "count_mean_minus_2std_to_mean_minus_1std": int(((score_1d >= lower_2) & (score_1d < lower_1) & valid_mask).sum().item()),
        "count_below_mean_minus_2std": int(((score_1d < lower_2) & valid_mask).sum().item()),
        "invalid_count": invalid_count
    }

    return bucket_dict

def grad_to_heat_colors(grad_accum, device="cuda", clamp_percentile=0.99):
    """
    grad_accum: [K,1] or [K,2]
        raw accumulated gradient (e.g. gaussians.xyz_gradient_accum)

    return: [K,3] RGB in [0,1]
    """
    if grad_accum.dim() == 2 and grad_accum.shape[1] == 1:
        grad_mag = grad_accum.detach().squeeze(1).clone()   # [K]
    else:
        grad_mag = torch.norm(grad_accum.detach(), dim=-1)  # [K]

    grad_mag[torch.isnan(grad_mag)] = 0.0
    grad_mag[torch.isinf(grad_mag)] = 0.0
    grad_mag = torch.clamp_min(grad_mag, 0.0)

    colors = torch.zeros((grad_mag.shape[0], 3), device=device)

    if grad_mag.numel() == 0:
        return colors

    # 너무 큰 outlier 때문에 전부 파랗게 보이는 것 방지
    gmax = torch.quantile(grad_mag, clamp_percentile) if grad_mag.numel() > 1 else grad_mag.max()
    gmax = torch.clamp_min(gmax, 1e-8)
    grad_clamped = torch.clamp(grad_mag, max=gmax)

    gmin = grad_clamped.min()
    norm = (grad_clamped - gmin) / (gmax - gmin + 1e-8)

    cmap = cm.get_cmap("jet")
    rgb_np = cmap(norm.detach().cpu().numpy())[:, :3]
    rgb = torch.tensor(rgb_np, dtype=torch.float32, device=device)

    colors[:] = rgb
    return colors

def norm_ratio_map_to_heatmap(norm_ratio_map):
    if norm_ratio_map.dim() == 3 and norm_ratio_map.shape[0] == 1:
        norm_ratio_map = norm_ratio_map.squeeze(0)
    if norm_ratio_map.dim() != 2:
        raise ValueError()
    
    ratio = norm_ratio_map.detach().float().cpu()
    # ratio = torch.clamp(ratio, 0.0, 1.0)

    cmap = cm.get_cmap("jet")
    rgb_np = cmap(ratio.numpy())[:,:,:3]
    rgb = torch.tensor(rgb_np, dtype=torch.float32).permute(2, 0, 1)
    
    return rgb

def norm_ratio_map_to_heatcolors(norm_ratio_map, sim_valid_mask, device="cuda"):
    if norm_ratio_map.dim() == 3 and norm_ratio_map.shape[0] == 1:
        norm_ratio_map = norm_ratio_map.squeeze(0)
    if norm_ratio_map.dim() != 2:
        raise ValueError()
    
    if sim_valid_mask.dim() == 2:
        valid = sim_valid_mask.squeeze(1).bool()
    else:
        valid = sim_valid_mask.bool()
    
    ratio = norm_ratio_map[valid].detach().float().cpu()
    # ratio = torch.clamp(ratio, 0.0, 1.0)

    r = ratio.detach().float().cpu()
    print("min:", r.min().item())
    print("max:", r.max().item())

    ratio -= 0.5

    cmap = cm.get_cmap("jet")
    rgb_np = cmap(ratio.numpy())[:,:,:3]
    rgb = torch.tensor(rgb_np, dtype=torch.float32, device=device)
    colors = torch.zeros((rgb_np.shape[0], 3), device=device)
    colors[valid] = rgb[valid]
    # invalid Gaussian은 회색
    colors[~valid] = 0.5
    
    return rgb

def save_all_views_priority_similarity_renders(
    scene,
    gaussians,
    pipe,
    background,
    dataset,
    opt,
    iteration,
    current_top_percent,
    mode="render_py"
):
    """
    Save priority-score and similarity-score renders for all train/test views
    at the current iteration.
    """
    # -----------------------------------------
    # 1. current global scores 계산
    # -----------------------------------------
    grads = gaussians.xyz_gradient_accum / (gaussians.denom + 1e-8)
    grads[torch.isnan(grads)] = 0.0
    grads[torch.isinf(grads)] = 0.0

    white_background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

    normalized_grads, clip_ref, percentile_ref = gaussians.compute_normalized_gradients(
        grads,
        adc_grad_threshold=opt.densify_grad_threshold,
        top_percent=0.02
    )

    priority_score = gaussians.compute_priority_score(
        normalized_grads=normalized_grads,
        similarity_score=gaussians.get_similarity_score
    )
    scaling = gaussians.get_scaling.detach()

    max_scale = scaling.max(dim=1).values
    min_scale = scaling.min(dim=1).values

    # scale_ratio = min_scale / (max_scale + 1e-12)
    volume = torch.prod(scaling, dim=1)
    scale_ratio = torch.log(volume + 1e-12)    

    similarity_score = gaussians.get_similarity_score

    # norm_ratio_score = gaussians.get_norm_ratio_score

    # valid mask
    sim_valid_mask = ((gaussians.sim_view_count > 0) & (gaussians.get_similarity_score > 0.0))
    # sim_valid_mask = (gaussians.denom > 0)
    prio_valid_mask = (gaussians.sim_view_count > 0)
    # sim_valid_mask = (similarity_score == 0)

    # current_top_percent = gaussians.get_dynamic_priority_top_percent(
    #     current_iteration=iteration,
    #     densify_until_iter=opt.densify_until_iter,
    #     percentile_ref=percentile_ref,
    #     adc_grad_threshold=opt.densify_grad_threshold,
    #     start_percent=0.05,
    #     end_percent=0.0015
    # )

    # semantic_prune_mask_old = gaussians.get_semantic_prune_mask()
    # selected_pts_mask_old, num_select, priority_cutoff = gaussians.get_topk_priority_mask(
    #     priority_score=priority_score,
    #     top_percent=current_top_percent,
    #     require_similarity_valid=True,
    #     require_grad_valid=True
    # )

    # 실제 densification 대상과 맞추려면 semantic prune 제외까지 반영
    # selected_pts_mask_old = selected_pts_mask_old & (~semantic_prune_mask_old)

    # -----------------------------------------
    # 2. Gaussian color 생성
    # -----------------------------------------
    similarity_colors = similarity_to_heat_colors(
        similarity_score,
        sim_valid_mask,
        device="cuda"
    )
    similarity_bucket_colors = similarity_to_bucket_colors(similarity_score, sim_valid_mask, current_top_percent, device="cuda")

    similarity_dist_colors = similarity_to_bucket_colors_by_distribution(similarity_score,sim_valid_mask, device='cuda')

    priority_colors = priority_to_heat_colors(
        scale_ratio,
        prio_valid_mask,
        device="cuda"
    )

    grad_colors = grad_to_heat_colors(
        grad_accum=grads,
        device="cuda",
        clamp_percentile=0.99
    )

    # norm_ratio_color = norm_ratio_map_to_heatcolors(
    #     norm_ratio_score,
    #     sim_valid_mask,
    #     device="cuda"
    # )


    # -----------------------------------------
    # 3. 저장 폴더 준비
    # -----------------------------------------
    priority_root = os.path.join(scene.model_path, "priority_score", f"iter_{iteration:06d}")
    similarity_root = os.path.join(scene.model_path, "similarity_score", f"iter_{iteration:06d}")
    # gradient_root = os.path.join(scene.model_path, "adc_threshold_binary", f"iter_{iteration:06d}")
    grad_root = os.path.join(scene.model_path, "grad_heamap", f"iter_{iteration:06d}")
    # norm_root = os.path.join(scene.model_path, "norm_ratio", f"iter_{iteration:06d}")

    os.makedirs(priority_root, exist_ok=True)
    os.makedirs(similarity_root, exist_ok=True)
    # os.makedirs(grad_root, exist_ok=True)
    # os.makedirs(norm_root, exist_ok=True)

    view_configs = get_consistent_view_configs(scene, mode=mode)

    for config in view_configs:
        split_name = config["name"]
        views = config["views"]

        split_priority_dir = os.path.join(priority_root, split_name)
        split_similarity_dir = os.path.join(similarity_root, split_name)
        # split_gradient_dir = os.path.join(gradient_root, split_name)
        split_grad_dir = os.path.join(grad_root, split_name)
        # split_norm_dir = os.path.join(norm_root, split_name)
    

        os.makedirs(split_priority_dir, exist_ok=True)
        os.makedirs(split_similarity_dir, exist_ok=True)
        # os.makedirs(split_gradient_dir, exist_ok=True)
        os.makedirs(split_grad_dir, exist_ok=True)
        # os.makedirs(split_norm_dir, exist_ok=True)

        for idx, view in enumerate(views):

            # render.py와 동일하게 00000.png 형식 사용
            file_name = f"{idx:05d}.png"

            # priority render
            prio_render_pkg = render(
                view,
                gaussians,
                pipe,
                background,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
                override_color=similarity_dist_colors
            )

            prio_img = prio_render_pkg["render"].detach().clamp(0, 1)
            vutils.save_image(prio_img, os.path.join(split_priority_dir, file_name))

            # similarity render
            sim_render_pkg = render(
                view,
                gaussians,
                pipe,
                background,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
                override_color=similarity_colors
            )

            sim_img = sim_render_pkg["render"].detach().clamp(0, 1)
            vutils.save_image(sim_img, os.path.join(split_similarity_dir, file_name))

            render_pkg = render(view, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, override_color=grad_colors)
            grad_img = render_pkg["render"].detach().clamp(0, 1)
            vutils.save_image(grad_img, os.path.join(split_grad_dir, file_name))

            gt_img = view.original_image[0:3, :, :].cuda()
            rgb_render_pkg = render(view, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            rgb_render = rgb_render_pkg["render"].detach().clamp(0,1)

            # ratio_map = dino_sim.norm_ratio_map(rgb_render.unsqueeze(0), gt_img.unsqueeze(0)).squeeze(0)
            # ratio_heatmap = norm_ratio_map_to_heatmap(ratio_map)
            # norm_ratio_render_pkg = render(view, gaussians, pipe, bg_color=white_background, use_trained_exp=SPARSE_ADAM_AVAILABLE, override_color=norm_ratio_color)
            # ratio_heatmap = norm_ratio_render_pkg["render"].detach()
            # vutils.save_image(ratio_heatmap, os.path.join(split_norm_dir, file_name))


    print(f"[ITER {iteration}] Saved all-view priority renders to {priority_root}")
    print(f"[ITER {iteration}] Saved all-view similarity renders to {similarity_root}")

def save_similarity_scale_statistics_json(
    scene,
    gaussians,
    iteration,
):
    """
    Save similarity score and scale ratio statistics.

    statistics.json

    Each iteration stores:
    {
        iteration,
        similarity_score[],
        scale_ratio[]
    }
    """

    similarity = gaussians.get_similarity_score.detach().squeeze(1)
    valid_mask = gaussians.sim_view_count.detach().squeeze(1) > 0

    scaling = gaussians.get_scaling.detach()

    max_scale = scaling.max(dim=1).values
    min_scale = scaling.min(dim=1).values

    scale_ratio = min_scale / (max_scale + 1e-12)

    similarity = similarity[valid_mask]
    scale_ratio = scale_ratio[valid_mask]

    points = []

    for sim, ratio in zip(similarity.cpu().tolist(), scale_ratio.cpu().tolist()):
        points.append({
            "similarity": sim,
            "scale_ratio": ratio
        })

    stats_entry = {
        "iteration": int(iteration),
        "points": points
    }

    json_path = os.path.join(scene.model_path, "similarity_scale_statistics.json")

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = []

    # 같은 iteration이 있으면 삭제
    data = [d for d in data if d["iteration"] != iteration]

    data.append(stats_entry)

    data.sort(key=lambda x: x["iteration"])

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[ITER {iteration}] Saved similarity-scale statistics.")

def save_similarity_statistics_json(
    scene,
    gaussians,
    iteration,
    view_score,
    visible_mask,
):
    """
    매 iteration마다 두 종류의 similarity 저장

    1. view_similarity
       현재 iteration/view에서 계산된 instantaneous similarity

    2. accumulated_similarity
       현재 100-step window에서 지금까지 누적된 similarity
       = sim_score_accum / sim_view_count

    실제 Gaussian similarity state는 변경하지 않음.
    """

    json_path = os.path.join(
        scene.model_path,
        "similarity_iteration_statistics.json"
    )

    with torch.no_grad():

        # =====================================================
        # 1. 현재 VIEW의 similarity
        # =====================================================

        current_view_score = (
            view_score
            .detach()
            .reshape(-1)
        )

        current_visible_mask = (
            visible_mask
            .detach()
            .reshape(-1)
            .bool()
        )

        # 혹시 nan/inf가 있으면 제외
        view_valid_mask = (
            current_visible_mask
            & torch.isfinite(current_view_score)
        )

        if view_valid_mask.any():

            valid_view_similarity = current_view_score[
                view_valid_mask
            ]

            view_mean = (
                valid_view_similarity.mean().item()
            )

            view_std = (
                valid_view_similarity
                .std(unbiased=False)
                .item()
            )

            view_min = (
                valid_view_similarity.min().item()
            )

            view_max = (
                valid_view_similarity.max().item()
            )

            view_valid_count = int(
                view_valid_mask.sum().item()
            )

        else:

            view_mean = None
            view_std = None
            view_min = None
            view_max = None
            view_valid_count = 0


        # =====================================================
        # 2. 현재까지 누적된 similarity
        # =====================================================

        sim_accum = (
            gaussians.sim_score_accum
            .detach()
            .reshape(-1)
        )

        sim_count = (
            gaussians.sim_view_count
            .detach()
            .reshape(-1)
        )

        accum_valid_mask = (
            (sim_count > 0)
            & torch.isfinite(sim_accum)
        )

        if accum_valid_mask.any():

            accumulated_similarity = (
                sim_accum[accum_valid_mask]
                /
                (
                    sim_count[accum_valid_mask]
                    + 1e-8
                )
            )

            accum_mean = (
                accumulated_similarity.mean().item()
            )

            accum_std = (
                accumulated_similarity
                .std(unbiased=False)
                .item()
            )

            accum_min = (
                accumulated_similarity.min().item()
            )

            accum_max = (
                accumulated_similarity.max().item()
            )

            accum_valid_count = int(
                accum_valid_mask.sum().item()
            )

        else:

            accum_mean = None
            accum_std = None
            accum_min = None
            accum_max = None
            accum_valid_count = 0

        ema = (
            gaussians.similarity_ema
            .detach()
            .reshape(-1)
        )

        ema_valid_mask = (
            gaussians.sim_view_count
            .detach()
            .reshape(-1)
            > 0
        )

        if ema_valid_mask.any():

            ema_valid = ema[
                ema_valid_mask
            ]

            ema_mean = ema_valid.mean().item()

            ema_std = (
                ema_valid
                .std(unbiased=False)
                .item()
            )

        else:

            ema_mean = None
            ema_std = None
            
    # =========================================================
    # 3. JSON
    # =========================================================

    record = {

        "iteration": int(iteration),

        "view_similarity": {
            "mean": view_mean,
            "std": view_std,
            "min": view_min,
            "max": view_max,
            "valid_count": view_valid_count
        },

        "accumulated_similarity": {
            "mean": accum_mean,
            "std": accum_std,
            "min": accum_min,
            "max": accum_max,
            "valid_count": accum_valid_count
        },

        "ema_similarity": {
            "mean": ema_mean,
            "std": ema_std
        }
    }


    # =========================================================
    # 4. JSONL append
    # =========================================================

    json_path = os.path.join(
        scene.model_path,
        "similarity_iteration_statistics.json"
    )

    with open(json_path, "a") as f:
        f.write(
            json.dumps(record) + "\n"
        )

def save_final_similarity_comparison_json(
    scene,
    gaussians,
    iteration,
):
    """
    100-step window 종료 후

    1. 기존 finalize_similarity()로 계산된 similarity
    2. EMA similarity

    의 전체 Gaussian 평균 통계만 저장.
    """

    json_path = os.path.join(
        scene.model_path,
        "final_similarity_comparison.json"
    )

    with torch.no_grad():

        # -----------------------------------------------------
        # 기존 finalized similarity
        # -----------------------------------------------------
        final_similarity = (
            gaussians.get_similarity_score
            .detach()
            .reshape(-1)
        )

        # -----------------------------------------------------
        # EMA similarity
        # -----------------------------------------------------
        ema_similarity = (
            gaussians.similarity_ema
            .detach()
            .reshape(-1)
        )

        # -----------------------------------------------------
        # valid Gaussian
        # 기존 similarity가 실제 계산된 Gaussian만 사용
        # -----------------------------------------------------
        valid_mask = (
            (final_similarity > 0.0)
            & torch.isfinite(final_similarity)
            & torch.isfinite(ema_similarity)
        )

        if not valid_mask.any():
            print(
                f"[ITER {iteration}] "
                "No valid similarity scores."
            )
            return

        final_valid = final_similarity[valid_mask]
        ema_valid = ema_similarity[valid_mask]

        # -----------------------------------------------------
        # 통계
        # -----------------------------------------------------
        final_mean = final_valid.mean().item()
        final_std = final_valid.std(unbiased=False).item()

        ema_mean = ema_valid.mean().item()
        ema_std = ema_valid.std(unbiased=False).item()

        difference = ema_valid - final_valid

        record = {
            "iteration": int(iteration),

            "final_similarity": {
                "mean": float(final_mean),
                "std": float(final_std)
            },

            "ema_similarity": {
                "mean": float(ema_mean),
                "std": float(ema_std)
            },

            "difference": {
                "mean": float(
                    difference.mean().item()
                ),
                "absolute_mean": float(
                    difference.abs().mean().item()
                ),
                "ema_higher_ratio": float(
                    (difference > 0)
                    .float()
                    .mean()
                    .item()
                )
            },

            "valid_count": int(
                valid_mask.sum().item()
            )
        }

    # JSONL append
    with open(json_path, "a") as f:
        f.write(
            json.dumps(record) + "\n"
        )

    print(
        f"[ITER {iteration}] "
        f"Final={final_mean:.4f} | "
        f"EMA={ema_mean:.4f} | "
        f"Diff={ema_mean-final_mean:+.4f}"
    )
####

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 13_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
