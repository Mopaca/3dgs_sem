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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    per_patch_dict = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}
            per_patch_dict[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}
                per_patch_dict[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    render = renders[idx]
                    gt = gts[idx]
                    image_name = image_names[idx]
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                    patch_results = compute_patch_metrics(render, gt, grid_size=4)
                    per_patch_dict[scene_dir][method][image_name] = patch_results

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_patch.json", 'w') as fp:
                json.dump(per_patch_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

def compute_patch_metrics(render, gt, grid_size=4):
    """
    render, gt: [1, 3, H, W]
    grid_size: 3 -> 3x3, 4 -> 4x4

    return:
        {
            "patch_0_0": {"SSIM": ..., "PSNR": ..., "LPIPS": ...},
            ...
        }
    """

    _, _, H, W = render.shape

    patch_results = {}

    for row in range(grid_size):
        for col in range(grid_size):

            # 마지막 patch가 남는 pixel까지 포함하도록 계산
            h_start = (row * H) // grid_size
            h_end = ((row + 1) * H) // grid_size

            w_start = (col * W) // grid_size
            w_end = ((col + 1) * W) // grid_size

            render_patch = render[
                :, :, h_start:h_end, w_start:w_end
            ]

            gt_patch = gt[
                :, :, h_start:h_end, w_start:w_end
            ]

            patch_ssim = ssim(
                render_patch,
                gt_patch
            ).item()

            patch_psnr = psnr(
                render_patch,
                gt_patch
            ).item()

            patch_lpips = lpips(
                render_patch,
                gt_patch,
                net_type='vgg'
            ).item()

            patch_name = f"patch_{row}_{col}"

            patch_results[patch_name] = {
                "SSIM": patch_ssim,
                "PSNR": patch_psnr,
                "LPIPS": patch_lpips,

                # 나중에 위치 확인용
                "pixel_range": {
                    "x": [w_start, w_end],
                    "y": [h_start, h_end]
                }
            }

    return patch_results

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
