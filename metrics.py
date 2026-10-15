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
import matplotlib.pyplot as plt

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
                    patch_vis_dir = (method_dir / "patch_metrics")
                    patch_vis_dir.mkdir(parents=True,exist_ok=True)
                    save_patch_metric_visualization(render=render, patch_results=patch_results, save_path=patch_vis_dir / image_name, grid_size=4)

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

def save_patch_metric_visualization(
    render,
    patch_results,
    save_path,
    grid_size=4
):
    """
    render:
        torch.Tensor [1, 3, H, W]

    patch_results:
        compute_patch_metrics()의 결과

    save_path:
        저장할 png 경로

    grid_size:
        3 -> 3x3
        4 -> 4x4
    """

    # [1, 3, H, W] -> [H, W, 3]
    image = (
        render[0]
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    H, W, _ = image.shape

    # 원본 이미지 비율 유지
    dpi = 100

    fig = plt.figure(
        figsize=(W / dpi, H / dpi),
        dpi=dpi
    )

    ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(image)

    # ----------------------------------------
    # Grid 그리기
    # ----------------------------------------

    for i in range(1, grid_size):

        x = (i * W) // grid_size
        y = (i * H) // grid_size

        ax.axvline(
            x=x,
            linewidth=1.5,
            color="red"
        )

        ax.axhline(
            y=y,
            linewidth=1.5,
            color="red"
        )

    # 이미지 바깥 테두리
    ax.plot(
        [0, W - 1, W - 1, 0, 0],
        [0, 0, H - 1, H - 1, 0],
        linewidth=1.5,
        color="red"
    )

    # ----------------------------------------
    # Patch metric 표시
    # ----------------------------------------

    for row in range(grid_size):
        for col in range(grid_size):

            patch_name = f"patch_{row}_{col}"

            result = patch_results[patch_name]

            psnr_value = result["PSNR"]
            ssim_value = result["SSIM"]
            lpips_value = result["LPIPS"]

            # patch 시작 위치
            x_start = (col * W) // grid_size
            y_start = (row * H) // grid_size

            # 글씨를 경계에서 약간 안쪽에 배치
            margin_x = W * 0.008
            margin_y = H * 0.015

            text_x = x_start + margin_x
            text_y = y_start + margin_y

            metric_text = (
                f"PSNR: {psnr_value:.4f}\n"
                f"SSIM: {ssim_value:.4f}\n"
                f"LPIPS: {lpips_value:.4f}"
            )

            ax.text(
                text_x,
                text_y,
                metric_text,

                fontsize=max(6, int(W / 160)),

                color="white",

                verticalalignment="top",
                horizontalalignment="left",

                # 흰 글씨가 밝은 하늘에서도 보이도록
                bbox=dict(
                    facecolor="black",
                    alpha=0.35,
                    edgecolor="none",
                    pad=1.5
                )
            )

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    ax.axis("off")

    plt.savefig(
        save_path,
        dpi=dpi,
        bbox_inches=None,
        pad_inches=0
    )

    plt.close(fig)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
