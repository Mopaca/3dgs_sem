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

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2
import torch
import random ###

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    image = Image.open(cam_info.image_path)

    if cam_info.depth_path != "":
        try:
            if is_nerf_synthetic:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
            else:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)

        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None
        
    orig_w, orig_h = image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
    

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=image, invdepthmap=invdepthmap,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


# #####
# def get_camera_center(cam):
#     """
#     Return camera center as torch.Tensor of shape [3].

#     Priority:
#     1) cam.camera_center
#     2) inverse(world_view_transform)[:3, 3]
#     """
#     if hasattr(cam, "camera_center"):
#         center = cam.camera_center
#         if not torch.is_tensor(center):
#             center = torch.tensor(center, dtype=torch.float32, device="cuda")
#         return center.float()

#     if hasattr(cam, "world_view_transform"):
#         w2c = cam.world_view_transform
#         if not torch.is_tensor(w2c):
#             w2c = torch.tensor(w2c, dtype=torch.float32, device="cuda")
#         c2w = torch.inverse(w2c)
#         return c2w[:3, 3].float()

#     raise AttributeError("Camera object must have either 'camera_center' or 'world_view_transform'.")

# def farthest_point_sampling_from_seed(cameras, seed_idx, target_count):
#     n = len(cameras)
#     target_count = max(1, min(target_count, n))

#     centers = torch.stack([get_camera_center(cam) for cam in cameras], dim=0)  # [N, 3]

#     selected = [seed_idx]
#     selected_mask = torch.zeros(n, dtype=torch.bool, device=centers.device)
#     selected_mask[seed_idx] = True

#     # min distance to selected set
#     dist = torch.norm(centers - centers[seed_idx:seed_idx+1], dim=1)

#     while len(selected) < target_count:
#         dist[selected_mask] = -1.0
#         next_idx = torch.argmax(dist).item()

#         selected.append(next_idx)
#         selected_mask[next_idx] = True

#         new_dist = torch.norm(centers - centers[next_idx:next_idx+1], dim=1)
#         dist = torch.minimum(dist, new_dist)

#     return selected

# def select_quarter_views_with_random_seed_and_fps(viewpoint_stack):
#     """
#     1) Randomly pick one seed camera
#     2) Run FPS from that seed
#     3) Select 1/4 of all train cameras
#     4) Return selected camera list
#     """
#     train_cams = viewpoint_stack
#     num_views = len(train_cams)

#     if num_views == 0:
#         return []

#     target_count = max(1, num_views // 4)
#     seed_idx = random.randint(0, num_views - 1)

#     selected_indices = farthest_point_sampling_from_seed(
#         cameras=train_cams,
#         seed_idx=seed_idx,
#         target_count=target_count,
#     )

#     selected_cams = [train_cams[i] for i in selected_indices]
#     return selected_cams, selected_indices, seed_idx

# #####