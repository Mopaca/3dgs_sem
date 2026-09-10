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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import json ###
import os ###

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self._similarity_aux = torch.empty(0) ###
        self.sim_score_accum = torch.empty(0) ###
        self.sim_view_count = torch.empty(0) ###
        self.similarity_score = torch.empty(0) ###

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._similarity_aux, ###
            self.sim_score_accum, ###
            self.sim_view_count, ###
            self.similarity_score, ###
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self._similarity_aux, ###
        sim_score_accum, ###
        sim_view_count, ###
        similarity_score) = model_args ###
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.sim_score_accum = sim_score_accum ###
        self.sim_view_count = sim_view_count ###
        self.similarity_score = similarity_score ###
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

#####
    @property
    def get_similarity_aux(self):
        return self._similarity_aux

    @property
    def get_similarity_num(self):
        return self._similarity_aux[:, 0:1]
    
    @property
    def get_similarity_den(self):
        return self._similarity_aux[:, 1:2]
    
    @property
    def get_similarity_score(self):
        return self.similarity_score

    @property
    def get_sim_view_count(self):
        return self.sim_view_count
    
    @property
    def get_norm_ratio_accum(self):
        return self.norm_ratio_accum

    @property
    def get_norm_ratio_score(self):
        return self.norm_ratio_score

#####

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._similarity_aux = nn.Parameter(torch.zeros((self.get_xyz.shape[0], 3), device="cuda").requires_grad_(True)) ###
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.sim_score_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") ###
        self.sim_view_count = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") ###
        self.similarity_score = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") ###
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.sim_score_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") ###
        self.sim_view_count = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") ###
        self.similarity_score = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") ###

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

#####
    def reset_similarity_accum(self):
        self.sim_score_accum.zero_()
        self.sim_view_count.zero_()
        self.similarity_score.zero_()
    
    def reset_similarity_aux(self):
        with torch.no_grad():
            self._similarity_aux.zero_()
        if self._similarity_aux.grad is not None:
            self._similarity_aux.grad.zero_()
#####

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._similarity_aux = nn.Parameter(torch.zeros((self.get_xyz.shape[0], 3), device="cuda").requires_grad_(True)) ###

        self.sim_score_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") ###
        self.sim_view_count = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") ###
        self.similarity_score = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") ###

        self.active_sh_degree = self.max_sh_degree

#####
    def accumulate_similarity(self, view_score, visible_mask):
        self.sim_score_accum += view_score * visible_mask
        self.sim_view_count += visible_mask
        # self.sim_score_accum += view_score

    def finalize_similarity_score(self, eps=1e-8):
        self.similarity_score = self.sim_score_accum / (self.sim_view_count)
        # self.similarity_score = self.sim_score_accum / self.denom
        self.similarity_score[self.similarity_score.isnan()] = 0.0
        self.similarity_score[self.similarity_score.isinf()] = 0.0
        return self.similarity_score
#####

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

#####
    def compute_normalized_gradients(self, grads, adc_grad_threshold, top_percent=0.05, eps=1e-8):
        if grads.dim() == 2 and grads.shape[1] == 1:
            grad_mag = grads.clone()
        else:
            grad_mag = torch.norm(grads, dim=-1, keepdim=True)

        grad_mag = grad_mag.detach().clone()
        grad_mag[torch.isnan(grad_mag)] = 0.0
        grad_mag[torch.isinf(grad_mag)] = 0.0
        grad_mag = torch.clamp_min(grad_mag, 0.0)

        valid = grad_mag.squeeze(1) > 0

        if valid.any():
            valid_grads = grad_mag[valid].squeeze(1)
            q = 1.0 - top_percent
            percentile_ref = torch.quantile(valid_grads, q)
        else:
            percentile_ref = torch.tensor(0.0, device=grad_mag.device, dtype=grad_mag.dtype)
        
        adc_ref = torch.tensor(adc_grad_threshold, device=grad_mag.device, dtype=grad_mag.dtype)

        clip_ref = torch.maximum(percentile_ref, adc_ref)
        clip_ref = torch.clamp_min(clip_ref, eps)
        normalized_grads = torch.clamp(grad_mag, max=clip_ref) / clip_ref
        normalized_threshold = torch.clamp(adc_ref / clip_ref, max=1.0)
        
        return normalized_grads, clip_ref, percentile_ref
    
    def compute_priority_score(self, normalized_grads, similarity_score, normalized_scale=None, alpha=0.7, beta=0.3, gamma=0.1):
        y = normalized_grads.detach().clone()
        s = similarity_score.detach().clone()

        y = torch.clamp(y, 0.0, 1.0)
        s = torch.clamp(s, 0.0, 1.0)

        x = 1.0 - s

        # priority_score = alpha * x3 + beta * y * (0.5 - x3)
        # priority_score = alpha * x + beta * y * (0.5 - x)
        # base_priority = alpha * x + beta * y
        # if normalized_scale is None:
        #     return base_priority
        
        # priority_score = base_priority * (1.0 + gamma * normalized_scale)

        base_priority = alpha * x + beta * y

        return base_priority
        # priority_score = base_priority * (1.0 + gamma * normalized_scale)
        # return priority_score

    
    def get_dynamic_similarity_threshold(
        self,
        current_iteration,
        densify_from_iter,
        densify_until_iter,
        start_threshold=0.05,
        end_threshold=0.5
    ):
        if current_iteration <= densify_from_iter:
            return start_threshold

        if current_iteration >= densify_until_iter:
            return end_threshold

        progress = (current_iteration - densify_from_iter) / float(max(1, densify_until_iter - densify_from_iter))
        progress = max(0.0, min(1.0, progress))

        current_threshold = start_threshold + (end_threshold - start_threshold) * progress
        return current_threshold
    
    def compute_normalized_scale(
    self,
    scene_extent,
    scale_ref_multiplier=0.02,
    eps=1e-8
    ):
        """
        Compute normalized Gaussian scale for priority weighting.

        Representative scale of each Gaussian:
            sigma_k = max(self.get_scaling[k])

        Normalization reference:
            sigma_ref = scale_ref_multiplier * scene_extent

        Normalized scale:
            n_k = min(sigma_k, sigma_ref) / sigma_ref

        Args:
            scene_extent: float
            scale_ref_multiplier: float
                Recommended default: 0.02
                (larger than split/clone threshold 0.01)
            eps: small constant for numerical stability

        Returns:
            normalized_scale: [K, 1] tensor in [0, 1]
            scale_ref: scalar tensor
            raw_scale: [K, 1] tensor
        """
        raw_scale = torch.max(self.get_scaling, dim=1).values.unsqueeze(1)   # [K,1]

        raw_scale = raw_scale.detach().clone()
        raw_scale[torch.isnan(raw_scale)] = 0.0
        raw_scale[torch.isinf(raw_scale)] = 0.0
        raw_scale = torch.clamp_min(raw_scale, 0.0)

        scale_ref = torch.tensor(
            scale_ref_multiplier * scene_extent,
            device=raw_scale.device,
            dtype=raw_scale.dtype
        )
        scale_ref = torch.clamp_min(scale_ref, eps)

        normalized_scale = torch.clamp(raw_scale, max=scale_ref) / scale_ref

        return normalized_scale, scale_ref, raw_scale
    
    def save_similarity_statistics(self, iteration, sim_mean, sim_std, densify_lower, densify_upper, prune_lower, prune_upper, save_dir):

        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join( save_dir, "similarity_log.json")

        record = {
            "iteration": int(iteration),
            "mean": float(sim_mean),
            "std": float(sim_std),
            "densify_lower": float(densify_lower),
            "densify_upper": float(densify_upper),
            "prune_lower": float(prune_lower),
            "prune_upper": float(prune_upper)
        }

        if os.path.exists(json_path):
            with open(json_path,"r") as f:
                data=json.load(f)
        else:
            data=[]
        data.append(record)
        with open(json_path,"w") as f:
            json.dump(data,f,indent=4)

    def save_adc_statistics(self, iteration, densify_grad, densify_sim, densify_union, prune_grad, prune_sim, prune_union, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(save_dir, "adc_log.json")

        densify_grad = densify_grad.sum().item()
        densify_sim = densify_sim.sum().item()
        densify_union = densify_union.sum().item()
        densify_inter = (densify_grad + densify_sim) - densify_union

        prune_grad = prune_grad.sum().item()
        prune_sim = prune_sim.sum().item()
        prune_union = prune_union.sum().item()
        prune_inter = (prune_grad + prune_sim) - prune_union

        record = {
            "iteration": int(iteration),
            "densify": None,
            "prune": None,
        }

        record["densify"] = {
            "gradient_only": int(densify_grad - densify_inter),
            "intersection": int(densify_inter),
            "similarity_only": int(densify_sim - densify_inter),
            "union": int(densify_union)
        }

        record["prune"] = {
            "gradient_only": int(prune_grad - prune_inter),
            "intersection": int(prune_inter),
            "similarity_only": int(prune_sim - prune_inter),
            "union": int(prune_union)
        }

        if os.path.exists(json_path):
            with open(json_path,"r") as f:
                data=json.load(f)
        else:
            data=[]
        data = [d for d in data if d["iteration"] != iteration]
        data.append(record)
        with open(json_path,"w") as f:
            json.dump(data,f,indent=4)
#####

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

        self.sim_score_accum = self.sim_score_accum[valid_points_mask] ###
        self.sim_view_count = self.sim_view_count[valid_points_mask] ###
        self.similarity_score = self.similarity_score[valid_points_mask] ###
        self._similarity_aux = nn.Parameter(self._similarity_aux[valid_points_mask].detach().requires_grad_(True)) ###


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        new_count = new_xyz.shape[0] ###

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        self.sim_score_accum = torch.cat([self.sim_score_accum, torch.zeros((new_count, 1), device="cuda")], dim=0) ###
        self.sim_view_count = torch.cat([self.sim_view_count, torch.zeros((new_count, 1), device="cuda")], dim=0) ###
        self.similarity_score = torch.cat([self.similarity_score, torch.zeros((new_count, 1), device="cuda")], dim=0) ###
        self._similarity_aux = nn.Parameter(torch.cat([self._similarity_aux.detach(), torch.zeros((new_count, 3), device="cuda")], dim=0).requires_grad_(True)) ###

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):

        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii

        old_max_radii2D = self.max_radii2D.clone()
        old_num_points = self.get_xyz.shape[0]

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            current_num_points = self.get_xyz.shape[0]
            max_radii2D_padded = torch.zeros(current_num_points, dtype=old_max_radii2D.dtype, device=old_max_radii2D.device)
            max_radii2D_padded[:old_num_points] = (old_max_radii2D)
            print(max_radii2D_padded.mean(), max_radii2D_padded.max(), max_radii2D_padded.min())
            big_points_vs = max_radii2D_padded > max_screen_size
            print(big_points_vs.sum())
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

#####
    def get_semantic_prune_mask_by_percentile(self, prune_percent=0.02, require_valid=True):
        """
        similarity score 하위 prune_percent 비율의 Gaussian을 prune 대상으로 선택.

        Args:
            prune_percent: float
                예: 0.02 -> 하위 2%
            require_valid: bool
                True이면 sim_view_count > 0 인 Gaussian만 후보로 사용

        Returns:
            semantic_prune_mask_old: [K] bool
                현재(old) Gaussian 기준 prune mask
            prune_cutoff: scalar tensor
                하위 prune_percent 경계 similarity 값
        """
        sim_score = self.get_similarity_score.squeeze(1).detach()   # [K]
        K = sim_score.shape[0]

        if require_valid:
            candidate_mask = (self.sim_view_count.squeeze(1) > 0)
        else:
            candidate_mask = torch.ones(K, dtype=torch.bool, device=sim_score.device)

        candidate_mask &= (sim_score > 0.0)

        semantic_prune_mask_old = torch.zeros(K, dtype=torch.bool, device=sim_score.device)

        num_candidates = int(candidate_mask.sum().item())
        if num_candidates == 0:
            prune_cutoff = torch.tensor(float("nan"), device=sim_score.device, dtype=sim_score.dtype)
            return semantic_prune_mask_old, prune_cutoff

        num_prune = max(1, int(num_candidates * prune_percent))
        num_prune = min(num_prune, num_candidates)

        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)   # [Nc]
        candidate_scores = sim_score[candidate_mask]                                    # [Nc]

        # similarity 낮은 순서대로 하위 num_prune개 선택
        lowk_vals, lowk_idx_in_candidates = torch.topk(
            candidate_scores,
            k=num_prune,
            largest=False,
            sorted=False
        )

        prune_indices = candidate_indices[lowk_idx_in_candidates]
        semantic_prune_mask_old[prune_indices] = True

        prune_cutoff = lowk_vals.max()   # 하위 2%의 상한값

        return semantic_prune_mask_old, prune_cutoff
    
    def get_semantic_prune_mask(self, require_valid=True, sim_threshold=0.0):
        """
        similarity score <= sim_threshold 인 Gaussian을 semantic prune 대상으로 선택.

        Args:
            require_valid: bool
                True이면 sim_view_count > 0 인 Gaussian만 후보로 사용
            sim_threshold: float
                기본값 0.0

        Returns:
            semantic_prune_mask_old: [K] bool
            prune_count: int
        """
        sim_score = self.get_similarity_score.squeeze(1).detach()   # [K]

        if require_valid:
            valid_mask = (self.sim_view_count.squeeze(1) > 0)
        else:
            valid_mask = torch.ones_like(sim_score, dtype=torch.bool)

        semantic_prune_mask_old = valid_mask & (sim_score <= sim_threshold)
        prune_count = int(semantic_prune_mask_old.sum().item())

        return semantic_prune_mask_old, prune_count
        
    def semantic_prune(self, prune_percent=0.01, require_valid=True):
        # semantic_prune_mask, prune_cutoff = self.get_semantic_prune_mask_by_percentile(prune_percent=prune_percent, require_valid=require_valid)
        semantic_prune_mask, prune_cutoff = self.get_semantic_prune_mask(prune_percent=prune_percent, require_valid=require_valid)
        num_semantic_prune = int(semantic_prune_mask.sum().item())
        if num_semantic_prune > 0:
            self.prune_points(semantic_prune_mask)
        return num_semantic_prune, prune_cutoff

    def get_topk_priority_mask(
        self,
        priority_score,
        top_percent,
        require_similarity_valid=True,
        require_grad_valid=True
    ):
        """
        Select exactly top-k Gaussians by priority score.

        Args:
            priority_score: [K,1]
            top_percent: float, e.g. 0.02 means top 2%
            require_similarity_valid: only allow sim_view_count > 0
            require_grad_valid: only allow denom > 0

        Returns:
            selected_pts_mask_old: [K] bool mask over current/original Gaussians
            num_select: int
        """
        score = priority_score.detach().squeeze(1)   # [K]
        K = score.shape[0]

        candidate_mask = torch.ones(K, dtype=torch.bool, device=score.device)

        if require_similarity_valid:
            candidate_mask &= (self.sim_view_count.squeeze(1) > 0)

        if require_grad_valid:
            candidate_mask &= (self.get_opacity.squeeze(1) > 0.005)

        selected_pts_mask_old = torch.zeros(K, dtype=torch.bool, device=score.device)

        num_candidates = int(candidate_mask.sum().item())
        if num_candidates == 0:
            return selected_pts_mask_old, 0

        num_select = max(1, int(num_candidates * top_percent))
        num_select = min(num_select, num_candidates)

        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)   # [Nc]
        candidate_scores = score[candidate_mask]                                        # [Nc]

        topk_vals, topk_idx_in_candidates = torch.topk(
            candidate_scores,
            k=num_select,
            largest=True,
            sorted=False
        )

        selected_indices = candidate_indices[topk_idx_in_candidates]
        selected_pts_mask_old[selected_indices] = True

        priority_cutoff = topk_vals.min()

        return selected_pts_mask_old, num_select, priority_cutoff

    def densify_and_clone_by_selected_mask(self, selected_pts_mask_old, scene_extent):
        """
        Clone selected small Gaussians.
        selected_pts_mask_old: [K_old] bool
        """
        n_init_points = self.get_xyz.shape[0]

        padded_mask = torch.zeros((n_init_points), device="cuda", dtype=torch.bool)
        padded_mask[:selected_pts_mask_old.shape[0]] = selected_pts_mask_old

        selected_pts_mask = torch.logical_and(
            padded_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )

        if selected_pts_mask.sum() == 0:
            return

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_tmp_radii
        )

    def densify_and_split_by_selected_mask(self, selected_pts_mask_old, scene_extent, N=2):
        """
        Split selected large Gaussians.
        selected_pts_mask_old: [K_old] bool
        """
        n_init_points = self.get_xyz.shape[0]

        padded_mask = torch.zeros((n_init_points), device="cuda", dtype=torch.bool)
        padded_mask[:selected_pts_mask_old.shape[0]] = selected_pts_mask_old

        selected_pts_mask = torch.logical_and(
            padded_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )

        if selected_pts_mask.sum() == 0:
            return

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacity,
            new_scaling, new_rotation, new_tmp_radii
        )

        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)
        ))
        self.prune_points(prune_filter)

    def densify_and_prune_by_priority(
        self,
        iteration,
        densify_until_iter,
        adc_grad_threshold,
        min_opacity,
        extent,
        max_screen_size,
        radii,
        top_percent,
        alpha=0.7,
        beta=0.3,
        grad_top_percent=0.05,
        reset_opacity = False,
        split_N=2
    ):
        """
        Priority-score-based densification using exact top-k selection.

        Steps:
        1) compute averaged ADC gradient
        2) normalize gradient
        3) compute priority score from normalized gradient + similarity score
        4) select exact top n% Gaussians by priority
        5) clone small ones / split large ones
        6) prune using original opacity + size rules
        """
        grads = self.xyz_gradient_accum / (self.denom + 1e-8)
        grads[torch.isnan(grads)] = 0.0

        normalized_grads, clip_ref, percentile_ref = self.compute_normalized_gradients(
            grads,
            adc_grad_threshold=adc_grad_threshold,
            top_percent=grad_top_percent
        )

        priority_score = self.compute_priority_score(
            normalized_grads=normalized_grads,
            similarity_score=self.get_similarity_score,
            normalized_scale=None,
            alpha=alpha,
            beta=beta
        )

        current_top_percent = self.get_dynamic_priority_top_percent(
            current_iteration=iteration,
            densify_until_iter=densify_until_iter,
            percentile_ref=percentile_ref,
            adc_grad_threshold=adc_grad_threshold,
            start_percent=top_percent,
            end_percent=0.01)
        
        current_top_percent = top_percent

        self.tmp_radii = radii

        semantic_prune_mask_old, prune_cutoff = self.get_semantic_prune_mask_by_percentile(prune_percent=0.01, require_valid=False)
        # semantic_prune_mask_old, prune_cutoff = self.get_semantic_prune_mask(require_valid=False, sim_threshold=0.0)
        

        selected_pts_mask_old, num_select, priority_cutoff = self.get_topk_priority_mask(
            priority_score=priority_score,
            top_percent=current_top_percent,
            require_similarity_valid=True,
            require_grad_valid=True
        )


        selected_pts_mask_old = selected_pts_mask_old & (~semantic_prune_mask_old)

        # 먼저 clone
        self.densify_and_clone_by_selected_mask(
            selected_pts_mask_old=selected_pts_mask_old,
            scene_extent=extent
        )

        # 그 다음 split
        self.densify_and_split_by_selected_mask(
            selected_pts_mask_old=selected_pts_mask_old,
            scene_extent=extent,
            N=split_N
        )

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs),
                big_points_ws
            )

        current_num_points = self.get_xyz.shape[0]
        semantic_prune_mask_padded = torch.zeros(current_num_points, device="cuda", dtype=torch.bool)
        semantic_prune_mask_padded[:semantic_prune_mask_old.shape[0]] = semantic_prune_mask_old
        # prune_mask = torch.logical_or(prune_mask, semantic_prune_mask_padded)
        self.prune_points(prune_mask)

        self.tmp_radii = None
        torch.cuda.empty_cache()

        return current_top_percent, priority_cutoff, reset_opacity
    
    def farthest_point_sampling(self, xyz, num_samples):
        """
        xyz: [N, 3]
        num_samples: 선택할 점 개수
        Returns:
            sampled_idx: [num_samples]
        """

        device = xyz.device
        num_points = xyz.shape[0]

        if num_samples <= 0 or num_points == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        num_samples = min(num_samples, num_points)
        sampled_idx = torch.empty(num_samples, dtype=torch.long, device=device)

        min_distances = torch.full((num_points,), float("inf"), device=device, dtype=xyz.dtype)

        # 첫 번째 점은 랜덤 선택
        farthest = torch.randint(low=0, high=num_points, size=(1,), device=device).item()

        for i in range(num_samples):
            sampled_idx[i] = farthest
            selected_xyz = xyz[farthest].unsqueeze(0)
            distances = torch.sum((xyz - selected_xyz) ** 2, dim=1)
            min_distances = torch.minimum(min_distances, distances)
            farthest = torch.argmax(min_distances).item()

        return sampled_idx

    def hybrid_scale_fps_sampling(self, candidate_indices, final_ratio=0.01, scale_ratio=0.05):
        if candidate_indices.numel() == 0:
            return candidate_indices

        device = candidate_indices.device

        num_candidate = candidate_indices.numel()
        num_scale = max(1, int(num_candidate * scale_ratio))
        num_final = max(1, int(num_candidate * final_ratio))

        num_scale = min(num_scale, num_candidate)
        num_final = min(num_final, num_scale)

        # gaussian_scale = torch.max(self.get_scaling.detach(), dim=1).values
        gaussian_scale = torch.prod(self.get_scaling, dim=1)
        gaussian_scale = torch.log(gaussian_scale + 1e-12)
        candidate_scale = gaussian_scale[candidate_indices]
        _, top_idx = torch.topk(candidate_scale, k=num_scale, largest=True)
        scale_selected = candidate_indices[top_idx]

        xyz = self.get_xyz.detach()[scale_selected]
        fps_local_idx = self.farthest_point_sampling(xyz, num_final)
        final_indices = scale_selected[fps_local_idx]

        return final_indices

    def build_semantic_densify_mask(self, sim_score, sim_valid_mask, iteration, global_sample_ratio):
        """
        similarity 분포를 이용하여 semantic densification mask 생성

        Returns
        -------
        semantic_densify_mask : [N] bool
        """

        semantic_densify_mask = torch.zeros_like(sim_score, dtype=torch.bool)

        if not sim_valid_mask.any():
            return semantic_densify_mask

        sim_valid = sim_score[sim_valid_mask]

        sim_mean = sim_valid.mean()
        sim_std = sim_valid.std(unbiased=False)


        # ----------------------------
        # 세 개의 similarity 구간
        # ----------------------------

        mask_high = (sim_valid_mask & (sim_score >= sim_mean - 2.0 * sim_std) & (sim_score < sim_mean - 1.0 * sim_std))

        mask_mid = (sim_valid_mask & (sim_score >= sim_mean - 3.0 * sim_std) & (sim_score < sim_mean - 2.0 * sim_std))

        mask_low = (sim_valid_mask & (sim_score < sim_mean - 3.0 * sim_std))

        masks = [mask_high, mask_mid, mask_low]

        counts = torch.tensor(
            [m.sum().item() for m in masks],
            dtype=torch.float32,
            device=sim_score.device
        )

        total_candidate = counts.sum()

        if total_candidate <= 0:
            return semantic_densify_mask

        # ----------------------------
        # 구간 비율
        # ----------------------------

        region_ratio = counts / sim_score.numel()
        valid_region = counts > 0

        inv_ratio = torch.zeros_like(region_ratio)

        # sampling ratio
        # ratio not normalized
        # sample_ratio = 1.0 / (region_ratio * 100.0)

        # ratio normalized
        inv_ratio[valid_region] = (1.0 / region_ratio[valid_region])
        sample_ratio = inv_ratio / inv_ratio.sum()

        if inv_ratio.sum() > 0:
            sample_ratio = (inv_ratio / inv_ratio.sum())
        else:
            return semantic_densify_mask

        sample_ratio = (sample_ratio * global_sample_ratio).clamp(max=1.0)

        # ----------------------------
        # 각 구간 sampling
        # ----------------------------

        gaussian_scale = torch.prod(self.get_scaling.detach(), dim=1)

        for mask, ratio in zip(masks, sample_ratio):
            candidate_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
            num_candidate = candidate_indices.numel()
            if num_candidate == 0:
                continue

            #
            # 여기 핵심
            #
            num_sample = max(1, int(num_candidate * ratio.item()))
            num_sample = min(num_sample, num_candidate)

            #random sampling
            perm = torch.randperm(num_candidate, device=sim_score.device)
            selected_indices = candidate_indices[perm[:num_sample]]
            semantic_densify_mask[selected_indices] = True

        return semantic_densify_mask
    
    def densify_and_prune_by_similarity(self, max_grad, min_opacity, extent, max_screen_size, radii, iteration, model_path):
        
        grads = self.xyz_gradient_accum / (self.denom + 1e-8)
        grads[torch.isnan(grads)] = 0.0
        grads[torch.isinf(grads)] = 0.0

        if grads.dim() == 2 and grads.shape[1] == 1:
            grad_mag = grads.squeeze(1)
        else:
            grad_mag = torch.norm(grads, dim=-1)

        grad_mask = grad_mag >= max_grad

        sim_score = self.get_similarity_score.squeeze(1).detach()
        sim_valid_mask = ((self.sim_view_count.squeeze(1) > 0) & (sim_score >0))

        semantic_densify_mask = self.build_semantic_densify_mask(sim_score, sim_valid_mask, iteration, global_sample_ratio=0.5)
        selected_pts_mask_old = grad_mask | semantic_densify_mask

        self.tmp_radii = radii

        self.densify_and_clone_by_selected_mask(selected_pts_mask_old, extent)
        self.densify_and_split_by_selected_mask(selected_pts_mask_old, extent)
        
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)


        self.prune_points(prune_mask)

        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()
    
#####
