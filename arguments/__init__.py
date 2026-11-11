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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        self.sim_start_iter = 3000 ###
        self.sim_interval = 10 ###
        # ---- semantic densification (decisive experiment); defaults = proposed method ----
        self.sem_signal = "dino"          # guidance map: dino | ssim | l1 | l1_blur
        self.sem_mode = "ours"            # ours (goodness, scale weight, stratified) | r9 ([9]-style error rule)
        self.sem_select = "stratified"    # ours mode: stratified | random_matched | radius_matched
        self.sem_view_norm = "auto"       # within-view aggregation: auto | mean | sum   (auto: ours=mean, r9=sum)
        self.sem_view_agg = "auto"        # across-view aggregation: auto | mean | min | max (auto: ours=mean, r9=max)
        self.sem_lambda_scale = 0.3       # lambda_s of the scale weight (0 disables it)
        self.sem_scale_mode = "linear"    # linear: S*(1 - lambda_s*R)  |  power: S^(1 + lambda_s*R)
        self.sem_radius_min = 1.0         # R_min (px)
        self.sem_radius_max = 20.0        # R_max (px)
        self.sem_sample_ratio = 0.3       # rho of the stratified sampling
        self.sem_r9_threshold = 0.1       # r9 mode: error threshold
        self.sem_r9_cap = 0.05            # r9 mode: max fraction of Gaussians added per step
        self.sem_dino_chunk = 0           # >0: upsample DINOv2 features in channel chunks of this size (less VRAM)
        self.sem_blur_sigma = 4.0         # l1_blur: Gaussian blur sigma (px)
        self.sem_shadow = "none"          # extra signal back-projected through the 3rd aux channel (logging only)
        self.sem_split = "adc"            # adc (densify_and_split_by_selected_mask) | mask2
        self.sem_log = False              # write sem_log.jsonl / sem_snapshots
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
