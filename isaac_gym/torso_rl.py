import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, \
    to_torch, get_axis_params, torch_rand_float, tensor_clamp, compute_heading_and_up, compute_rot, normalize_angle

from isaacgymenvs.tasks.base.vec_task import VecTask

class TorsoClimb(VecTask):
    
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        pass

    def create_sim(self):
        pass

    def _create_ground_plane(self):
        pass

    def _create_envs(self, num_envs, spacing, num_per_row):
        pass

    def compute_reward(self, actions):
        pass

    def compute_observations(self):
        pass

    def reset_idx(self, env_ids):
        pass

    def pre_physics_step(self, actions):
        pass

    def post_physics_step(self):
        pass

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_torso_reward():
    pass

@torch.jit.script
def compute_torso_observations():
    pass