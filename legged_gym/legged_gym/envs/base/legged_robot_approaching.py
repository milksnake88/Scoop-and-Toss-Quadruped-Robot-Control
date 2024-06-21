# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import *
from legged_gym.utils.helpers import class_to_dict
from scipy.spatial.transform import Rotation as R
from .toss_and_load_robot_config import TossAndLoadRobotCfg
from .toss_and_load_robot import TossAndLoadRobot

#from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

def euler_from_quaternion(quat_angle):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = torch.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = torch.clip(t2, -1, 1)
        pitch_y = torch.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = torch.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z # in radians

def quaternion_to_6D_matrix(base_quat): # q = self.root_states[:, 3:7]
    # Extract the values from root_states
    x, y, z, w = torch.unbind(base_quat, -1)
    two = 2.0 / (base_quat*base_quat).sum(-1)
    matrix = torch.stack(
        (
            1-two*(y*y+z*z),
            two*(x*y-z*w),
            two*(x*y+z*w),
            1-two*(x*x+z*z),
            two*(x*z-y*w),
            two*(y*z+x*w),
        ),
        -1,
    )

    return matrix


class LeggedRobotApproaching(TossAndLoadRobot):
    def _update_goals(self): #TODO: 여기서 새롭게 정의
        self.target_pos_rel = self.prop_root_states[:, :3] - self.rigid_body_states[:, self.shovel_index, 0:3]
        self.target_pos_rel_norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        self.target_vec_norm = self.target_pos_rel / (self.target_pos_rel_norm + 1e-5)
        self.target_yaw = torch.atan2(self.target_vec_norm[:, 1], self.target_vec_norm[:, 0])

        self.robot_to_prop_rel = self.prop_root_states[:, :3] - self.robot_root_states[:, :3]
        self.robot_to_prop_rel_norm = torch.norm(self.robot_to_prop_rel, dim=-1, keepdim=True)
        self.robot_to_prop_vec_norm = self.robot_to_prop_rel / (self.robot_to_prop_rel_norm + 1e-5)

        horizontal_distance = torch.norm(self.robot_to_prop_vec_norm[:, :2], dim=-1)
        self.target_pitch = torch.atan2(self.robot_to_prop_vec_norm[:, 2], horizontal_distance)

        # pos_of_prop_wrt_a1_base
        prop_root_pos = self.prop_root_states[:, 0:3]
        pos_of_prop_wrt_robot_base = self.get_transformed_position(point_global=prop_root_pos)

        camera_pos = torch.tensor(self.cfg.depth.position, dtype=torch.float, device=self.device, requires_grad=False)
        camera_to_prop_rel = pos_of_prop_wrt_robot_base - camera_pos
        camera_to_prop_rel_norm = torch.norm(camera_to_prop_rel, dim=-1, keepdim=True)
        camera_to_prop_vec_norm = camera_to_prop_rel / (camera_to_prop_rel_norm + 1e-5)
        # Calculate horizontal and vertical angles
        self.horizontal_angle = torch.abs(torch.atan2(camera_to_prop_vec_norm[:, 1], camera_to_prop_vec_norm[:, 0]))
        horizontal_distance = torch.norm(camera_to_prop_vec_norm[:, :2], dim=-1)
        self.vertical_angle = torch.abs(torch.atan2(camera_to_prop_vec_norm[:, 2], horizontal_distance))


    def check_termination(self):
        """ Check if environments need to be reset
        """
        reset = torch.norm(self.contact_forces[:, self.robot_base_index, :], dim=1) > 1.
        reset = reset | torch.any(torch.norm(self.contact_forces[:, self.calf_indices, :], dim=2) > 1., dim=1)
        reset = reset | torch.any(torch.norm(self.contact_forces[:, self.thigh_indices, :], dim=2) > 1., dim=1)
        bed_contact_force_norm = torch.norm(self.contact_forces[:, self.robot_bed_index, :], dim=1)
        reset = reset | (bed_contact_force_norm > 120.).bool()
        reset = reset | (self.target_pos_rel_norm < 0.08).squeeze().bool()
        time_out = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        reset = reset | time_out

        self.reset_buf[:] = reset

    def reset_idx(self, env_ids): #TODO: 정리
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_robot_root_vel[:] = 0.
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.  # reset obs history buffer TODO no 0s
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.reach_goal_timer[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.episode_length_buf[env_ids] = 0

        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _resample_commands(self, env_ids): #TODO: 여기서 새롭게 정의
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 2] *= torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_clip

        # set small commands to zero
        self.commands[env_ids, :2] *= torch.abs(self.commands[env_ids, 0:1]) > self.cfg.commands.lin_vel_clip


    ################## parkour rewards ##################

    def _reward_tracking_goal_vel(self):
        #cur_vel = self.rigid_body_states[:, self.shovel_index, 7:9]
        cur_vel = self.robot_root_states[:, 7:9]
        #rew = torch.minimum(torch.sum(self.target_vec_norm[:, :2] * cur_vel, dim=-1), self.commands[:, 0]) / (self.commands[:, 0] + 1e-5)
        rew = torch.minimum(torch.sum(self.robot_to_prop_vec_norm[:, :2] * cur_vel, dim=-1), self.commands[:, 0])
        return rew

    def _reward_tracking_yaw(self):
        shovel_cur_quat = self.rigid_body_states[:, self.shovel_index, 3:7]
        _, _, yaw = euler_from_quaternion(shovel_cur_quat)
        rew = torch.exp(-torch.abs(self.target_yaw - yaw))
        return rew

    """
    def _reward_tracking_pitch(self):
        distance_scaling = torch.exp(-self.robot_to_prop_rel_norm/0.5).squeeze(-1)
        rew = distance_scaling * torch.exp(-torch.abs(self.target_pitch - self.pitch))
        return rew
    """

    def _reward_looking_at_prop(self):
        hfov_rad = torch.deg2rad(torch.tensor(self.cfg.depth.horizontal_fov/2, dtype=torch.float, device=self.device, requires_grad=False))
        vfov_rad = torch.deg2rad(torch.tensor(self.cfg.depth.vertical_fov/2, dtype=torch.float, device=self.device, requires_grad=False))
        rew = (self.horizontal_angle <= hfov_rad) & (self.vertical_angle <= vfov_rad)
        return rew

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_joint_indices] - self.default_dof_pos[:, self.hip_joint_indices]), dim=1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1)

