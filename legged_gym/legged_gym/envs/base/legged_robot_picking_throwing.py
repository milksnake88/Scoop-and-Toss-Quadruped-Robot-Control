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

class LeggedRobotPickingThrowing(TossAndLoadRobot):
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.landing[env_ids] = 0

    #----------------------------------------
    def _init_buffers(self):
        super()._init_buffers()
        self.landing = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)


    def check_termination(self):
        """ Check if environments need to be reset
        """
        reset = torch.norm(self.contact_forces[:, self.robot_base_index, :], dim=1) > 1.
        reset = reset | torch.any(torch.norm(self.contact_forces[:, self.calf_indices, :], dim=2) > 1., dim=1)
        reset = reset | torch.any(torch.norm(self.contact_forces[:, self.thigh_indices, :], dim=2) > 1., dim=1)
        bed_contact_force_norm = torch.norm(self.contact_forces[:, self.robot_bed_index, :], dim=1)
        reset = reset | (bed_contact_force_norm > 120.).bool()

        progress_cond = self.episode_length_buf > ( 0.6 / self.dt)
        prop_contact = torch.norm(self.contact_forces[:, self.prop_index, :], dim=1) > 0.
        not_landing = ~self.landing.bool()
        not_bed_contact = ~bed_contact_force_norm.bool()
        reset = reset | (progress_condition & prop_contact & not_landing & not_bed_contact)

        time_out = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        reset = reset | time_out

        self.reset_buf[:] = reset


    ################## parkour rewards ##################

    def _reward_landing(self):
        rew = torch.norm(self.contact_forces[:, self.robot_bed_bottom_index, :], dim=1) > 0.
        return rew

    def _reward_throwing(self):
        bed_position = self.rigid_body_state[:, self.robot_bed_bottom_index, 0:3]
        prop_position = self.prop_root_states[:, 0:3]
        bed_prop_distance = torch.norm(prop_position - bed_position, dim=1)
        rew = torch.exp(-bed_prop_distance/0.25) * prop_position[:, 2]
        return rew

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1)


