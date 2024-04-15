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

from legged_gym.envs.base.toss_and_load_robot_config import TossAndLoadRobotCfg, TossAndLoadRobotCfgPPO


class A1PickingThrowingCfg( TossAndLoadRobotCfg ):
    class env( TossAndLoadRobotCfg.env ):
        episode_length_s = 10
    class init_state( TossAndLoadRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.35] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': -0.0 ,  # [rad]
            'RR_hip_joint': -0.0,   # [rad]

            'FL_thigh_joint': 0.67,     # [rad]
            'RL_thigh_joint': 0.67,   # [rad]
            'FR_thigh_joint': 0.67,     # [rad]
            'RR_thigh_joint': 0.67,   # [rad]

            'FL_calf_joint': -1.3,   # [rad]
            'RL_calf_joint': -1.2,    # [rad]
            'FR_calf_joint': -1.3,  # [rad]
            'RR_calf_joint': -1.2,    # [rad]
        }

    class control( TossAndLoadRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 60.}  # [N*m/rad]
        damping = {'joint': 3}     # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class asset( TossAndLoadRobotCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1_new.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1_with_shovel_description/urdf/a1_with_shovel.urdf'
        foot_name = "foot"
        thigh_name = "thigh"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]#, "thigh", "calf"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        collapse_fixed_joints = False
        replace_cylinder_with_capsule = False
        thickness = 0.003 # Thickness of the collision shapes. Sets how far objects should come to rest from the surface of this body
        vhacd_enabled = True # Whether convex decomposition is enabled. Used only with PhysX.

    class rewards( TossAndLoadRobotCfg.rewards ):
        class scales( TossAndLoadRobotCfg.rewards.scales):
            # landing rewards
            landing = 100
            # throwing rewards
            throwing = 30
            # regularization rewards
            torques = -0.000025
            action_rate = -0.001
            dof_acc = -0.0001

        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

    class prop( TossAndLoadRobotCfg.prop ):
        class init_state( TossAndLoadRobotCfg.prop.init_state):
            pos = [0.34, 0.13, 0.025] # x,y,z [m]
            rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
            lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
            ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

class A1PickingThrowingCfgPPO( TossAndLoadRobotCfgPPO ):
    class algorithm( TossAndLoadRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( TossAndLoadRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_a1'

