import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.a1_with_shovel import A1WithShovel
from isaacgymenvs.tasks.a1_with_shovel_dagger_passive_joint import A1WithShovelDaggerPassiveJoint
from isaacgymenvs.tasks.a1_without_shovel import A1WithoutShovel

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

class A1ApproachingExtremeParkourVersion(A1WithShovelDaggerPassiveJoint):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        self.prop_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.prop_handles[0], "prop")
        self.commands = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands[:, 0] = 1.0


    def compute_reward(self):
        # torque penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * -0.000025

        # action rate penalty
        action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        rew_action_rate = -0.0009 * action_rate

        # joint acceleration penalty
        joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1)
        rew_joint_acc = -0.00009 * joint_acc

        # angular velocity penalty
        # a1_ang_vel = torch.norm(self.a1_root_states[:, 10:13], dim=1)
        # rew_ang_vel = -0.000003 * a1_ang_vel

        base_quat = self.a1_root_states[:, 3:7]
        base_lin_vel = quat_rotate_inverse(base_quat, self.a1_root_states[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, self.a1_root_states[:, 10:13])

        # reward tracking goal vel
        target_pos_rel = self.prop_root_states[:, :2] - self.a1_root_states[:, :2]
        target_pos_rel = self.prop_root_states[:, :3] - self.a1_root_states[:, :3]
        target_pos_rel_norm = torch.norm(target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = target_pos_rel / (target_pos_rel_norm + 1e-5)
        #self.draw_lines(self.a1_root_states[:, :3]+target_vec_norm, self.a1_root_states[:, :3])
        cur_vel = self.a1_root_states[:, 7:9]
        #self.draw_lines(self.a1_root_states[0, :3], self.a1_root_states[0, :3]+cur_vel[0, :])
        #print(torch.norm(target_vec_norm, dim=-1))
        rew_tracking_goal_vel = torch.minimum(torch.sum(target_vec_norm[:, :2] * cur_vel, dim=-1), self.commands[:, 0] + 1e-5)
        #print(torch.sum(target_vec_norm * cur_vel, dim=-1))
        rew_into_1m = target_pos_rel_norm < 1
        print("rew intdasa", rew_into_1m)

        cur_vel_norm = torch.norm(cur_vel, dim=-1, keepdim=True)
        cur_vel_normed = cur_vel / cur_vel_norm
        # 목표 벡터와 현재 속도 벡터의 코사인 유사도 계산
        cos_similarity = torch.sum(target_vec_norm[:, :2] * cur_vel_normed, dim=1)
        #self.draw_lines(self.a1_root_states[1, :3], self.a1_root_states[1, :3]+cur_vel_normed[1, :])

        # reward tracking yaw
        target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        _, pitch, yaw = euler_from_quaternion(base_quat)
        rew_tracking_yaw = torch.exp(-torch.abs(target_yaw - yaw))

        # reward tracking pitch
        horizontal_distance = torch.norm(target_vec_norm[:, :2], dim=-1)
        #print(horizontal_distance)
        #print(target_vec_norm[:, 2])
        target_pitch = torch.atan2(target_vec_norm[:, 2], horizontal_distance)
        #print(target_pitch)
        exponential_scaling = torch.exp(-target_pos_rel_norm/0.5).squeeze(-1)
        rew_tracking_pitch = exponential_scaling * torch.exp(-torch.abs(target_pitch - pitch))
        #print("ddd", rew_tracking_pitch)
        # reward ang vel xy


        #rew_hip_pos = torch.sum(torch.square(self.dof_pos[:, self.hip_joint_indices] - self.default_dof_pos[:, self.hip_joint_indices]), dim=1)

        # reward ang vel xy
        rew_ang_vel_xy = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)

        #total_reward = -1. * lin_vel_error -1. * rew_lateral_deviation + 0.05 * rew_base_ang_vel -1e-5 * rew_energy + 2.
        #total_reward = -1. * lin_vel_error -1. * rew_lateral_deviation + -1e-5 * rew_energy + 2.
        #total_reward = 30 * torch.exp(-a1_to_prop_dis/2) + rew_torque + rew_action_rate + rew_joint_acc
        #total_reward = 3 * rew_tracking_goal_vel + 0.4 * rew_tracking_yaw -0.05 * rew_ang_vel_xy -0.5 * rew_foot_stumble + rew_torque + rew_action_rate + rew_joint_acc
        total_reward = 0.4 * rew_tracking_yaw -0.05 * rew_ang_vel_xy + rew_torque + rew_action_rate + rew_joint_acc + rew_tracking_pitch
        #print("rew_tracking_goal_vel", 3*rew_tracking_goal_vel)
        #print("rew_tracking_yaw",  0.4 * rew_tracking_yaw)
        #print("rew_foot_stumble", -1 * rew_foot_stumble)
        total_reward = torch.clip(total_reward, 0., None)
        self.rew_buf[:] = total_reward.detach()

        reset = torch.norm(self.contact_forces[:, self.trunk_index, :], dim=1) > 1.
        reset = reset | torch.any(torch.norm(self.contact_forces[:, self.calf_indices, :], dim=2) > 1., dim=1)
        reset = reset | torch.any(torch.norm(self.contact_forces[:, self.thigh_indices, :], dim=2) > 1., dim=1)
        bed_contact_force_norm = torch.norm(self.contact_forces[:, self.bed_index, :], dim=1)
        reset = reset | (bed_contact_force_norm > 120.).bool()
        reset = reset | (target_pos_rel_norm < 1).squeeze().bool()
        time_out = self.progress_buf >= self.max_episode_length/3- 1  # no terminal reward for time-outs
        reset = reset | time_out

        self.reset_buf[:] = reset


