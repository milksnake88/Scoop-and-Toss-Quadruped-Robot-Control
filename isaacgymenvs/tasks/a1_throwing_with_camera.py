import numpy as np
import os
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

#from isaacgymenvs.tasks.a1_with_shovel import A1WithShovel
#from isaacgymenvs.tasks.a1_with_shovel_dagger_passive_joint import A1WithShovelDaggerPassiveJoint
#from isaacgymenvs.tasks.a1_with_shovel_dagger import A1WithShovelDagger
from isaacgymenvs.tasks.a1_with_shovel_passive_joint_with_camera import A1WithShovelPassiveJointWithCamera

class A1ThrowingWithCamera(A1WithShovelPassiveJointWithCamera):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        self.bed_bottom_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "bed_bottom")
        self.prop_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.prop_handles[0], "prop")
        self.bed_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "bed")
        self.shovel_bottom_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FL_shovel_bottom")
        self.shovel_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FL_shovel")
        self.base_up_vector = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_up_vector[:, 2] = 1. # bed joint position in base frame
        self.bed_position_local = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.bed_position_local[:, 2] = 0.06 # bed joint position in base frame

    def compute_reward(self):
        # landing reward
        rew_landing = torch.zeros_like(self.landing, dtype=torch.float, device=self.device, requires_grad=False)
        for env_idx in range(self.num_envs):
            if (torch.norm(self.contact_forces[env_idx, self.bed_bottom_index, :]) >0.):
                self.landing[env_idx] += 1
                rew_landing[env_idx] = 100

        # torque penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * -0.000025

        # distance reward
        bed_position = self.get_transformed_position(point_local=self.bed_position_local)
        prop_position = self.prop_root_states[:, 0:3]
        bed_prop_distance = torch.norm(prop_position - bed_position, dim=1)
        rew_bed_prop_distance = torch.exp(-bed_prop_distance/0.25)

        # throwing up reward
        box_position_upward = self.prop_root_states[:, 2]
        rew_box_position_upward = box_position_upward

        # action rate penalty
        action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        rew_action_rate = -0.0009 * action_rate

        # joint acceleration penalty
        joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1)
        rew_joint_acc = -0.00009 * joint_acc

        base_quat = self.a1_root_states[:, 3:7]
        base_ang_vel = quat_rotate_inverse(base_quat, self.a1_root_states[:, 10:13])
        base_ang_vel_z_square = torch.sqrt(torch.square(base_ang_vel[:, 2]))
        rew_base_ang_vel_z = -0.00001 * base_ang_vel_z_square

        # picking reward
        shovel_bottom_contact_forces = torch.norm(self.contact_forces[:, self.shovel_bottom_index, :], dim=1)
        prop_contact_forces = torch.norm(self.contact_forces[:, self.prop_index, :], dim=1)
        contact_differences = torch.sqrt(torch.square(shovel_bottom_contact_forces-prop_contact_forces))
        rew_prop_is_picked = (shovel_bottom_contact_forces>0.) & (prop_contact_forces>0.) & (contact_differences<0.01)
        picked_ids = rew_prop_is_picked.nonzero(as_tuple=False).squeeze(-1)
        self.picked[picked_ids] += 1

        # total
        total_reward = 100 * rew_prop_is_picked + rew_landing + (30 * rew_bed_prop_distance * rew_box_position_upward) + rew_action_rate + rew_joint_acc + rew_torque
        """
        print("rew_prop_is_picked = ", rew_prop_is_picked)
        print("rew_landing = ", rew_landing)
        print("rew_bed_prop_distance = ", rew_bed_prop_distance)
        print("rew_box_position_upward = ", rew_box_position_upward)
        print("rew_action_rate = ", rew_action_rate)
        print("rew_joint_acc = ", rew_joint_acc)
        print("rew_torque = ", rew_torque)
        print("total_reward = ", total_reward)
        """
        total_reward = torch.clip(total_reward, 0., None)
        self.rew_buf[:] = total_reward.detach()

        # reset agents
        reset = torch.norm(self.contact_forces[:, self.trunk_index, :], dim=1) > 1.
        reset = reset | torch.any(torch.norm(self.contact_forces[:, self.calf_indices, :], dim=2) > 1., dim=1)
        reset = reset | torch.any(torch.norm(self.contact_forces[:, self.thigh_indices, :], dim=2) > 1., dim=1)
        bed_contact_force_norm = torch.norm(self.contact_forces[:, self.bed_index, :], dim=1)
        reset = reset | (bed_contact_force_norm > 120.).bool()

        shovel_contact_force_norm = torch.norm(self.contact_forces[:, self.shovel_index, :], dim=1)
        for env_idx in range(self.num_envs):
            if self.progress_buf[env_idx] > 3 /self.dt:
                if (torch.norm(self.contact_forces[env_idx, self.prop_index, :]) > 0.) & ~self.landing[env_idx].bool() & ~bed_contact_force_norm[env_idx].bool():
                    print("ddddddddddddddddddddddddddvreset")
                    reset[env_idx] = True
                elif self.landing[env_idx] > 5 / self.dt:
                    self.curriculum_update[env_idx] = 1
                    reset[env_idx] = True

        """
        # for curriculum learning
        for env_idx in range(self.num_envs):
            if self.landing[env_idx] > 5 / self.dt:
                self.curriculum_update[env_idx] = 1
                reset[env_idx] = True
            if self.progress_buf[env_idx] > ?? / self.dt:
                if (torch.norm(self.contact_forces[env_idx, self.prop_index, :]) > 0.) & ~self.landing[env_idx].bool() & ~bed_contact_force_norm[env_idx].bool() & ~self.picked[env_idx].bool() & ~shovel_contact_force_norm[env_idx].bool():
                    reset[env_idx] = True
        """

        time_out = self.progress_buf >= self.max_episode_length/5 - 1  # no terminal reward for time-outs
        reset = reset | time_out

        self.reset_buf[:] = reset

