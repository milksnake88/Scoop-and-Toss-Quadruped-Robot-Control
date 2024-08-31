import numpy as np
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from torch.nn.functional import normalize

from isaacgymenvs.tasks.a1_with_shovel import A1WithShovel
from isaacgymenvs.tasks.a1_with_shovel_passive_joint import A1WithShovelPassiveJoint
from isaacgymenvs.tasks.a1_with_shovel_dagger_passive_joint import A1WithShovelDaggerPassiveJoint
from isaacgymenvs.tasks.a1_with_shovel_passive_joint_with_camera import A1WithShovelPassiveJointWithCamera
from isaacgymenvs.tasks.a1_with_shovel_dagger_fixed_joint import A1WithShovelDaggerFixedJoint

class A1Throwing(A1WithShovelDaggerPassiveJoint):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        self.bed_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "bed")
        self.bed_bottom_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "bed_bottom")
        self.prop_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.prop_handles[0], "prop")
        self.shovel_bottom_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FL_shovel_bottom")
        self.shovel_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FL_shovel")
        self.bed_position_local = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.bed_position_local[:, 2] = 0.06 #0.057 # bed joint position in base frame
        self.base_up_vector = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_up_vector[:, 2] = 1. # bed joint position in base frame

    def draw_lines(self, start, end):
        # type: (Tensor, Tensor) -> None

        start_np = start.squeeze().cpu().numpy()
        end_np = end.squeeze().cpu().numpy()

        line_vertices = [[start_np[0], start_np[1], start_np[2]],
                         [end_np[0], end_np[1], end_np[2]]]
        num_lines = 1
        line_colors = [1., 0., 0.]

        self.gym.clear_lines(self.viewer)
        self.gym.add_lines(self.viewer, self.envs[0], num_lines, line_vertices, line_colors)


    def get_global_position(self, point_local=None, vector_local=None):
        base_pos = self.a1_root_states[:, 0:3].squeeze().cpu().numpy()
        base_ori = self.a1_root_states[:, 3:7].squeeze().cpu().numpy()

        global_position = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        if (self.num_envs == 1):
            base_pos = base_pos.reshape((1, *base_pos.shape))
            base_ori = base_ori.reshape((1, *base_ori.shape))
        for i in range(self.num_envs):
            transform = gymapi.Transform(gymapi.Vec3(base_pos[i][0], base_pos[i][1], base_pos[i][2]),
                                         gymapi.Quat(base_ori[i][0], base_ori[i][1], base_ori[i][2], base_ori[i][3]))
            if point_local is not None:
                transformed_position = transform.transform_point(gymapi.Vec3(point_local[i][0], point_local[i][1], point_local[i][2]))
                global_position[i] = torch.tensor([[transformed_position.x, transformed_position.y, transformed_position.z]],  dtype=torch.float, device=self.device, requires_grad=False)
            if vector_local is not None:
                transformed_vector = transform.transform_vector(gymapi.Vec3(vector_local[i][0], vector_local[i][1], vector_local[i][2]))
                global_position[i] = torch.tensor([[transformed_vector.x, transformed_vector.y, transformed_vector.z]],  dtype=torch.float, device=self.device, requires_grad=False)

        return global_position


    def compute_bed_prop_distance(self):
        bed_position = self.get_global_position(point_local=self.bed_position_local)
        #print("1", bed_position)
        #print("2", self.rb_states[:, self.bed_bottom_index, 0:3])
        box_position = self.prop_root_states[:, 0:3]
        distance = torch.norm(box_position - bed_position, dim=1)

        #check
        #if self.num_envs == 1:
            #self.draw_lines(bed_position, box_position)
        camera_position = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        camera_position[:, 0] = 0.27
        camera_position[:, 2] = 0.0075
        camera_position[:, 1] = 0.033
        cam_pos = self.get_global_position(camera_position)

        #self.draw_lines(cam_pos, box_position)

        return distance

    def compute_box_position_along_bed(self):
        bed_normal_vector = self.get_global_position(vector_local=self.base_up_vector)

        bed_prop_distance = self.compute_bed_prop_distance()

        bed_position = self.get_global_position(point_local=self.bed_position_local)
        box_position = self.prop_root_states[:, 0:3]
        bed_to_box_vector = box_position - bed_position

        box_position_in_normal_direction = torch.einsum('ij,ij->i', bed_to_box_vector, bed_normal_vector)

        box_position_in_plane_direction = torch.sqrt(torch.square(bed_prop_distance)-torch.square(box_position_in_normal_direction))

        #print("1 ", bed_prop_distance)
        #print(box_position_in_normal_direction)
        #print(box_position_in_plane_direction)

        return box_position_in_normal_direction, box_position_in_plane_direction

    def compute_reward(self):

        box_position_in_noraml_direction, box_position_in_plane_direction = self.compute_box_position_along_bed()
        # landing reward
        #print(box_position_in_noraml_direction)
        rew_landing = torch.zeros_like(self.landing, dtype=torch.float, device=self.device, requires_grad=False)
        for env_idx in range(self.num_envs):
            if (0<box_position_in_noraml_direction[env_idx] < 0.05) & (box_position_in_plane_direction[env_idx] < 0.12):
                self.landing[env_idx] += 1
                rew_landing[env_idx] = 100

        #print(self.landing)

        # torque penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * -0.000025
        #print("torque", rew_torque)

        # distance reward
        bed_prop_distance = self.compute_bed_prop_distance()
        #print("2", bed_prop_distance)
        #bed_position = self.rb_states[:, self.bed_bottom_index, 0:3]
        #box_position = self.prop_root_states[:, 0:3]
        #distance = torch.norm(box_position - bed_position, dim=1)
        #print("1", distance)
        #rew_bed_prop_distance = torch.exp(-bed_prop_distance/0.25)
        #print(rew_bed_prop_distance)

        # throwing up reward
        box_position_upward = self.prop_root_states[:, 2]
        rew_box_position_upward = box_position_upward
        #print(box_position_upward)

        # action rate penalty
        action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        rew_action_rate = -0.0009 * action_rate
        #print("action", rew_action_rate)

        base_quat = self.a1_root_states[:, 3:7]
        base_ang_vel = quat_rotate_inverse(base_quat, self.a1_root_states[:, 10:13])
        base_ang_vel_z_square = torch.sqrt(torch.square(base_ang_vel[:, 2]))
        rew_base_ang_vel_z = -0.00002 * base_ang_vel_z_square

        # joint acceleration penalty
        joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1)
        rew_joint_acc = -0.00009 * joint_acc
        #print("joint", rew_joint_acc)

        #total_reward = rew_landing + rew_torque + (30 * rew_bed_prop_distance + rew_box_position_upward) + rew_action_rate + rew_joint_acc + rew_base_ang_vel_z
        #print("distance", rew_bed_prop_distance)
        #print("upward", rew_box_position_upward)
        #total_reward = torch.clip(total_reward, 0., None)
        #self.rew_buf[:] = total_reward.detach()

        # distance reward
        a1_root_position = self.a1_root_states[:, 0:3]
        prop_root_position = self.prop_root_states[:, 0:3]
        a1_prop_distance = torch.norm(a1_root_position-prop_root_position, dim=1)



        #rew_a1_prop_distance = torch.clip(4-torch.abs(0.1-a1_prop_distance), 0, None)
        rew_a1_prop_distance = 50 * torch.exp(-a1_prop_distance/0.5)
        #print(rew_a1_prop_distance)
        #print(self.contact_forces[:, self.bed_index, :])
        # reset agents
        #print("shovel_bottom_contact_forces = ", torch.norm(self.contact_forces[:, self.shovel_bottom_index, :], dim=1))
        #print("prop_contact_forces = ", torch.norm(self.contact_forces[:, self.prop_index, :], dim=1))

        # picking policy
        shovel_bottom_contact_forces = torch.norm(self.contact_forces[:, self.shovel_bottom_index, :], dim=1)
        prop_contact_forces = torch.norm(self.contact_forces[:, self.prop_index, :], dim=1)
        contact_differences = torch.sqrt(torch.square(shovel_bottom_contact_forces-prop_contact_forces))
        rew_prop_is_picked = (shovel_bottom_contact_forces>0.) & (prop_contact_forces>0.) & (contact_differences<0.01)

        # shovel contact reward
        shovel_contact_force_norm = torch.norm(self.contact_forces[:, self.shovel_index, :], dim=1)
        shovel_prop_contact_differences = torch.sqrt(torch.square(shovel_contact_force_norm-prop_contact_forces))
        rew_shovel_prop_contact = (shovel_contact_force_norm>0.) & (prop_contact_forces>0.) & (shovel_prop_contact_differences<0.005)

        #total_reward = rew_landing + rew_torque + (30 * rew_bed_prop_distance * rew_box_position_upward) + rew_joint_acc
        #total_reward = 100 * rew_prop_is_picked + rew_torque + rew_joint_acc + rew_action_rate
        #total_reward = torch.clip(total_reward, 0., None)

        reset = torch.norm(self.contact_forces[:, self.trunk_index, :], dim=1) > 1.
        #print("trunk", reset)
        #reset = reset | torch.any(torch.norm(self.contact_forces[:, self.calf_indices, :], dim=2) > 1., dim=1)
        #print("calf", reset)
        reset = reset | torch.any(torch.norm(self.contact_forces[:, self.thigh_indices, :], dim=2) > 1., dim=1)
        #print("thigh", reset)
        bed_contact_force_norm = torch.norm(self.contact_forces[:, self.bed_index, :], dim=1)
        reset = reset | (bed_contact_force_norm > 120.).bool()
        #print("bed_contact_force", bed_contact_force_norm)
        #print("bed", reset)

        for env_idx in range(self.num_envs):
            if self.progress_buf[env_idx] > 0.7 / self.dt:
                if (torch.norm(self.contact_forces[env_idx, self.prop_index, :]) > 0.) & ~ self.landing[env_idx].bool() & ~ bed_contact_force_norm[env_idx].bool():
                    reset[env_idx] = True
                    print(env_idx, "prop", reset)
                elif self.landing[env_idx] > 5 / self.dt:
                    reset[env_idx] = True
                    self.randing_cnt[env_idx] += 1
                    print(env_idx, "landing", reset)

        time_out = self.progress_buf >= self.max_episode_length / 5- 1  # no terminal reward for time-outs
        reset = reset | time_out

        self.reset_buf[:] = reset



