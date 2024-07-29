import numpy as np
import os
import torch
torch.set_printoptions(profile="full")

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask

from typing import Tuple, Dict
class A1MoE(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg

        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["alive"] = self.cfg["env"]["learn"]["aliveRewardScale"]
        self.rew_scales["landing"] = self.cfg["env"]["learn"]["landingRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

         # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # base init state
        base_pos = self.cfg["env"]["baseInitState"]["pos"]
        base_rot = self.cfg["env"]["baseInitState"]["rot"]
        base_v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        base_v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        base_state = base_pos + base_rot + base_v_lin + base_v_ang
        self.base_init_state = base_state

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        # box init state. TODO: add to cfg file
        #box_pos = [0.24, 0.13, 0.08]
        box_pos = [3., 0.13, 0.025]
        box_rot = [0., 0., 0., 1.]
        box_v_lin = [0., 0., 0.]
        box_v_ang = [0., 0., 0.]
        box_state = box_pos + box_rot + box_v_lin + box_v_ang
        self.box_init_state = box_state
        self.num_boxes = 4

        self.env_space = self.cfg["env"]["envSpacing"]
        self.grid_size = self.cfg["env"]["objectMap"]["gridSize"]
        self.sigma = self.cfg["env"]["objectMap"]["sigma"]

        self.cfg["env"]["numObservations"] = 3 + 48
        self.cfg["env"]["numActions"] = 12 + 12

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # other
        self.dt = self.sim_params.dt # 0.02
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"] # 50, episode length in seconds
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.actors_per_env = self.gym.get_sim_actor_count(self.sim) // self.num_envs
        self.dofs_per_env = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.dofs_per_env, 2)[..., 0]
        self.rb_states = gymtorch.wrap_tensor(_rb_states).view(self.num_envs, -1, 13)
        self.dof_vel = self.dof_state.view(self.num_envs, self.dofs_per_env, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)
        self.a1_root_states = self.root_states.view(self.num_envs, self.actors_per_env, 13)[:, 0, :]
        self.prop_root_states = self.root_states.view(self.num_envs, self.actors_per_env, 13)[:, 1:self.num_boxes+1, :]

        self.all_actor_indices = torch.arange(self.actors_per_env * self.num_envs, dtype=torch.int32, device=self.device).view(self.num_envs, self.actors_per_env)

        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        # initialize some data used later on
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.base_lin_vel_before = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.bed_position_local = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.bed_position_local[:, 2] = 0.06 # bed joint position in base frame

        foot_names = [s for s in self.body_names if "foot" in s]
        self.foot_indices = torch.zeros(len(foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        thigh_names = [s for s in self.body_names if "thigh" in s] # thigh and thigh_shoulder
        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        calf_names = [s for s in self.body_names if "calf" in s]
        self.calf_indices = torch.zeros(len(calf_names), dtype=torch.long, device=self.device, requires_grad=False)

        for i in range(len(foot_names)):
            self.foot_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], foot_names[i])
        for i in range(len(thigh_names)):
            self.thigh_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], thigh_names[i])
        for i in range(len(calf_names)):
            self.calf_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], calf_names[i])
        self.trunk_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "trunk")

        self.last_actions = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel, dtype=torch.float, device=self.device, requires_grad=False)
        self.landing = torch.zeros(self.num_envs, self.num_boxes, dtype=torch.float, device=self.device, requires_grad=False)
        self.picked = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        self.bed_bottom_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "bed_bottom")
        self.prop_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.prop_handles[0], "prop")
        self.bed_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "bed")
        self.shovel_bottom_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FL_shovel_bottom")
        self.bed_position_local = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.bed_position_local[:, 2] = 0.06 # bed joint position in base frame

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)


    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')

        # load a1 asset
        a1_asset_file = "urdf/a1_with_shovel_description/urdf/a1_with_shovel.urdf"
        a1_asset_options = gymapi.AssetOptions()
        a1_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        a1_asset_options.replace_cylinder_with_capsule = False
        a1_asset_options.flip_visual_attachments = True
        a1_asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        a1_asset_options.thickness = 0.003 #Thickness of the collision shapes. Sets how far objects should come to rest from the surface of this body
        a1_asset_options.disable_gravity = False
        a1_asset_options.vhacd_enabled= True

        a1_asset = self.gym.load_asset(self.sim, asset_root, a1_asset_file, a1_asset_options)
        self.num_dof = self.gym.get_asset_dof_count(a1_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(a1_asset) # 25

        a1_start_pose = gymapi.Transform()
        a1_start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        a1_start_pose.r = gymapi.Quat(*self.base_init_state[3:7])

        self.body_names = self.gym.get_asset_rigid_body_names(a1_asset)
        self.dof_names = self.gym.get_asset_dof_names(a1_asset)
        a1_dof_props = self.gym.get_asset_dof_properties(a1_asset)
        for i in range(self.num_dof):
            a1_dof_props['driveMode'][i] = 1 #gymapi.DOF_MODE_POS
            a1_dof_props['stiffness'][i] = 60.
            a1_dof_props['damping'][i] = 3.

        body_dict = self.gym.get_asset_rigid_body_dict(a1_asset)
        """
        {'base': 0, 'trunk': 1,
        'FL_hip': 2, 'FL_thigh_shoulder': 3, 'FL_thigh': 4, 'FL_calf': 5, 'FL_foot': 6, 'FL_shovel': 7,
        'FR_hip': 8, 'FR_thigh_shoulder': 9, 'FR_thigh': 10, 'FR_calf': 11, 'FR_foot': 12,
        'RL_hip': 13, 'RL_thigh_shoulder': 14, 'RL_thigh': 15, 'RL_calf': 16, 'RL_foot': 17,
        'RR_hip': 18, 'RR_thigh_shoulder': 19, 'RR_thigh': 20, 'RR_calf': 21, 'RR_foot': 22,
        'bed': 23, 'bed_bottom': 24, 'imu_link': 25}
        """
        a1_body_shape_indices = self.gym.get_asset_rigid_body_shape_indices(a1_asset)
        a1_body_shape_props = self.gym.get_asset_rigid_shape_properties(a1_asset)

        thigh_names = [s for s in self.body_names if "thigh" in s] # thigh and thigh_shoulder
        thigh_indices = [body_dict[thigh_names[i]] for i in range(len(thigh_names))]

        for thigh_idx in thigh_indices:
            for i in range(a1_body_shape_indices[thigh_idx].count):
                a1_body_shape_props[a1_body_shape_indices[thigh_idx].start + i].filter = 1

        FL_shovel_index = body_dict["FL_shovel"]
        a1_body_shape_props[FL_shovel_index].friction = 0.5

        hip_names = [s for s in self.dof_names if "hip" in s]
        self.hip_joint_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(hip_names)):
            self.hip_joint_indices[i] = self.gym.find_asset_dof_index(a1_asset, hip_names[i])

        # create box asset
        box_size = 0.04
        box_asset_options = gymapi.AssetOptions()
        box_asset_options.density = 1500 # kg/m^3
        box_asset_options.fix_base_link = False
        box_asset_options.disable_gravity = False
        box_asset = self.gym.create_box(self.sim, box_size, box_size, box_size, box_asset_options)
        box_pose = gymapi.Transform()

        # create env
        self.envs = []
        self.a1_handles = []
        self.a1_indices = []
        self.a1_init_state = []
        self.prop_handles = []
        self.prop_indices = []
        self.prop_init_state = []

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env_ptr)

            a1_handle = self.gym.create_actor(env_ptr, a1_asset, a1_start_pose,  "a1", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, a1_handle, a1_dof_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, a1_handle, a1_body_shape_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, a1_handle)
            self.a1_handles.append(a1_handle)
            a1_idx = self.gym.get_actor_index(env_ptr, a1_handle, gymapi.DOMAIN_SIM)
            self.a1_indices.append(a1_idx)
            self.a1_init_state.append(self.base_init_state)

            for j in range(self.num_boxes):
                box_pose.p = gymapi.Vec3(*self.box_init_state[:3]) + gymapi.Vec3(j*box_size, j*box_size, 0)
                box_pose.r = gymapi.Quat(*self.box_init_state[3:7])
                prop_handle = self.gym.create_actor(env_ptr, box_asset, box_pose, "prop" + str(j), i, 0, 0)
                self.prop_handles.append(prop_handle)
                prop_idx = self.gym.get_actor_index(env_ptr, prop_handle, gymapi.DOMAIN_SIM)
                self.prop_indices.append(prop_idx)
                self.prop_init_state.append(self.box_init_state)

        self.a1_indices = to_torch(self.a1_indices, dtype=torch.long, device=self.device)
        self.a1_init_state = to_torch(self.a1_init_state, dtype=torch.float, device=self.device).view(self.num_envs, 13)
        self.prop_indices = to_torch(self.prop_indices, dtype=torch.long, device=self.device)
        self.prop_init_state = to_torch(self.prop_init_state, dtype=torch.float, device=self.device).view(self.num_envs, -1, 13)

    def pre_physics_step(self, actions):
        # resets
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)
        targets = 0.5 * self.actions + self.default_dof_pos

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.progress_buf += 1

        self.compute_observations()
        self.compute_reward()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

    def compute_reward(self, actions):
        pass

    def quaternion_to_6D_matrix(self, base_quat): # q = self.root_states[:, 3:7]
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

    def get_transformed_position(self, point_local=None, vector_local=None, point_global=None):
        base_pos = self.a1_root_states[:, 0:3].squeeze().cpu().numpy()
        base_ori = self.a1_root_states[:, 3:7].squeeze().cpu().numpy()

        result = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        if (self.num_envs == 1):
            base_pos = base_pos.reshape((1, *base_pos.shape))
            base_ori = base_ori.reshape((1, *base_ori.shape))
        for i in range(self.num_envs):
            transform = gymapi.Transform(gymapi.Vec3(base_pos[i][0], base_pos[i][1], base_pos[i][2]),
                                         gymapi.Quat(base_ori[i][0], base_ori[i][1], base_ori[i][2], base_ori[i][3]))
            transform_inverse = transform.inverse()
            if point_local is not None:
                transformed_position = transform.transform_point(gymapi.Vec3(point_local[i][0], point_local[i][1], point_local[i][2]))
                result[i] = torch.tensor([[transformed_position.x, transformed_position.y, transformed_position.z]],  dtype=torch.float, device=self.device, requires_grad=False)
            if vector_local is not None:
                transformed_vector = transform.transform_vector(gymapi.Vec3(vector_local[i][0], vector_local[i][1], vector_local[i][2]))
                result[i] = torch.tensor([[transformed_vector.x, transformed_vector.y, transformed_vector.z]],  dtype=torch.float, device=self.device, requires_grad=False)
            if point_global is not None:
                transformed_position = transform_inverse.transform_point(gymapi.Vec3(point_global[i][0], point_global[i][1], point_global[i][2]))
                result[i] = torch.tensor([[transformed_position.x, transformed_position.y, transformed_position.z]],  dtype=torch.float, device=self.device, requires_grad=False)
        return result


    def compute_observations(self):
        closest_prop_positions = self.get_closest_prop_position()
        # 1. quaternion
        base_quat = self.a1_root_states[:, 3:7] # quaternion
        rot_matrix = self.quaternion_to_6D_matrix(base_quat)
        # 2. gyroscope
        base_ang_vel = quat_rotate_inverse(base_quat, self.a1_root_states[:, 10:13])
        # 3. accelerometer
        base_lin_vel = quat_rotate_inverse(base_quat, self.a1_root_states[:, 7:10])
        accelerometer = ((base_lin_vel - self.base_lin_vel_before) / self.dt) - quat_rotate_inverse(base_quat, self.gravity_vec)
        self.base_lin_vel_before = base_lin_vel

        self.obs_buf[:] = torch.cat((closest_prop_positions,
                                     rot_matrix,
                                     base_ang_vel,
                                     accelerometer,
                                     self.dof_pos,
                                     self.dof_vel,
                                     self.actions,
                                     ), dim=-1)

        # extra obs for experts
        shovel_bottom_pos = self.rb_states[:, self.shovel_bottom_index, 0:3]
        self.extras["shovel_bottom_pos"] = shovel_bottom_pos

    def get_closest_prop_position(self):
        prop_root_pos = self.prop_root_states[:, :, 0:3]
        base_pos = self.a1_root_states[:, None, 0:3]
        distances = torch.norm(prop_root_pos - base_pos, dim=-1)
        min_distance_indices = torch.argmin(distances, dim=1)
        closest_prop_positions = prop_root_pos[torch.arange(self.num_envs), min_distance_indices]
        print(prop_root_pos)
        print(closest_prop_positions)
        return closest_prop_positions

    def gaussian_kernel(self, x, y, sigma=1.0):
        return torch.exp(-torch.sum((x-y)**2, dim=-1) / (2*sigma**2))

    def compute_ditance_map(self):
        linspace = torch.linspace(-self.env_space*2, self.env_space*2, self.grid_size, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(linspace, linspace)
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)

        prop_root_pos = self.prop_root_states[:, :, 0:3]
        prop_root_pos_wrt_a1_base = torch.zeros_like(prop_root_pos, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_boxes):
            prop_root_pos_wrt_a1_base[:, i, :] = self.get_transformed_position(point_global=prop_root_pos[:, i, :])

        grid_expanded = grid.unsqueeze(0).unsqueeze(2)
        pos_xy_expanded = prop_root_pos_wrt_a1_base[:, :, 0:2].unsqueeze(1)

        distances = self.gaussian_kernel(grid_expanded, pos_xy_expanded, self.sigma) # (num_evns, grid_size*grid_size, num_boxes)
        object_map = distances.sum(dim=-1).reshape(self.num_envs, self.grid_size*self.grid_size)

        # self.visualize_object_map(object_map.reshape(self.num_envs, self.grid_size, self.grid_size), prop_root_pos_wrt_a1_base[:, :, 0:2],  self.prop_root_states[:, :, 0:2])

        return object_map

    def reset_idx(self, env_ids):
        self.compute_ditance_map()
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0.75, 1.25, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        # reset root state for all actors in selected envs
        self.root_states[self.a1_indices[env_ids]] = self.a1_init_state[env_ids].clone()

        random_prop_init_pos = self.prop_init_state[env_ids].clone()

        for i in range(env_ids.size(0)):
            random_prop_init_pos[i, :, 0:2] = (torch.rand(self.num_boxes, 2) - 0.5) * torch.tensor([2*self.env_space-0.5, 2*self.env_space-0.5])
            self.root_states[self.prop_indices[i*self.num_boxes:i*self.num_boxes+self.num_boxes]] = random_prop_init_pos[i]

        actor_indices = self.all_actor_indices[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(actor_indices), len(actor_indices))

        # reset DOF state for a1 in selected envs
        a1_indice = self.a1_indices[env_ids].to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(a1_indice), len(a1_indice))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.base_lin_vel_before[env_ids, :] = torch.tensor([0., 0., 0.], dtype=torch.float, device=self.device, requires_grad=False)

        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.landing[env_ids] = 0.
        self.picked[env_ids] = 0.
