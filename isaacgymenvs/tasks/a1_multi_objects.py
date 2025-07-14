import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask

from typing import Tuple, Dict
class A1MultiObjects(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg

        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
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

        self.env_space = self.cfg["env"]["envSpacing"]
        self.cfg["env"]["numObservations"] = 49 #80*3 + 49
        self.cfg["env"]["numActions"] = 12

        # box init state. TODO: add to cfg file
        box_pos = [0.34, 0.13, 0.025]
        box_rot = [0., 0., 0., 1.]
        box_v_lin = [0., 0., 0.]
        box_v_ang = [0., 0., 0.]
        box_state = box_pos + box_rot + box_v_lin + box_v_ang
        self.box_init_state = box_state
        self.num_props = 6
        self.num_boxes = 6

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
        #self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces)
        self.a1_contact_forces= self.contact_forces.view(self.num_envs, -1, 3)[:, :self.num_bodies, :]
        self.prop_contact_forces = self.contact_forces.view(self.num_envs, -1, 3)[:, self.num_bodies:, :]
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)
        self.a1_root_states = self.root_states.view(self.num_envs, self.actors_per_env, 13)[:, 0, :]
        self.prop_root_states = self.root_states.view(self.num_envs, self.actors_per_env, 13)[:, 1:self.num_props+1, :]

        self.all_actor_indices = torch.arange(self.actors_per_env * self.num_envs, dtype=torch.int32, device=self.device).view(self.num_envs, self.actors_per_env)

        self.FL_shovel_joint_index = self.gym.find_actor_dof_handle(self.envs[0], self.a1_handles[0], "FL_shovel_joint")

        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            if name=="FL_shovel_joint":
                continue
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
        self.shovel_bottom_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FL_shovel_bottom")

        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel, dtype=torch.float, device=self.device, requires_grad=False)
        self.landing = torch.zeros(self.num_envs, self.num_boxes, dtype=torch.float, device=self.device, requires_grad=False)
        self.landing_now = torch.zeros(self.num_envs, self.num_boxes, dtype=torch.float, device=self.device, requires_grad=False)
        self.num_landing = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.total_num_landing = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.picked = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.metric1 = 0.
        self.metric2 = 0.
        self.metric3 = 0.
        self.time = 0.
        self.cnt = 0.

        self.landing_times = [[] for _ in range(self.num_envs)]
        self.avg_packing_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # 이전 스텝의 담긴 여부 저장 (물체가 방금 담겼는지 판별하기 위함)
        self.prev_landing_now = torch.zeros_like(self.landing_now, dtype=torch.float, device=self.device, requires_grad=False)

        self.randing_cnt = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_cnt = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.update_cnt = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.offset_forward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.offset_backward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.offset_right = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.offset_left = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.reset_progress_s = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.reset_progress_s[:] = 1.0
        self.offset_degree = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.offset_degree[:] = 10

        self.cube_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.prop_handles[0], "object") #27
        self.bucket_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.prop_handles[1], "bucket") #28
        self.banana_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.prop_handles[2], "banana") #29
        self.mug_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.prop_handles[3], "mug") #30
        self.brick_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.prop_handles[4], "foam_brick") #31
        self.can_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.prop_handles[5], "potted_meat_can")

        self.prop_num = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.prop_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.prop_weight = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

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
        a1_asset_file = "urdf/a1_with_shovel_description/urdf/a1_with_shovel_passive_joint_black.urdf"
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
        a1_FL_shovel_joint_index = self.gym.find_asset_dof_index(a1_asset, "FL_shovel_joint")
        for i in range(self.num_dof):
            a1_dof_props['driveMode'][i] = 1 #gymapi.DOF_MODE_POS
            a1_dof_props['stiffness'][i] = 60.
            a1_dof_props['damping'][i] = 3.
        a1_dof_props['stiffness'][a1_FL_shovel_joint_index] = 0.0001
        a1_dof_props['damping'][a1_FL_shovel_joint_index] = 0.0001

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

        # create objects asset
        cube_asset_file = "urdf/objects/cube_multicolor.urdf"
        bucket_asset_file = "urdf/objects/bucket.urdf"
        banana_asset_file = "urdf/ycb/011_banana/011_banana.urdf"
        mug_asset_file = "urdf/ycb/025_mug/025_mug.urdf"
        brick_asset_file = "urdf/ycb/061_foam_brick/061_foam_brick.urdf"
        can_asset_file = "urdf/ycb/010_potted_meat_can/010_potted_meat_can.urdf"

        object_files = []
        object_files.append(cube_asset_file)
        object_files.append(bucket_asset_file)
        object_files.append(banana_asset_file)
        object_files.append(mug_asset_file)
        object_files.append(brick_asset_file)
        object_files.append(can_asset_file)
        object_files.append(object_files)

        object_asset_options = gymapi.AssetOptions()
        object_asset_options.fix_base_link = False
        object_asset_options.disable_gravity = False
        object_asset_options.use_mesh_materials = True

        object_assets = []
        object_assets.append(self.gym.load_asset(self.sim, asset_root, cube_asset_file, object_asset_options))
        object_assets.append(self.gym.load_asset(self.sim, asset_root, bucket_asset_file, object_asset_options))
        object_assets.append(self.gym.load_asset(self.sim, asset_root, banana_asset_file, object_asset_options))
        object_assets.append(self.gym.load_asset(self.sim, asset_root, mug_asset_file, object_asset_options))
        object_assets.append(self.gym.load_asset(self.sim, asset_root, brick_asset_file, object_asset_options))
        object_assets.append(self.gym.load_asset(self.sim, asset_root, can_asset_file, object_asset_options))

        object_pose = gymapi.Transform()
        object_pose.p = gymapi.Vec3(*self.box_init_state[:3]) # set manually
        object_pose.r = gymapi.Quat(*self.box_init_state[3:7])

        # create env
        self.envs = []
        self.a1_handles = []
        self.a1_indices = []
        self.a1_init_state = []

        self.prop_handles = []
        self.cube_handles = []
        self.bucket_handles = []
        self.banana_handles = []
        self.mug_handles = []
        self.brick_handles = []
        self.can_handles = []

        self.prop_indices = []
        self.cube_indices = []
        self.bucket_indices = []
        self.banana_indices = []
        self.mug_indices = []
        self.brick_indices = []
        self.can_indices = []

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

            cube_handle = self.gym.create_actor(env_ptr, object_assets[0], object_pose, "cube", i, 0, 0)
            bucket_handle = self.gym.create_actor(env_ptr, object_assets[1], object_pose, "bucket", i, 0, 0)
            banana_handle = self.gym.create_actor(env_ptr, object_assets[2], object_pose, "banana", i, 0, 0)
            mug_handle = self.gym.create_actor(env_ptr, object_assets[3], object_pose, "mug", i, 0, 0)
            brick_handle = self.gym.create_actor(env_ptr, object_assets[4], object_pose, "brick", i, 0, 0)
            can_handle = self.gym.create_actor(env_ptr, object_assets[5], object_pose, "can", i, 0, 0)
            self.prop_handles.append(cube_handle)
            self.prop_handles.append(bucket_handle)
            self.prop_handles.append(banana_handle)
            self.prop_handles.append(mug_handle)
            self.prop_handles.append(brick_handle)
            self.prop_handles.append(can_handle)
            cube_idx = self.gym.get_actor_index(env_ptr, cube_handle, gymapi.DOMAIN_SIM)
            bucket_idx = self.gym.get_actor_index(env_ptr, bucket_handle, gymapi.DOMAIN_SIM)
            banana_idx = self.gym.get_actor_index(env_ptr, banana_handle, gymapi.DOMAIN_SIM)
            mug_idx = self.gym.get_actor_index(env_ptr, mug_handle, gymapi.DOMAIN_SIM)
            brick_idx = self.gym.get_actor_index(env_ptr, brick_handle, gymapi.DOMAIN_SIM)
            can_idx = self.gym.get_actor_index(env_ptr, can_handle, gymapi.DOMAIN_SIM)
            self.prop_indices.append(cube_idx)
            self.prop_indices.append(bucket_idx)
            self.prop_indices.append(banana_idx)
            self.prop_indices.append(mug_idx)
            self.prop_indices.append(brick_idx)
            self.prop_indices.append(can_idx)
            self.cube_indices.append(cube_idx) #[1, 7, 13, 19, 25, 31, 37, ...]
            self.bucket_indices.append(bucket_idx) #[2, 8, 14, 20, 26, 32, 38, ...]
            self.banana_indices.append(banana_idx) #[3, 9, 15, 21, 27, 33, 39, ...]
            self.mug_indices.append(mug_idx) #[4, 10, 16, 22, 28, 34, 40, ...]
            self.brick_indices.append(brick_idx) #[5, 11, 17, 23, 29, 35, 41, ...]
            self.can_indices.append(can_idx)

            self.prop_init_state.append(self.box_init_state)
            self.prop_init_state.append(self.box_init_state)
            self.prop_init_state.append(self.box_init_state)
            self.prop_init_state.append(self.box_init_state)
            self.prop_init_state.append(self.box_init_state)
            self.prop_init_state.append(self.box_init_state)

        self.a1_indices = to_torch(self.a1_indices, dtype=torch.long, device=self.device)
        self.a1_init_state = to_torch(self.a1_init_state, dtype=torch.float, device=self.device).view(self.num_envs, 13)
        self.prop_indices = to_torch(self.prop_indices, dtype=torch.long, device=self.device)
        self.cube_indices = to_torch(self.cube_indices, dtype=torch.long, device=self.device)
        self.bucket_indices = to_torch(self.bucket_indices, dtype=torch.long, device=self.device)
        self.banana_indices = to_torch(self.banana_indices, dtype=torch.long, device=self.device)
        self.mug_indices = to_torch(self.mug_indices, dtype=torch.long, device=self.device)
        self.brick_indices = to_torch(self.brick_indices, dtype=torch.long, device=self.device)
        self.can_indices = to_torch(self.can_indices, dtype=torch.long, device=self.device)
        self.prop_init_state = to_torch(self.prop_init_state, dtype=torch.float, device=self.device).view(self.num_envs, -1, 13)

    def pre_physics_step(self, actions):
        # resets
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)
        tensor_to_insert = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        action = torch.cat((self.actions[:, :self.FL_shovel_joint_index], tensor_to_insert, self.actions[:, self.FL_shovel_joint_index:]), dim=1)
        targets = 0.5 * action + self.default_dof_pos

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

        self.track_packing_times()

        # 현재 landing_now 값을 prev_landing_now 에 저장 (다음 스텝에서 비교하기 위해)
        self.prev_landing_now[:] = self.landing_now

    def track_packing_times(self):
        newly_landed = (self.landing_now == 1) & (self.prev_landing_now == 0) #현재 담긴 물체들
        current_time = self.time

        for env_idx in range(self.num_envs):
            for obj_idx in range(self.num_boxes):
                if newly_landed[env_idx, obj_idx]: # 새롭게 담긴 경우
                    if len(self.landing_times[env_idx]) == 0:
                        last_time = 0 # 첫 물체면 에피소드 시작 시간
                    else:
                        last_time = self.landing_times[env_idx][-1] # 이전 물체 담긴 시간

                    self.landing_times[env_idx].append(current_time - last_time) # 시간 차이 저장


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

    def get_transformed_position(self, root_states, point_local=None, vector_local=None, point_global=None):
        base_pos = root_states[:, 0:3].squeeze().cpu().numpy()
        base_ori = root_states[:, 3:7].squeeze().cpu().numpy()

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
        #_, closest_prop_pos = self.get_closest_prop_position()
        base_pos = self.a1_root_states[:, 0:3]
        base_quat = self.a1_root_states[:, 3:7] # quaternion
        # states from imu(quaternion, gyroscope, accelerometer)
        # 1. quaternion
        rot_matrix = self.quaternion_to_6D_matrix(base_quat)
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)
        # 2. gyroscope
        base_ang_vel = quat_rotate_inverse(base_quat, self.a1_root_states[:, 10:13])
        # 3. accelerometer
        base_lin_vel = quat_rotate_inverse(base_quat, self.a1_root_states[:, 7:10])
        accelerometer = ((base_lin_vel - self.base_lin_vel_before) / self.dt) - quat_rotate_inverse(base_quat, self.gravity_vec)
        self.base_lin_vel_before = base_lin_vel
        # pos_of_prop_wrt_a1_base
        prop_root_pos = self.prop_root_states[torch.arange(self.num_envs), self.prop_num, 0:3]
        pos_of_prop_wrt_a1_base = self.get_transformed_position(self.a1_root_states, point_global=prop_root_pos)
        # distance between shovel to prop
        shovel_bottom_pos = self.rb_states[:, self.shovel_bottom_index, 0:3]
        shovel_to_prop_dis = torch.norm(prop_root_pos - shovel_bottom_pos, dim=1, keepdim=True)

        self.dof_pos_new = torch.cat((self.dof_pos[:, :self.FL_shovel_joint_index], self.dof_pos[:, self.FL_shovel_joint_index+1:]), dim=1)
        self.dof_vel_new = torch.cat((self.dof_vel[:, :self.FL_shovel_joint_index], self.dof_vel[:, self.FL_shovel_joint_index+1:]), dim=1)

        self.obs_buf[:] = torch.cat((pos_of_prop_wrt_a1_base,
                                     shovel_to_prop_dis,
                                     projected_gravity,
                                     base_ang_vel,
                                     accelerometer,
                                     self.dof_pos_new,
                                     self.dof_vel_new,
                                     self.actions,
                                     ), dim=-1)

        # extra obs for experts
        self.time += self.dt

    def update_curriculum(self, env_idx):
        self.offset_forward[env_idx] += 0.1
        self.offset_backward[env_idx] -= 0.05
        self.offset_right[env_idx] += 0.08
        self.offset_left[env_idx] -= 0.08
        self.update_cnt[env_idx] += 1
        self.reset_progress_s[env_idx] += 0.5
        self.offset_degree[env_idx] += 5

    def get_closest_prop_position(self, threshold=0.1):
        prop_root_pos = self.prop_root_states[:, :, 0:3]  # 모든 물체 위치
        base_pos = self.a1_root_states[:, None, 0:3]  # 로봇 위치

        distances = torch.norm(prop_root_pos - base_pos, dim=-1)  # 거리 계산
        distances[self.landing.bool()] = float('inf')  # 착지한 물체는 무시

        # 현재 가장 가까운 물체 찾기
        new_min_distance_indices = torch.argmin(distances, dim=1)
        new_closest_prop_positions = prop_root_pos[torch.arange(self.num_envs), new_min_distance_indices]

        # 🛠 이전 프레임의 인덱스가 없으면 초기화
        if not hasattr(self, "prev_min_distance_indices"):
            self.prev_min_distance_indices = new_min_distance_indices.clone()  # 첫 실행 시 현재 값을 사용

        prev_positions = prop_root_pos[torch.arange(self.num_envs), self.prev_min_distance_indices]
        prev_distances = torch.norm(prev_positions - base_pos.squeeze(1), dim=-1)
        new_distances = torch.norm(new_closest_prop_positions - base_pos.squeeze(1), dim=-1)

        # 히스테리시스 적용: 거리가 비슷하면 기존 물체 유지
        keep_prev = (new_distances - prev_distances).abs() < threshold
        new_min_distance_indices[keep_prev] = self.prev_min_distance_indices[keep_prev]


        # 값 업데이트
        self.prev_min_distance_indices = new_min_distance_indices.clone()
        closest_prop_positions = prop_root_pos[torch.arange(self.num_envs), new_min_distance_indices]
        return new_min_distance_indices, closest_prop_positions
        #return self.prop_num, self.prop_root_states[:, self.prop_num, 0:3]


    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0.75, 1.25, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.25, 0.25, (len(env_ids), self.num_dof), device=self.device)

        #self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids]
        self.dof_vel[env_ids] = 0.

        # reset root state for all actors in selected envs
        # self.root_states[self.a1_indices[env_ids]] = self.a1_init_state[env_ids].clone()
        random_a1_init_state = self.a1_init_state[env_ids].clone()
        #theta_deg = torch.FloatTensor(1).uniform_(-5, 5)
        # 각도를 라디안으로 변환
        #theta = theta_deg * (torch.pi / 180.0)
        #random_a1_init_state[:, 3:7] = torch.tensor([0., 0., torch.sin(theta / 2), torch.cos(theta / 2)])
        self.root_states[self.a1_indices[env_ids]] = random_a1_init_state

        #self.prop_num[env_ids] = np.random.randint(0, 5)
        self.prop_num[env_ids] = np.random.randint(0, self.num_props)
        random_prop_init_pos = self.prop_init_state[env_ids].clone()
        random_prop_num = self.prop_num[env_ids].clone()

        for i in range(env_ids.size(0)):
            #random_prop_init_pos[i, :, 0:2] = (torch.rand(self.num_props, 2) - 0.5) * torch.tensor([2*self.env_space-0.5, 2*self.env_space-0.5])
            for j in range(self.num_props):
                random_prop_init_pos[i, j, 0:2] = torch.tensor([self.env_space-0.2*j, self.env_space-0.2*j], device=self.device)
            r = (1 - 0.5) * torch.rand(self.num_boxes, device=self.device)
            theta = 2 * torch.pi * torch.rand(self.num_boxes, device=self.device)
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            random_prop_init_pos[i, random_prop_num[i], 0:2] = torch.tensor([0.35, 0.], device=self.device)
            random_prop_init_pos[i, random_prop_num[i], 0] = x[i]
            random_prop_init_pos[i, random_prop_num[i], 1] = y[i]
            self.root_states[self.prop_indices[env_ids[i]*self.num_props:env_ids[i]*self.num_props+self.num_props]] = random_prop_init_pos[i]
            """
            if self.randing_cnt[env_ids[i]]:
                self.episode_cnt[env_ids[i]] += 1
            if (self.episode_cnt[env_ids[i]]>10) & ((self.randing_cnt[env_ids[i]]/self.episode_cnt[env_ids[i]])>0.8):
                self.update_curriculum(env_ids[i])
                self.episode_cnt[env_ids[i]] = 0
                self.randing_cnt[env_ids[i]] = 0
                print("env ", env_ids[i], " update_curr", self.update_cnt[env_ids[i]])
            prop_position_offset_x = torch_rand_float(self.offset_backward[env_ids[i]], self.offset_forward[env_ids[i]], (1, 1), device=self.device)
            prop_position_offset_y = torch_rand_float(self.offset_left[env_ids[i]], self.offset_right[env_ids[i]], (1, 1), device=self.device)
            random_prop_init_pos[i, random_prop_num[i], 0:2] = torch.tensor([0.34, 0.13], device=self.device)
            random_prop_init_pos[i, random_prop_num[i], 0] += prop_position_offset_x.item()
            random_prop_init_pos[i, random_prop_num[i], 1] += prop_position_offset_y.item()
            self.root_states[self.prop_indices[env_ids[i]*self.num_props:env_ids[i]*self.num_props+self.num_props]] = random_prop_init_pos[i]
            """
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

        for env_idx in env_ids:
            if len(self.landing_times[env_idx]) > 0:
                self.avg_packing_time[env_idx] = sum(self.landing_times[env_idx]) / len(self.landing_times[env_idx])
            else:
                self.avg_packing_time[env_idx] = 0 # 물체를 하나도 담지 못한 경우 제외

        landing_10s = self.landing > 10/self.dt
        landing = landing_10s & self.landing_now.bool()
        temp = self.landing_now.sum(dim=-1)

        #1 담은 개수
        self.metric1 += self.num_landing

        #2 평균 담기 시간
        self.metric2 += self.avg_packing_time

        #3 누락 물체 개수
        self.metric3 += (self.num_landing - temp)
        print("episod: ", self.cnt)
        print(self.metric1)
        print(self.metric2)
        print(self.metric3)

        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.landing[env_ids] = 0.
        self.picked[env_ids] = 0.
        self.num_landing[env_ids] = 0.
        self.landing_now[env_ids] = 0.
        self.time = 0.
        self.cnt += 1
        self.landing_times = [[] for _ in range(self.num_envs)]
        self.prev_landing_now.zero_()


        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.landing[env_ids] = 0.
        self.picked[env_ids] = 0.

        mask_cube = self.prop_num[env_ids] == 0
        mask_bucket = self.prop_num[env_ids] == 1
        mask_banana = self.prop_num[env_ids] == 2
        mask_mug = self.prop_num[env_ids] == 3
        mask_brick = self.prop_num[env_ids] == 4
        mask_can = self.prop_num[env_ids] == 5

        self.prop_index[env_ids[mask_cube]] = self.cube_index
        self.prop_index[env_ids[mask_bucket]] = self.bucket_index
        self.prop_index[env_ids[mask_banana]] = self.banana_index
        self.prop_index[env_ids[mask_mug]] = self.mug_index
        self.prop_index[env_ids[mask_brick]] = self.brick_index
        self.prop_index[env_ids[mask_can]] = self.can_index
