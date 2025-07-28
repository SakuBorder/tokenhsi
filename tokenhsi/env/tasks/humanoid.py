# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
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

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from utils import torch_utils

from env.tasks.base_task import BaseTask

class Humanoid(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        self._pd_control = self.cfg["env"]["pdControl"]
        # print(self._pd_control )
        # import ipdb;ipdb.set_trace()
        self.power_scale = self.cfg["env"]["powerScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._local_root_obs_policy = self.cfg["env"]["localRootObsPolicy"]
        self._root_height_obs = self.cfg["env"].get("rootHeightObs", True)
        self._root_height_obs_policy = self.cfg["env"].get("rootHeightObsPolicy", True)
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]
        
        key_bodies = self.cfg["env"]["keyBodies"]
        self._setup_character_props(key_bodies)

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
         
        super().__init__(cfg=self.cfg)
        
        self.dt = self.control_freq_inv * sim_params.dt
        
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        # sensors_per_env = 2
        # self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)
        if hasattr(self, '_has_foot_sensors') and self._has_foot_sensors:
            sensors_per_env = 2
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)
        else:
            # 没有传感器的情况
            sensors_per_env = 0
            self.vec_sensor_tensor = torch.zeros((self.num_envs, 0), device=self.device)

            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()
        
        self._humanoid_root_states = self._root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])[..., 0, :]
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:, 7:13] = 0
        print("=== 初始根状态检查 ===")
        print(f"_humanoid_root_states: {self._humanoid_root_states[0]}")
        print(f"_initial_humanoid_root_states: {self._initial_humanoid_root_states[0]}")

        # 检查是否有异常值
        if torch.any(torch.isnan(self._initial_humanoid_root_states)):
            print("WARNING: 初始状态包含NaN!")
        if torch.any(torch.abs(self._initial_humanoid_root_states[:, 7:13]) > 0.1):
            print("WARNING: 初始速度不为零!")

        self._humanoid_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 1]
        
        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)
        
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)

        self._initial_humanoid_rigid_body_states = rigid_body_state_reshaped[..., :self.num_bodies, :].clone()
        self._initial_humanoid_rigid_body_states[..., 7:13] = 0

        self._rigid_body_pos = rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        print("torch.min(self._rigid_body_pos[...,2])=",torch.min(self._rigid_body_pos[...,2]))
        self._rigid_body_rot = rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = rigid_body_state_reshaped[..., :self.num_bodies, 10:13]

        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., :self.num_bodies, :]
        
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        
        self._build_termination_heights()
        
        contact_bodies = self.cfg["env"]["contactBodies"]
        self._key_body_ids = self._build_key_body_ids_tensor(key_bodies)
        self._contact_body_ids = self._build_contact_body_ids_tensor(contact_bodies)
        
        if self.viewer != None:
            self._init_camera()
            self._camera_follow_flag = True
            
        return

    def get_obs_size(self):
        return self._num_obs

    def get_action_size(self):
        return self._num_actions

    def get_num_actors_per_env(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        return

    def reset(self, env_ids=None):
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self._reset_envs(env_ids)
        return

    def set_char_color(self, col, env_ids):
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            handle = self.humanoid_handles[env_id]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(col[0], col[1], col[2]))

        return

    def _reset_envs(self, env_ids):
        if (len(env_ids) > 0):
            self._reset_actors(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
        return

    def _reset_env_tensors(self, env_ids):
        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _setup_character_props(self, key_bodies):
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        if (asset_file == "mjcf/humanoid/amp_humanoid.xml"):
            self._dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
            self._dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
            self._dof_obs_size = 72
            self._num_actions = 28
            self._num_actions_joint = self._num_actions
            self._num_obs = 1 + 15 * (3 + 6 + 3 + 3) - 3
        elif (asset_file == "mjcf/humanoid/phys_humanoid.xml") or (asset_file == "mjcf/humanoid/phys_humanoid_v2.xml") or (asset_file == "mjcf/humanoid/phys_humanoid_v3.xml") or (asset_file == "mjcf/humanoid/phys_humanoid_v3_box_foot.xml"):
            self._dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
            self._dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 20, 23, 26, 29, 32]
            self._dof_obs_size = 72
            self._num_actions = 28 + 2 * 2
            self._num_actions_joint = self._num_actions
            self._num_obs = 1 + 15 * (3 + 6 + 3 + 3) - 3
        # elif (asset_file == "mjcf/g1/g1_29dof.xml") :
        #     self._dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
        #     self._dof_offsets = [0, 3, 4, 7, 10, 11, 14, 17, 18, 21, 24, 25, 28]
        #     self._dof_obs_size = 72
        #     self._num_actions = 28 + 2 * 2
        #     self._num_actions_joint = self._num_actions
        #     self._num_obs = 1 + 15 * (3 + 6 + 3 + 3) - 3
        elif (asset_file == "mjcf/tai5/tai5.xml") :
            # TAI5机器人的15个关节体对应的DOF分布
            self._dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
            
            # TAI5的DOF偏移表 - 根据实际的关节结构
            # body 0 (base_link): 无DOF（freejoint不算）

            # body 1 (WAIST_Y_S): 3 DOF (WAIST_Y, WAIST_R, WAIST_P)
            # body 2 (NECK_Y_S): 1 DOF (NECK_Y) 
            # body 3 (R_SHOULDER_P_S): 3 DOF (R_SHOULDER_P, R_SHOULDER_R, R_SHOULDER_Y)
            # body 4 (R_ELBOW_Y_S): 1 DOF (R_ELBOW_Y)

            # body 5 (R_WRIST_R_S): 3 DOF (R_WRIST_P, R_WRIST_Y, R_WRIST_R)

            # body 6 (L_SHOULDER_P_S): 3 DOF (L_SHOULDER_P, L_SHOULDER_R, L_SHOULDER_Y)
            # body 7 (L_ELBOW_Y_S): 1 DOF (L_ELBOW_Y)

            # body 8 (L_WRIST_R_S): 3 DOF (L_WRIST_P, L_WRIST_Y, L_WRIST_R)

            # body 9 (R_HIP_P_S): 3 DOF (R_HIP_P, R_HIP_R, R_HIP_Y)
            # body 10 (R_KNEE_P_S): 1 DOF (R_KNEE_P)
            # body 11 (R_ANKLE_R_S): 2 DOF (R_ANKLE_P, R_ANKLE_R)
            # body 12 (L_HIP_P_S): 3 DOF (L_HIP_P, L_HIP_R, L_HIP_Y)
            # body 13 (L_KNEE_P_S): 1 DOF (L_KNEE_P)
            # body 14 (L_ANKLE_R_S): 2 DOF (L_ANKLE_P, L_ANKLE_R)
            # self.tai5_node_mapping = {
            #     'base_link': 0, 'WAIST_Y_S': 1, 'NECK_Y_S': 2, 'R_SHOULDER_P_S': 3, 'R_ELBOW_Y_S': 4,
            #     'R_WRIST_R_S': 5, 'L_SHOULDER_P_S': 6, 'L_ELBOW_Y_S': 7, 'L_WRIST_R_S': 8,
            #     'R_HIP_Y_S': 9, 'R_KNEE_P_S': 10, 'R_ANKLE_R_S': 11, 'L_HIP_Y_S': 12, 'L_KNEE_P_S': 13, 'L_ANKLE_R_S': 14
            # }
            # self.tai5_node_mapping = {
            #      'WAIST_Y_S': 1, 'NECK_Y_S': 2, 'R_SHOULDER_P_S': 3, 'R_ELBOW_Y_S': 4,
            #      'L_SHOULDER_P_S': 6, 'L_ELBOW_Y_S': 7, 
            #     'R_HIP_Y_S': 9, 'R_KNEE_P_S': 10, 'R_ANKLE_R_S': 11, 'L_HIP_Y_S': 12, 'L_KNEE_P_S': 13, 'L_ANKLE_R_S': 14
            # }
            # [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
            self._dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 19, 22, 25, 27, 30]
            #                  ^  ^  ^  ^  ^  ^   ^   ^   ^   ^   ^   ^   ^   ^   ^
            #                  0  1  2  3  4  5   6   7   8   9  10  11  12  13  14
            
            self._dof_obs_size = 72
            self._num_actions = 30
            self._num_actions_joint = self._num_actions  
            self._num_obs = 1 + 15 * (3 + 6 + 3 + 3) - 3

        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert(False)

        return

    # def _build_termination_heights(self):
    #     head_term_height = 0.3

    #     termination_height = self.cfg["env"]["terminationHeight"]
    #     self._termination_heights = np.array([termination_height] * self.num_bodies)

    #     head_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "head")
    #     self._termination_heights[head_id] = max(head_term_height, self._termination_heights[head_id])
    #     self._termination_heights = to_torch(self._termination_heights, device=self.device)
    #     return
    def _build_termination_heights(self):
        head_term_height = 0.3
        termination_height = self.cfg["env"]["terminationHeight"]
        self._termination_heights = np.array([termination_height] * self.num_bodies)

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        
        if asset_file == "mjcf/tai5/tai5.xml":
            # tai5的15个主要节点列表
            tai5_nodes = ['base_link', 'WAIST_Y_S', 'NECK_Y_S', 'R_SHOULDER_P_S', 'R_ELBOW_Y_S', 
                        'R_WRIST_R_S', 'L_SHOULDER_P_S', 'L_ELBOW_Y_S', 'L_WRIST_R_S', 
                        'R_HIP_Y_S', 'R_KNEE_P_S', 'R_ANKLE_R_S', 'L_HIP_Y_S', 'L_KNEE_P_S', 'L_ANKLE_R_S']
            
            head_name = "NECK_Y_S"  # tai5的头部/颈部
            
            # 直接使用列表索引
            if head_name in tai5_nodes:
                head_id = tai5_nodes.index(head_name)  # NECK_Y_S 对应索引 2
                print(f"tai5 head '{head_name}' mapped to index {head_id}")
                self._termination_heights[head_id] = max(head_term_height, self._termination_heights[head_id])
            else:
                print(f"Warning: Could not find head body '{head_name}' in tai5 nodes")
                
        else:
            head_name = "head"      # 标准humanoid的头部
            
            # 非tai5模型使用原来的方法
            head_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], head_name)
            if head_id != -1:
                self._termination_heights[head_id] = max(head_term_height, self._termination_heights[head_id])
            else:
                print(f"Warning: Could not find head body '{head_name}'")
        
        self._termination_heights = to_torch(self._termination_heights, device=self.device)
        return
    

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.cfg["env"]["asset"]["assetFileName"]

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # asset_options.fix_base_link = True
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        
        # 智能传感器创建 - 根据资产类型选择合适的足部关节
        asset_file_full = self.cfg["env"]["asset"]["assetFileName"]
        foot_sensor_mapping = {
            "mjcf/tai5/tai5.xml": ("R_ANKLE_R_S", "L_ANKLE_R_S"),
            "mjcf/humanoid/amp_humanoid.xml": ("right_foot", "left_foot"),
            "mjcf/humanoid/phys_humanoid.xml": ("right_foot", "left_foot"),
            "mjcf/humanoid/phys_humanoid_v2.xml": ("right_foot", "left_foot"),
            "mjcf/humanoid/phys_humanoid_v3.xml": ("right_foot", "left_foot"),
            "mjcf/humanoid/phys_humanoid_v3_box_foot.xml": ("right_foot", "left_foot"),
            "mjcf/g1/g1_29dof.xml": ("right_foot", "left_foot"),
        }
        
        if asset_file_full in foot_sensor_mapping:
            right_foot_name, left_foot_name = foot_sensor_mapping[asset_file_full]
            right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, right_foot_name)
            left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, left_foot_name)
            
            if right_foot_idx != -1 and left_foot_idx != -1:
                sensor_pose = gymapi.Transform()
                self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
                self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)
                self._has_foot_sensors = True
                print(f"Successfully created foot sensors for {asset_file_full}: {right_foot_name}, {left_foot_name}")
            else:
                self._has_foot_sensors = False
                print(f"Warning: Could not find foot bodies for {asset_file_full}")
                print(f"  Expected: {right_foot_name} (idx: {right_foot_idx}), {left_foot_name} (idx: {left_foot_idx})")
                # 打印所有可用的刚体名称以帮助调试
                print("  Available rigid bodies:")
                for i in range(self.gym.get_asset_rigid_body_count(humanoid_asset)):
                    body_name = self.gym.get_asset_rigid_body_name(humanoid_asset, i)
                    print(f"    {i}: {body_name}")
        else:
            # 尝试默认的足部名称
            right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
            left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
            
            if right_foot_idx != -1 and left_foot_idx != -1:
                sensor_pose = gymapi.Transform()
                self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
                self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)
                self._has_foot_sensors = True
                print(f"Created default foot sensors for {asset_file_full}")
            else:
                self._has_foot_sensors = False
                print(f"Warning: No foot sensor mapping defined for {asset_file_full} and default names not found")
                print(f"  Please add mapping to foot_sensor_mapping dictionary")

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        
        # **在这里添加tai5的特殊处理**
        # 检查是否是tai5模型
        is_tai5 = "tai5" in asset_file_full.lower()
        
        if is_tai5:
            print(f"Detected tai5 model: {asset_file_full}")
            
            # 获取原始数量
            original_num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
            original_num_shapes = self.gym.get_asset_rigid_shape_count(humanoid_asset)
            original_num_dof = self.gym.get_asset_dof_count(humanoid_asset)
            
            print(f"tai5 original: bodies={original_num_bodies}, shapes={original_num_shapes}, dof={original_num_dof}")
            
            # 使用完整的身体数量，不要限制为15
            self.num_bodies = original_num_bodies  # 使用完整的31个身体
            self.num_shapes = original_num_shapes
            
            # num_dof已经在_setup_character_props中设置了，这里验证一下
            if hasattr(self, 'num_dof'):
                print(f"tai5 num_dof already set to: {self.num_dof}")
            else:
                self.num_dof = min(original_num_dof, 30)  # 后备方案
                print(f"tai5 num_dof set as fallback to: {self.num_dof}")
            
            print(f"tai5 final: bodies={self.num_bodies}, shapes={self.num_shapes}, dof={self.num_dof}")
            
            self.is_tai5_model = True
            
        else:
            # 非tai5模型，使用原始数量
            self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
            self.num_shapes = self.gym.get_asset_rigid_shape_count(humanoid_asset)
            self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
            
            print(f"Using full model: bodies={self.num_bodies}, shapes={self.num_shapes}, dof={self.num_dof}")
            self.is_tai5_model = False
            


        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self._build_env(i, env_ptr, humanoid_asset)
            self.envs.append(env_ptr)

        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        if (self._pd_control):
            self._build_pd_action_offset_scale()
        # import ipdb;ipdb.set_trace()    
        print(f"=== 调试tai5身体部件 ===")
        total_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        print(f"资产总身体数量: {total_bodies}")
        
        for i in range(total_bodies):
            body_name = self.gym.get_asset_rigid_body_name(humanoid_asset, i)
            print(f"身体 {i}: {body_name}")
        
        print(f"使用的身体数量: {self.num_bodies}")
        print("=========================")
        return
    # def _create_envs(self, num_envs, spacing, num_per_row):
    #     lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    #     upper = gymapi.Vec3(spacing, spacing, spacing)

    #     asset_root = self.cfg["env"]["asset"]["assetRoot"]
    #     asset_file = self.cfg["env"]["asset"]["assetFileName"]

    #     asset_path = os.path.join(asset_root, asset_file)
    #     asset_root = os.path.dirname(asset_path)
    #     asset_file = os.path.basename(asset_path)

    #     asset_options = gymapi.AssetOptions()
    #     asset_options.angular_damping = 0.01
    #     asset_options.max_angular_velocity = 100.0
    #     asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    #     #asset_options.fix_base_link = True
    #     humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    #     actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
    #     motor_efforts = [prop.motor_effort for prop in actuator_props]
        
    #     # create force sensors at the feet
    #     right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
    #     left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
    #     sensor_pose = gymapi.Transform()

    #     self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
    #     self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

    #     self.max_motor_effort = max(motor_efforts)
    #     self.motor_efforts = to_torch(motor_efforts, device=self.device)

    #     self.torso_index = 0
    #     self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
    #     self.num_shapes = self.gym.get_asset_rigid_shape_count(humanoid_asset)
    #     self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)

    #     self.humanoid_handles = []
    #     self.envs = []
    #     self.dof_limits_lower = []
    #     self.dof_limits_upper = []
        
    #     for i in range(self.num_envs):
    #         # create env instance
    #         env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
    #         self._build_env(i, env_ptr, humanoid_asset)
    #         self.envs.append(env_ptr)

    #     dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
    #     for j in range(self.num_dof):
    #         if dof_prop['lower'][j] > dof_prop['upper'][j]:
    #             self.dof_limits_lower.append(dof_prop['upper'][j])
    #             self.dof_limits_upper.append(dof_prop['lower'][j])
    #         else:
    #             self.dof_limits_lower.append(dof_prop['lower'][j])
    #             self.dof_limits_upper.append(dof_prop['upper'][j])

    #     self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
    #     self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

    #     if (self._pd_control):
    #         self._build_pd_action_offset_scale()

    #     return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id
        col_filter = self._get_humanoid_collision_filter()
        segmentation_id = 0

        start_pose = gymapi.Transform()
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        if (asset_file == "mjcf/humanoid/amp_humanoid.xml"):
            self._char_h = 0.89 # perfect number
        elif (asset_file == "mjcf/humanoid/phys_humanoid.xml") or (asset_file == "mjcf/humanoid/phys_humanoid_v2.xml"):
            self._char_h = 0.92 # perfect number
        elif (asset_file == "mjcf/humanoid/phys_humanoid_v3.xml") or (asset_file == "mjcf/humanoid/phys_humanoid_v3_box_foot.xml"):
            self._char_h = 0.94
        elif (asset_file == "mjcf/g1/g1_29dof.xml") :
            self._char_h = 0.94
        elif (asset_file == "mjcf/tai5/tai5.xml"):
            self._char_h = 0.96  # TAI5的高度（根据MJCF中base_link的pos="0 0 0.96"）
        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert(False)
        start_pose.p = gymapi.Vec3(*get_axis_params(self._char_h, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", col_group, col_filter, segmentation_id)
        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        # # 为TAI5设置特殊的DOF属性（在创建actor之后）
        # if (asset_file == "mjcf/tai5/tai5.xml"):
        #     dof_props = self.gym.get_actor_dof_properties(env_ptr, humanoid_handle)
            
        #     # 大幅降低刚度和增加阻尼
        #     for i in range(len(dof_props['damping'])):
        #         dof_props['damping'][i] = 10.0   # 增加阻尼
        #         dof_props['stiffness'][i] = 50.0  # 大幅降低刚度
                
        #         # 对于某些关键关节可以设置更低的值
        #         if i < 6:  # 假设前6个是腰部关节
        #             dof_props['stiffness'][i] = 20.0
            
        #     self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_props)
        #     print(f"Applied TAI5 low-gain DOF properties")

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))

        if (self._pd_control):
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

        self.humanoid_handles.append(humanoid_handle)
        return
    def _build_pd_action_offset_scale(self):
        num_joints = len(self._dof_offsets) - 1
        
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            dof_offset = self._dof_offsets[j]
            dof_size = self._dof_offsets[j + 1] - self._dof_offsets[j]

            if (dof_size == 3):
                curr_low = lim_low[dof_offset:(dof_offset + dof_size)]
                curr_high = lim_high[dof_offset:(dof_offset + dof_size)]
                curr_low = np.max(np.abs(curr_low))
                curr_high = np.max(np.abs(curr_high))
                curr_scale = max([curr_low, curr_high])
                curr_scale = 1.2 * curr_scale
                curr_scale = min([curr_scale, np.pi])

                lim_low[dof_offset:(dof_offset + dof_size)] = -curr_scale
                lim_high[dof_offset:(dof_offset + dof_size)] = curr_scale
                
                #lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                #lim_high[dof_offset:(dof_offset + dof_size)] = np.pi


            elif (dof_size == 1):
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)
                
                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] =  curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return

    def _get_humanoid_collision_filter(self):
        """
        Setting the collision filter to 0 will enable collisions between all shapes in the actor. 
        Setting the collision filter to anything > 0 will disable all self collisions.
        """
        # return 1
        if self.cfg["env"]["enableSelfCollisionDetection"]:
            return 0
        else:
            return 1

    def _compute_reward(self, actions):
        self.rew_buf[:] = compute_humanoid_reward(self.obs_buf)
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights)
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _compute_observations(self, env_ids=None):
        obs = self._compute_humanoid_obs(env_ids)

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs

        return

    def _compute_humanoid_obs(self, env_ids=None):
        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
        else:
            body_pos = self._kinematic_humanoid_rigid_body_states[env_ids, :, 0:3]
            body_rot = self._kinematic_humanoid_rigid_body_states[env_ids, :, 3:7]
            body_vel = self._kinematic_humanoid_rigid_body_states[env_ids, :, 7:10]
            body_ang_vel = self._kinematic_humanoid_rigid_body_states[env_ids, :, 10:13]

        # TAI5特殊处理：只使用前15个身体进行观察计算
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        if asset_file == "mjcf/tai5/tai5.xml":
            # 只使用前15个关键身体，保持与motion数据的一致性
            body_pos = body_pos[:, :15, :]
            body_rot = body_rot[:, :15, :]
            body_vel = body_vel[:, :15, :]
            body_ang_vel = body_ang_vel[:, :15, :]

        obs = compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, 
                                                self._local_root_obs_policy,
                                                self._root_height_obs_policy)
        return obs

    def _reset_actors(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        return

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        if (self._pd_control):
            pd_tar = self._action_to_pd_targets(self.actions)

            # tracking initial state of each env (use for debug)
            if self.cfg["env"]["enableTrackInitState"]:
                pd_tar = self._every_env_init_dof_pos.clone()

            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

        return

    def post_physics_step(self):
        # root_heights = self._rigid_body_pos[:, 0, 2]
        # flying_envs = torch.where(root_heights > 1.8)[0]
        # if len(flying_envs) > 0:
        #     print(f"Flying envs detected: {flying_envs}")
        #     print(f"Heights: {root_heights[flying_envs]}")
        #     print(f"Root states: {self._humanoid_root_states[flying_envs]}")
        #     max_contact_forces = torch.max(torch.abs(self._contact_forces), dim=-1)[0]
        #     max_per_env = torch.max(max_contact_forces, dim=-1)[0]
            
        #     abnormal_contact = torch.where(max_per_env > 500.0)[0]  # 超过100N认为异常
        #     if len(abnormal_contact) > 0:
        #         print(f"异常接触力环境: {abnormal_contact[:5]}")  # 只打印前5个
        #         print(f"最大接触力: {max_per_env[abnormal_contact[:5]]}")
                
        #         # 检查具体是哪个身体部位有问题
        #         for env_id in abnormal_contact[:2]:  # 只检查前2个
        #             body_forces = torch.abs(self._contact_forces[env_id])
        #             problem_bodies = torch.where(torch.max(body_forces, dim=-1)[0] > 100.0)[0]
        #             print(f"环境{env_id}问题身体部位: {problem_bodies}")
        self.progress_buf += 1

        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()
        
        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def render(self, sync_frame_time=False):
        if self.viewer:
            self._update_camera()

        super().render(sync_frame_time)
        return

    # def _build_key_body_ids_tensor(self, key_body_names):
    #     env_ptr = self.envs[0]
    #     actor_handle = self.humanoid_handles[0]
    #     body_ids = []
    #     # import ipdb;ipdb.set_trace()
    #     for body_name in key_body_names:
    #         body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
    #         assert(body_id != -1)
    #         body_ids.append(body_id)

    #     body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
    #     import ipdb;ipdb.set_trace()
    #     return body_ids
    def _build_key_body_ids_tensor(self, key_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        
        # 检查是否是tai5模型
        # 可以通过检查第一个节点名称或其他方式判断
        first_body_name = key_body_names[0] if key_body_names else ""
        
        # tai5的15个主要节点映射
        tai5_15_nodes = [
            'base_link', 'WAIST_Y_S', 'NECK_Y_S', 'R_SHOULDER_P_S', 'R_ELBOW_Y_S', 
            'R_WRIST_R_S', 'L_SHOULDER_P_S', 'L_ELBOW_Y_S', 'L_WRIST_R_S', 
            'R_HIP_Y_S', 'R_KNEE_P_S', 'R_ANKLE_R_S', 'L_HIP_Y_S', 'L_KNEE_P_S', 'L_ANKLE_R_S'
        ]
        
        # 检查是否是tai5格式的body名称
        if any(name in tai5_15_nodes for name in key_body_names):
            print("Detected tai5 format, using direct index mapping")
            body_ids = []
            for body_name in key_body_names:
                if body_name in tai5_15_nodes:
                    # 直接使用在tai5_15_nodes中的索引
                    body_id = tai5_15_nodes.index(body_name)
                    print(f"tai5 body '{body_name}' -> index {body_id}")
                else:
                    # 如果找不到，使用gym查找（后备方案）
                    body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
                    if body_id == -1:
                        print(f"Warning: Body '{body_name}' not found, using index 0")
                        body_id = 0
                body_ids.append(body_id)
        else:
            # 非tai5格式，使用原来的方法
            body_ids = []
            for body_name in key_body_names:
                body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
                assert(body_id != -1)
                body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        print(f"Final body_ids: {body_ids}")
        # import ipdb;ipdb.set_trace()
        return body_ids
    
    def _build_contact_body_ids_tensor(self, contact_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in contact_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()
        
        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], 
                              self._cam_prev_char_pos[1] - 3.0, 
                              1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1],
                                 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()
        
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], 
                                  char_root_pos[1] + cam_delta[1], 
                                  cam_pos[2])

        if self._camera_follow_flag:
            self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return
    

    
    def _fetch_humanoid_rigid_body_pos_rot_states(self, env_ids=None):
        if env_ids is None:
            env_ids = to_torch(np.arange(self.num_envs), dtype=torch.long, device=self.device)

        rigid_body_pos_rot_states = torch.cat((self._rigid_body_pos[env_ids], self._rigid_body_rot[env_ids]), dim=-1)
        kinematic_buffer = self._kinematic_humanoid_rigid_body_states[env_ids, :, 0:7]
        
        # For the two variables that have just been reset, 
        # the values ​​will not be updated, so read the correct values ​​from the buffer
        mask = (self.progress_buf[env_ids] == 0)
        rigid_body_pos_rot_states[mask] = kinematic_buffer[mask]
        return rigid_body_pos_rot_states

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def dof_to_obs(pose, dof_obs_size, dof_offsets):
    # type: (Tensor, int, List[int]) -> Tensor
    joint_obs_size = 6
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # assume this is a spherical joint
        # import ipdb;ipdb.set_trace()
        # print(dof_size)
        if (dof_size == 3):
            joint_pose_q = torch_utils.exp_map_to_quat(joint_pose)
        elif (dof_size == 2):
            # 2DOF关节支持（踝关节：pitch + roll）
            axis1 = torch.tensor([1.0, 0.0, 0.0], dtype=joint_pose.dtype, device=pose.device)  # pitch
            axis2 = torch.tensor([0.0, 0.0, 1.0], dtype=joint_pose.dtype, device=pose.device)  # roll
            
            q1 = quat_from_angle_axis(joint_pose[..., 0], axis1)
            q2 = quat_from_angle_axis(joint_pose[..., 1], axis2)
            joint_pose_q = torch_utils.quat_mul(q1, q2)
        elif (dof_size == 1):
            axis = torch.tensor([0.0, 1.0, 0.0], dtype=joint_pose.dtype, device=pose.device)
            joint_pose_q = quat_from_angle_axis(joint_pose[..., 0], axis)
        else:
            joint_pose_q = None
            assert(False), "Unsupported joint type"
        # import ipdb;ipdb.set_trace()
        joint_dof_obs = torch_utils.quat_to_tan_norm(joint_pose_q)
        # print(f"Joint {j}: dof_size={dof_size}, joint_pose_q.shape={joint_pose_q.shape}")
        # print(f"joint_dof_obs.shape={joint_dof_obs.shape}")
        dof_obs[:, (j * joint_obs_size):((j + 1) * joint_obs_size)] = joint_dof_obs

    assert((num_joints * joint_obs_size) == dof_obs_size)

    return dof_obs

@torch.jit.script
def compute_humanoid_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos,
                                  local_root_obs, root_height_obs, dof_obs_size, dof_offsets):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)

    obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs

@torch.jit.script
def compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    
    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
    
    if (not local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
    
    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])
    
    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel), dim=-1)
    return obs


@torch.jit.script
def compute_humanoid_reward(obs_buf):
    # type: (Tensor) -> Tensor
    reward = torch.ones_like(obs_buf[:, 0])
    return reward

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated