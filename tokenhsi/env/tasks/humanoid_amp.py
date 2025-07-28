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

import os
from enum import Enum
import numpy as np
import torch

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymapi, gymutil
from env.tasks.humanoid import Humanoid, dof_to_obs
from utils import gym_util
from utils.motion_lib import MotionLib
from isaacgym.torch_utils import *

from utils import torch_utils


# # ----------Sim 参数-----------
# sim_params = gymapi.SimParams()
# sim_params.up_axis             = gymapi.UP_AXIS_Z        # Z 轴朝上
# sim_params.gravity             = gymapi.Vec3(0.0, 0.0, 0.0)  # 先把重力关掉
# sim_params.substeps            = 2                       # 可选：更快渲染
# sim_params.use_gpu_pipeline    = True

# # ----------Viewer-----------
# args = gymutil.parse_arguments(description="debug viewer")  # 读取 --headless
# args.headless = False        # <<< 打开 GUI
class HumanoidAMP(Humanoid):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        # configs for task
        self._enable_task_obs = cfg["env"]["enableTaskObs"]
        
        # traj following task
        self._num_traj_samples = cfg["env"]["numTrajSamples"]
        self._traj_sample_timestep = cfg["env"]["trajSampleTimestep"]
        self._speed_min = cfg["env"]["speedMin"]
        self._speed_max = cfg["env"]["speedMax"]
        self._accel_max = cfg["env"]["accelMax"]
        self._sharp_turn_prob = cfg["env"]["sharpTurnProb"]
        self._sharp_turn_angle = cfg["env"]["sharpTurnAngle"]
        self._fail_dist = 4.0

        # configs for amp
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidAMP.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._power_reward = cfg["env"]["power_reward"]
        self._power_coefficient = cfg["env"]["power_coefficient"]

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        motion_file = cfg['env']['motion_file']
        self._load_motion(motion_file)

        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        
        self._amp_obs_demo_buf = None

        # tensors for enableTrackInitState
        self._every_env_init_dof_pos = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=torch.float)

        # tensors for fixing obs bug
        self._kinematic_humanoid_rigid_body_states = torch.zeros((self.num_envs, self.num_bodies, 13), device=self.device, dtype=torch.float)

        return

    def post_physics_step(self):
        super().post_physics_step()
        
        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat

        return

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def fetch_amp_obs_demo(self, num_samples):

        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
        
        motion_ids = self._motion_lib.sample_motions(num_samples)
        
        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
        motion_times0 += truncate_time

        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())

        return amp_obs_demo_flat

    def build_amp_obs_demo(self, motion_ids, motion_times0):
        dt = self.dt

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel,
                                              dof_pos, dof_vel, key_pos,
                                              self._local_root_obs, self._root_height_obs,
                                              self._dof_obs_size, self._dof_offsets)
        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        return
        
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)
        # import ipdb;ipdb.set_trace()

        if (asset_file == "mjcf/humanoid/amp_humanoid.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        elif (asset_file == "mjcf/humanoid/phys_humanoid.xml") or (asset_file == "mjcf/humanoid/phys_humanoid_v2.xml") or (asset_file == "mjcf/humanoid/phys_humanoid_v3.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 2 * 2 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        elif (asset_file == "mjcf/g1/g1_29dof.xml") :
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
            # import ipdb;ipdb.set_trace()
        elif (asset_file == "mjcf/tai5/tai5.xml") or (asset_file == "mjcf/tai5/tai5.urdf"):
            # TAI5有30个DOF，所以观察空间需要相应调整
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 30 + 3 * num_key_bodies
            #                           ^    ^                    ^    ^
            #                           |    |                    |    └─ 关键关节位置
            #                           |    |                    └─ TAI5的30个DOF速度
            #                           |    └─ DOF观察空间(72)
            #                           └─ 根部状态(13)
        else:
            print("Unsupported character config file: ",format(asset_file))
            assert(False)

        return

    def _load_motion(self, motion_file):
        # assert(self._dof_offsets[-1] == self.num_dof)

        ext = os.path.splitext(motion_file)[1]
        if (ext == ".yaml"):
            self._motion_lib = MotionLib(motion_file=motion_file,
                                         skill=self.cfg["env"]["skill"],
                                         dof_body_ids=self._dof_body_ids,
                                         dof_offsets=self._dof_offsets,
                                         key_body_ids=self._key_body_ids.cpu().numpy(), 
                                         device=self.device)
        else:
            raise NotImplementedError

        return
    
    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super()._reset_envs(env_ids)
        self._init_amp_obs(env_ids)

        return

    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidAMP.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidAMP.StateInit.Start
              or self._state_init == HumanoidAMP.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == HumanoidAMP.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return
    
    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids

        if (len(self._reset_default_env_ids) > 0):
            self._kinematic_humanoid_rigid_body_states[self._reset_default_env_ids] = self._initial_humanoid_rigid_body_states[self._reset_default_env_ids]

        self._every_env_init_dof_pos[self._reset_default_env_ids] = self._initial_dof_pos[env_ids] # for "enableTrackInitState"

        return

    def _reset_ref_state_init(self, env_ids):
        """修正后的TAI5运动重定向"""
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        
        if (self._state_init == HumanoidAMP.StateInit.Random
            or self._state_init == HumanoidAMP.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self._state_init == HumanoidAMP.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
            = self._motion_lib.get_motion_state(motion_ids, motion_times)
        
        root_pos[:, 2] += 0.1
        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times

        if (len(self._reset_ref_env_ids) > 0):
            body_pos, body_rot, body_vel, body_ang_vel \
                = self._motion_lib.get_motion_state_max(self._reset_ref_motion_ids, self._reset_ref_motion_times)
            
            # ===== TAI5运动重定向处理 =====
            asset_file = self.cfg["env"]["asset"]["assetFileName"]
            if asset_file == "mjcf/tai5/tai5.xml":
                # 🔥 使用新的FK方法
                full_body_states = self._tai5_complete_fk_retargeting(
                    body_pos, body_rot, body_vel, body_ang_vel, 
                    root_pos, root_rot, root_vel, root_ang_vel
                )
                self._kinematic_humanoid_rigid_body_states[self._reset_ref_env_ids] = full_body_states
            else:
                # 非TAI5模型，使用原来的方法
                self._kinematic_humanoid_rigid_body_states[self._reset_ref_env_ids] = torch.cat((body_pos, body_rot, body_vel, body_ang_vel), dim=-1)
        
        self._every_env_init_dof_pos[self._reset_ref_env_ids] = dof_pos
        return

    def _tai5_complete_fk_retargeting(self, body_pos, body_rot, body_vel, body_ang_vel, 
                                    root_pos, root_rot, root_vel, root_ang_vel):
        """完整的TAI5 FK重定向：参考SMPL FK实现"""
        
        batch_size = len(self._reset_ref_env_ids)
        
        # 创建完整的身体状态tensor
        full_body_states = torch.zeros((batch_size, self.num_bodies, 13), 
                                    device=self.device, dtype=torch.float)
        
        # TAI5的层次结构定义（基于MJCF文件）
        tai5_hierarchy = self._get_tai5_hierarchy()
        
        # Motion数据到TAI5模型的映射
        motion_to_model_mapping = {
            0: 0,   # base_link
            1: 13,  # WAIST_Y_S  
            2: 23,  # NECK_Y_S
            3: 24,  # R_SHOULDER_P_S
            4: 27,  # R_ELBOW_Y_S
            5: 30,  # R_WRIST_R_S
            6: 16,  # L_SHOULDER_P_S
            7: 19,  # L_ELBOW_Y_S
            8: 22,  # L_WRIST_R_S
            9: 7,   # R_HIP_P_S
            10: 10, # R_KNEE_P_S
            11: 12, # R_ANKLE_R_S
            12: 1,  # L_HIP_P_S
            13: 4,  # L_KNEE_P_S
            14: 6,  # L_ANKLE_R_S
        }
        
        # 1. 直接映射已知的15个关键身体
        for motion_idx, model_idx in motion_to_model_mapping.items():
            if motion_idx < body_pos.shape[1] and model_idx < self.num_bodies:
                full_body_states[:, model_idx, 0:3] = body_pos[:, motion_idx, :]      # position
                full_body_states[:, model_idx, 3:7] = body_rot[:, motion_idx, :]      # rotation
                full_body_states[:, model_idx, 7:10] = body_vel[:, motion_idx, :]     # velocity
                full_body_states[:, model_idx, 10:13] = body_ang_vel[:, motion_idx, :] # angular velocity
        
        # 2. 使用FK计算剩余身体的位置和速度
        self._tai5_forward_kinematics(full_body_states, tai5_hierarchy, motion_to_model_mapping)
        
        return full_body_states

    def _get_tai5_hierarchy(self):
        """获取TAI5的层次结构（父子关系）"""
        # 基于TAI5 MJCF文件的层次结构
        hierarchy = {
            # 身体索引: 父身体索引
            0: -1,   # base_link (root)
            
            # 左腿链
            1: 0,    # L_HIP_P_S -> base_link
            2: 1,    # L_HIP_R_S -> L_HIP_P_S
            3: 2,    # L_HIP_Y_S -> L_HIP_R_S
            4: 3,    # L_KNEE_P_S -> L_HIP_Y_S
            5: 4,    # L_ANKLE_P_S -> L_KNEE_P_S
            6: 5,    # L_ANKLE_R_S -> L_ANKLE_P_S
            
            # 右腿链
            7: 0,    # R_HIP_P_S -> base_link
            8: 7,    # R_HIP_R_S -> R_HIP_P_S
            9: 8,    # R_HIP_Y_S -> R_HIP_R_S
            10: 9,   # R_KNEE_P_S -> R_HIP_Y_S
            11: 10,  # R_ANKLE_P_S -> R_KNEE_P_S
            12: 11,  # R_ANKLE_R_S -> R_ANKLE_P_S
            
            # 躯干链
            13: 0,   # WAIST_Y_S -> base_link
            14: 13,  # WAIST_R_S -> WAIST_Y_S
            15: 14,  # WAIST_P_S -> WAIST_R_S
            
            # 左臂链
            16: 15,  # L_SHOULDER_P_S -> WAIST_P_S
            17: 16,  # L_SHOULDER_R_S -> L_SHOULDER_P_S
            18: 17,  # L_SHOULDER_Y_S -> L_SHOULDER_R_S
            19: 18,  # L_ELBOW_Y_S -> L_SHOULDER_Y_S
            20: 19,  # L_WRIST_P_S -> L_ELBOW_Y_S
            21: 20,  # L_WRIST_Y_S -> L_WRIST_P_S
            22: 21,  # L_WRIST_R_S -> L_WRIST_Y_S
            
            # 头部
            23: 15,  # NECK_Y_S -> WAIST_P_S
            
            # 右臂链
            24: 15,  # R_SHOULDER_P_S -> WAIST_P_S
            25: 24,  # R_SHOULDER_R_S -> R_SHOULDER_P_S
            26: 25,  # R_SHOULDER_Y_S -> R_SHOULDER_R_S
            27: 26,  # R_ELBOW_Y_S -> R_SHOULDER_Y_S
            28: 27,  # R_WRIST_P_S -> R_ELBOW_Y_S
            29: 28,  # R_WRIST_Y_S -> R_WRIST_P_S
            30: 29,  # R_WRIST_R_S -> R_WRIST_Y_S
        }
        return hierarchy

    def _tai5_forward_kinematics(self, full_body_states, hierarchy, mapped_indices):
        """TAI5前向运动学：参考SMPL FK实现思路"""
        
        mapped_set = set(mapped_indices.values())
        
        # 获取关节偏移量（基于MJCF pos属性）
        joint_offsets = self._get_tai5_joint_offsets()
        
        # 按层次顺序处理每个身体
        for body_idx in range(self.num_bodies):
            parent_idx = hierarchy.get(body_idx, -1)
            
            # 跳过根节点和已经映射的节点
            if parent_idx == -1 or body_idx in mapped_set:
                continue
                
            # 确保父节点已经处理
            if parent_idx not in mapped_set:
                # 如果父节点也未映射，需要先处理父节点
                self._process_unmapped_parent(full_body_states, parent_idx, hierarchy, mapped_set, joint_offsets)
            
            # 使用FK计算当前节点
            self._compute_child_from_parent_fk(full_body_states, body_idx, parent_idx, joint_offsets)

    def _process_unmapped_parent(self, full_body_states, parent_idx, hierarchy, mapped_set, joint_offsets):
        """递归处理未映射的父节点"""
        grandparent_idx = hierarchy.get(parent_idx, -1)
        
        if grandparent_idx != -1 and grandparent_idx not in mapped_set:
            # 递归处理祖父节点
            self._process_unmapped_parent(full_body_states, grandparent_idx, hierarchy, mapped_set, joint_offsets)
        
        # 现在处理父节点
        if grandparent_idx != -1:
            self._compute_child_from_parent_fk(full_body_states, parent_idx, grandparent_idx, joint_offsets)

    def _compute_child_from_parent_fk(self, full_body_states, child_idx, parent_idx, joint_offsets):
        """基于父节点计算子节点的位置和速度（核心FK逻辑）"""
        
        # 获取父节点状态
        parent_pos = full_body_states[:, parent_idx, 0:3]       # 位置
        parent_rot = full_body_states[:, parent_idx, 3:7]       # 旋转(四元数)
        parent_vel = full_body_states[:, parent_idx, 7:10]      # 线速度
        parent_ang_vel = full_body_states[:, parent_idx, 10:13] # 角速度
        
        # 获取相对偏移
        relative_offset = joint_offsets.get(child_idx, torch.zeros(3, device=self.device))
        batch_size = parent_pos.shape[0]
        relative_offset_batch = relative_offset.unsqueeze(0).expand(batch_size, -1)
        
        # 🔥 位置计算：子位置 = 父位置 + 旋转(相对偏移)
        child_pos = parent_pos + quat_rotate(parent_rot, relative_offset_batch)
        
        # 🔥 速度计算：子速度 = 父速度 + 角速度 × 相对偏移
        # 这是刚体运动学的核心公式！
        induced_vel = torch.cross(parent_ang_vel, relative_offset_batch, dim=-1)
        child_vel = parent_vel + induced_vel
        
        # 旋转继承（简化处理，可以根据关节类型调整）
        child_rot = parent_rot  # 对于大部分关节，旋转与父节点相同
        child_ang_vel = parent_ang_vel * 0.95  # 角速度稍微衰减
        
        # 更新子节点状态
        full_body_states[:, child_idx, 0:3] = child_pos
        full_body_states[:, child_idx, 3:7] = child_rot
        full_body_states[:, child_idx, 7:10] = child_vel
        full_body_states[:, child_idx, 10:13] = child_ang_vel

    def _get_tai5_joint_offsets(self):
        """获取TAI5所有关节的相对偏移量"""
        # 基于MJCF文件中的pos属性定义
        offsets = {
            # 左腿链偏移
            2: torch.tensor([0.058, 0.044, 0.0], device=self.device, dtype=torch.float),
            3: torch.tensor([-0.058, 0.0, -0.1972], device=self.device, dtype=torch.float),
            5: torch.tensor([-0.032248, -0.058, -0.36859], device=self.device, dtype=torch.float),
            
            # 右腿链偏移
            8: torch.tensor([0.058, -0.044, 0.0], device=self.device, dtype=torch.float),
            9: torch.tensor([-0.058, 0.0, -0.1972], device=self.device, dtype=torch.float),
            11: torch.tensor([-0.032248, 0.058, -0.36859], device=self.device, dtype=torch.float),
            
            # 躯干链偏移
            14: torch.tensor([-0.0495, 0.0, 0.127], device=self.device, dtype=torch.float),
            15: torch.tensor([0.0495, 0.0465, 0.0], device=self.device, dtype=torch.float),
            
            # 左臂链偏移
            17: torch.tensor([0.054, 0.066, 0.0], device=self.device, dtype=torch.float),
            18: torch.tensor([-0.054, 0.1395, 0.0], device=self.device, dtype=torch.float),
            20: torch.tensor([0.0, 0.1281, -0.051], device=self.device, dtype=torch.float),
            21: torch.tensor([9.0984e-05, 0.14468, 0.0425], device=self.device, dtype=torch.float),
            
            # 右臂链偏移
            25: torch.tensor([0.054, -0.066, 0.0], device=self.device, dtype=torch.float),
            26: torch.tensor([-0.054, -0.1395, 0.0], device=self.device, dtype=torch.float),
            28: torch.tensor([0.0, -0.1281, -0.051], device=self.device, dtype=torch.float),
            29: torch.tensor([9.0984e-05, -0.14468, 0.0425], device=self.device, dtype=torch.float),
            
            # 对于没有明确偏移的关节，使用默认值
        }
        
        # 为所有关节提供默认偏移
        for i in range(self.num_bodies):
            if i not in offsets:
                offsets[i] = torch.tensor([0.0, 0.0, 0.0], device=self.device, dtype=torch.float)
        
        return offsets
        
    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)

        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        if (len(self._reset_ref_env_ids) > 0):
            self._init_amp_obs_ref(self._reset_ref_env_ids, self._reset_ref_motion_ids,
                                   self._reset_ref_motion_times)
        
        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, 
                                              dof_pos, dof_vel, key_pos, 
                                              self._local_root_obs, self._root_height_obs, 
                                              self._dof_obs_size, self._dof_offsets)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return
    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return
    
    def _compute_amp_observations(self, env_ids=None):
        if (env_ids is None):
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
            
            # TAI5特殊处理：只使用前15个身体
            asset_file = self.cfg["env"]["asset"]["assetFileName"]
            if asset_file == "mjcf/tai5/tai5.xml":
                root_pos = self._rigid_body_pos[:, 0, :]
                root_rot = self._rigid_body_rot[:, 0, :]
                root_vel = self._rigid_body_vel[:, 0, :]
                root_ang_vel = self._rigid_body_ang_vel[:, 0, :]
            else:
                root_pos = self._rigid_body_pos[:, 0, :]
                root_rot = self._rigid_body_rot[:, 0, :]
                root_vel = self._rigid_body_vel[:, 0, :]
                root_ang_vel = self._rigid_body_ang_vel[:, 0, :]
                
            self._curr_amp_obs_buf[:] = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel,
                                                            self._dof_pos, self._dof_vel, key_body_pos,
                                                            self._local_root_obs, self._root_height_obs, 
                                                            self._dof_obs_size, self._dof_offsets)
        else:
            # 对于重置的环境，使用kinematic状态
            kinematic_rigid_body_pos = self._kinematic_humanoid_rigid_body_states[:, :, 0:3]
            key_body_pos = kinematic_rigid_body_pos[:, self._key_body_ids, :]
            
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(self._kinematic_humanoid_rigid_body_states[env_ids, 0, 0:3],
                                                                self._kinematic_humanoid_rigid_body_states[env_ids, 0, 3:7],
                                                                self._kinematic_humanoid_rigid_body_states[env_ids, 0, 7:10],
                                                                self._kinematic_humanoid_rigid_body_states[env_ids, 0, 10:13],
                                                                self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                self._local_root_obs, self._root_height_obs, 
                                                                self._dof_obs_size, self._dof_offsets)
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
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
