import os
import torch
import numpy as np
import random
from enum import Enum

from tokenhsi.env.tasks.multi_task.humanoid_traj_sit_carry_climb import HumanoidTrajSitCarryClimb
from utils import torch_utils
from isaacgym.torch_utils import *
from isaacgym import gymapi, gymtorch


class HumanoidStateTransition4Skills(HumanoidTrajSitCarryClimb):
    """
    四个基础技能之间状态转移的训练任务类
    支持traj、sit、carry、climb技能间的两两转移
    """
    
    class TransitionPhase(Enum):
        SOURCE_SKILL = 0      # 执行源技能阶段
        TRANSITION = 1        # 状态转移阶段  
        TARGET_SKILL = 2      # 执行目标技能阶段
    
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # 状态转移相关配置
        self._enable_state_transition = cfg["env"].get("enableStateTransition", True)
        self._transition_mode = cfg["env"].get("transitionMode", "pairwise")
        self._transition_pairs = cfg["env"].get("transitionPairs", [])
        
        # 时间配置
        self._transition_start_time = cfg["env"].get("transitionStartTime", 5.0)
        self._transition_duration = cfg["env"].get("transitionDuration", 10.0) 
        self._transition_end_time = cfg["env"].get("transitionEndTime", 15.0)
        
        # 奖励权重配置
        self._transition_reward_weights = cfg["env"].get("transitionRewardWeights", {
            "source_skill": 0.3,
            "target_skill": 0.5, 
            "transition_smooth": 0.2
        })
        
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        
        # 状态转移相关张量
        self._transition_phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._source_skill_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._target_skill_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._transition_progress = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self._transition_start_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # 技能状态存储（用于平滑过渡）
        self._source_skill_state = torch.zeros(self.num_envs, self.get_obs_size(), device=self.device)
        self._target_skill_state = torch.zeros(self.num_envs, self.get_obs_size(), device=self.device)
        
        # 转移对映射
        self._transition_pair_mapping = self._build_transition_mapping()
        
        print(f"初始化状态转移训练，支持 {len(self._transition_pairs)} 个转移对")
    
    def _build_transition_mapping(self):
        """构建技能转移对的映射"""
        mapping = {}
        for i, (from_skill, to_skill) in enumerate(self._transition_pairs):
            mapping[i] = (from_skill, to_skill)
        return mapping
    
    def _reset_envs(self, env_ids):
        """重置环境时随机选择转移对"""
        super()._reset_envs(env_ids)
        
        if len(env_ids) > 0:
            # 随机选择转移对
            transition_indices = torch.randint(0, len(self._transition_pairs), 
                                             (len(env_ids),), device=self.device)
            
            for i, env_id in enumerate(env_ids):
                trans_idx = transition_indices[i].item()
                source_skill, target_skill = self._transition_pair_mapping[trans_idx]
                
                self._source_skill_id[env_id] = source_skill
                self._target_skill_id[env_id] = target_skill
                self._transition_phase[env_id] = self.TransitionPhase.SOURCE_SKILL.value
                self._transition_progress[env_id] = 0.0
                self._transition_start_step[env_id] = 0
            
            # 设置初始任务mask为源技能
            self._update_task_masks_for_transition(env_ids)
    
    def _update_task_masks_for_transition(self, env_ids):
        """根据转移阶段更新任务mask"""
        for env_id in env_ids:
            current_phase = self._transition_phase[env_id].item()
            
            if current_phase == self.TransitionPhase.SOURCE_SKILL.value:
                # 源技能阶段
                skill_id = self._source_skill_id[env_id].item()
                self._task_mask[env_id, :] = False
                self._task_mask[env_id, skill_id] = True
                
            elif current_phase == self.TransitionPhase.TARGET_SKILL.value:
                # 目标技能阶段
                skill_id = self._target_skill_id[env_id].item()
                self._task_mask[env_id, :] = False
                self._task_mask[env_id, skill_id] = True
                
            elif current_phase == self.TransitionPhase.TRANSITION.value:
                # 转移阶段：根据进度混合两个技能
                source_id = self._source_skill_id[env_id].item()
                target_id = self._target_skill_id[env_id].item()
                progress = self._transition_progress[env_id].item()
                
                self._task_mask[env_id, :] = False
                # 使用进度作为权重混合两个技能
                self._task_mask[env_id, source_id] = 1.0 - progress
                self._task_mask[env_id, target_id] = progress
    
    def _update_transition_state(self):
        """更新状态转移进度"""
        current_time = self.progress_buf.float() * self.dt
        
        # 检查是否开始转移
        start_transition_mask = (current_time >= self._transition_start_time) & \
                               (self._transition_phase == self.TransitionPhase.SOURCE_SKILL.value)
        
        if start_transition_mask.sum() > 0:
            env_ids = start_transition_mask.nonzero(as_tuple=False).squeeze(-1)
            self._transition_phase[env_ids] = self.TransitionPhase.TRANSITION.value
            self._transition_start_step[env_ids] = self.progress_buf[env_ids]
            
            # 保存源技能状态
            self._source_skill_state[env_ids] = self.obs_buf[env_ids].clone()
            
            print(f"开始状态转移，环境数量: {len(env_ids)}")
        
        # 更新转移进度
        in_transition_mask = (self._transition_phase == self.TransitionPhase.TRANSITION.value)
        if in_transition_mask.sum() > 0:
            env_ids = in_transition_mask.nonzero(as_tuple=False).squeeze(-1)
            elapsed_steps = self.progress_buf[env_ids] - self._transition_start_step[env_ids]
            elapsed_time = elapsed_steps.float() * self.dt
            
            # 计算转移进度 (0.0 -> 1.0)
            self._transition_progress[env_ids] = torch.clamp(
                elapsed_time / self._transition_duration, 0.0, 1.0
            )
            
            # 检查是否完成转移
            complete_transition_mask = (elapsed_time >= self._transition_duration)
            if complete_transition_mask.sum() > 0:
                complete_env_ids = env_ids[complete_transition_mask]
                self._transition_phase[complete_env_ids] = self.TransitionPhase.TARGET_SKILL.value
                self._transition_progress[complete_env_ids] = 1.0
                
                print(f"完成状态转移，环境数量: {len(complete_env_ids)}")
        
        # 更新任务masks
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._update_task_masks_for_transition(all_env_ids)
    
    def _compute_transition_reward(self):
        """计算状态转移奖励"""
        reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        # 源技能阶段奖励
        source_mask = (self._transition_phase == self.TransitionPhase.SOURCE_SKILL.value)
        if source_mask.sum() > 0:
            source_reward = self._compute_skill_reward(self._source_skill_id[source_mask])
            reward[source_mask] = source_reward * self._transition_reward_weights["source_skill"]
        
        # 目标技能阶段奖励
        target_mask = (self._transition_phase == self.TransitionPhase.TARGET_SKILL.value)
        if target_mask.sum() > 0:
            target_reward = self._compute_skill_reward(self._target_skill_id[target_mask])
            reward[target_mask] = target_reward * self._transition_reward_weights["target_skill"]
        
        # 转移阶段奖励
        transition_mask = (self._transition_phase == self.TransitionPhase.TRANSITION.value)
        if transition_mask.sum() > 0:
            transition_env_ids = transition_mask.nonzero(as_tuple=False).squeeze(-1)
            
            # 混合源技能和目标技能奖励
            source_reward = self._compute_skill_reward(self._source_skill_id[transition_env_ids])
            target_reward = self._compute_skill_reward(self._target_skill_id[transition_env_ids])
            
            progress = self._transition_progress[transition_env_ids]
            mixed_reward = (1.0 - progress) * source_reward + progress * target_reward
            
            # 添加平滑转移奖励
            smooth_reward = self._compute_smooth_transition_reward(transition_env_ids)
            
            total_transition_reward = (
                mixed_reward * (self._transition_reward_weights["source_skill"] + 
                              self._transition_reward_weights["target_skill"]) +
                smooth_reward * self._transition_reward_weights["transition_smooth"]
            )
            
            reward[transition_env_ids] = total_transition_reward
        
        return reward
    
    def _compute_skill_reward(self, skill_ids):
        """根据技能ID计算对应的技能奖励"""
        # 这里应该调用各个技能的具体奖励函数
        # 简化实现，返回基础奖励
        return torch.ones_like(skill_ids, dtype=torch.float, device=self.device)
    
    def _compute_smooth_transition_reward(self, env_ids):
        """计算平滑转移奖励"""
        # 计算姿态变化的平滑度
        current_root_pos = self._humanoid_root_states[env_ids, 0:3]
        prev_root_pos = self._prev_root_pos[env_ids]
        
        # 根据速度变化计算平滑度奖励
        velocity_change = torch.norm(current_root_pos - prev_root_pos, dim=-1) / self.dt
        
        # 鼓励平滑的速度变化
        smooth_reward = torch.exp(-torch.clamp(velocity_change - 1.0, 0.0, float('inf')))
        
        return smooth_reward
    
    def _compute_reward(self, actions):
        """重写奖励计算，支持状态转移"""
        if self._enable_state_transition:
            transition_reward = self._compute_transition_reward()
            self.rew_buf[:] = transition_reward
        else:
            super()._compute_reward(actions)
        
        # 添加功率惩罚
        if self._power_reward:
            power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim=-1)
            power_penalty = -self._power_coefficient * power
            self.rew_buf += power_penalty
    
    def post_physics_step(self):
        """物理步骤后处理"""
        # 更新转移状态
        if self._enable_state_transition:
            self._update_transition_state()
        
        super().post_physics_step()
        
        # 添加转移相关的额外信息
        if self._enable_state_transition:
            self.extras["transition_phase"] = self._transition_phase.clone()
            self.extras["transition_progress"] = self._transition_progress.clone()
            self.extras["source_skill"] = self._source_skill_id.clone()
            self.extras["target_skill"] = self._target_skill_id.clone()
    
    def get_transition_info(self):
        """获取状态转移信息，用于调试和监控"""
        return {
            "transition_pairs": self._transition_pairs,
            "num_transitions": len(self._transition_pairs),
            "current_phases": self._transition_phase.cpu().numpy(),
            "transition_progress": self._transition_progress.cpu().numpy(),
            "source_skills": self._source_skill_id.cpu().numpy(),
            "target_skills": self._target_skill_id.cpu().numpy()
        }