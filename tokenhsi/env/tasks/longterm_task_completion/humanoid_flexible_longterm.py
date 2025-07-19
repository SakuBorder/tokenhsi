# tokenhsi/env/tasks/longterm_task_completion/humanoid_flexible_longterm.py

import torch
import numpy as np
import random
from enum import Enum

from tokenhsi.env.tasks.longterm_task_completion.humanoid_longterm_4basicskills import HumanoidLongTerm4BasicSkills
from utils import torch_utils
from isaacgym.torch_utils import *

class HumanoidFlexibleLongTerm(HumanoidLongTerm4BasicSkills):
    """
    灵活长技能组合系统
    可以使用预训练的状态转移模型执行任意顺序的技能序列
    """
    
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        
        # 技能执行配置
        self._skill_execution_time = cfg["env"].get("skillExecutionTime", 15.0)  # 每个技能的执行时间
        self._transition_time = cfg["env"].get("transitionTime", 3.0)            # 技能间转移时间
        self._enable_random_sequences = cfg["env"].get("enableRandomSequences", True)  # 是否支持随机序列
        
        # 预定义的技能序列（可选）
        self._predefined_sequences = cfg["env"].get("predefinedSequences", [])
        
        # 技能序列长度范围
        self._min_sequence_length = cfg["env"].get("minSequenceLength", 2)
        self._max_sequence_length = cfg["env"].get("maxSequenceLength", 6)
        
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        
        # 技能序列管理
        self._skill_sequences = torch.zeros(self.num_envs, self._max_sequence_length, dtype=torch.long, device=self.device)
        self._sequence_lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._current_skill_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._skill_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._in_transition = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 状态转移模型相关（假设已加载预训练的转移模型）
        self._transition_model_loaded = False
        self._enable_state_transition = cfg["env"].get("enableStateTransition", True)
        
        print(f"初始化灵活长技能系统，支持{self._min_sequence_length}-{self._max_sequence_length}长度的技能序列")
    
    def _generate_skill_sequence(self, env_id):
        """为指定环境生成技能序列"""
        if self._predefined_sequences and not self._enable_random_sequences:
            # 使用预定义序列
            sequence = random.choice(self._predefined_sequences)
            sequence_length = len(sequence)
        else:
            # 生成随机序列
            sequence_length = random.randint(self._min_sequence_length, self._max_sequence_length)
            
            if self._enable_random_sequences:
                # 完全随机序列（可能包含重复技能）
                sequence = [random.randint(0, 3) for _ in range(sequence_length)]
            else:
                # 避免连续重复的序列
                sequence = []
                prev_skill = -1
                for _ in range(sequence_length):
                    available_skills = [i for i in range(4) if i != prev_skill]
                    skill = random.choice(available_skills)
                    sequence.append(skill)
                    prev_skill = skill
        
        return sequence, sequence_length
    
    def _reset_skill_sequence(self, env_ids):
        """为指定环境重置技能序列"""
        for env_id in env_ids:
            env_id_int = env_id.item() if isinstance(env_id, torch.Tensor) else env_id
            
            sequence, length = self._generate_skill_sequence(env_id_int)
            
            # 填充序列到tensor
            self._sequence_lengths[env_id_int] = length
            self._skill_sequences[env_id_int, :length] = torch.tensor(sequence, device=self.device)
            self._skill_sequences[env_id_int, length:] = -1  # 无效标记
            
            # 重置执行状态
            self._current_skill_index[env_id_int] = 0
            self._skill_timer[env_id_int] = 0.0
            self._in_transition[env_id_int] = False
            
            # 设置初始任务mask
            current_skill = sequence[0]
            self._task_mask[env_id_int, :] = False
            self._task_mask[env_id_int, current_skill] = True
    
    def _update_skill_execution(self):
        """更新技能执行状态"""
        dt = self.dt
        self._skill_timer += dt
        
        # 检查是否需要进入转移阶段
        skill_complete_mask = (self._skill_timer >= self._skill_execution_time) & (~self._in_transition)
        
        if skill_complete_mask.sum() > 0:
            complete_env_ids = skill_complete_mask.nonzero(as_tuple=False).squeeze(-1)
            
            for env_id in complete_env_ids:
                env_id_int = env_id.item()
                current_index = self._current_skill_index[env_id_int].item()
                sequence_length = self._sequence_lengths[env_id_int].item()
                
                if current_index < sequence_length - 1:
                    # 还有下一个技能，进入转移阶段
                    self._in_transition[env_id_int] = True
                    self._skill_timer[env_id_int] = 0.0
                    
                    if self._enable_state_transition:
                        # 启用状态转移：混合当前技能和下一个技能
                        current_skill = self._skill_sequences[env_id_int, current_index].item()
                        next_skill = self._skill_sequences[env_id_int, current_index + 1].item()
                        
                        # 计算转移进度
                        transition_progress = min(self._skill_timer[env_id_int].item() / self._transition_time, 1.0)
                        
                        # 设置混合的任务mask
                        self._task_mask[env_id_int, :] = False
                        self._task_mask[env_id_int, current_skill] = 1.0 - transition_progress
                        self._task_mask[env_id_int, next_skill] = transition_progress
                    else:
                        # 直接切换到下一个技能
                        self._current_skill_index[env_id_int] += 1
                        next_skill = self._skill_sequences[env_id_int, current_index + 1].item()
                        self._task_mask[env_id_int, :] = False
                        self._task_mask[env_id_int, next_skill] = True
                        self._in_transition[env_id_int] = False
                else:
                    # 序列完成，重新生成新序列
                    self._reset_skill_sequence([env_id_int])
        
        # 检查转移阶段是否完成
        transition_complete_mask = (self._skill_timer >= self._transition_time) & self._in_transition
        
        if transition_complete_mask.sum() > 0:
            complete_env_ids = transition_complete_mask.nonzero(as_tuple=False).squeeze(-1)
            
            for env_id in complete_env_ids:
                env_id_int = env_id.item()
                current_index = self._current_skill_index[env_id_int].item()
                
                # 完成转移，切换到下一个技能
                self._current_skill_index[env_id_int] += 1
                next_skill = self._skill_sequences[env_id_int, current_index + 1].item()
                
                self._task_mask[env_id_int, :] = False
                self._task_mask[env_id_int, next_skill] = True
                self._in_transition[env_id_int] = False
                self._skill_timer[env_id_int] = 0.0
        
        # 更新转移阶段中的任务mask（平滑过渡）
        if self._enable_state_transition:
            transition_mask = self._in_transition
            if transition_mask.sum() > 0:
                transition_env_ids = transition_mask.nonzero(as_tuple=False).squeeze(-1)
                
                for env_id in transition_env_ids:
                    env_id_int = env_id.item()
                    current_index = self._current_skill_index[env_id_int].item()
                    
                    current_skill = self._skill_sequences[env_id_int, current_index].item()
                    next_skill = self._skill_sequences[env_id_int, current_index + 1].item()
                    
                    # 计算转移进度
                    transition_progress = min(self._skill_timer[env_id_int].item() / self._transition_time, 1.0)
                    
                    # 更新混合的任务mask
                    self._task_mask[env_id_int, :] = False
                    self._task_mask[env_id_int, current_skill] = 1.0 - transition_progress
                    self._task_mask[env_id_int, next_skill] = transition_progress
    
    def _reset_envs(self, env_ids):
        """重置环境时生成新的技能序列"""
        super()._reset_envs(env_ids)
        
        if len(env_ids) > 0:
            self._reset_skill_sequence(env_ids)
    
    def _compute_reward(self, actions):
        """计算奖励，支持状态转移奖励"""
        if self._enable_state_transition and self._in_transition.sum() > 0:
            # 在转移阶段，使用转移奖励
            base_reward = super()._compute_base_task_rewards()
            transition_reward = self._compute_transition_rewards()
            
            # 混合基础奖励和转移奖励
            reward = base_reward.clone()
            transition_mask = self._in_transition
            reward[transition_mask] = 0.7 * base_reward[transition_mask] + 0.3 * transition_reward[transition_mask]
            
            self.rew_buf[:] = reward
        else:
            # 正常技能执行阶段
            super()._compute_reward(actions)
    
    def _compute_transition_rewards(self):
        """计算状态转移奖励"""
        transition_mask = self._in_transition
        if transition_mask.sum() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        
        transition_env_ids = transition_mask.nonzero(as_tuple=False).squeeze(-1)
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # 计算转移平滑性奖励
        for env_id in transition_env_ids:
            env_id_int = env_id.item()
            
            # 1. 姿态变化平滑性
            current_pos = self._humanoid_root_states[env_id_int, 0:3]
            prev_pos = self._prev_root_pos[env_id_int]
            
            velocity = torch.norm(current_pos - prev_pos) / self.dt
            # 鼓励适中的移动速度（1-2 m/s）
            target_velocity = 1.5
            velocity_reward = torch.exp(-torch.abs(velocity - target_velocity))
            
            # 2. 角度变化平滑性
            current_rot = self._humanoid_root_states[env_id_int, 3:7]
            prev_rot = self._prev_root_rot[env_id_int]
            
            # 计算角度变化
            rot_diff = torch.abs(torch.sum(current_rot * prev_rot))  # 四元数点积
            rotation_reward = torch.clamp(rot_diff, 0.8, 1.0)  # 鼓励小的旋转变化
            
            # 3. 关节角度平滑性
            if hasattr(self, '_prev_dof_pos'):
                dof_change = torch.norm(self._dof_pos[env_id_int] - self._prev_dof_pos[env_id_int])
                joint_reward = torch.exp(-dof_change)
            else:
                joint_reward = torch.tensor(1.0, device=self.device)
            
            # 组合转移奖励
            transition_reward = 0.4 * velocity_reward + 0.3 * rotation_reward + 0.3 * joint_reward
            rewards[env_id_int] = transition_reward
        
        return rewards
    
    def post_physics_step(self):
        """物理步骤后处理"""
        # 记录前一帧状态用于转移奖励计算
        if not hasattr(self, '_prev_dof_pos'):
            self._prev_dof_pos = self._dof_pos.clone()
        else:
            self._prev_dof_pos[:] = self._dof_pos.clone()
        
        # 更新技能执行状态
        self._update_skill_execution()
        
        super().post_physics_step()
        
        # 添加技能序列信息到extras
        self.extras["skill_sequences"] = self._skill_sequences.clone()
        self.extras["sequence_lengths"] = self._sequence_lengths.clone() 
        self.extras["current_skill_index"] = self._current_skill_index.clone()
        self.extras["skill_timer"] = self._skill_timer.clone()
        self.extras["in_transition"] = self._in_transition.clone()
    
    def get_current_skill_info(self):
        """获取当前技能执行信息"""
        current_skills = []
        for env_id in range(self.num_envs):
            current_index = self._current_skill_index[env_id].item()
            sequence_length = self._sequence_lengths[env_id].item()
            
            if current_index < sequence_length:
                current_skill = self._skill_sequences[env_id, current_index].item()
                skill_names = ["traj", "sit", "carry", "climb"]
                current_skills.append(skill_names[current_skill])
            else:
                current_skills.append("completed")
        
        return current_skills
    
    def set_skill_sequence(self, env_id, sequence):
        """手动设置指定环境的技能序列"""
        if isinstance(sequence, list):
            sequence_length = len(sequence)
            
            # 验证序列有效性
            if sequence_length > self._max_sequence_length:
                raise ValueError(f"序列长度 {sequence_length} 超过最大限制 {self._max_sequence_length}")
            
            for skill in sequence:
                if skill not in [0, 1, 2, 3]:
                    raise ValueError(f"无效的技能ID: {skill}，应该在0-3之间")
            
            # 设置序列
            self._sequence_lengths[env_id] = sequence_length
            self._skill_sequences[env_id, :sequence_length] = torch.tensor(sequence, device=self.device)
            self._skill_sequences[env_id, sequence_length:] = -1
            
            # 重置执行状态
            self._current_skill_index[env_id] = 0
            self._skill_timer[env_id] = 0.0
            self._in_transition[env_id] = False
            
            # 设置初始任务mask
            self._task_mask[env_id, :] = False
            self._task_mask[env_id, sequence[0]] = True
            
            print(f"环境 {env_id} 设置技能序列: {[['traj', 'sit', 'carry', 'climb'][s] for s in sequence]}")
    
    def get_skill_sequence(self, env_id):
        """获取指定环境的技能序列"""
        sequence_length = self._sequence_lengths[env_id].item()
        sequence = self._skill_sequences[env_id, :sequence_length].cpu().numpy().tolist()
        skill_names = ["traj", "sit", "carry", "climb"]
        return [skill_names[s] for s in sequence]