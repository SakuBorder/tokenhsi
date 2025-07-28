import torch
import numpy as np
from enum import Enum
import random

# 直接继承多任务基础类，而不是长期任务完成类
from tokenhsi.env.tasks.multi_task.humanoid_traj_sit_carry_climb import HumanoidTrajSitCarryClimb
from utils import torch_utils
from isaacgym.torch_utils import *

class HumanoidStateTransition4Skills(HumanoidTrajSitCarryClimb):
    """
    增强版四技能状态转移训练环境
    支持所有技能对之间的状态转移学习
    直接继承多任务基础类，避免长期任务配置依赖
    """
    
    class TransitionPhase(Enum):
        SOURCE_SKILL = 0      # 执行源技能阶段
        TRANSITION = 1        # 状态转移阶段  
        TARGET_SKILL = 2      # 执行目标技能阶段
    
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # 所有可能的技能转移对 (4个技能的两两组合，共12对)
        self._all_transition_pairs = [
            (0, 1), (0, 2), (0, 3),  # traj -> sit, carry, climb  
            (1, 0), (1, 2), (1, 3),  # sit -> traj, carry, climb
            (2, 0), (2, 1), (2, 3),  # carry -> traj, sit, climb
            (3, 0), (3, 1), (3, 2)   # climb -> traj, sit, carry
        ]
        
        # 状态转移训练的时间配置
        self._source_skill_duration = cfg["env"].get("sourceSkillDuration", 8.0)  # 源技能执行时间
        self._transition_duration = cfg["env"].get("transitionDuration", 4.0)     # 转移阶段时间
        self._target_skill_duration = cfg["env"].get("targetSkillDuration", 8.0)  # 目标技能执行时间
        
        # 奖励权重
        self._transition_reward_weights = cfg["env"].get("transitionRewardWeights", {
            "source_skill": 0.3,
            "target_skill": 0.5, 
            "transition_smooth": 0.2
        })
        
        # 调用父类初始化，使用多任务基础类
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        
        # 状态转移相关张量
        self._transition_phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._source_skill_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._target_skill_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._transition_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        # 转移期间的状态记录（用于平滑奖励计算）
        self._source_end_state = torch.zeros(self.num_envs, self.get_obs_size(), device=self.device)
        self._target_start_state = torch.zeros(self.num_envs, self.get_obs_size(), device=self.device)
        
        print(f"初始化状态转移训练，支持 {len(self._all_transition_pairs)} 个转移对")
    
    def _reset_envs(self, env_ids):
        """重置环境时随机选择转移对和初始阶段"""
        super()._reset_envs(env_ids)
        
        if len(env_ids) > 0:
            # 为每个环境随机选择一个转移对
            for env_id in env_ids:
                transition_pair = random.choice(self._all_transition_pairs)
                source_skill, target_skill = transition_pair
                
                self._source_skill_id[env_id] = source_skill
                self._target_skill_id[env_id] = target_skill
                
                # 随机选择起始阶段（可以从任一阶段开始）
                start_phase = random.choice([
                    self.TransitionPhase.SOURCE_SKILL.value,
                    self.TransitionPhase.TRANSITION.value,
                    self.TransitionPhase.TARGET_SKILL.value
                ])
                self._transition_phase[env_id] = start_phase
                self._transition_timer[env_id] = 0.0
            
            # 更新任务masks
            self._update_task_masks_for_transition(env_ids)
    
    def _update_task_masks_for_transition(self, env_ids):
        """根据转移阶段和进度更新任务mask"""
        for env_id in env_ids:
            current_phase = self._transition_phase[env_id].item()
            source_id = self._source_skill_id[env_id].item()
            target_id = self._target_skill_id[env_id].item()
            
            # 清除之前的任务mask
            if hasattr(self, '_task_mask'):
                self._task_mask[env_id, :] = False
                
                if current_phase == self.TransitionPhase.SOURCE_SKILL.value:
                    # 源技能阶段：执行源技能
                    self._task_mask[env_id, source_id] = True
                    
                elif current_phase == self.TransitionPhase.TARGET_SKILL.value:
                    # 目标技能阶段：执行目标技能
                    self._task_mask[env_id, target_id] = True
                    
                elif current_phase == self.TransitionPhase.TRANSITION.value:
                    # 转移阶段：根据时间进度混合两个技能
                    progress = min(self._transition_timer[env_id].item() / self._transition_duration, 1.0)
                    
                    # 使用连续的权重进行技能混合
                    source_weight = 1.0 - progress
                    target_weight = progress
                    
                    self._task_mask[env_id, source_id] = source_weight
                    self._task_mask[env_id, target_id] = target_weight
    
    def _update_transition_state(self):
        """更新状态转移进度和阶段切换"""
        dt = self.dt
        self._transition_timer += dt
        
        # 检查阶段切换条件
        for phase in [self.TransitionPhase.SOURCE_SKILL, self.TransitionPhase.TRANSITION, self.TransitionPhase.TARGET_SKILL]:
            phase_mask = (self._transition_phase == phase.value)
            
            if phase_mask.sum() > 0:
                env_ids = phase_mask.nonzero(as_tuple=False).squeeze(-1)
                
                if phase == self.TransitionPhase.SOURCE_SKILL:
                    # 源技能阶段 -> 转移阶段
                    transition_mask = self._transition_timer[env_ids] >= self._source_skill_duration
                    if transition_mask.sum() > 0:
                        transition_env_ids = env_ids[transition_mask]
                        self._transition_phase[transition_env_ids] = self.TransitionPhase.TRANSITION.value
                        self._transition_timer[transition_env_ids] = 0.0
                        
                        # 记录源技能结束时的状态
                        self._source_end_state[transition_env_ids] = self.obs_buf[transition_env_ids].clone()
                        
                elif phase == self.TransitionPhase.TRANSITION:
                    # 转移阶段 -> 目标技能阶段
                    transition_mask = self._transition_timer[env_ids] >= self._transition_duration
                    if transition_mask.sum() > 0:
                        transition_env_ids = env_ids[transition_mask]
                        self._transition_phase[transition_env_ids] = self.TransitionPhase.TARGET_SKILL.value
                        self._transition_timer[transition_env_ids] = 0.0
                        
                        # 记录目标技能开始时的状态
                        self._target_start_state[transition_env_ids] = self.obs_buf[transition_env_ids].clone()
                        
                elif phase == self.TransitionPhase.TARGET_SKILL:
                    # 目标技能阶段完成 -> 重新开始循环
                    reset_mask = self._transition_timer[env_ids] >= self._target_skill_duration
                    if reset_mask.sum() > 0:
                        reset_env_ids = env_ids[reset_mask]
                        self._reset_envs(reset_env_ids)
        
        # 更新所有环境的任务masks
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._update_task_masks_for_transition(all_env_ids)
    
    def _compute_transition_reward(self):
        """计算状态转移奖励"""
        reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        # 基础技能奖励（根据当前执行的技能）
        base_task_reward = self._compute_base_task_rewards()
        
        # 源技能阶段
        source_mask = (self._transition_phase == self.TransitionPhase.SOURCE_SKILL.value)
        if source_mask.sum() > 0:
            reward[source_mask] = base_task_reward[source_mask] * self._transition_reward_weights["source_skill"]
        
        # 目标技能阶段  
        target_mask = (self._transition_phase == self.TransitionPhase.TARGET_SKILL.value)
        if target_mask.sum() > 0:
            reward[target_mask] = base_task_reward[target_mask] * self._transition_reward_weights["target_skill"]
        
        # 转移阶段
        transition_mask = (self._transition_phase == self.TransitionPhase.TRANSITION.value)
        if transition_mask.sum() > 0:
            transition_env_ids = transition_mask.nonzero(as_tuple=False).squeeze(-1)
            progress = torch.clamp(self._transition_timer[transition_env_ids] / self._transition_duration, 0.0, 1.0)
            
            # 混合基础奖励
            mixed_reward = base_task_reward[transition_env_ids]
            
            # 添加平滑转移奖励
            smooth_reward = self._compute_smooth_transition_reward(transition_env_ids, progress)
            
            total_transition_reward = (
                mixed_reward * (self._transition_reward_weights["source_skill"] + 
                              self._transition_reward_weights["target_skill"]) / 2.0 +
                smooth_reward * self._transition_reward_weights["transition_smooth"]
            )
            
            reward[transition_env_ids] = total_transition_reward
        
        return reward
    
    def _compute_base_task_rewards(self):
        """计算基础任务奖励（复用原有的奖励函数）"""
        # 直接调用父类的奖励计算
        return self.rew_buf.clone()
    
    def _compute_smooth_transition_reward(self, env_ids, progress):
        """计算平滑转移奖励"""
        if len(env_ids) == 0:
            return torch.zeros(0, device=self.device)
        
        # 1. 姿态连续性奖励
        current_pose = self._humanoid_root_states[env_ids, 0:7]  # 位置+方向
        prev_pose = torch.cat([self._prev_root_pos[env_ids], self._prev_root_rot[env_ids]], dim=-1)
        
        pose_change = torch.norm(current_pose - prev_pose, dim=-1)
        pose_continuity_reward = torch.exp(-5.0 * pose_change)
        
        # 2. 速度平滑性奖励
        current_vel = (self._humanoid_root_states[env_ids, 0:3] - self._prev_root_pos[env_ids]) / self.dt
        vel_magnitude = torch.norm(current_vel, dim=-1)
        
        # 鼓励适中的速度变化
        target_vel = 1.5 + 1.0 * progress  # 转移过程中逐渐加速
        vel_smooth_reward = torch.exp(-torch.abs(vel_magnitude - target_vel))
        
        # 3. 关节角度平滑性
        if hasattr(self, '_prev_dof_pos'):
            dof_change = torch.norm(self._dof_pos[env_ids] - self._prev_dof_pos[env_ids], dim=-1)
            joint_smooth_reward = torch.exp(-2.0 * dof_change)
        else:
            joint_smooth_reward = torch.ones_like(pose_continuity_reward)
        
        # 综合平滑性奖励
        smooth_reward = 0.4 * pose_continuity_reward + 0.3 * vel_smooth_reward + 0.3 * joint_smooth_reward
        
        return smooth_reward
    
    def _compute_reward(self, actions):
        """重写奖励计算，使用状态转移奖励"""
        # 先调用父类计算基础奖励
        super()._compute_reward(actions)
        
        # 然后用状态转移奖励替换
        transition_reward = self._compute_transition_reward()
        self.rew_buf[:] = transition_reward
        
        # 添加功率惩罚
        if self._power_reward:
            power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim=-1)
            power_penalty = -self._power_coefficient * power
            self.rew_buf += power_penalty

    
    
    def _update_transition_state(self):
        """更新状态转移进度和阶段切换"""
        dt = self.dt
        self._transition_timer += dt
        
        # 检查阶段切换条件
        for phase in [self.TransitionPhase.SOURCE_SKILL, self.TransitionPhase.TRANSITION, self.TransitionPhase.TARGET_SKILL]:
            phase_mask = (self._transition_phase == phase.value)
            
            if phase_mask.sum() > 0:
                env_ids = phase_mask.nonzero(as_tuple=False).squeeze(-1)
                
                if phase == self.TransitionPhase.SOURCE_SKILL:
                    # 源技能阶段 -> 转移阶段
                    transition_mask = self._transition_timer[env_ids] >= self._source_skill_duration
                    if transition_mask.sum() > 0:
                        transition_env_ids = env_ids[transition_mask]
                        self._transition_phase[transition_env_ids] = self.TransitionPhase.TRANSITION.value
                        self._transition_timer[transition_env_ids] = 0.0
                        
                        # 记录源技能结束时的状态
                        self._source_end_state[transition_env_ids] = self.obs_buf[transition_env_ids].clone()
                        
                elif phase == self.TransitionPhase.TRANSITION:
                    # 转移阶段 -> 目标技能阶段
                    transition_mask = self._transition_timer[env_ids] >= self._transition_duration
                    if transition_mask.sum() > 0:
                        transition_env_ids = env_ids[transition_mask]
                        self._transition_phase[transition_env_ids] = self.TransitionPhase.TARGET_SKILL.value
                        self._transition_timer[transition_env_ids] = 0.0
                        
                        # 记录目标技能开始时的状态
                        self._target_start_state[transition_env_ids] = self.obs_buf[transition_env_ids].clone()
                        
                elif phase == self.TransitionPhase.TARGET_SKILL:
                    # 目标技能阶段完成 -> 重新开始循环
                    reset_mask = self._transition_timer[env_ids] >= self._target_skill_duration
                    if reset_mask.sum() > 0:
                        reset_env_ids = env_ids[reset_mask]
                        self._reset_envs(reset_env_ids)
        
        # 更新所有环境的任务masks
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._update_task_masks_for_transition(all_env_ids)
    
    def _compute_transition_reward(self):
        """计算状态转移奖励"""
        reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        # 基础技能奖励（根据当前执行的技能）
        base_task_reward = self._compute_base_task_rewards()
        
        # 源技能阶段
        source_mask = (self._transition_phase == self.TransitionPhase.SOURCE_SKILL.value)
        if source_mask.sum() > 0:
            reward[source_mask] = base_task_reward[source_mask] * self._transition_reward_weights["source_skill"]
        
        # 目标技能阶段  
        target_mask = (self._transition_phase == self.TransitionPhase.TARGET_SKILL.value)
        if target_mask.sum() > 0:
            reward[target_mask] = base_task_reward[target_mask] * self._transition_reward_weights["target_skill"]
        
        # 转移阶段
        transition_mask = (self._transition_phase == self.TransitionPhase.TRANSITION.value)
        if transition_mask.sum() > 0:
            transition_env_ids = transition_mask.nonzero(as_tuple=False).squeeze(-1)
            progress = torch.clamp(self._transition_timer[transition_env_ids] / self._transition_duration, 0.0, 1.0)
            
            # 混合基础奖励
            mixed_reward = base_task_reward[transition_env_ids]
            
            # 添加平滑转移奖励
            smooth_reward = self._compute_smooth_transition_reward(transition_env_ids, progress)
            
            total_transition_reward = (
                mixed_reward * (self._transition_reward_weights["source_skill"] + 
                              self._transition_reward_weights["target_skill"]) / 2.0 +
                smooth_reward * self._transition_reward_weights["transition_smooth"]
            )
            
            reward[transition_env_ids] = total_transition_reward
        
        return reward
    
    def _compute_smooth_transition_reward(self, env_ids, progress):
        """计算平滑转移奖励"""
        if len(env_ids) == 0:
            return torch.zeros(0, device=self.device)
        
        # 1. 姿态连续性奖励
        current_pose = self._humanoid_root_states[env_ids, 0:7]  # 位置+方向
        prev_pose = torch.cat([self._prev_root_pos[env_ids], self._prev_root_rot[env_ids]], dim=-1)
        
        pose_change = torch.norm(current_pose - prev_pose, dim=-1)
        pose_continuity_reward = torch.exp(-5.0 * pose_change)
        
        # 2. 速度平滑性奖励
        current_vel = (self._humanoid_root_states[env_ids, 0:3] - self._prev_root_pos[env_ids]) / self.dt
        vel_magnitude = torch.norm(current_vel, dim=-1)
        
        # 鼓励适中的速度变化
        target_vel = 1.5 + 1.0 * progress  # 转移过程中逐渐加速
        vel_smooth_reward = torch.exp(-torch.abs(vel_magnitude - target_vel))
        
        # 3. 关节角度平滑性
        dof_change = torch.norm(self._dof_pos[env_ids] - self._prev_dof_pos[env_ids], dim=-1) if hasattr(self, '_prev_dof_pos') else 0.0
        joint_smooth_reward = torch.exp(-2.0 * dof_change) if isinstance(dof_change, torch.Tensor) else torch.ones_like(pose_continuity_reward)
        
        # 综合平滑性奖励
        smooth_reward = 0.4 * pose_continuity_reward + 0.3 * vel_smooth_reward + 0.3 * joint_smooth_reward
        
        return smooth_reward
    
    def _compute_reward(self, actions):
        """重写奖励计算，使用状态转移奖励"""
        transition_reward = self._compute_transition_reward()
        self.rew_buf[:] = transition_reward
        
        # 添加功率惩罚
        if self._power_reward:
            power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim=-1)
            power_penalty = -self._power_coefficient * power
            self.rew_buf += power_penalty
    
    def post_physics_step(self):
        """物理步骤后处理"""
        # 记录前一帧状态
        if not hasattr(self, '_prev_dof_pos'):
            self._prev_dof_pos = self._dof_pos.clone()
        else:
            self._prev_dof_pos[:] = self._dof_pos.clone()
        
        # 更新转移状态
        self._update_transition_state()
        
        super().post_physics_step()
        
        # 添加转移相关的额外信息到extras
        self.extras["transition_phase"] = self._transition_phase.clone()
        self.extras["transition_timer"] = self._transition_timer.clone()
        self.extras["source_skill"] = self._source_skill_id.clone()
        self.extras["target_skill"] = self._target_skill_id.clone()
    
    def get_transition_info(self):
        """获取状态转移信息，用于调试和监控"""
        return {
            "all_transition_pairs": self._all_transition_pairs,
            "num_transitions": len(self._all_transition_pairs),
            "current_phases": self._transition_phase.cpu().numpy(),
            "transition_timers": self._transition_timer.cpu().numpy(),
            "source_skills": self._source_skill_id.cpu().numpy(),
            "target_skills": self._target_skill_id.cpu().numpy()
        }