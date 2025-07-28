#!/usr/bin/env python3
"""
TAI5模型最小化测试 - 检查模型是否在初始化时就有问题
只加载TAI5 XML，不进行任何复杂操作，观察初始状态
"""

import os
import sys
import numpy as np

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import torch
class TAI5MinimalTest:
    def __init__(self):
        # 初始化gym
        self.gym = gymapi.acquire_gym()
        
        # 基础配置
        self.device = "cuda:0"
        self.num_envs = 4  # 少量环境便于观察
        
        # 创建仿真
        self._create_sim()
        self._create_envs()
        
        # 获取状态张量
        self._setup_tensors()
        
        print("=== TAI5最小化测试初始化完成 ===")
        self._print_initial_states()
    
    def _create_sim(self):
        """创建仿真环境"""
        # 仿真参数
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0  # 60Hz
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        
        # 物理引擎设置
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.bounce_threshold_velocity = 0.5
        sim_params.physx.friction_offset_threshold = 0.01
        sim_params.physx.friction_correlation_distance = 0.025
        sim_params.physx.num_threads = 0
        sim_params.physx.use_gpu = True
        
        # 创建仿真
        compute_device_id = 0
        graphics_device_id = 0
        self.sim = self.gym.create_sim(compute_device_id, graphics_device_id, 
                                     gymapi.SIM_PHYSX, sim_params)
        
        if self.sim is None:
            print("*** 创建仿真失败 ***")
            sys.exit()
        
        # 创建地面
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)
        
        print("✓ 仿真环境创建成功")
    
    def _create_envs(self):
        """创建环境和TAI5模型"""
        # 加载TAI5资产
        asset_root = "/home/dy/dy/code/tokenhsi/tokenhsi/data/assets/"  # 根据你的路径调整
        asset_file = "mjcf/tai5/tai5.xml"  # TAI5 XML文件路径
        
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.fix_base_link = False  # 允许自由移动
        
        print(f"正在加载TAI5资产: {os.path.join(asset_root, asset_file)}")
        
        try:
            self.tai5_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
            print("✓ TAI5资产加载成功")
        except Exception as e:
            print(f"*** TAI5资产加载失败: {e} ***")
            sys.exit()
        
        # 获取资产信息
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.tai5_asset)
        self.num_shapes = self.gym.get_asset_rigid_shape_count(self.tai5_asset)
        self.num_dof = self.gym.get_asset_dof_count(self.tai5_asset)
        
        print(f"TAI5资产信息: bodies={self.num_bodies}, shapes={self.num_shapes}, dof={self.num_dof}")
        
        # 打印所有身体部位名称
        print("身体部位列表:")
        for i in range(self.num_bodies):
            body_name = self.gym.get_asset_rigid_body_name(self.tai5_asset, i)
            print(f"  {i}: {body_name}")
        
        # 打印所有DOF名称
        print("DOF列表:")
        for i in range(self.num_dof):
            dof_name = self.gym.get_asset_dof_name(self.tai5_asset, i)
            print(f"  {i}: {dof_name}")
        
        # 创建环境
        spacing = 2.0
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        self.envs = []
        self.tai5_handles = []
        
        for i in range(self.num_envs):
            # 创建环境
            env = self.gym.create_env(self.sim, lower, upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env)
            
            # 设置TAI5初始位置
            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(0.0, 0.0, 0.96)  # 根据XML中的base_link位置
            start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
            # 创建TAI5角色
            tai5_handle = self.gym.create_actor(env, self.tai5_asset, start_pose, 
                                              f"tai5_{i}", i, 1, 0)
            self.tai5_handles.append(tai5_handle)
            
            # 设置颜色以便区分
            colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
            color = colors[i % len(colors)]
            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env, tai5_handle, j, gymapi.MESH_VISUAL,
                                            gymapi.Vec3(color[0], color[1], color[2]))
        
        print(f"✓ 创建了{self.num_envs}个TAI5环境")
    
    def _setup_tensors(self):
        """设置状态张量"""
        self.gym.prepare_sim(self.sim)
        
        # 获取状态张量
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        
        # 刷新张量
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
        # 包装张量
        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        self._dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        self._rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self._contact_forces = gymtorch.wrap_tensor(contact_force_tensor)
        
        # 重塑张量
        self._humanoid_root_states = self._root_states.view(self.num_envs, -1)[:, :13]
        
        dofs_per_env = self._dof_states.shape[0] // self.num_envs
        self._dof_pos = self._dof_states.view(self.num_envs, dofs_per_env, 2)[:, :self.num_dof, 0]
        self._dof_vel = self._dof_states.view(self.num_envs, dofs_per_env, 2)[:, :self.num_dof, 1]
        
        bodies_per_env = self._rigid_body_states.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_states.view(self.num_envs, bodies_per_env, 13)
        self._rigid_body_pos = rigid_body_state_reshaped[:, :self.num_bodies, 0:3]
        self._rigid_body_rot = rigid_body_state_reshaped[:, :self.num_bodies, 3:7]
        self._rigid_body_vel = rigid_body_state_reshaped[:, :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = rigid_body_state_reshaped[:, :self.num_bodies, 10:13]
        
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[:, :self.num_bodies, :]
        
        print("✓ 状态张量设置完成")
    
    def _print_initial_states(self):
        """打印初始状态信息"""
        print("\n=== 初始状态分析 ===")
        
        # 根部状态
        for i in range(self.num_envs):
            root_pos = self._humanoid_root_states[i, 0:3]
            root_rot = self._humanoid_root_states[i, 3:7]
            root_vel = self._humanoid_root_states[i, 7:10]
            root_ang_vel = self._humanoid_root_states[i, 10:13]
            
            print(f"\n环境 {i}:")
            print(f"  根位置: [{root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f}]")
            print(f"  根旋转: [{root_rot[0]:.3f}, {root_rot[1]:.3f}, {root_rot[2]:.3f}, {root_rot[3]:.3f}]")
            print(f"  根速度: [{root_vel[0]:.3f}, {root_vel[1]:.3f}, {root_vel[2]:.3f}]")
            print(f"  根角速度: [{root_ang_vel[0]:.3f}, {root_ang_vel[1]:.3f}, {root_ang_vel[2]:.3f}]")
        
        # DOF状态
        print(f"\nDOF状态:")
        print(f"  DOF位置范围: [{torch.min(self._dof_pos):.3f}, {torch.max(self._dof_pos):.3f}]")
        print(f"  DOF速度范围: [{torch.min(self._dof_vel):.3f}, {torch.max(self._dof_vel):.3f}]")
        
        # 检查是否有NaN
        if torch.any(torch.isnan(self._humanoid_root_states)):
            print("⚠️  检测到根状态中有NaN值!")
        
        if torch.any(torch.isnan(self._dof_pos)) or torch.any(torch.isnan(self._dof_vel)):
            print("⚠️  检测到DOF状态中有NaN值!")
        
        # 检查高度
        heights = self._rigid_body_pos[:, 0, 2]  # 基座高度
        print(f"\n基座高度: {heights}")
        flying_envs = torch.where(heights > 1.5)[0]
        if len(flying_envs) > 0:
            print(f"⚠️  初始化时检测到飞行环境: {flying_envs}")
        else:
            print("✓ 所有环境初始高度正常")
        
        # 检查接触力
        max_contact_forces = torch.max(torch.abs(self._contact_forces), dim=-1)[0]
        max_per_env = torch.max(max_contact_forces, dim=-1)[0]
        print(f"\n初始最大接触力: {max_per_env}")
        
        if torch.any(max_per_env > 10.0):
            abnormal_envs = torch.where(max_per_env > 10.0)[0]
            print(f"⚠️  初始化时检测到异常接触力环境: {abnormal_envs}")
        else:
            print("✓ 初始接触力正常")
    
    def run_test(self, num_steps=100):
        """运行测试"""
        print(f"\n=== 开始运行{num_steps}步测试 ===")
        
        # 创建查看器（可选）
        viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if viewer is None:
            print("无法创建查看器，继续无头模式测试")
        else:
            # 设置相机
            cam_pos = gymapi.Vec3(3, 3, 2)
            cam_target = gymapi.Vec3(0, 0, 1)
            self.gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
        
        step_count = 0
        problem_detected = False
        
        while step_count < num_steps:
            # 仿真步进
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            
            # 刷新状态
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            
            # 检查问题
            heights = self._rigid_body_pos[:, 0, 2]
            flying_envs = torch.where(heights > 1.8)[0]
            
            max_contact_forces = torch.max(torch.abs(self._contact_forces), dim=-1)[0]
            max_per_env = torch.max(max_contact_forces, dim=-1)[0]
            abnormal_contact = torch.where(max_per_env > 800.0)[0]
            
            # 每10步报告一次
            if step_count % 10 == 0:
                print(f"步骤 {step_count}: 高度范围[{torch.min(heights):.2f}, {torch.max(heights):.2f}], "
                      f"最大接触力[{torch.max(max_per_env):.1f}N]")
            
            # 检测问题
            if len(flying_envs) > 0:
                print(f"⚠️  步骤{step_count}: 检测到飞行环境 {flying_envs}")
                print(f"   飞行高度: {heights[flying_envs]}")
                problem_detected = True
            
            if len(abnormal_contact) > 0:
                print(f"⚠️  步骤{step_count}: 异常接触力环境 {abnormal_contact}")
                print(f"   接触力: {max_per_env[abnormal_contact]}")
                
                # 检查具体身体部位
                for env_id in abnormal_contact[:2]:
                    body_forces = torch.abs(self._contact_forces[env_id])
                    problem_bodies = torch.where(torch.max(body_forces, dim=-1)[0] > 100.0)[0]
                    print(f"   环境{env_id}问题身体部位: {problem_bodies}")
                
                problem_detected = True
            
            # 渲染
            if viewer is not None:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(viewer, self.sim, True)
                self.gym.poll_viewer_events(viewer)
                
                # 检查窗口关闭
                if self.gym.query_viewer_has_closed(viewer):
                    break
            
            step_count += 1
        
        print(f"\n=== 测试完成 ===")
        if problem_detected:
            print("❌ 检测到问题，TAI5模型配置可能存在问题")
        else:
            print("✅ 未检测到明显问题，TAI5模型基本稳定")
        
        if viewer is not None:
            self.gym.destroy_viewer(viewer)
    
    def cleanup(self):
        """清理资源"""
        self.gym.destroy_sim(self.sim)

def main():
    """主函数"""
    print("TAI5模型最小化测试启动")
    
    try:
        # 创建测试实例
        test = TAI5MinimalTest()
        
        # 运行测试
        test.run_test(num_steps=200)
        
        # 清理
        test.cleanup()
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()