import sys
sys.path.append("./")

import numpy as np
import torch

from lpanlib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from lpanlib.poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from lpanlib.poselib.core.rotation3d import quat_mul, quat_from_angle_axis, quat_mul_norm, quat_rotate, quat_identity

def process_amass_seq(fname, output_path):

    # load raw params from AMASS dataset
    raw_params = dict(np.load(fname, allow_pickle=True))

    poses = raw_params["poses"]
    trans = raw_params["trans"]

    # downsample from 120hz to 30hz
    source_fps = raw_params["mocap_frame_rate"]
    target_fps = 30
    skip = int(source_fps // target_fps)
    poses = poses[::skip]
    trans = trans[::skip]

    # extract 24 SMPL joints from 55 SMPL-X joints
    joints_to_use = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 40]
    )
    joints_to_use = np.arange(0, 156).reshape((-1, 3))[joints_to_use].reshape(-1)
    poses = poses[:, joints_to_use]

    required_params = {}
    required_params["poses"] = poses
    required_params["trans"] = trans
    required_params["fps"] = target_fps
    
    # save
    np.save(output_path, required_params)
    
    return

def project_joints(motion):
    """ This is the original function used by ASE, designed for amp_humanoid.xml """

    right_upper_arm_id = motion.skeleton_tree._node_indices["right_upper_arm"]
    right_lower_arm_id = motion.skeleton_tree._node_indices["right_lower_arm"]
    right_hand_id = motion.skeleton_tree._node_indices["right_hand"]
    left_upper_arm_id = motion.skeleton_tree._node_indices["left_upper_arm"]
    left_lower_arm_id = motion.skeleton_tree._node_indices["left_lower_arm"]
    left_hand_id = motion.skeleton_tree._node_indices["left_hand"]
    
    right_thigh_id = motion.skeleton_tree._node_indices["right_thigh"]
    right_shin_id = motion.skeleton_tree._node_indices["right_shin"]
    right_foot_id = motion.skeleton_tree._node_indices["right_foot"]
    left_thigh_id = motion.skeleton_tree._node_indices["left_thigh"]
    left_shin_id = motion.skeleton_tree._node_indices["left_shin"]
    left_foot_id = motion.skeleton_tree._node_indices["left_foot"]
    
    device = motion.global_translation.device

    # right arm
    right_upper_arm_pos = motion.global_translation[..., right_upper_arm_id, :]
    right_lower_arm_pos = motion.global_translation[..., right_lower_arm_id, :]
    right_hand_pos = motion.global_translation[..., right_hand_id, :]
    right_shoulder_rot = motion.local_rotation[..., right_upper_arm_id, :]
    right_elbow_rot = motion.local_rotation[..., right_lower_arm_id, :]
    
    right_arm_delta0 = right_upper_arm_pos - right_lower_arm_pos
    right_arm_delta1 = right_hand_pos - right_lower_arm_pos
    right_arm_delta0 = right_arm_delta0 / torch.norm(right_arm_delta0, dim=-1, keepdim=True)
    right_arm_delta1 = right_arm_delta1 / torch.norm(right_arm_delta1, dim=-1, keepdim=True)
    right_elbow_dot = torch.sum(-right_arm_delta0 * right_arm_delta1, dim=-1)
    right_elbow_dot = torch.clamp(right_elbow_dot, -1.0, 1.0)
    right_elbow_theta = torch.acos(right_elbow_dot)
    right_elbow_q = quat_from_angle_axis(-torch.abs(right_elbow_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                            device=device, dtype=torch.float32))
    
    right_elbow_local_dir = motion.skeleton_tree.local_translation[right_hand_id]
    right_elbow_local_dir = right_elbow_local_dir / torch.norm(right_elbow_local_dir)
    right_elbow_local_dir_tile = torch.tile(right_elbow_local_dir.unsqueeze(0), [right_elbow_rot.shape[0], 1])
    right_elbow_local_dir0 = quat_rotate(right_elbow_rot, right_elbow_local_dir_tile)
    right_elbow_local_dir1 = quat_rotate(right_elbow_q, right_elbow_local_dir_tile)
    right_arm_dot = torch.sum(right_elbow_local_dir0 * right_elbow_local_dir1, dim=-1)
    right_arm_dot = torch.clamp(right_arm_dot, -1.0, 1.0)
    right_arm_theta = torch.acos(right_arm_dot)
    right_arm_theta = torch.where(right_elbow_local_dir0[..., 1] <= 0, right_arm_theta, -right_arm_theta)
    right_arm_q = quat_from_angle_axis(right_arm_theta, right_elbow_local_dir.unsqueeze(0))
    right_shoulder_rot = quat_mul(right_shoulder_rot, right_arm_q)
    
    # left arm
    left_upper_arm_pos = motion.global_translation[..., left_upper_arm_id, :]
    left_lower_arm_pos = motion.global_translation[..., left_lower_arm_id, :]
    left_hand_pos = motion.global_translation[..., left_hand_id, :]
    left_shoulder_rot = motion.local_rotation[..., left_upper_arm_id, :]
    left_elbow_rot = motion.local_rotation[..., left_lower_arm_id, :]
    
    left_arm_delta0 = left_upper_arm_pos - left_lower_arm_pos
    left_arm_delta1 = left_hand_pos - left_lower_arm_pos
    left_arm_delta0 = left_arm_delta0 / torch.norm(left_arm_delta0, dim=-1, keepdim=True)
    left_arm_delta1 = left_arm_delta1 / torch.norm(left_arm_delta1, dim=-1, keepdim=True)
    left_elbow_dot = torch.sum(-left_arm_delta0 * left_arm_delta1, dim=-1)
    left_elbow_dot = torch.clamp(left_elbow_dot, -1.0, 1.0)
    left_elbow_theta = torch.acos(left_elbow_dot)
    left_elbow_q = quat_from_angle_axis(-torch.abs(left_elbow_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))

    left_elbow_local_dir = motion.skeleton_tree.local_translation[left_hand_id]
    left_elbow_local_dir = left_elbow_local_dir / torch.norm(left_elbow_local_dir)
    left_elbow_local_dir_tile = torch.tile(left_elbow_local_dir.unsqueeze(0), [left_elbow_rot.shape[0], 1])
    left_elbow_local_dir0 = quat_rotate(left_elbow_rot, left_elbow_local_dir_tile)
    left_elbow_local_dir1 = quat_rotate(left_elbow_q, left_elbow_local_dir_tile)
    left_arm_dot = torch.sum(left_elbow_local_dir0 * left_elbow_local_dir1, dim=-1)
    left_arm_dot = torch.clamp(left_arm_dot, -1.0, 1.0)
    left_arm_theta = torch.acos(left_arm_dot)
    left_arm_theta = torch.where(left_elbow_local_dir0[..., 1] <= 0, left_arm_theta, -left_arm_theta)
    left_arm_q = quat_from_angle_axis(left_arm_theta, left_elbow_local_dir.unsqueeze(0))
    left_shoulder_rot = quat_mul(left_shoulder_rot, left_arm_q)
    
    # right leg
    right_thigh_pos = motion.global_translation[..., right_thigh_id, :]
    right_shin_pos = motion.global_translation[..., right_shin_id, :]
    right_foot_pos = motion.global_translation[..., right_foot_id, :]
    right_hip_rot = motion.local_rotation[..., right_thigh_id, :]
    right_knee_rot = motion.local_rotation[..., right_shin_id, :]
    
    right_leg_delta0 = right_thigh_pos - right_shin_pos
    right_leg_delta1 = right_foot_pos - right_shin_pos
    right_leg_delta0 = right_leg_delta0 / torch.norm(right_leg_delta0, dim=-1, keepdim=True)
    right_leg_delta1 = right_leg_delta1 / torch.norm(right_leg_delta1, dim=-1, keepdim=True)
    right_knee_dot = torch.sum(-right_leg_delta0 * right_leg_delta1, dim=-1)
    right_knee_dot = torch.clamp(right_knee_dot, -1.0, 1.0)
    right_knee_theta = torch.acos(right_knee_dot)
    right_knee_q = quat_from_angle_axis(torch.abs(right_knee_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))
    
    right_knee_local_dir = motion.skeleton_tree.local_translation[right_foot_id]
    right_knee_local_dir = right_knee_local_dir / torch.norm(right_knee_local_dir)
    right_knee_local_dir_tile = torch.tile(right_knee_local_dir.unsqueeze(0), [right_knee_rot.shape[0], 1])
    right_knee_local_dir0 = quat_rotate(right_knee_rot, right_knee_local_dir_tile)
    right_knee_local_dir1 = quat_rotate(right_knee_q, right_knee_local_dir_tile)
    right_leg_dot = torch.sum(right_knee_local_dir0 * right_knee_local_dir1, dim=-1)
    right_leg_dot = torch.clamp(right_leg_dot, -1.0, 1.0)
    right_leg_theta = torch.acos(right_leg_dot)
    right_leg_theta = torch.where(right_knee_local_dir0[..., 1] >= 0, right_leg_theta, -right_leg_theta)
    right_leg_q = quat_from_angle_axis(right_leg_theta, right_knee_local_dir.unsqueeze(0))
    right_hip_rot = quat_mul(right_hip_rot, right_leg_q)
    
    # left leg
    left_thigh_pos = motion.global_translation[..., left_thigh_id, :]
    left_shin_pos = motion.global_translation[..., left_shin_id, :]
    left_foot_pos = motion.global_translation[..., left_foot_id, :]
    left_hip_rot = motion.local_rotation[..., left_thigh_id, :]
    left_knee_rot = motion.local_rotation[..., left_shin_id, :]
    
    left_leg_delta0 = left_thigh_pos - left_shin_pos
    left_leg_delta1 = left_foot_pos - left_shin_pos
    left_leg_delta0 = left_leg_delta0 / torch.norm(left_leg_delta0, dim=-1, keepdim=True)
    left_leg_delta1 = left_leg_delta1 / torch.norm(left_leg_delta1, dim=-1, keepdim=True)
    left_knee_dot = torch.sum(-left_leg_delta0 * left_leg_delta1, dim=-1)
    left_knee_dot = torch.clamp(left_knee_dot, -1.0, 1.0)
    left_knee_theta = torch.acos(left_knee_dot)
    left_knee_q = quat_from_angle_axis(torch.abs(left_knee_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))
    
    left_knee_local_dir = motion.skeleton_tree.local_translation[left_foot_id]
    left_knee_local_dir = left_knee_local_dir / torch.norm(left_knee_local_dir)
    left_knee_local_dir_tile = torch.tile(left_knee_local_dir.unsqueeze(0), [left_knee_rot.shape[0], 1])
    left_knee_local_dir0 = quat_rotate(left_knee_rot, left_knee_local_dir_tile)
    left_knee_local_dir1 = quat_rotate(left_knee_q, left_knee_local_dir_tile)
    left_leg_dot = torch.sum(left_knee_local_dir0 * left_knee_local_dir1, dim=-1)
    left_leg_dot = torch.clamp(left_leg_dot, -1.0, 1.0)
    left_leg_theta = torch.acos(left_leg_dot)
    left_leg_theta = torch.where(left_knee_local_dir0[..., 1] >= 0, left_leg_theta, -left_leg_theta)
    left_leg_q = quat_from_angle_axis(left_leg_theta, left_knee_local_dir.unsqueeze(0))
    left_hip_rot = quat_mul(left_hip_rot, left_leg_q)
    

    new_local_rotation = motion.local_rotation.clone()
    new_local_rotation[..., right_upper_arm_id, :] = right_shoulder_rot
    new_local_rotation[..., right_lower_arm_id, :] = right_elbow_q
    new_local_rotation[..., left_upper_arm_id, :] = left_shoulder_rot
    new_local_rotation[..., left_lower_arm_id, :] = left_elbow_q
    
    new_local_rotation[..., right_thigh_id, :] = right_hip_rot
    new_local_rotation[..., right_shin_id, :] = right_knee_q
    new_local_rotation[..., left_thigh_id, :] = left_hip_rot
    new_local_rotation[..., left_shin_id, :] = left_knee_q
    
    new_local_rotation[..., left_hand_id, :] = quat_identity([1])
    new_local_rotation[..., right_hand_id, :] = quat_identity([1])

    new_sk_state = SkeletonState.from_rotation_and_root_translation(motion.skeleton_tree, new_local_rotation, motion.root_translation, is_local=True)
    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=motion.fps)
    
    return new_motion

def project_joints_simple(motion):
    """ This is the our revised function used by TokenHSI, designed for phys_humanoid_v3.xml 

    The difference is that we only project the arms, not the legs.
    The reason is that the leg joints have been modified to 3 DoF spherical joints.

    """

    right_upper_arm_id = motion.skeleton_tree._node_indices["right_upper_arm"]
    right_lower_arm_id = motion.skeleton_tree._node_indices["right_lower_arm"]
    right_hand_id = motion.skeleton_tree._node_indices["right_hand"]
    left_upper_arm_id = motion.skeleton_tree._node_indices["left_upper_arm"]
    left_lower_arm_id = motion.skeleton_tree._node_indices["left_lower_arm"]
    left_hand_id = motion.skeleton_tree._node_indices["left_hand"]
    
    right_thigh_id = motion.skeleton_tree._node_indices["right_thigh"]
    right_shin_id = motion.skeleton_tree._node_indices["right_shin"]
    right_foot_id = motion.skeleton_tree._node_indices["right_foot"]
    left_thigh_id = motion.skeleton_tree._node_indices["left_thigh"]
    left_shin_id = motion.skeleton_tree._node_indices["left_shin"]
    left_foot_id = motion.skeleton_tree._node_indices["left_foot"]
    
    device = motion.global_translation.device

    # right arm
    right_upper_arm_pos = motion.global_translation[..., right_upper_arm_id, :]
    right_lower_arm_pos = motion.global_translation[..., right_lower_arm_id, :]
    right_hand_pos = motion.global_translation[..., right_hand_id, :]
    right_shoulder_rot = motion.local_rotation[..., right_upper_arm_id, :]
    right_elbow_rot = motion.local_rotation[..., right_lower_arm_id, :]
    
    right_arm_delta0 = right_upper_arm_pos - right_lower_arm_pos
    right_arm_delta1 = right_hand_pos - right_lower_arm_pos
    right_arm_delta0 = right_arm_delta0 / torch.norm(right_arm_delta0, dim=-1, keepdim=True)
    right_arm_delta1 = right_arm_delta1 / torch.norm(right_arm_delta1, dim=-1, keepdim=True)
    right_elbow_dot = torch.sum(-right_arm_delta0 * right_arm_delta1, dim=-1)
    right_elbow_dot = torch.clamp(right_elbow_dot, -1.0, 1.0)
    right_elbow_theta = torch.acos(right_elbow_dot)
    right_elbow_q = quat_from_angle_axis(-torch.abs(right_elbow_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                            device=device, dtype=torch.float32))
    
    right_elbow_local_dir = motion.skeleton_tree.local_translation[right_hand_id]
    right_elbow_local_dir = right_elbow_local_dir / torch.norm(right_elbow_local_dir)
    right_elbow_local_dir_tile = torch.tile(right_elbow_local_dir.unsqueeze(0), [right_elbow_rot.shape[0], 1])
    right_elbow_local_dir0 = quat_rotate(right_elbow_rot, right_elbow_local_dir_tile)
    right_elbow_local_dir1 = quat_rotate(right_elbow_q, right_elbow_local_dir_tile)
    right_arm_dot = torch.sum(right_elbow_local_dir0 * right_elbow_local_dir1, dim=-1)
    right_arm_dot = torch.clamp(right_arm_dot, -1.0, 1.0)
    right_arm_theta = torch.acos(right_arm_dot)
    right_arm_theta = torch.where(right_elbow_local_dir0[..., 1] <= 0, right_arm_theta, -right_arm_theta)
    right_arm_q = quat_from_angle_axis(right_arm_theta, right_elbow_local_dir.unsqueeze(0))
    right_shoulder_rot = quat_mul(right_shoulder_rot, right_arm_q)
    
    # left arm
    left_upper_arm_pos = motion.global_translation[..., left_upper_arm_id, :]
    left_lower_arm_pos = motion.global_translation[..., left_lower_arm_id, :]
    left_hand_pos = motion.global_translation[..., left_hand_id, :]
    left_shoulder_rot = motion.local_rotation[..., left_upper_arm_id, :]
    left_elbow_rot = motion.local_rotation[..., left_lower_arm_id, :]
    
    left_arm_delta0 = left_upper_arm_pos - left_lower_arm_pos
    left_arm_delta1 = left_hand_pos - left_lower_arm_pos
    left_arm_delta0 = left_arm_delta0 / torch.norm(left_arm_delta0, dim=-1, keepdim=True)
    left_arm_delta1 = left_arm_delta1 / torch.norm(left_arm_delta1, dim=-1, keepdim=True)
    left_elbow_dot = torch.sum(-left_arm_delta0 * left_arm_delta1, dim=-1)
    left_elbow_dot = torch.clamp(left_elbow_dot, -1.0, 1.0)
    left_elbow_theta = torch.acos(left_elbow_dot)
    left_elbow_q = quat_from_angle_axis(-torch.abs(left_elbow_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))

    left_elbow_local_dir = motion.skeleton_tree.local_translation[left_hand_id]
    left_elbow_local_dir = left_elbow_local_dir / torch.norm(left_elbow_local_dir)
    left_elbow_local_dir_tile = torch.tile(left_elbow_local_dir.unsqueeze(0), [left_elbow_rot.shape[0], 1])
    left_elbow_local_dir0 = quat_rotate(left_elbow_rot, left_elbow_local_dir_tile)
    left_elbow_local_dir1 = quat_rotate(left_elbow_q, left_elbow_local_dir_tile)
    left_arm_dot = torch.sum(left_elbow_local_dir0 * left_elbow_local_dir1, dim=-1)
    left_arm_dot = torch.clamp(left_arm_dot, -1.0, 1.0)
    left_arm_theta = torch.acos(left_arm_dot)
    left_arm_theta = torch.where(left_elbow_local_dir0[..., 1] <= 0, left_arm_theta, -left_arm_theta)
    left_arm_q = quat_from_angle_axis(left_arm_theta, left_elbow_local_dir.unsqueeze(0))
    left_shoulder_rot = quat_mul(left_shoulder_rot, left_arm_q)

    new_local_rotation = motion.local_rotation.clone()
    new_local_rotation[..., right_upper_arm_id, :] = right_shoulder_rot
    new_local_rotation[..., right_lower_arm_id, :] = right_elbow_q
    new_local_rotation[..., left_upper_arm_id, :] = left_shoulder_rot
    new_local_rotation[..., left_lower_arm_id, :] = left_elbow_q
    
    new_local_rotation[..., left_hand_id, :] = quat_identity([1])
    new_local_rotation[..., right_hand_id, :] = quat_identity([1])

    new_sk_state = SkeletonState.from_rotation_and_root_translation(motion.skeleton_tree, new_local_rotation, motion.root_translation, is_local=True)
    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=motion.fps)

    return new_motion

def project_joints_g1(motion):
    """ 
    专门为G1机器人设计的投影函数
    G1的关节命名和结构与humanoid模型不同，需要适配
    """
    
    # G1的关节索引映射（需要根据实际的skeleton结构确定）
    try:
        # 手臂关节
        left_shoulder_pitch_id = motion.skeleton_tree._node_indices["left_shoulder_pitch_link"]
        left_shoulder_roll_id = motion.skeleton_tree._node_indices["left_shoulder_roll_link"] 
        left_shoulder_yaw_id = motion.skeleton_tree._node_indices["left_shoulder_yaw_link"]
        left_elbow_id = motion.skeleton_tree._node_indices["left_elbow_link"]
        left_wrist_roll_id = motion.skeleton_tree._node_indices["left_wrist_roll_link"]
        left_wrist_pitch_id = motion.skeleton_tree._node_indices["left_wrist_pitch_link"]
        left_wrist_yaw_id = motion.skeleton_tree._node_indices["left_wrist_yaw_link"]
        
        right_shoulder_pitch_id = motion.skeleton_tree._node_indices["right_shoulder_pitch_link"]
        right_shoulder_roll_id = motion.skeleton_tree._node_indices["right_shoulder_roll_link"]
        right_shoulder_yaw_id = motion.skeleton_tree._node_indices["right_shoulder_yaw_link"]
        right_elbow_id = motion.skeleton_tree._node_indices["right_elbow_link"]
        right_wrist_roll_id = motion.skeleton_tree._node_indices["right_wrist_roll_link"]
        right_wrist_pitch_id = motion.skeleton_tree._node_indices["right_wrist_pitch_link"]
        right_wrist_yaw_id = motion.skeleton_tree._node_indices["right_wrist_yaw_link"]
        
        # 腿部关节
        left_hip_pitch_id = motion.skeleton_tree._node_indices["left_hip_pitch_link"]
        left_hip_roll_id = motion.skeleton_tree._node_indices["left_hip_roll_link"]
        left_hip_yaw_id = motion.skeleton_tree._node_indices["left_hip_yaw_link"]
        left_knee_id = motion.skeleton_tree._node_indices["left_knee_link"]
        left_ankle_pitch_id = motion.skeleton_tree._node_indices["left_ankle_pitch_link"]
        left_ankle_roll_id = motion.skeleton_tree._node_indices["left_ankle_roll_link"]
        
        right_hip_pitch_id = motion.skeleton_tree._node_indices["right_hip_pitch_link"]
        right_hip_roll_id = motion.skeleton_tree._node_indices["right_hip_roll_link"]
        right_hip_yaw_id = motion.skeleton_tree._node_indices["right_hip_yaw_link"]
        right_knee_id = motion.skeleton_tree._node_indices["right_knee_link"]
        right_ankle_pitch_id = motion.skeleton_tree._node_indices["right_ankle_pitch_link"]
        right_ankle_roll_id = motion.skeleton_tree._node_indices["right_ankle_roll_link"]
        
    except KeyError as e:
        print(f"Warning: Could not find joint {e} in G1 skeleton, skipping projection")
        return motion
    
    device = motion.global_translation.device
    new_local_rotation = motion.local_rotation.clone()
    
    # 1. 处理手臂 - 简化版本，主要处理肘关节
    try:
        # 左臂投影 - 从肩膀到手腕计算肘关节角度
        left_shoulder_pos = motion.global_translation[..., left_shoulder_yaw_id, :]
        left_elbow_pos = motion.global_translation[..., left_elbow_id, :]
        left_wrist_pos = motion.global_translation[..., left_wrist_pitch_id, :]  # 使用wrist_pitch作为末端
        
        left_arm_delta0 = left_shoulder_pos - left_elbow_pos
        left_arm_delta1 = left_wrist_pos - left_elbow_pos
        left_arm_delta0 = left_arm_delta0 / torch.norm(left_arm_delta0, dim=-1, keepdim=True)
        left_arm_delta1 = left_arm_delta1 / torch.norm(left_arm_delta1, dim=-1, keepdim=True)
        left_elbow_dot = torch.sum(-left_arm_delta0 * left_arm_delta1, dim=-1)
        left_elbow_dot = torch.clamp(left_elbow_dot, -1.0, 1.0)
        left_elbow_theta = torch.acos(left_elbow_dot)
        left_elbow_q = quat_from_angle_axis(-torch.abs(left_elbow_theta), 
                                          torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                          device=device, dtype=torch.float32))
        new_local_rotation[..., left_elbow_id, :] = left_elbow_q
        
        # 右臂投影
        right_shoulder_pos = motion.global_translation[..., right_shoulder_yaw_id, :]
        right_elbow_pos = motion.global_translation[..., right_elbow_id, :]
        right_wrist_pos = motion.global_translation[..., right_wrist_pitch_id, :]
        
        right_arm_delta0 = right_shoulder_pos - right_elbow_pos
        right_arm_delta1 = right_wrist_pos - right_elbow_pos
        right_arm_delta0 = right_arm_delta0 / torch.norm(right_arm_delta0, dim=-1, keepdim=True)
        right_arm_delta1 = right_arm_delta1 / torch.norm(right_arm_delta1, dim=-1, keepdim=True)
        right_elbow_dot = torch.sum(-right_arm_delta0 * right_arm_delta1, dim=-1)
        right_elbow_dot = torch.clamp(right_elbow_dot, -1.0, 1.0)
        right_elbow_theta = torch.acos(right_elbow_dot)
        right_elbow_q = quat_from_angle_axis(-torch.abs(right_elbow_theta), 
                                           torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                           device=device, dtype=torch.float32))
        new_local_rotation[..., right_elbow_id, :] = right_elbow_q
        
    except Exception as e:
        print(f"Warning: Arm projection failed for G1: {e}")
    
    # 2. 处理腿部 - G1有复杂的3DOF髋关节，简化处理膝关节
    try:
        # 左腿膝关节投影
        left_hip_pos = motion.global_translation[..., left_hip_yaw_id, :]  # 使用hip_yaw作为髋部位置
        left_knee_pos = motion.global_translation[..., left_knee_id, :]
        left_ankle_pos = motion.global_translation[..., left_ankle_pitch_id, :]
        
        left_leg_delta0 = left_hip_pos - left_knee_pos
        left_leg_delta1 = left_ankle_pos - left_knee_pos
        left_leg_delta0 = left_leg_delta0 / torch.norm(left_leg_delta0, dim=-1, keepdim=True)
        left_leg_delta1 = left_leg_delta1 / torch.norm(left_leg_delta1, dim=-1, keepdim=True)
        left_knee_dot = torch.sum(-left_leg_delta0 * left_leg_delta1, dim=-1)
        left_knee_dot = torch.clamp(left_knee_dot, -1.0, 1.0)
        left_knee_theta = torch.acos(left_knee_dot)
        left_knee_q = quat_from_angle_axis(torch.abs(left_knee_theta), 
                                         torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                         device=device, dtype=torch.float32))
        new_local_rotation[..., left_knee_id, :] = left_knee_q
        
        # 右腿膝关节投影
        right_hip_pos = motion.global_translation[..., right_hip_yaw_id, :]
        right_knee_pos = motion.global_translation[..., right_knee_id, :]
        right_ankle_pos = motion.global_translation[..., right_ankle_pitch_id, :]
        
        right_leg_delta0 = right_hip_pos - right_knee_pos
        right_leg_delta1 = right_ankle_pos - right_knee_pos
        right_leg_delta0 = right_leg_delta0 / torch.norm(right_leg_delta0, dim=-1, keepdim=True)
        right_leg_delta1 = right_leg_delta1 / torch.norm(right_leg_delta1, dim=-1, keepdim=True)
        right_knee_dot = torch.sum(-right_leg_delta0 * right_leg_delta1, dim=-1)
        right_knee_dot = torch.clamp(right_knee_dot, -1.0, 1.0)
        right_knee_theta = torch.acos(right_knee_dot)
        right_knee_q = quat_from_angle_axis(torch.abs(right_knee_theta), 
                                          torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                          device=device, dtype=torch.float32))
        new_local_rotation[..., right_knee_id, :] = right_knee_q
        
    except Exception as e:
        print(f"Warning: Leg projection failed for G1: {e}")
    
    # 3. 重置手腕和踝关节为单位四元数（可选）
    try:
        new_local_rotation[..., left_wrist_yaw_id, :] = quat_identity([1])
        new_local_rotation[..., right_wrist_yaw_id, :] = quat_identity([1])
        new_local_rotation[..., left_ankle_roll_id, :] = quat_identity([1])
        new_local_rotation[..., right_ankle_roll_id, :] = quat_identity([1])
    except:
        pass
    
    # 创建新的运动状态
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree, new_local_rotation, motion.root_translation, is_local=True)
    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=motion.fps)
    
    return new_motion
