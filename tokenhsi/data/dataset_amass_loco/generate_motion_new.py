import sys
sys.path.append("./")

import os
import os.path as osp
import torch
import numpy as np
import torchgeometry as tgm
from tqdm import tqdm

from lpanlib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from lpanlib.poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from lpanlib.poselib.core.rotation3d import quat_mul, quat_from_angle_axis, quat_mul_norm, quat_rotate, quat_identity

from body_models.model_loader import get_body_model

from lpanlib.isaacgym_utils.vis.api import vis_motion_use_scenepic_animation
from lpanlib.others.colors import name_to_rgb

from tokenhsi.data.data_utils import project_joints, project_joints_simple, project_joints_g1

def get_g1_joint_mapping(g1_skeleton, smpl_skeleton):
    """
    动态获取G1和SMPL之间的关节映射
    """
    # 打印G1的关节信息以便调试
    print("G1 skeleton nodes:")
    for i, name in enumerate(g1_skeleton.node_names):
        print(f"  {i}: {name}")
    
    # 基于G1的关节名称，建立与SMPL的映射关系
    # 这里需要根据实际的G1关节名称进行调整
    g1_to_smpl_mapping = []
    
    # 尝试自动匹配一些关键关节
    key_mappings = {
        # G1关节名 -> SMPL关节索引
        "pelvis": 0,  # SMPL Pelvis
        "torso_link": 3,  # SMPL Torso
        "waist_yaw_link": 6,  # SMPL Spine
        "left_hip_pitch_link": 1,  # SMPL L_Hip
        "right_hip_pitch_link": 2,  # SMPL R_Hip
        "left_knee_link": 4,  # SMPL L_Knee
        "right_knee_link": 5,  # SMPL R_Knee
        "left_ankle_pitch_link": 7,  # SMPL L_Ankle
        "right_ankle_pitch_link": 8,  # SMPL R_Ankle
        "left_shoulder_pitch_link": 16,  # SMPL L_Shoulder
        "right_shoulder_pitch_link": 17,  # SMPL R_Shoulder
        "left_elbow_link": 18,  # SMPL L_Elbow
        "right_elbow_link": 19,  # SMPL R_Elbow
        "head_link":15,
        "left_wrist_roll_link":20,
        "right_wrist_roll_link":21
    }
    
    # 为G1的每个关节找到对应的SMPL关节
    for i, g1_joint_name in enumerate(g1_skeleton.node_names):
        smpl_idx = 0  # 默认映射到pelvis
        
        # 查找匹配的映射
        for key, value in key_mappings.items():
            if key in g1_joint_name.lower():
                smpl_idx = value
                break
        
        g1_to_smpl_mapping.append(smpl_idx)
    
    return np.array(g1_to_smpl_mapping)

joints_to_use = {
    "from_smpl_original_to_amp_humanoid": np.array([0, 6, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7]),
    "from_smpl_original_to_phys_humanoid_v3": np.array([0, 6, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7]),
}


def visualize_g1_tpose_method2(g1_tpose):

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    
    global_positions = g1_tpose.global_translation.numpy()
    skeleton_tree = g1_tpose.skeleton_tree
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(global_positions[:, 0], global_positions[:, 1], global_positions[:, 2], 
               c='red', s=100, alpha=0.8, label='Joints')
    
    for i, name in enumerate(skeleton_tree.node_names):
        ax.text(global_positions[i, 0], global_positions[i, 1], global_positions[i, 2], 
                f'{i}:{name}', fontsize=8)
    
    # 
    parent_indices = skeleton_tree.parent_indices.numpy()
    for i, parent_idx in enumerate(parent_indices):
        if parent_idx >= 0:  
            ax.plot([global_positions[parent_idx, 0], global_positions[i, 0]],
                   [global_positions[parent_idx, 1], global_positions[i, 1]],
                   [global_positions[parent_idx, 2], global_positions[i, 2]], 
                   'b-', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('G1 Robot T-pose Visualization')
    ax.legend()
    
    max_range = np.array([global_positions[:,0].max()-global_positions[:,0].min(),
                         global_positions[:,1].max()-global_positions[:,1].min(),
                         global_positions[:,2].max()-global_positions[:,2].min()]).max() / 2.0
    mid_x = (global_positions[:,0].max()+global_positions[:,0].min()) * 0.5
    mid_y = (global_positions[:,1].max()+global_positions[:,1].min()) * 0.5
    mid_z = (global_positions[:,2].max()+global_positions[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()
    print("可视化完成")

if __name__ == "__main__":

    # load skeleton of smpl_humanoid
    smpl_humanoid_xml_path = osp.join(osp.dirname(__file__), "../assets/mjcf/humanoid/smpl_humanoid.xml")
    smpl_humanoid_skeleton = SkeletonTree.from_mjcf(smpl_humanoid_xml_path)

    # load skeleton of amp_humanoid
    amp_humanoid_xml_path = osp.join(osp.dirname(__file__), "../assets/mjcf/humanoid/amp_humanoid.xml")
    amp_humanoid_skeleton = SkeletonTree.from_mjcf(amp_humanoid_xml_path)

    # load skeleton of phys_humanoid_v3
    phys_humanoid_v3_xml_path = osp.join(osp.dirname(__file__), "../assets/mjcf/humanoid/phys_humanoid_v3.xml")
    phys_humanoid_v3_skeleton = SkeletonTree.from_mjcf(phys_humanoid_v3_xml_path)

    # load skeleton of G1
    g1_xml_path = osp.join(osp.dirname(__file__), "../assets/mjcf/g1/g1_29dof.xml")
    g1_skeleton = SkeletonTree.from_mjcf(g1_xml_path)
    
    # load skeleton of smpl_original
    bm = get_body_model("SMPL", "NEUTRAL", batch_size=1, debug=False)
    jts_global_trans = bm().joints[0, :24, :].cpu().detach().numpy()
    jts_local_trans = np.zeros_like(jts_global_trans)
    for i in range(jts_local_trans.shape[0]):
        parent = bm.parents[i]
        if parent == -1:
            jts_local_trans[i] = jts_global_trans[i]
        else:
            jts_local_trans[i] = jts_global_trans[i] - jts_global_trans[parent]

    skel_dict = smpl_humanoid_skeleton.to_dict()
    skel_dict["node_names"] = [
        "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee", "Spine", "L_Ankle", "R_Ankle",
        "Chest", "L_Toe", "R_Toe", "Neck", "L_Thorax", "R_Thorax", "Head", "L_Shoulder", "R_Shoulder", 
        "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
    ]
    skel_dict["parent_indices"]["arr"] = bm.parents.numpy()
    skel_dict["local_translation"]["arr"] = jts_local_trans
    smpl_original_skeleton = SkeletonTree.from_dict(skel_dict)

    # 动态获取G1的关节映射
    g1_joint_mapping = get_g1_joint_mapping(g1_skeleton, smpl_original_skeleton)
    joints_to_use["from_smpl_original_to_g1_29dof"] = g1_joint_mapping

    # create tposes
    smpl_original_tpose = SkeletonState.zero_pose(smpl_original_skeleton)
    
    amp_humanoid_tpose = SkeletonState.zero_pose(amp_humanoid_skeleton)
    local_rotation = amp_humanoid_tpose.local_rotation
    local_rotation[amp_humanoid_skeleton.index("left_upper_arm")] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
        local_rotation[amp_humanoid_skeleton.index("left_upper_arm")]
    )
    local_rotation[amp_humanoid_skeleton.index("right_upper_arm")] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
        local_rotation[amp_humanoid_skeleton.index("right_upper_arm")]
    )

    phys_humanoid_v3_tpose = SkeletonState.zero_pose(phys_humanoid_v3_skeleton)
    
    g1_tpose = SkeletonState.zero_pose(g1_skeleton)
    local_rotation = g1_tpose.local_rotation
    visualize_g1_tpose_method2(g1_tpose)
    # 根据G1的结构调整手臂姿态（如果需要）
#   - L_Elbow: "[0, -np.pi/2, 0]"
#   - R_Elbow: "[0, np.pi/2, 0]"
    left_shoulder_idx = g1_skeleton.index("left_shoulder_pitch_link")
    right_shoulder_idx = g1_skeleton.index("right_shoulder_pitch_link")
    L_Elbow_idx = g1_skeleton.index("left_elbow_link")
    R_Elbow_idx = g1_skeleton.index("right_elbow_link")

    
    local_rotation[left_shoulder_idx] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
        local_rotation[left_shoulder_idx]
    )
    local_rotation[right_shoulder_idx] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
        local_rotation[right_shoulder_idx]
        )
    local_rotation[L_Elbow_idx] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([0.0, -1.0, 0.0]), degree=True), 
        local_rotation[L_Elbow_idx]
    )
    local_rotation[R_Elbow_idx] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True), 
        local_rotation[R_Elbow_idx]
        )
    # except:
    #     pass

    local_rotation = phys_humanoid_v3_tpose.local_rotation
    local_rotation[phys_humanoid_v3_skeleton.index("left_upper_arm")] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
        local_rotation[phys_humanoid_v3_skeleton.index("left_upper_arm")]
    )
    local_rotation[phys_humanoid_v3_skeleton.index("right_upper_arm")] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
        local_rotation[phys_humanoid_v3_skeleton.index("right_upper_arm")]
    )

    # input/output dirs
    input_dir = osp.join(osp.dirname(__file__), "smpl_params")
    output_dir = osp.join(osp.dirname(__file__), "motions")

    os.makedirs(output_dir, exist_ok=True)

    data_list = os.listdir(input_dir)
    pbar = tqdm(data_list)
    for fname in pbar:
        pbar.set_description(fname)

        subset_name = fname.split("+__+")[0]
        subject = fname.split("+__+")[1]
        action = fname.split("+__+")[2][:-4]

        curr_output_dir = osp.join(output_dir, subset_name, fname[:-4])

        os.makedirs(curr_output_dir, exist_ok=True)

        # load SMPL params
        raw_params = np.load(osp.join(input_dir, fname), allow_pickle=True).item()
        poses = torch.tensor(raw_params["poses"], dtype=torch.float32)
        trans = torch.tensor(raw_params["trans"], dtype=torch.float32)
        fps = raw_params["fps"]

        # compute world absolute position of root joint
        trans = bm(
            global_orient=poses[:, 0:3], 
            body_pose=poses[:, 3:72],
            transl=trans[:, :],
        ).joints[:, 0, :].cpu().detach()

        poses = poses.reshape(-1, 24, 3)

        # angle axis ---> quaternion
        poses_quat = tgm.angle_axis_to_quaternion(poses.reshape(-1, 3)).reshape(poses.shape[0], -1, 4)

        # switch quaternion order
        # wxyz -> xyzw
        poses_quat = poses_quat[:, :, [1, 2, 3, 0]]

        # generate motion
        skeleton_state = SkeletonState.from_rotation_and_root_translation(smpl_original_skeleton, poses_quat, trans, is_local=True)
        motion = SkeletonMotion.from_skeleton_state(skeleton_state, fps=fps)

        ################ retarget ################

        configs = {
            "g1_29dof": {
                "skeleton": g1_skeleton,
                "xml_path": g1_xml_path,
                "tpose": g1_tpose,
                "joints_to_use": joints_to_use["from_smpl_original_to_g1_29dof"],
                "root_height_offset": 0.1,
            },
        }

        ###### retargeting ######
        for k, v in configs.items():
            
            # 对于G1，我们需要特殊处理
            if k == "g1_29dof":
                # 获取SMPL的全局旋转
                smpl_global_rotations = skeleton_state.global_rotation.clone()  # shape: [T, 24, 4]
                
                # 创建G1大小的旋转矩阵
                num_frames = smpl_global_rotations.shape[0]
                num_g1_joints = len(g1_skeleton.node_names)
                
                # 初始化G1的全局旋转
                target_final_global_rotation = torch.zeros(num_frames, num_g1_joints, 4)
                target_final_global_rotation[..., 3] = 1.0  # 设置w=1 (单位四元数)
                
                # 使用映射关系设置旋转
                for g1_idx in range(num_g1_joints):
                    smpl_idx = v["joints_to_use"][g1_idx]
                    if smpl_idx < smpl_global_rotations.shape[1]:
                        # 获取对应的SMPL旋转
                        smpl_rotation = smpl_global_rotations[:, smpl_idx, :]
                        
                        # 获取G1 T-pose的对应旋转
                        g1_tpose_rotation = v["tpose"].global_rotation[g1_idx:g1_idx+1, :]
                        
                        # 对齐旋转
                        aligned_rotation = quat_mul_norm(
                            torch.tensor([-0.5, -0.5, -0.5, 0.5]), g1_tpose_rotation
                        )
                        
                        # 应用SMPL旋转
                        target_final_global_rotation[:, g1_idx, :] = quat_mul_norm(
                            smpl_rotation, aligned_rotation.expand(num_frames, -1)
                        )
                
                target_final_root_translation = skeleton_state.root_translation.clone()
                
            else:
                # 原有的处理方式（用于其他humanoid模型）
                target_origin_global_rotation = v["tpose"].global_rotation.clone()

                target_aligned_global_rotation = quat_mul_norm( 
                    torch.tensor([-0.5, -0.5, -0.5, 0.5]), target_origin_global_rotation
                )

                target_final_global_rotation = quat_mul_norm(
                    skeleton_state.global_rotation.clone()[..., v["joints_to_use"], :], 
                    target_aligned_global_rotation.clone()
                )
                target_final_root_translation = skeleton_state.root_translation.clone()

            # 创建新的骨架状态
            new_skeleton_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree=v["skeleton"],
                r=target_final_global_rotation,
                t=target_final_root_translation,
                is_local=False,
            ).local_repr()
            new_motion = SkeletonMotion.from_skeleton_state(new_skeleton_state, fps=fps)

            new_motion_params_root_trans = new_motion.root_translation.clone()
            new_motion_params_local_rots = new_motion.local_rotation.clone()

            # check foot-ground penetration
            if "stair" not in fname:
                min_h = torch.min(new_motion.global_translation[:, :, 2], dim=-1)[0].mean()
            else:
                min_h = torch.min(new_motion.global_translation[:, :, 2], dim=-1)[0].min()

            for i in range(new_motion.global_translation.shape[0]):
                new_motion_params_root_trans[i, 2] += -min_h

            # adjust the height of the root to avoid ground penetration
            root_height_offset = v["root_height_offset"]
            new_motion_params_root_trans[:, 2] += root_height_offset

            # update new_motion
            new_skeleton_state = SkeletonState.from_rotation_and_root_translation(
                v["skeleton"], new_motion_params_local_rots, new_motion_params_root_trans, is_local=True
            )
            new_motion = SkeletonMotion.from_skeleton_state(new_skeleton_state, fps=fps)

            # 应用关节投影
            if k == "amp_humanoid":
                new_motion = project_joints(new_motion)
            elif k == "phys_humanoid" or k == "phys_humanoid_v2" or k == "phys_humanoid_v3":
                new_motion = project_joints_simple(new_motion)
            elif k == "g1_29dof":
                new_motion = project_joints_g1(new_motion)
            else:
                pass

            # save retargeted motion
            save_dir = osp.join(curr_output_dir, k)
            os.makedirs(save_dir, exist_ok=True)
            save_path = osp.join(save_dir, "ref_motion.npy")
            new_motion.to_file(save_path)
            # plot_skeleton_motion_interactive(motion)
            # scenepic animation
            vis_motion_use_scenepic_animation(
                asset_filename=v["xml_path"],
                rigidbody_global_pos=new_motion.global_translation,
                rigidbody_global_rot=new_motion.global_rotation,
                fps=fps,
                up_axis="z",
                color=name_to_rgb['AliceBlue'] * 255,
                output_path=osp.join(save_dir, "ref_motion_render.html"),
            )