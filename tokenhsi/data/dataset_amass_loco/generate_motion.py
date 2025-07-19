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

from tokenhsi.data.data_utils import project_joints, project_joints_simple,project_joints_g1

joints_to_use = {
    "from_smpl_original_to_amp_humanoid": np.array([0, 6, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7]),
    "from_smpl_original_to_phys_humanoid_v3": np.array([0, 6, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7]),
    "from_smpl_original_to_g1": np.array([0, 6, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7])
}


# G1 T-pose 3D可视化代码

# 方法1: 使用内置的poselib可视化功能
def visualize_g1_tpose_method1(g1_tpose):
    """
    方法1: 使用poselib内置的可视化功能
    这是最简单直接的方法
    """
    from lpanlib.poselib.visualization.common import plot_skeleton_state
    
    print("=== 方法1: 使用poselib内置可视化 ===")
    print("正在显示G1 T-pose...")
    
    # 直接使用内置的plot函数
    plot_skeleton_state(g1_tpose)
    print("T-pose可视化窗口已打开")


# 方法2: 使用matplotlib 3D绘制
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

def create_g1_tpose_corrected(g1_skeleton):

    g1_tpose = SkeletonState.zero_pose(g1_skeleton)
    local_rotation = g1_tpose.local_rotation.clone()

    print(f"G1骨架关节数量: {len(g1_skeleton.node_names)}")
    print(f"G1关节名称列表: {g1_skeleton.node_names}")

    try:
        left_arm_idx = g1_skeleton.index("left_upper_arm")
        print(f"找到左臂关节: 'left_upper_arm' 索引为 {left_arm_idx}")
        
        right_arm_idx = g1_skeleton.index("right_upper_arm")
        print(f"找到右臂关节: 'right_upper_arm' 索引为 {right_arm_idx}")
        
        left_rotation = quat_from_angle_axis(
            angle=torch.tensor([90.0]), 
            axis=torch.tensor([1.0, 0.0, 0.0]), 
            degree=True
        )
        local_rotation[left_arm_idx] = quat_mul(
            left_rotation,
            local_rotation[left_arm_idx]
        )
        print(f"已设置左臂T-pose旋转: +90度绕X轴")

        # 设置右臂关节的T-pose旋转（向下90度，参考H1版本）
        right_rotation = quat_from_angle_axis(
            angle=torch.tensor([-90.0]), 
            axis=torch.tensor([1.0, 0.0, 0.0]), 
            degree=True
        )
        local_rotation[right_arm_idx] = quat_mul(
            right_rotation,
            local_rotation[right_arm_idx]
        )
        print(f"已设置右臂T-pose旋转: -90度绕X轴")
        
    except KeyError as e:
        print(f"关节名称查找失败: {e}")
        print("尝试使用备用方法...")
        
        if len(g1_skeleton.node_names) == 15:
            left_arm_idx = 6   # left_upper_arm应该在索引6
            right_arm_idx = 3  # right_upper_arm应该在索引3
            
            print(f"使用预期顺序 - 左臂索引: {left_arm_idx}, 右臂索引: {right_arm_idx}")
            print(f"实际关节名称 - 左臂: {g1_skeleton.node_names[left_arm_idx]}, 右臂: {g1_skeleton.node_names[right_arm_idx]}")
            
            if ("arm" in g1_skeleton.node_names[left_arm_idx].lower() and 
                "left" in g1_skeleton.node_names[left_arm_idx].lower()):
                
                left_rotation = quat_from_angle_axis(
                    angle=torch.tensor([90.0]), 
                    axis=torch.tensor([1.0, 0.0, 0.0]), 
                    degree=True
                )
                local_rotation[left_arm_idx] = quat_mul(
                    left_rotation,
                    local_rotation[left_arm_idx]
                )
                print(f"已设置左臂({g1_skeleton.node_names[left_arm_idx]})T-pose旋转")
            else:
                print(f"警告: 左臂关节名称不符合预期: {g1_skeleton.node_names[left_arm_idx]}")
                
            if ("arm" in g1_skeleton.node_names[right_arm_idx].lower() and 
                "right" in g1_skeleton.node_names[right_arm_idx].lower()):
                
                # 设置右臂T-pose旋转
                right_rotation = quat_from_angle_axis(
                    angle=torch.tensor([-90.0]), 
                    axis=torch.tensor([1.0, 0.0, 0.0]), 
                    degree=True
                )
                local_rotation[right_arm_idx] = quat_mul(
                    right_rotation,
                    local_rotation[right_arm_idx]
                )
                print(f"已设置右臂({g1_skeleton.node_names[right_arm_idx]})T-pose旋转")
            else:
                print(f"警告: 右臂关节名称不符合预期: {g1_skeleton.node_names[right_arm_idx]}")
        else:
            print(f"错误: G1骨架关节数量不是15个，实际为{len(g1_skeleton.node_names)}个")
            print("无法设置T-pose，将使用零姿态")

    try:
        g1_tpose = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree=g1_skeleton,
            r=local_rotation,
            t=torch.zeros(3),  
            is_local=True
        )
        print("G1 T-pose创建成功")
        
        print(f"T-pose验证:")
        print(f"  根节点位置: {g1_tpose.root_translation}")
        if len(g1_skeleton.node_names) >= 7:
            print(f"  左臂关节旋转: {local_rotation[6] if len(local_rotation) > 6 else 'N/A'}")
        if len(g1_skeleton.node_names) >= 4:
            print(f"  右臂关节旋转: {local_rotation[3] if len(local_rotation) > 3 else 'N/A'}")
            
    except Exception as e:
        print(f"T-pose创建失败: {e}")
        print("使用原始零姿态")
        g1_tpose = SkeletonState.zero_pose(g1_skeleton)

    print("=" * 50)
    return g1_tpose

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
    g1_xml_path = osp.join(osp.dirname(__file__), "../assets/mjcf/g1/g1_29dof.xml")
    g1_skeleton = SkeletonTree.from_mjcf_g1(g1_xml_path) 

    # import ipdb; ipdb.set_trace()
    # print(g1_skeleton.node_names)
    print("=== Skeleton Analysis ===")
    print(f"amp_humanoid nodes: {len(amp_humanoid_skeleton.node_names)}")
    print(f"amp_humanoid names: {amp_humanoid_skeleton.node_names}")
    print()
    print(f"phys_humanoid_v3 nodes: {len(phys_humanoid_v3_skeleton.node_names)}")  
    print(f"phys_humanoid_v3 names: {phys_humanoid_v3_skeleton.node_names}")
    print()
    print(f"g1 nodes: {len(g1_skeleton.node_names)}")
    print(f"g1 names: {g1_skeleton.node_names}")



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
    local_rotation = phys_humanoid_v3_tpose.local_rotation
    local_rotation[phys_humanoid_v3_skeleton.index("left_upper_arm")] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
        local_rotation[phys_humanoid_v3_skeleton.index("left_upper_arm")]
    )
    local_rotation[phys_humanoid_v3_skeleton.index("right_upper_arm")] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
        local_rotation[phys_humanoid_v3_skeleton.index("right_upper_arm")]
    )
    g1_tpose = create_g1_tpose_corrected(g1_skeleton)
    visualize_g1_tpose_method2(g1_tpose)

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

        # plot_skeleton_motion_interactive(motion)

        ################ retarget ################

        configs = {
            # "amp_humanoid": {
            #     "skeleton": amp_humanoid_skeleton,
            #     "xml_path": amp_humanoid_xml_path,
            #     "tpose": amp_humanoid_tpose,
            #     "joints_to_use": joints_to_use["from_smpl_original_to_amp_humanoid"],
            #     "root_height_offset": 0.05,
            # },
            # "phys_humanoid_v3": {
            #     "skeleton": phys_humanoid_v3_skeleton,
            #     "xml_path": phys_humanoid_v3_xml_path,
            #     "tpose": phys_humanoid_v3_tpose,
            #     "joints_to_use": joints_to_use["from_smpl_original_to_phys_humanoid_v3"],
            #     "root_height_offset": 0.07,
            # },

            "g1_29dof": {
                "skeleton": g1_skeleton,
                "xml_path": g1_xml_path,
                "tpose": g1_tpose,
                "joints_to_use": joints_to_use["from_smpl_original_to_g1"],
                "root_height_offset": 0.07,
            },
        }

        ###### retargeting ######
        ###### retargeting ######
        for k, v in configs.items():


            target_origin_global_rotation = v["tpose"].global_rotation.clone()

            target_aligned_global_rotation = quat_mul_norm( 
                torch.tensor([-0.5, -0.5, -0.5, 0.5]), target_origin_global_rotation
            )
            
            target_final_global_rotation = quat_mul_norm(
                skeleton_state.global_rotation.clone()[..., v["joints_to_use"], :], target_aligned_global_rotation.clone()
            )
            target_final_root_translation = skeleton_state.root_translation.clone()

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
            new_skeleton_state = SkeletonState.from_rotation_and_root_translation(v["skeleton"], new_motion_params_local_rots, new_motion_params_root_trans, is_local=True)
            new_motion = SkeletonMotion.from_skeleton_state(new_skeleton_state, fps=fps)


            if k == "amp_humanoid":
                new_motion = project_joints(new_motion)
            elif k == "phys_humanoid" or k == "phys_humanoid_v2" or k == "phys_humanoid_v3":
                new_motion = project_joints_simple(new_motion)
            elif k == "g1_29dof":
                print(f"\nApplying project_joints_g1...")
                new_motion = project_joints_g1(new_motion)
                

            else:
                pass

            # save retargeted motion
            save_dir = osp.join(curr_output_dir, k)
            os.makedirs(save_dir, exist_ok=True)
            save_path = osp.join(save_dir, "ref_motion.npy")
            new_motion.to_file(save_path)

            for axis, name in enumerate(['X', 'Y', 'Z']):
                motion_range = new_motion.global_translation[:, :, axis]
                # print(f"  {name}-axis: min={motion_range.min():.3f}, max={motion_range.max():.3f}, range={motion_range.max()-motion_range.min():.3f}")
            
            # print("Key joint movement ranges:")
            for joint_name in ['left_thigh', 'right_thigh', 'left_shin', 'right_shin']:
                if joint_name in g1_skeleton.node_names:
                    joint_idx = g1_skeleton.index(joint_name)
                    joint_pos = new_motion.global_translation[:, joint_idx, :]
                    y_range = joint_pos[:, 1].max() - joint_pos[:, 1].min()
                    # print(f"  {joint_name} Y-range: {y_range:.3f}")
            print("=== Retargeted Motion 检查 ===")
            print("global_translation shape:", new_motion.global_translation.shape)
            print("global_translation min/max:", new_motion.global_translation.min().item(), new_motion.global_translation.max().item())
            print("是否全为零:", torch.allclose(new_motion.global_translation, torch.zeros_like(new_motion.global_translation)))

            # 打印根节点轨迹
            print("根节点轨迹 (前5帧):", new_motion.root_translation[:5])


            print(f"\n=== Joint Mapping Check for {k} ===")
            smpl_joints_used = joints_to_use["from_smpl_original_to_g1"]
            g1_joint_order = g1_skeleton.node_names
            
            print("SMPL -> G1 mapping:")
            for i, smpl_idx in enumerate(smpl_joints_used):
                g1_joint_name = g1_joint_order[i]
                smpl_joint_name = smpl_original_skeleton.node_names[smpl_idx]
                print(f"  SMPL[{smpl_idx}] {smpl_joint_name} -> G1[{i}] {g1_joint_name}")
                smpl_pos = skeleton_state.global_translation[0, smpl_joints_used]
                g1_pos = new_motion.global_translation[0]
                diff = torch.norm(smpl_pos - g1_pos, dim=-1)
                print("SMPL vs G1 第一帧位置差异:", diff)

            # plot_skeleton_motion_interactive(new_motion)
            


            # # scenepic animation
            vis_motion_use_scenepic_animation(
                asset_filename=v["xml_path"],
                rigidbody_global_pos=new_motion.global_translation,
                rigidbody_global_rot=new_motion.global_rotation,
                fps=fps,
                up_axis="z",
                color=name_to_rgb['AliceBlue'] * 255,
                output_path=osp.join(save_dir, "ref_motion_render.html"),
            )

