import sys
sys.path.append("./")

import os
import os.path as osp
import torch
import numpy as np
import torchgeometry as tgm
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from lpanlib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from lpanlib.poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from lpanlib.poselib.core.rotation3d import quat_mul, quat_from_angle_axis, quat_mul_norm, quat_rotate, quat_identity

from body_models.model_loader import get_body_model

from lpanlib.isaacgym_utils.vis.api import vis_motion_use_scenepic_animation
from lpanlib.others.colors import name_to_rgb

from tokenhsi.data.data_utils import project_joints, project_joints_simple,project_joints_tai5

joints_to_use = {
    "from_smpl_original_to_amp_humanoid": np.array([0, 6, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7]),
    "from_smpl_original_to_phys_humanoid_v3": np.array([0, 6, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7]),
    "from_smpl_original_to_tai5": np.array([0, 6, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7]),
}
def visualize_skeleton_static(skeleton_state, title="Skeleton Visualization", save_path=None):
    """
    静态可视化单个骨架帧
    """
    global_positions = skeleton_state.global_translation.numpy()
    skeleton_tree = skeleton_state.skeleton_tree
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制关节点
    ax.scatter(global_positions[:, 0], global_positions[:, 1], global_positions[:, 2], 
               c='red', s=100, alpha=0.8, label='Joints')
    
    # 添加关节标签
    for i, name in enumerate(skeleton_tree.node_names):
        ax.text(global_positions[i, 0], global_positions[i, 1], global_positions[i, 2], 
                f'{i}:{name}', fontsize=8)
    
    # 绘制骨骼连接
    parent_indices = skeleton_tree.parent_indices.numpy()
    for i, parent_idx in enumerate(parent_indices):
        if parent_idx >= 0:
            ax.plot([global_positions[parent_idx, 0], global_positions[i, 0]],
                   [global_positions[parent_idx, 1], global_positions[i, 1]],
                   [global_positions[parent_idx, 2], global_positions[i, 2]], 
                   'b-', linewidth=2, alpha=0.7)
    
    # 设置坐标轴
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    
    # 设置等比例显示
    max_range = np.array([global_positions[:,0].max()-global_positions[:,0].min(),
                         global_positions[:,1].max()-global_positions[:,1].min(),
                         global_positions[:,2].max()-global_positions[:,2].min()]).max() / 2.0
    mid_x = (global_positions[:,0].max()+global_positions[:,0].min()) * 0.5
    mid_y = (global_positions[:,1].max()+global_positions[:,1].min()) * 0.5
    mid_z = (global_positions[:,2].max()+global_positions[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
def visualize_skeleton_motion_simple(motion, frame_step=10, save_path=None):
    """
    简单的运动可视化 - 显示多个关键帧
    """
    num_frames = motion.global_translation.shape[0]
    frames_to_show = list(range(0, num_frames, frame_step))
    if len(frames_to_show) > 8:  # 最多显示8帧
        frames_to_show = frames_to_show[:8]
    
    fig = plt.figure(figsize=(16, 12))
    
    for i, frame_idx in enumerate(frames_to_show):
        ax = fig.add_subplot(2, 4, i+1, projection='3d')
        
        # 获取当前帧的位置
        global_positions = motion.global_translation[frame_idx].numpy()
        skeleton_tree = motion.skeleton_tree
        
        # 绘制关节点
        ax.scatter(global_positions[:, 0], global_positions[:, 1], global_positions[:, 2], 
                   c='red', s=50, alpha=0.8)
        
        # 绘制骨骼连接
        parent_indices = skeleton_tree.parent_indices.numpy()
        for j, parent_idx in enumerate(parent_indices):
            if parent_idx >= 0:
                ax.plot([global_positions[parent_idx, 0], global_positions[j, 0]],
                       [global_positions[parent_idx, 1], global_positions[j, 1]],
                       [global_positions[parent_idx, 2], global_positions[j, 2]], 
                       'b-', linewidth=1.5, alpha=0.7)
        
        ax.set_title(f'Frame {frame_idx}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # 设置相同的坐标范围
        all_positions = motion.global_translation.numpy()
        max_range = np.array([all_positions[:,:,0].max()-all_positions[:,:,0].min(),
                             all_positions[:,:,1].max()-all_positions[:,:,1].min(),
                             all_positions[:,:,2].max()-all_positions[:,:,2].min()]).max() / 2.0
        mid_x = (all_positions[:,:,0].max()+all_positions[:,:,0].min()) * 0.5
        mid_y = (all_positions[:,:,1].max()+all_positions[:,:,1].min()) * 0.5
        mid_z = (all_positions[:,:,2].max()+all_positions[:,:,2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def visualize_skeleton_animation(motion, fps=30, save_path=None, show_trail=True, show_display=True):
    """
    动画可视化骨架运动
    """
    global_positions_all = motion.global_translation.numpy()
    skeleton_tree = motion.skeleton_tree
    parent_indices = skeleton_tree.parent_indices.numpy()
    
    # 使用非交互式后端避免显示窗口
    if not show_display:
        import matplotlib
        matplotlib.use('Agg')
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置固定的坐标范围
    all_positions = global_positions_all
    max_range = np.array([all_positions[:,:,0].max()-all_positions[:,:,0].min(),
                         all_positions[:,:,1].max()-all_positions[:,:,1].min(),
                         all_positions[:,:,2].max()-all_positions[:,:,2].min()]).max() / 2.0
    mid_x = (all_positions[:,:,0].max()+all_positions[:,:,0].min()) * 0.5
    mid_y = (all_positions[:,:,1].max()+all_positions[:,:,1].min()) * 0.5
    mid_z = (all_positions[:,:,2].max()+all_positions[:,:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # 初始化绘图元素
    joints_plot = ax.scatter([], [], [], c='red', s=80, alpha=0.8)
    lines = []
    trails = []
    
    # 创建骨骼连接线
    for i, parent_idx in enumerate(parent_indices):
        if parent_idx >= 0:
            line, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.7)
            lines.append((line, parent_idx, i))
    
    # 创建轨迹线（如果需要）
    if show_trail:
        for i in range(len(skeleton_tree.node_names)):
            trail, = ax.plot([], [], [], 'g-', alpha=0.3, linewidth=1)
            trails.append(trail)
    
    def animate(frame):
        # 更新关节位置
        global_positions = global_positions_all[frame]
        joints_plot._offsets3d = (global_positions[:, 0], 
                                 global_positions[:, 1], 
                                 global_positions[:, 2])
        
        # 更新骨骼连接
        for line, parent_idx, child_idx in lines:
            line.set_data_3d([global_positions[parent_idx, 0], global_positions[child_idx, 0]],
                            [global_positions[parent_idx, 1], global_positions[child_idx, 1]],
                            [global_positions[parent_idx, 2], global_positions[child_idx, 2]])
        
        # 更新轨迹
        if show_trail and frame > 10:
            trail_length = min(frame, 30)  # 显示最近30帧的轨迹
            for i, trail in enumerate(trails):
                trail_data = global_positions_all[frame-trail_length:frame+1, i, :]
                trail.set_data_3d(trail_data[:, 0], trail_data[:, 1], trail_data[:, 2])
        
        ax.set_title(f'Frame: {frame}/{len(global_positions_all)-1} - {skeleton_tree.node_names[0] if skeleton_tree.node_names else "Motion"}')
        
        return [joints_plot] + [line for line, _, _ in lines] + trails
    
    # 降采样帧数以减少GIF大小和生成时间
    total_frames = len(global_positions_all)
    max_frames = min(total_frames, 120)  # 最多120帧
    frame_step = max(1, total_frames // max_frames)
    frame_indices = list(range(0, total_frames, frame_step))
    
    # 创建动画
    anim = FuncAnimation(fig, animate, frames=frame_indices, 
                        interval=1000//fps, blit=False, repeat=True)
    
    if save_path:
        try:
            print(f"Generating GIF animation with {len(frame_indices)} frames...")
            anim.save(save_path, writer='pillow', fps=fps, dpi=80)
            print(f"Animation saved to {save_path}")
        except Exception as e:
            print(f"Failed to save animation: {e}")
    
    if show_display:
        plt.show()
    else:
        plt.close(fig)  # 关闭图形避免内存泄漏
    
    return anim

def save_motion_gif(motion, save_dir, filename="ref_motion_render.gif", fps=20):
    """
    专门用于保存运动GIF的函数，不显示可视化窗口
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    
    # 不显示窗口，只保存GIF
    anim = visualize_skeleton_animation(
        motion, 
        fps=fps, 
        save_path=save_path, 
        show_trail=True, 
        show_display=False
    )
    
    return save_path

def visualize_motion_trajectory(motion, joint_names=None, save_path=None):
    """
    可视化关节轨迹
    """
    global_positions = motion.global_translation.numpy()
    skeleton_tree = motion.skeleton_tree
    
    if joint_names is None:
        # 默认显示一些重要关节
        important_joints = ['pelvis', 'torso', 'head', 'left_foot', 'right_foot', 
                           'left_hand', 'right_hand']
        joint_indices = []
        for joint_name in important_joints:
            try:
                idx = skeleton_tree.index(joint_name)
                joint_indices.append((idx, joint_name))
            except:
                continue
        
        # 如果没找到，就用前几个关节
        if not joint_indices:
            joint_indices = [(i, name) for i, name in enumerate(skeleton_tree.node_names[:7])]
    else:
        joint_indices = [(skeleton_tree.index(name), name) for name in joint_names]
    
    fig = plt.figure(figsize=(15, 10))
    
    # 3D轨迹图
    ax1 = fig.add_subplot(221, projection='3d')
    colors = plt.cm.tab10(np.linspace(0, 1, len(joint_indices)))
    
    for (idx, name), color in zip(joint_indices, colors):
        trajectory = global_positions[:, idx, :]
        ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                label=name, color=color, alpha=0.7)
        # 标记起点和终点
        ax1.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                   color=color, s=100, marker='o', alpha=0.8)
        ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                   color=color, s=100, marker='s', alpha=0.8)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Joint Trajectories')
    ax1.legend()
    
    # X-Y平面投影
    ax2 = fig.add_subplot(222)
    for (idx, name), color in zip(joint_indices, colors):
        trajectory = global_positions[:, idx, :]
        ax2.plot(trajectory[:, 0], trajectory[:, 1], label=name, color=color, alpha=0.7)
        ax2.scatter(trajectory[0, 0], trajectory[0, 1], color=color, s=50, marker='o')
        ax2.scatter(trajectory[-1, 0], trajectory[-1, 1], color=color, s=50, marker='s')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View (X-Y)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 高度变化
    ax3 = fig.add_subplot(223)
    time_steps = np.arange(global_positions.shape[0]) / motion.fps
    for (idx, name), color in zip(joint_indices, colors):
        height = global_positions[:, idx, 2]
        ax3.plot(time_steps, height, label=name, color=color, alpha=0.7)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Height Z (m)')
    ax3.set_title('Height Changes Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 速度分析
    ax4 = fig.add_subplot(224)
    for (idx, name), color in zip(joint_indices, colors):
        pos = global_positions[:, idx, :]
        velocity = np.sqrt(np.sum(np.diff(pos, axis=0)**2, axis=1)) * motion.fps
        time_steps_vel = np.arange(len(velocity)) / motion.fps
        ax4.plot(time_steps_vel, velocity, label=name, color=color, alpha=0.7)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Speed (m/s)')
    ax4.set_title('Joint Speeds Over Time')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def quick_motion_check(motion, save_dir=None):
    """
    快速检查运动数据的完整可视化
    """
    print(f"Motion Info:")
    print(f"  Frames: {motion.global_translation.shape[0]}")
    print(f"  Joints: {motion.global_translation.shape[1]}")
    print(f"  FPS: {motion.fps}")
    print(f"  Duration: {motion.global_translation.shape[0]/motion.fps:.2f}s")
    print(f"  Joint names: {motion.skeleton_tree.node_names}")
    
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存静态T-pose
        visualize_skeleton_static(
            motion.skeleton_tree.zero_pose(), 
            title="T-Pose",
            save_path=os.path.join(save_dir, "tpose.png")
        )
        
        # 保存关键帧
        visualize_skeleton_motion_simple(
            motion, 
            save_path=os.path.join(save_dir, "keyframes.png")
        )
        
        # 保存轨迹分析
        visualize_motion_trajectory(
            motion,
            save_path=os.path.join(save_dir, "trajectories.png")
        )
        
        print(f"Visualizations saved to {save_dir}")
    else:
        # 直接显示
        visualize_skeleton_static(motion.skeleton_tree.zero_pose(), "T-Pose")
        visualize_skeleton_motion_simple(motion)
        visualize_motion_trajectory(motion)

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

    # load skeleton of tai5
    tai5_xml_path = osp.join(osp.dirname(__file__), "../assets/mjcf/tai5/tai5.xml")
    tai5_skeleton = SkeletonTree.from_mjcf_tai5(tai5_xml_path)
    # import ipdb;ipdb.set_trace()
    # 添加调试函数到 SkeletonTree 类中
    # SkeletonTree.from_mjcf_tai5_improved = from_mjcf_tai5_improved

    # # 使用调试版本解析
    # print("开始调试TAI5骨架解析...")
    # tai5_skeleton, tai5_tpose_debug = SkeletonTree.test_tai5_parsing(tai5_xml_path)

    # # 使用调试后的tpose而不是重新创建
    # # tai5_tpose = SkeletonState.zero_pose(tai5_skeleton)  # 注释掉这行
    # tai5_tpose = tai5_tpose_debug  # 使用这个
    # import ipdb;ipdb.set_trace()
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


    # tai5_tpose = SkeletonState.zero_pose(tai5_skeleton)
    tai5_tpose = SkeletonTree.create_tai5_tpose_with_correct_root(tai5_skeleton)
    local_rotation = tai5_tpose.local_rotation
    # local_rotation[tai5_skeleton.index("L_SHOULDER_P_S")] = quat_mul(
    #     quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
    #     local_rotation[tai5_skeleton.index("L_SHOULDER_P_S")]
    # )
    # local_rotation[tai5_skeleton.index("R_SHOULDER_P_S")] = quat_mul(
    #     quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
    #     local_rotation[tai5_skeleton.index("R_SHOULDER_P_S")]
    # )
    # visualize_g1_tpose_method2(tai5_tpose)
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
            "tai5": {
                "skeleton": tai5_skeleton,
                "xml_path": tai5_xml_path,
                "tpose": tai5_tpose,
                "joints_to_use": joints_to_use["from_smpl_original_to_tai5"],
                "root_height_offset": 0.07,
            },
        }

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
            elif k == "tai5" :
                new_motion = project_joints_tai5(new_motion)
            else:
                pass

            # save retargeted motion
            save_dir = osp.join(curr_output_dir, k)
            os.makedirs(save_dir, exist_ok=True)
            save_path = osp.join(save_dir, "ref_motion.npy")
            new_motion.to_file(save_path)

            # visualize_skeleton_animation(new_motion, fps=30, show_trail=True)
            save_motion_gif(new_motion, save_dir, "ref_motion_render.gif", fps=20)


            print(f"Motion global translation shape: {new_motion.global_translation.shape}")
            print(f"Motion global rotation shape: {new_motion.global_rotation.shape}")
            print(f"Skeleton joints count: {len(new_motion.skeleton_tree.node_names)}")
            # import ipdb; ipdb.set_trace() 
            # # scenepic animation
            # vis_motion_use_scenepic_animation(
            #     asset_filename=v["xml_path"],
            #     rigidbody_global_pos=new_motion.global_translation,
            #     rigidbody_global_rot=new_motion.global_rotation,
            #     fps=fps,
            #     up_axis="z",
            #     color=name_to_rgb['AliceBlue'] * 255,
            #     output_path=osp.join(save_dir, "ref_motion_render.html"),
            # )