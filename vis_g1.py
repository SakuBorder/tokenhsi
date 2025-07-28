import sys
sys.path.append("./")

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

from lpanlib.poselib.skeleton.skeleton3d import SkeletonMotion
from lpanlib.poselib.visualization.common import plot_skeleton_motion_interactive


def visualize_motion_debug(motion_file):
    """调试版本的motion可视化，包含更多信息"""
    print(f"Loading motion from: {motion_file}")
    
    # 加载motion
    motion = SkeletonMotion.from_file(motion_file)
    
    # 打印详细信息
    print(f"\nMotion Information:")
    print(f"- Number of frames: {len(motion)}")
    print(f"- FPS: {motion.fps}")
    print(f"- Duration: {len(motion) / motion.fps:.2f} seconds")
    print(f"- Number of joints: {motion.skeleton_tree.num_joints}")
    print(f"- Joint names: {motion.skeleton_tree.node_names}")
    
    # 检查数据范围
    root_positions = motion.root_translation.numpy()
    global_positions = motion.global_translation.numpy()
    
    print(f"\nPosition ranges:")
    print(f"Root position:")
    print(f"  X: [{root_positions[:, 0].min():.3f}, {root_positions[:, 0].max():.3f}]")
    print(f"  Y: [{root_positions[:, 1].min():.3f}, {root_positions[:, 1].max():.3f}]")
    print(f"  Z: [{root_positions[:, 2].min():.3f}, {root_positions[:, 2].max():.3f}]")
    
    print(f"\nGlobal position (all joints):")
    print(f"  X: [{global_positions[:, :, 0].min():.3f}, {global_positions[:, :, 0].max():.3f}]")
    print(f"  Y: [{global_positions[:, :, 1].min():.3f}, {global_positions[:, :, 1].max():.3f}]")
    print(f"  Z: [{global_positions[:, :, 2].min():.3f}, {global_positions[:, :, 2].max():.3f}]")
    
    # 检查是否有NaN
    if np.any(np.isnan(global_positions)):
        print("WARNING: Found NaN values in positions!")
        
    return motion


def plot_motion_frame(motion, frame_idx=0, ax=None, elev=20, azim=45):
    """绘制motion的单个帧，并更新视角"""
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax.clear()
    
    # 获取该帧的全局位置
    positions = motion.global_translation[frame_idx].numpy()
    skeleton_tree = motion.skeleton_tree
    
    # 绘制关节点
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
               c='red', s=100, alpha=0.8)
    
    # 绘制骨骼连接
    parent_indices = skeleton_tree.parent_indices.numpy()
    for i, parent_idx in enumerate(parent_indices):
        if parent_idx >= 0:
            ax.plot([positions[parent_idx, 0], positions[i, 0]],
                   [positions[parent_idx, 1], positions[i, 1]],
                   [positions[parent_idx, 2], positions[i, 2]], 
                   'b-', linewidth=2, alpha=0.7)
    
    # 为一些关键关节添加标签
    key_joints = ['pelvis', 'torso', 'head', 'left_foot', 'right_foot', 
                  'left_hand', 'right_hand', 'left_elbow', 'right_elbow',
                  'left_knee', 'right_knee']
    
    for i, name in enumerate(skeleton_tree.node_names):
        for key in key_joints:
            if key in name.lower():
                ax.text(positions[i, 0], positions[i, 1], positions[i, 2], 
                       name, fontsize=8, alpha=0.7)
                break
    
    # 设置坐标轴
    all_positions = motion.global_translation.numpy()
    
    # 计算合理的显示范围
    x_range = [all_positions[:, :, 0].min() - 0.5, all_positions[:, :, 0].max() + 0.5]
    y_range = [all_positions[:, :, 1].min() - 0.5, all_positions[:, :, 1].max() + 0.5]
    z_range = [all_positions[:, :, 2].min() - 0.1, all_positions[:, :, 2].max() + 0.5]
    
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Frame {frame_idx}/{len(motion)-1}')
    
    # 添加地面参考
    xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], 2),
                         np.linspace(y_range[0], y_range[1], 2))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
    
    # 更新视角
    ax.view_init(elev=elev, azim=azim)
    
    return ax


def interactive_motion_viewer(motion_file):
    """交互式motion查看器，支持鼠标控制视角"""
    motion = visualize_motion_debug(motion_file)
    
    # 创建图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 初始化
    frame_idx = [0]
    playing = [False]
    
    # 绘制初始帧
    plot_motion_frame(motion, frame_idx[0], ax)
    
    # 添加滑块
    from matplotlib.widgets import Slider, Button
    
    # 滑块轴
    ax_slider = plt.axes([0.1, 0.02, 0.8, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, len(motion)-1, valinit=0, valstep=1)
    
    # 播放按钮
    ax_button = plt.axes([0.45, 0.08, 0.1, 0.04])
    button = Button(ax_button, 'Play/Pause')
    
    def update_frame(val):
        frame_idx[0] = int(slider.val)
        plot_motion_frame(motion, frame_idx[0], ax)
        plt.draw()
    
    def toggle_play(event):
        playing[0] = not playing[0]
        if playing[0]:
            animate()  # Start the animation loop when play is toggled on
    
    slider.on_changed(update_frame)
    button.on_clicked(toggle_play)
    
    # 动画循环
    def animate():
        while playing[0]:  # Ensure it continues looping while playing is True
            frame_idx[0] = (frame_idx[0] + 1) % len(motion)
            slider.set_val(frame_idx[0])
            plot_motion_frame(motion, frame_idx[0], ax)
            plt.pause(1.0 / motion.fps)
        plt.pause(0.1)  # Small delay when not playing
    
    # 键盘控制
    def on_key(event):
        if event.key == ' ':
            playing[0] = not playing[0]
            if playing[0]:
                animate()  # Start animation if space is pressed
        elif event.key == 'left':
            frame_idx[0] = max(0, frame_idx[0] - 1)
            slider.set_val(frame_idx[0])
        elif event.key == 'right':
            frame_idx[0] = min(len(motion) - 1, frame_idx[0] + 1)
            slider.set_val(frame_idx[0])
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # 鼠标控制视角
    def on_move(event):
        """ 鼠标移动时更新视角 """
        if event.button == 1:  # 左键按下时才更新视角
            # 获取当前视角并更新
            elev, azim = ax.get_proj()[0], ax.get_proj()[1]  # 获取当前视角
            # 更新视角
            ax.view_init(elev=elev + event.ydata * 0.1, azim=azim + event.xdata * 0.1)
            plt.draw()
    
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    
    print("\nControls:")
    print("- Click 'Play/Pause' button or press SPACE to play/pause")
    print("- Use slider to jump to specific frame")
    print("- Press LEFT/RIGHT arrow keys to go frame by frame")
    print("- Drag the mouse to change the view")
    
    plt.show()






def compare_skeleton_structures(motion_file):
    """比较和分析骨架结构"""
    motion = SkeletonMotion.from_file(motion_file)
    skeleton = motion.skeleton_tree
    
    print("\nSkeleton Structure Analysis:")
    print(f"Total joints: {skeleton.num_joints}")
    
    # 打印父子关系
    print("\nParent-Child relationships:")
    for i, (name, parent_idx) in enumerate(zip(skeleton.node_names, skeleton.parent_indices.numpy())):
        if parent_idx >= 0:
            parent_name = skeleton.node_names[parent_idx]
            print(f"  {i:2d}: {name:30s} <- {parent_idx:2d}: {parent_name}")
        else:
            print(f"  {i:2d}: {name:30s} (ROOT)")
    
    # 检查关节的局部偏移
    print("\nLocal translations (joint offsets):")
    local_trans = skeleton.local_translation.numpy()
    for i, name in enumerate(skeleton.node_names):
        offset = local_trans[i]
        length = np.linalg.norm(offset)
        if length > 0.01:  # 只显示有明显偏移的
            print(f"  {name:30s}: [{offset[0]:6.3f}, {offset[1]:6.3f}, {offset[2]:6.3f}] (length: {length:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize motion files with debugging")
    parser.add_argument("motion_file", type=str, nargs='?', 
                       help="Path to motion file (.npy)")
    parser.add_argument("--mode", type=str, default="interactive",
                       choices=["interactive", "debug", "structure"],
                       help="Visualization mode")
    
    args = parser.parse_args()
    
    # 默认路径
    if not args.motion_file:
        default_paths = [
            "tokenhsi/data/dataset_amass_loco/motions/ACCAD/ACCAD+__+Female1Walking_c3d+__+B3_-_walk1_stageii/g1_29dof/ref_motion.npy",
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                args.motion_file = path
                print(f"Using default file: {path}")
                break
    
    if not args.motion_file or not os.path.exists(args.motion_file):
        print("Please specify a valid motion file")
        exit(1)
    
    # 运行可视化
    if args.mode == "interactive":
        interactive_motion_viewer(args.motion_file)
    elif args.mode == "debug":
        motion = visualize_motion_debug(args.motion_file)
        # 使用原始的plot_skeleton_motion_interactive
        plot_skeleton_motion_interactive(motion, task_name=os.path.basename(args.motion_file))
    elif args.mode == "structure":
        compare_skeleton_structures(args.motion_file)