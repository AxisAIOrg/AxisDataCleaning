import json
import numpy as np

# === 修复: Python 3.13 / NumPy 2.0 兼容性 ===
if not hasattr(np, 'disp'):
    np.disp = print

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import roboticstoolbox as rtb

def load_trajectory_data(file_path):
    """读取数据 (带时间戳)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'trajectory_data' in data:
            steps = data['trajectory_data']['steps']
        else:
            steps = data
            
        joint_names = [f"franka/panda_joint{i+1}" for i in range(7)]
        q_list = []
        t_list = []
        
        for step in steps:
            if 'state' in step: joints = step['state']['robot_joints']
            else: joints = step['robot_joints']
            q = [joints[name] for name in joint_names]
            q_list.append(q)

            if 'simulation_time' in step: t_list.append(step['simulation_time'])
            else: t_list.append(len(t_list) * 0.01) 

        return np.array(q_list), np.array(t_list)
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def compute_fk(q_matrix):
    """计算正运动学 (DH 模型)"""
    panda = rtb.models.DH.Panda()
    return panda.fkine(q_matrix).t

def visualize_complete_comparison(orig_pos, smooth_pos, time_array):
    """
    绘制完整的 3D 对比图：增强了原始轨迹的显示效果
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colormap = 'plasma'
    norm = plt.Normalize(time_array.min(), time_array.max())
    total_points = len(orig_pos)
    skip = max(1, total_points // 100)

    # =========================================
    # 1. 绘制原始轨迹 (高亮显示：深红色粗虚线)
    # =========================================
    # 【修改点】让原始线条更明显
    # color='crimson' (猩红色), linestyle='--' (虚线), linewidth=1.8 (加粗), alpha=0.9 (不透明)
    ax.plot(orig_pos[:, 0], orig_pos[:, 1], orig_pos[:, 2], 
            color='crimson', linestyle='--', linewidth=1.8, alpha=0.9,
            label='Original Line (Raw)', zorder=2) # zorder 稍微提高一点
    
    # 【修改点】让原始点也跟着变明显
    ax.scatter(orig_pos[::skip, 0], orig_pos[::skip, 1], orig_pos[::skip, 2], 
               color='crimson', marker='o', s=35, alpha=0.9, # 加大尺寸和不透明度
               label='Original Points', zorder=2)

    # =========================================
    # 2. 绘制平滑轨迹 (渐变线条 + 渐变散点)
    # =========================================
    # A) 制作渐变线条
    x, y, z = smooth_pos[:, 0], smooth_pos[:, 1], smooth_pos[:, 2]
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = Line3DCollection(segments, cmap=colormap, norm=norm)
    lc.set_array(time_array[:-1])
    lc.set_linewidth(3.5) # 平滑线也稍微再粗一点，保持对比
    lc.set_alpha(0.8)     # 稍微透明一点点，让原始线能透出来看清对比
    ax.add_collection(lc)

    # B) 画平滑点
    sc = ax.scatter(smooth_pos[::skip, 0], smooth_pos[::skip, 1], smooth_pos[::skip, 2],
               c=time_array[::skip], cmap=colormap, norm=norm,
               marker='o', s=45, alpha=1.0,
               label='Smoothed Points (Color=Time)', zorder=3)

    # =========================================
    # 3. 装饰图表
    # =========================================
    cbar = plt.colorbar(sc, ax=ax, pad=0.1, fraction=0.03)
    cbar.set_label('Simulation Time (s)', rotation=270, labelpad=15)

    # 起点终点标记
    ax.scatter(x[0], y[0], z[0], c='green', s=180, marker='^', label='Start', zorder=10, edgecolors='white')
    ax.scatter(x[-1], y[-1], z[-1], c='black', s=180, marker='x', label='End', zorder=10, linewidth=3)

    # 锁定坐标轴比例
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x, mid_y, mid_z = (x.max()+x.min())*0.5, (y.max()+y.min())*0.5, (z.max()+z.min())*0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_title('Franka End-Effector: Highlighting Original vs Smoothed Gradient', fontsize=14)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), markerscale=1.2)

    plt.tight_layout()
    plt.show()

def main():
    print("Reading Data...")
    q_orig, _ = load_trajectory_data('trajectory.json')
    q_smooth, t_smooth = load_trajectory_data('trajectory_smoothed.json')
    
    if q_orig is None or q_smooth is None: return

    print("Computing Kinematics (FK)...")
    xyz_orig = compute_fk(q_orig)
    xyz_smooth = compute_fk(q_smooth)
    
    print("Generating Enhanced Visualization...")
    if len(t_smooth) != len(xyz_smooth):
        t_smooth = np.linspace(t_smooth[0], t_smooth[-1], len(xyz_smooth))
        
    visualize_complete_comparison(xyz_orig, xyz_smooth, t_smooth)

if __name__ == "__main__":
    main()