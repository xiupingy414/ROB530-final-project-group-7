import zarr
import numpy as np
import open3d as o3d

# -----------------------------
# 1. 读取原始点云
# -----------------------------
z = zarr.open(
    r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/s1/lidar/hesai_points_undistorted/hesai_points_undistorted",
    mode='r'
)

all_points = z['points']
num_frames = all_points.shape[0]

print("num_frames in zarr =", num_frames)

# -----------------------------
# 2. 读取 KISS-ICP 输出的 poses
# -----------------------------
kiss_poses = np.load(
    r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/s1/lidar/results/kiss_icp_input_poses.npy"
)

print("kiss_poses.shape =", kiss_poses.shape)

# 防止帧数不一致
num_use = min(num_frames, len(kiss_poses))
print("num_use =", num_use)

# -----------------------------
# 3. 一些辅助函数
# -----------------------------
def make_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd

# -----------------------------
# 4. 用 KISS-ICP pose 拼全局地图
# -----------------------------
global_map = o3d.geometry.PointCloud()
trajectory_points = []

# 你可以调这个参数
frame_step = 10       # 每隔多少帧加一帧到地图里，防止太慢
voxel_size = 0.5      # 每帧先做一次下采样

for i in range(0, num_use, frame_step):
    pts = np.asarray(all_points[i], dtype=np.float32)   # [N,3]
    pose = kiss_poses[i]                                # [4,4]

    pcd = make_pcd(pts)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd.transform(pose)

    global_map += pcd
    trajectory_points.append(pose[:3, 3])

    #print(f"Added frame {i}/{num_use-1}")

# 全局再下采样一次，减少点数
global_map = global_map.voxel_down_sample(voxel_size=0.8)
global_map.paint_uniform_color([0.7, 0.7, 0.7])

# -----------------------------
# 5. 画轨迹
# -----------------------------
trajectory_points = np.array(trajectory_points)

traj_line = o3d.geometry.LineSet()
traj_line.points = o3d.utility.Vector3dVector(trajectory_points)

lines = [[i, i + 1] for i in range(len(trajectory_points) - 1)]
traj_line.lines = o3d.utility.Vector2iVector(lines)

colors = [[1, 0, 0] for _ in range(len(lines))]
traj_line.colors = o3d.utility.Vector3dVector(colors)

# 坐标轴
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=20,
    origin=[0, 0, 0]
)

# -----------------------------
# 6. 可视化
# -----------------------------
o3d.visualization.draw_geometries(
    [global_map, traj_line, axis],
    window_name="KISS-ICP Map"
)