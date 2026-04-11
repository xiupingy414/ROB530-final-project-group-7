import zarr
import numpy as np
import open3d as o3d

# 1. 打开数据
z = zarr.open(
    r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/lidar/hesai_points_undistorted/hesai_points_undistorted",
    mode='r'
)

all_points = z['points']
num_frames = all_points.shape[0]
print("min:", np.min(all_points[0], axis=0))
print("max:", np.max(all_points[0], axis=0))
print("num_frames =", num_frames)
print("single frame shape =", all_points[0].shape)

# -----------------------------
# 一些辅助函数
# -----------------------------
def make_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def preprocess_pcd(pcd, voxel_size=0.5):
    # 下采样，减少点数，让ICP更稳定更快
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return pcd_down

# -----------------------------
# 2. 初始化
# -----------------------------
# 全局地图
global_map = o3d.geometry.PointCloud()

# 当前累计位姿（世界坐标系下）
current_pose = np.eye(4)
T_init = np.eye(4)   # ICP初始位姿

# 第0帧作为初始帧
source_points = all_points[0]
source_pcd = make_pcd(source_points)
source_down = preprocess_pcd(source_pcd)

# 把第0帧先加入地图
source_world = make_pcd(source_points)
source_world.transform(current_pose)
global_map += source_world

# 保存轨迹（可选）
trajectory = [current_pose.copy()]

# ICP参数
threshold = 0.5   # 匹配距离阈值，后面可以调

# -----------------------------
# 3. 逐帧配准并累积建图
# -----------------------------
for i in range(1, num_frames):
    #print(f"Processing frame {i}/{num_frames-1}")

    target_points = all_points[i]
    target_pcd = make_pcd(target_points)
    target_down = preprocess_pcd(target_pcd)

    # ICP: source_down -> target_down
    reg = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        threshold,
        T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    T_source_to_target = reg.transformation

    #print("fitness =", reg.fitness, " rmse =", reg.inlier_rmse)
    #print("T =\n", T_source_to_target)

    # 累积位姿
    current_pose = current_pose @ np.linalg.inv(T_source_to_target)
    trajectory.append(current_pose.copy())

    # 把当前帧放到世界坐标系
    target_world = make_pcd(target_points)
    target_world.transform(current_pose)
    target_world = target_world.voxel_down_sample(voxel_size=1)

    # 加入全局地图
    if i % 50 == 0:
        global_map += target_world

    # 下一轮
    source_down = target_down
    T_init = T_source_to_target

# -----------------------------
# 4. 可视化最终地图
# -----------------------------
print("Mapping finished.")
# 可选：再做一次全局下采样，避免点太密
global_map = global_map.voxel_down_sample(voxel_size=1)
global_map.paint_uniform_color([0.7, 0.7, 0.7])
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=50,   # 坐标轴长度
    origin=[0, 0, 0]
)
# -----------------------------
# 生成轨迹线
# -----------------------------
traj_points = []
for pose in trajectory:
    traj_points.append(pose[:3, 3])   # 提取每个位姿的平移部分

traj_points = np.array(traj_points)

traj_line = o3d.geometry.LineSet()
traj_line.points = o3d.utility.Vector3dVector(traj_points)

lines = [[i, i + 1] for i in range(len(traj_points) - 1)]
traj_line.lines = o3d.utility.Vector2iVector(lines)

colors = [[1, 0, 0] for _ in range(len(lines))]   # 红色轨迹
traj_line.colors = o3d.utility.Vector3dVector(colors)

# 可视化
o3d.visualization.draw_geometries(
    [global_map, axis, traj_line],
    window_name="Global Map with Trajectory"
)