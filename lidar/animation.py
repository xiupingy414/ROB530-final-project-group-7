import zarr
import numpy as np
import open3d as o3d
import time

# 1. 打开数据
z = zarr.open(
    r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/lidar/hesai_points_undistorted/hesai_points_undistorted",
    mode='r'
)

all_points = z['points']
num_frames = all_points.shape[0]

print("num_frames =", num_frames)
print("single frame shape =", all_points[0].shape)

# 2. 先取第0帧
points = all_points[0]

# 可选：过滤无效点
mask = ~np.all(points == 0, axis=1)
points = points[mask]
mask = np.isfinite(points).all(axis=1)
points = points[mask]

# 3. 创建点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 4. 创建可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="LiDAR Animation", width=1200, height=800)
vis.add_geometry(pcd)

# 可选：设置点大小
render_option = vis.get_render_option()
render_option.point_size = 2.0

# 5. 播放动画
for i in range(num_frames):
    points = all_points[i]

    # 过滤
    mask = ~np.all(points == 0, axis=1)
    points = points[mask]
    mask = np.isfinite(points).all(axis=1)
    points = points[mask]

    # 更新点云
    pcd.points = o3d.utility.Vector3dVector(points)

    # 刷新显示
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    # 控制播放速度：0.1秒约等于10Hz
    time.sleep(0.1)

vis.destroy_window()