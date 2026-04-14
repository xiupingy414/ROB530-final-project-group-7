import numpy as np
import zarr
import matplotlib.pyplot as plt
from extended_kalman_filter import extended_kalman_filter
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------
# TF utilities
# ---------------------------------------------------------------
def quat_to_matrix(qw, qx, qy, qz):
    """四元数转旋转矩阵"""
    return R.from_quat([qx, qy, qz, qw]).as_matrix()


def make_T(rw, rx, ry, rz, tx, ty, tz):
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(rw, rx, ry, rz)
    T[:3, 3] = [tx, ty, tz]
    return T


def invert_T(T):
    R_mat = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_mat.T
    T_inv[:3, 3] = -R_mat.T @ t
    return T_inv


def apply_rigid(points, R_mat, t):
    return (R_mat @ points.T).T + t


# ---------------------------------------------------------------
# 正确的坐标变换（关键修改！）
# ---------------------------------------------------------------
# 根据TF树，正确的链应该是：
# base -> box_base (因为base是父帧，box_base是子帧)
# 所以 T_base_to_boxbase 才是正确的方向

# 从JSON：box_base相对于base的变换（这就是你有的）
T_boxbase_in_base = make_T(
    4.0026794885936924e-05, -0.9999999991989279, 0.0, 0.0,
    -0.0764038, -0.036122437224394885, 0.2802761091673168
)

# 所以 base -> box_base 是逆变换
T_base_to_boxbase = invert_T(T_boxbase_in_base)

# hesai_lidar -> box_base
T_lidar_in_boxbase = make_T(
    -0.004419949690193552, -0.7093301791430758, 0.7048516039326548, 0.003921407292290494,
    -0.04461184951630229, 0.3022381420105506, -0.01253994548350209
)

# 正确的链：lidar -> base = (base->boxbase) @ (boxbase->lidar)? 
# 等等，需要小心顺序

# 如果要得到 lidar_frame 中的点在 base_frame 中的坐标：
# p_base = T_base_to_boxbase @ T_boxbase_to_lidar? 不对

# 正确理解：
# T_lidar_in_boxbase 表示：点在lidar帧，转换到boxbase帧
# T_boxbase_in_base 表示：点在boxbase帧，转换到base帧
# 所以：点在lidar -> boxbase -> base
T_lidar_to_base = T_boxbase_in_base @ T_lidar_in_boxbase

# IMU: stim320_imu -> box_base
T_imu_in_boxbase = make_T(
    -0.005861732683034935, -0.999969253444768, -0.005117285966468607, 0.0009724399227390018,
    -0.28505604358365294, -0.07158336465343071, 0.15961588385259787
)

# IMU -> base
T_imu_to_base = T_boxbase_in_base @ T_imu_in_boxbase

print("=== 变换矩阵验证 ===")
print(f"T_lidar_to_base translation: {T_lidar_to_base[:3, 3]}")
print(f"T_imu_to_base translation: {T_imu_to_base[:3, 3]}")
print(f"T_base_to_boxbase translation: {T_base_to_boxbase[:3, 3]}")


# ---------------------------------------------------------------
# 简单的验证：检查第一帧LiDAR和Odometry的位置差异
# ---------------------------------------------------------------
def quick_check(lidar_first, odom_first):
    diff = lidar_first - odom_first
    print(f"\n第一帧差异: {diff}")
    print(f"距离: {np.linalg.norm(diff):.2f} m")
    if np.linalg.norm(diff) > 10:
        print("警告：差异过大，坐标系可能有问题！")


# ---------------------------------------------------------------
# 读取数据（保持不变）
# ---------------------------------------------------------------
print("\n加载数据...")
imu_data = zarr.open(
    r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/stim320_imu/stim320_imu", mode='r'
)
imu_lin_acc = imu_data['lin_acc'][:]
imu_ang_vel = imu_data['ang_vel'][:]
imu_ang_vel_cov = imu_data['ang_vel_cov'][:]
imu_lin_acc_cov = imu_data['lin_acc_cov'][:]
imu_timestamp = imu_data['timestamp'][:]

odom_data = zarr.open(
    r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/anymal_state_odometry/anymal_state_odometry", mode='r'
)
odom_pose_pos = odom_data['pose_pos'][:]
odom_timestamp = odom_data['timestamp'][:]

gt_data = zarr.open(
    r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/prism_position/prism_position", mode='r'
)
gt_pos = gt_data['point'][:]
gt_timestamp = gt_data['timestamp'][:]

lidar_poses = np.load(
    r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/results/latest/kiss_icp_input_poses.npy"
)

lidar_zarr = zarr.open(
    r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/lidar/hesai_points_undistorted/hesai_points_undistorted", mode='r'
)
lidar_timestamp = lidar_zarr['timestamp'][:]

n_lidar = min(len(lidar_poses), len(lidar_timestamp))
lidar_poses = lidar_poses[:n_lidar]
lidar_timestamp = lidar_timestamp[:n_lidar]

print(f"IMU: {len(imu_timestamp)} pts, Odom: {len(odom_timestamp)} pts, LiDAR: {n_lidar} pts, GT: {len(gt_timestamp)} pts")


# ---------------------------------------------------------------
# 变换LiDAR到base帧
# ---------------------------------------------------------------
def transform_lidar_to_base(lidar_poses, T_lidar_to_base):
    """将LiDAR位姿转换到base帧"""
    lidar_in_base = []
    for T_w_l in lidar_poses:
        # T_w_l 是 LiDAR 在世界帧的位姿（KISS-ICP输出）
        # 我们要得到 base 在世界帧的位姿
        # T_w_b = T_w_l @ T_lidar_to_base
        T_w_b = T_w_l @ T_lidar_to_base
        lidar_in_base.append(T_w_b[:3, 3])
    return np.array(lidar_in_base)


lidar_in_base = transform_lidar_to_base(lidar_poses, T_lidar_to_base)

# 检查
quick_check(lidar_in_base[0], odom_pose_pos[0])


# ---------------------------------------------------------------
# 变换IMU到base帧
# ---------------------------------------------------------------
def transform_imu_to_base(lin_acc, ang_vel, T_imu_to_base):
    R_mat = T_imu_to_base[:3, :3]
    lin_acc_base = (R_mat @ lin_acc.T).T
    ang_vel_base = (R_mat @ ang_vel.T).T
    return lin_acc_base, ang_vel_base


imu_lin_acc_base, imu_ang_vel_base = transform_imu_to_base(
    imu_lin_acc, imu_ang_vel, T_imu_to_base
)


# ---------------------------------------------------------------
# 简单对齐：让LiDAR和EKF的起点与Odometry对齐（仅用于可视化）
# ---------------------------------------------------------------
# 注意：这只是为了让它们在同一个起点开始，不改变相对运动
lidar_start_offset = odom_pose_pos[0] - lidar_in_base[0]
lidar_aligned_for_plot = lidar_in_base + lidar_start_offset

# EKF的初始状态也使用odometry
class Init:
    def __init__(self):
        v0 = (odom_pose_pos[1] - odom_pose_pos[0]) / (odom_timestamp[1] - odom_timestamp[0])
        self.x = np.array([
            odom_pose_pos[0][0], odom_pose_pos[0][1], odom_pose_pos[0][2],
            v0[0], v0[1], v0[2],
            1.0, 0.0, 0.0, 0.0
        ])
        self.Sigma = np.eye(10) * 0.1


# ---------------------------------------------------------------
# EKF System
# ---------------------------------------------------------------
class System:
    def __init__(self):
        self.Q = np.eye(10) * 0.01

    def A(self, x, u, dt):
        A = np.eye(10)
        A[0, 3] = dt
        A[1, 4] = dt
        A[2, 5] = dt

        qw, qx, qy, qz = x[6], x[7], x[8], x[9]
        ax, ay, az = u[0], u[1], u[2]

        A[3, 6] = 2 * (qw * ax - qz * ay + qy * az) * dt
        A[4, 6] = 2 * (qz * ax + qw * ay - qx * az) * dt
        A[5, 6] = 2 * (-qy * ax + qx * ay + qw * az) * dt

        A[3, 7] = 2 * (qx * ax + qy * ay + qz * az) * dt
        A[4, 7] = 2 * (qy * ax - qx * ay - qw * az) * dt
        A[5, 7] = 2 * (qz * ax + qw * ay - qx * az) * dt

        A[3, 8] = 2 * (-qy * ax + qx * ay + qw * az) * dt
        A[4, 8] = 2 * (qx * ax + qy * ay + qz * az) * dt
        A[5, 8] = 2 * (-qw * ax + qz * ay - qy * az) * dt

        A[3, 9] = 2 * (-qz * ax - qw * ay + qx * az) * dt
        A[4, 9] = 2 * (qw * ax - qz * ay + qy * az) * dt
        A[5, 9] = 2 * (qx * ax + qy * ay + qz * az) * dt

        return A


def process_noise_from_imu(acc_cov, gyro_cov, dt):
    Q = np.zeros((10, 10))
    Q[0:3, 0:3] = 0.25 * acc_cov * dt**4
    Q[3:6, 3:6] = acc_cov * dt**2
    Q[6:10, 6:10] = np.eye(4) * np.mean(np.diag(gyro_cov)) * dt**2
    return Q


# ---------------------------------------------------------------
# 运行EKF
# ---------------------------------------------------------------
system = System()
init = Init()
ekf = extended_kalman_filter(system, init)

trajectory = []
lidar_idx = 0

print("\n运行EKF...")
for k in range(1, min(len(imu_timestamp), 10000)):  # 限制长度便于调试
    u = np.hstack((imu_lin_acc_base[k], imu_ang_vel_base[k]))
    dt = imu_timestamp[k] - imu_timestamp[k - 1]
    
    if dt <= 0 or dt > 0.05:
        continue

    ekf.Q = process_noise_from_imu(imu_lin_acc_cov[k], imu_ang_vel_cov[k], dt)
    ekf.prediction(u, dt)

    while lidar_idx < n_lidar and lidar_timestamp[lidar_idx] <= imu_timestamp[k]:
        # 使用对齐后的LiDAR进行校正（起点对齐，保持相对运动）
        ekf.correction_lidar(lidar_aligned_for_plot[lidar_idx])
        lidar_idx += 1

    trajectory.append(ekf.x[0:3].copy())

trajectory = np.array(trajectory)


# ---------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 减去起点便于比较
odom_plot = odom_pose_pos - odom_pose_pos[0]
lidar_plot = lidar_aligned_for_plot - lidar_aligned_for_plot[0]
ekf_plot = trajectory - trajectory[0] if len(trajectory) > 0 else np.array([])
gt_plot = gt_pos - gt_pos[0]

ax.plot(ekf_plot[:, 0], ekf_plot[:, 1], ekf_plot[:, 2], 
        label='EKF (IMU + LiDAR)', color='blue', linewidth=2)
ax.plot(lidar_plot[:, 0], lidar_plot[:, 1], lidar_plot[:, 2], 
        label='LiDAR KISS-ICP', color='green', linewidth=1, alpha=0.7)
ax.plot(odom_plot[:, 0], odom_plot[:, 1], odom_plot[:, 2], 
        label='Odometry', color='orange', linewidth=1.5, linestyle='--')
ax.plot(gt_plot[:, 0], gt_plot[:, 1], gt_plot[:, 2], 
        label='Ground Truth', color='red', linewidth=2, linestyle='-.')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Trajectory Comparison')
ax.legend()
plt.show()

print(f"\n轨迹范围:")
print(f"Odometry: X [{odom_plot[:,0].min():.1f}, {odom_plot[:,0].max():.1f}]")
print(f"LiDAR:    X [{lidar_plot[:,0].min():.1f}, {lidar_plot[:,0].max():.1f}]")
print(f"GT:       X [{gt_plot[:,0].min():.1f}, {gt_plot[:,0].max():.1f}]")