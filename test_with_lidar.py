import numpy as np
import zarr
import matplotlib.pyplot as plt
from extended_kalman_filter import extended_kalman_filter


# ---------------------------------------------------------------
# TF: hesai_lidar → box_base → base
# ---------------------------------------------------------------
def quat_to_rot_tf(w, x, y, z):
    return np.array([
        [1-2*(y**2+z**2),   2*(x*y-z*w),     2*(x*z+y*w)],
        [  2*(x*y+z*w),   1-2*(x**2+z**2),   2*(y*z-x*w)],
        [  2*(x*z-y*w),     2*(y*z+x*w),   1-2*(x**2+y**2)]
    ])

def make_T(rw, rx, ry, rz, tx, ty, tz):
    T = np.eye(4)
    T[:3, :3] = quat_to_rot_tf(rw, rx, ry, rz)
    T[:3,  3] = [tx, ty, tz]
    return T

T_hesai_in_boxbase = make_T(
    -0.004419949690193552, -0.7093301791430758, 0.7048516039326548, 0.003921407292290494,
    -0.04461184951630229,   0.3022381420105506, -0.01253994548350209)
T_boxbase_in_base = make_T(
    4.0026794885936924e-05, -0.9999999991989279, 0.0, 0.0,
    -0.0764038, -0.036122437224394885, 0.2802761091673168)
R_hesai_in_base = (T_boxbase_in_base @ T_hesai_in_boxbase)[:3, :3]


# ---------------------------------------------------------------
# SVD rigid-body alignment: find R, t such that dst ≈ R @ src + t
# src, dst: (N, 3)
# ---------------------------------------------------------------
def svd_align(src, dst):
    mu_src = src.mean(0);  mu_dst = dst.mean(0)
    H = (src - mu_src).T @ (dst - mu_dst)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = mu_dst - R @ mu_src
    return R, t   # apply as: dst_approx = (R @ src.T).T + t


# ---------------------------------------------------------------
# GT outlier filter
# ---------------------------------------------------------------
def filter_gt(pos, ts, max_vel=3.0):
    dt  = np.diff(ts)
    vel = np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.maximum(dt, 1e-6)
    good = np.ones(len(pos), dtype=bool)
    good[1:]  &= vel < max_vel
    good[:-1] &= vel < max_vel
    return pos[good], ts[good]


class System:
    def __init__(self):
        self.Q = np.eye(10) * 0.01

    def A(self, x, u, dt):
        A = np.eye(10)
        A[0,3]=dt; A[1,4]=dt; A[2,5]=dt
        qw,qx,qy,qz = x[6],x[7],x[8],x[9]
        ax,ay,az = u[0],u[1],u[2]
        A[3,6]=2*( qw*ax-qz*ay+qy*az)*dt; A[4,6]=2*( qz*ax+qw*ay-qx*az)*dt; A[5,6]=2*(-qy*ax+qx*ay+qw*az)*dt
        A[3,7]=2*( qx*ax+qy*ay+qz*az)*dt; A[4,7]=2*( qy*ax-qx*ay-qw*az)*dt; A[5,7]=2*( qz*ax+qw*ay-qx*az)*dt
        A[3,8]=2*(-qy*ax+qx*ay+qw*az)*dt; A[4,8]=2*( qx*ax+qy*ay+qz*az)*dt; A[5,8]=2*(-qw*ax+qz*ay-qy*az)*dt
        A[3,9]=2*(-qz*ax-qw*ay+qx*az)*dt; A[4,9]=2*( qw*ax-qz*ay+qy*az)*dt; A[5,9]=2*( qx*ax+qy*ay+qz*az)*dt
        return A


# -----------------------------
# 1. 读取数据
# -----------------------------
imu_data = zarr.open(
    r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/stim320_imu/stim320_imu", mode='r')
imu_lin_acc     = imu_data['lin_acc'][:]
imu_ang_vel     = imu_data['ang_vel'][:]
imu_ang_vel_cov = imu_data['ang_vel_cov'][:]
imu_lin_acc_cov = imu_data['lin_acc_cov'][:]
imu_timestamp   = imu_data['timestamp'][:]

odom_data = zarr.open(
    r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/anymal_state_odometry/anymal_state_odometry", mode='r')
odom_pose_pos  = odom_data['pose_pos'][:]
odom_timestamp = odom_data['timestamp'][:]

gt_data      = zarr.open(
    r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/prism_position/prism_position", mode='r')
gt_pos       = gt_data['point'][:]
gt_timestamp = gt_data['timestamp'][:]

def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def transform_lidar_poses_to_base(lidar_poses, T_lidar_base):
    base_positions = []
    for k in range(len(lidar_poses)):
        T_world_lidar = lidar_poses[k]
        T_world_base = T_world_lidar @ T_lidar_base
        base_positions.append(T_world_base[:3, 3])
    return np.array(base_positions)

# 读取完整 pose
lidar_poses = np.load(
    r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/results/latest/kiss_icp_input_poses.npy"
)

lidar_zarr = zarr.open(
    r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/lidar/hesai_points_undistorted/hesai_points_undistorted",
    mode='r'
)
lidar_timestamp = lidar_zarr['timestamp'][:]

n_lidar = min(len(lidar_poses), len(lidar_timestamp))
lidar_poses = lidar_poses[:n_lidar]
lidar_timestamp = lidar_timestamp[:n_lidar]

T_base_lidar = T_boxbase_in_base @ T_hesai_in_boxbase
T_lidar_base = invert_T(T_base_lidar)

lidar_base = transform_lidar_poses_to_base(lidar_poses, T_lidar_base)

# 可选：如果你只是为了画图比较，可以再做一次首点对齐
lidar_aligned = lidar_base + (odom_pose_pos[0] - lidar_base[0])

print(f"After TF alignment:")
print(f"  odom  0→100: {odom_pose_pos[100] - odom_pose_pos[0]}")
print(f"  lidar 0→100: {lidar_aligned[100]  - lidar_aligned[0]}")


# -----------------------------
# 3. GT 过滤 + SVD 对齐到 odom 坐标系
# -----------------------------
gt_clean, gt_ts_clean = filter_gt(gt_pos, gt_timestamp, max_vel=3.0)
print(f"\nGT after outlier filter: {len(gt_clean)} / {len(gt_pos)} pts kept")

# 在时间重叠区间内，把 GT 插值到 odom 时间轴，然后用 SVD 求变换
t_start   = max(gt_ts_clean[0], odom_timestamp[0])
t_end     = min(gt_ts_clean[-1], odom_timestamp[-1])
mask_o    = (odom_timestamp >= t_start) & (odom_timestamp <= t_end)

odom_overlap = odom_pose_pos[mask_o]
t_overlap    = odom_timestamp[mask_o]

gt_interp = np.stack([
    np.interp(t_overlap, gt_ts_clean, gt_clean[:, i])
    for i in range(3)
], axis=1)

# SVD: GT → odom frame
R_gt, t_gt = svd_align(gt_interp, odom_overlap)
gt_in_odom = (R_gt @ gt_clean.T).T + t_gt

# 验证对齐质量
gt_check   = (R_gt @ gt_interp.T).T + t_gt
residual   = np.linalg.norm(gt_check - odom_overlap, axis=1)
print(f"SVD alignment residual: mean={residual.mean():.3f}m, max={residual.max():.3f}m")


# -----------------------------
# 4. 初始条件
# -----------------------------
class Init:
    def __init__(self):
        v0 = (odom_pose_pos[1] - odom_pose_pos[0]) / (odom_timestamp[1] - odom_timestamp[0])
        self.x = np.array([*odom_pose_pos[0], *v0, 1., 0., 0., 0.])
        self.Sigma = np.eye(10) * 0.1

def process_noise_from_imu(acc_cov, gyro_cov, dt):
    Q = np.zeros((10, 10))
    Q[0:3, 0:3]   = 0.25 * acc_cov * dt**4
    Q[3:6, 3:6]   = acc_cov * dt**2
    Q[6:10, 6:10] = np.eye(4) * np.mean(np.diag(gyro_cov)) * dt**2
    return Q


# -----------------------------
# 5. EKF
# -----------------------------
system = System()
init   = Init()
ekf    = extended_kalman_filter(system, init)

trajectory = []
lidar_idx  = 0

for k in range(1, len(imu_timestamp)):
    u  = np.hstack((imu_lin_acc[k], imu_ang_vel[k]))
    dt = imu_timestamp[k] - imu_timestamp[k - 1]
    ekf.Q = process_noise_from_imu(imu_lin_acc_cov[k], imu_ang_vel_cov[k], dt)
    ekf.prediction(u, dt)

    while lidar_idx < n_lidar and lidar_timestamp[lidar_idx] <= imu_timestamp[k]:
        ekf.correction_lidar(lidar_aligned[lidar_idx])
        lidar_idx += 1

    trajectory.append(ekf.x[0:3].copy())

trajectory = np.array(trajectory)
print(f"\nfinal position   = {ekf.x[0:3]}")
print(f"quat norm        = {np.linalg.norm(ekf.x[6:10]):.6f}")
print(f"lidar corrections= {lidar_idx} / {n_lidar}")


# -----------------------------
# 6. 画图（统一起点）
# -----------------------------
ekf_plot   = trajectory    - trajectory[0]
lidar_plot = lidar_aligned - lidar_aligned[0]
odom_plot  = odom_pose_pos - odom_pose_pos[0]
gt_plot    = gt_in_odom    - gt_in_odom[0]

fig = plt.figure(figsize=(12, 8))
ax  = fig.add_subplot(111, projection='3d')

ax.plot(ekf_plot[:, 0],   ekf_plot[:, 1],   ekf_plot[:, 2],
        label="EKF (IMU + LiDAR)", color='blue', linewidth=1.5)
ax.plot(lidar_plot[:, 0], lidar_plot[:, 1], lidar_plot[:, 2],
        label=f"LiDAR KISS-ICP ({n_lidar} pts)", color='green', linewidth=1.2)
ax.plot(odom_plot[:, 0],  odom_plot[:, 1],  odom_plot[:, 2],
        label=f"Odometry ({len(odom_pose_pos)} pts)", color='orange', linewidth=1.0, linestyle='--')
ax.plot(gt_plot[:, 0],    gt_plot[:, 1],    gt_plot[:, 2],
        label=f"Ground Truth ({len(gt_in_odom)} pts)", color='red', linewidth=1.5, linestyle='-.')

ax.scatter(*ekf_plot[0],  s=80, marker='o', color='blue', zorder=5)
ax.scatter(*gt_plot[0],   s=80, marker='*', color='red',  zorder=5)

ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
ax.set_title('Trajectory Comparison (TF-corrected + SVD GT alignment)')
ax.legend()
plt.tight_layout()
plt.show()