import numpy as np
import zarr
import matplotlib.pyplot as plt
from extended_kalman_filter import extended_kalman_filter


# ---------------------------------------------------------------
# TF utilities
# ---------------------------------------------------------------
def quat_to_rot_tf(w, x, y, z):
    return np.array([
        [1 - 2 * (y**2 + z**2),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x**2 + z**2),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
    ])


def make_T(rw, rx, ry, rz, tx, ty, tz):
    T = np.eye(4)
    T[:3, :3] = quat_to_rot_tf(rw, rx, ry, rz)
    T[:3, 3] = [tx, ty, tz]
    return T


def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def apply_rigid(points, R, t):
    return (R @ points.T).T + t


def invert_rigid(R, t):
    R_inv = R.T
    t_inv = -R.T @ t
    return R_inv, t_inv


# ---------------------------------------------------------------
# Static TF: hesai_lidar -> box_base -> base
# ---------------------------------------------------------------
T_hesai_in_boxbase = make_T(
    -0.004419949690193552, -0.7093301791430758, 0.7048516039326548, 0.003921407292290494,
    -0.04461184951630229,   0.3022381420105506, -0.01253994548350209
)

T_boxbase_in_base = make_T(
    4.0026794885936924e-05, -0.9999999991989279, 0.0, 0.0,
    -0.0764038, -0.036122437224394885, 0.2802761091673168
)

T_base_lidar = T_boxbase_in_base @ T_hesai_in_boxbase
T_lidar_base = invert_T(T_base_lidar)


# ---------------------------------------------------------------
# SVD rigid-body alignment
# find R, t such that dst ≈ R @ src + t
# ---------------------------------------------------------------
def svd_align(src, dst):
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    H = (src - mu_src).T @ (dst - mu_dst)
    U, _, Vt = np.linalg.svd(H)

    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = mu_dst - R @ mu_src
    return R, t


# ---------------------------------------------------------------
# GT outlier filter
# ---------------------------------------------------------------
def filter_gt(pos, ts, max_vel=3.0):
    dt = np.diff(ts)
    vel = np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.maximum(dt, 1e-6)

    good = np.ones(len(pos), dtype=bool)
    good[1:] &= vel < max_vel
    good[:-1] &= vel < max_vel
    return pos[good], ts[good]


# ---------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------
def interp_traj_to_time(src_pos, src_ts, dst_ts):
    return np.stack([
        np.interp(dst_ts, src_ts, src_pos[:, 0]),
        np.interp(dst_ts, src_ts, src_pos[:, 1]),
        np.interp(dst_ts, src_ts, src_pos[:, 2]),
    ], axis=1)


# ---------------------------------------------------------------
# LiDAR pose convention trials
# ---------------------------------------------------------------
def apply_pose_chain(lidar_poses, mode, T_base_lidar, T_lidar_base):
    """
    Return base positions from LiDAR poses using different pose conventions.

    mode options:
        A: T_w_b = T_w_l @ T_lidar_base
        B: T_w_b = T_base_lidar @ T_w_l
        C: T_w_b = inv(T_w_l) @ T_lidar_base
        D: T_w_b = T_base_lidar @ inv(T_w_l)
    """
    out = []

    for k in range(len(lidar_poses)):
        T_w_l = lidar_poses[k]

        if mode == "A":
            T_w_b = T_w_l @ T_lidar_base
        elif mode == "B":
            T_w_b = T_base_lidar @ T_w_l
        elif mode == "C":
            T_w_b = invert_T(T_w_l) @ T_lidar_base
        elif mode == "D":
            T_w_b = T_base_lidar @ invert_T(T_w_l)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        out.append(T_w_b[:3, 3])

    return np.array(out)


def rigid_align_over_overlap(src_pos, src_ts, dst_pos, dst_ts):
    """
    Align src trajectory to dst trajectory over overlapping timestamps.
    Returns aligned full src, plus R, t, overlap residual stats.
    """
    t_start = max(src_ts[0], dst_ts[0])
    t_end = min(src_ts[-1], dst_ts[-1])

    mask_dst = (dst_ts >= t_start) & (dst_ts <= t_end)
    dst_overlap = dst_pos[mask_dst]
    t_overlap = dst_ts[mask_dst]

    src_interp = interp_traj_to_time(src_pos, src_ts, t_overlap)

    R, t = svd_align(src_interp, dst_overlap)
    src_aligned = apply_rigid(src_pos, R, t)

    src_overlap_aligned = apply_rigid(src_interp, R, t)
    residual = np.linalg.norm(src_overlap_aligned - dst_overlap, axis=1)

    return src_aligned, R, t, residual.mean(), residual.max()


def score_lidar_mode(lidar_pos, lidar_ts, odom_pos, odom_ts):
    _, R_tmp, t_tmp, mean_err, max_err = rigid_align_over_overlap(
        lidar_pos, lidar_ts, odom_pos, odom_ts
    )
    return mean_err, max_err, R_tmp, t_tmp


# ---------------------------------------------------------------
# EKF system
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

        A[3, 6] = 2 * ( qw * ax - qz * ay + qy * az) * dt
        A[4, 6] = 2 * ( qz * ax + qw * ay - qx * az) * dt
        A[5, 6] = 2 * (-qy * ax + qx * ay + qw * az) * dt

        A[3, 7] = 2 * ( qx * ax + qy * ay + qz * az) * dt
        A[4, 7] = 2 * ( qy * ax - qx * ay - qw * az) * dt
        A[5, 7] = 2 * ( qz * ax + qw * ay - qx * az) * dt

        A[3, 8] = 2 * (-qy * ax + qx * ay + qw * az) * dt
        A[4, 8] = 2 * ( qx * ax + qy * ay + qz * az) * dt
        A[5, 8] = 2 * (-qw * ax + qz * ay - qy * az) * dt

        A[3, 9] = 2 * (-qz * ax - qw * ay + qx * az) * dt
        A[4, 9] = 2 * ( qw * ax - qz * ay + qy * az) * dt
        A[5, 9] = 2 * ( qx * ax + qy * ay + qz * az) * dt

        return A


def process_noise_from_imu(acc_cov, gyro_cov, dt):
    Q = np.zeros((10, 10))
    Q[0:3, 0:3] = 0.25 * acc_cov * dt**4
    Q[3:6, 3:6] = acc_cov * dt**2
    Q[6:10, 6:10] = np.eye(4) * np.mean(np.diag(gyro_cov)) * dt**2
    return Q


# ---------------------------------------------------------------
# 1. Read data
# ---------------------------------------------------------------
imu_data = zarr.open(
    r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/s1/stim320_imu/stim320_imu",
    mode='r'
)
imu_lin_acc = imu_data['lin_acc'][:]
imu_ang_vel = imu_data['ang_vel'][:]
imu_ang_vel_cov = imu_data['ang_vel_cov'][:]
imu_lin_acc_cov = imu_data['lin_acc_cov'][:]
imu_timestamp = imu_data['timestamp'][:]

odom_data = zarr.open(
    r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/s1/anymal_state_odometry/anymal_state_odometry",
    mode='r'
)
odom_pose_pos = odom_data['pose_pos'][:]
odom_timestamp = odom_data['timestamp'][:]

gt_data = zarr.open(
    r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/s1/prism_position/prism_position",
    mode='r'
)
gt_pos = gt_data['point'][:]
gt_timestamp = gt_data['timestamp'][:]

lidar_poses = np.load(
    r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/s1/results/kiss_icp_input_poses.npy"
)

lidar_zarr = zarr.open(
    r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/s1/lidar/hesai_points_undistorted/hesai_points_undistorted",
    mode='r'
)
lidar_timestamp = lidar_zarr['timestamp'][:]

n_lidar = min(len(lidar_poses), len(lidar_timestamp))
lidar_poses = lidar_poses[:n_lidar]
lidar_timestamp = lidar_timestamp[:n_lidar]

print(f"IMU   : {len(imu_timestamp)} pts")
print(f"Odom  : {len(odom_timestamp)} pts")
print(f"LiDAR : {len(lidar_timestamp)} pts")
print(f"GT    : {len(gt_timestamp)} pts")


# ---------------------------------------------------------------
# 2. Clean GT
# ---------------------------------------------------------------
gt_clean, gt_ts_clean = filter_gt(gt_pos, gt_timestamp, max_vel=3.0)
print(f"\nGT after outlier filter: {len(gt_clean)} / {len(gt_pos)} pts kept")


# ---------------------------------------------------------------
# 3. LiDAR -> choose best pose convention, align to odom
# ---------------------------------------------------------------
candidate_modes = ["A", "B", "C", "D"]
mode_results = {}

for mode in candidate_modes:
    lidar_base_candidate = apply_pose_chain(
        lidar_poses, mode, T_base_lidar, T_lidar_base
    )

    mean_err, max_err, R_tmp, t_tmp = score_lidar_mode(
        lidar_base_candidate, lidar_timestamp, odom_pose_pos, odom_timestamp
    )

    mode_results[mode] = {
        "mean_err": mean_err,
        "max_err": max_err,
        "R_align": R_tmp,
        "t_align": t_tmp,
        "traj_raw": lidar_base_candidate,
    }

best_mode = min(mode_results, key=lambda m: mode_results[m]["mean_err"])
best = mode_results[best_mode]

print("\nLiDAR mode comparison:")
for mode in candidate_modes:
    print(
        f"  mode {mode}: mean={mode_results[mode]['mean_err']:.3f} m, "
        f"max={mode_results[mode]['max_err']:.3f} m"
    )

print(f"\nChosen LiDAR mode: {best_mode}")

lidar_base = best["traj_raw"]
lidar_in_odom = apply_rigid(lidar_base, best["R_align"], best["t_align"])

print("\nAfter TF + LiDAR->odom rigid alignment:")
print("  odom  0→100:", odom_pose_pos[100] - odom_pose_pos[0])
print("  lidar 0→100:", lidar_in_odom[100] - lidar_in_odom[0])


# ---------------------------------------------------------------
# 4. GT -> odom alignment
#    find R_gt_to_odom, t_gt_to_odom such that:
#    odom ≈ R_gt_to_odom @ gt + t_gt_to_odom
# ---------------------------------------------------------------
gt_in_odom, R_gt_to_odom, t_gt_to_odom, gt_mean_res, gt_max_res = rigid_align_over_overlap(
    gt_clean, gt_ts_clean, odom_pose_pos, odom_timestamp
)

print(f"\nGT -> odom alignment residual: mean={gt_mean_res:.3f} m, max={gt_max_res:.3f} m")

# inverse: odom -> GT
R_odom_to_gt, t_odom_to_gt = invert_rigid(R_gt_to_odom, t_gt_to_odom)

# move odom-like trajectories into GT frame
odom_in_gt = apply_rigid(odom_pose_pos, R_odom_to_gt, t_odom_to_gt)
lidar_in_gt = apply_rigid(lidar_in_odom, R_odom_to_gt, t_odom_to_gt)

# ---------------------------------------------------------------
# 5. EKF in odom-like frame
# ---------------------------------------------------------------
class Init:
    def __init__(self):
        v0 = (odom_pose_pos[1] - odom_pose_pos[0]) / (odom_timestamp[1] - odom_timestamp[0])
        self.x = np.array([
            odom_pose_pos[0][0], odom_pose_pos[0][1], odom_pose_pos[0][2],
            v0[0], v0[1], v0[2],
            1.0, 0.0, 0.0, 0.0
        ])
        self.Sigma = np.eye(10) * 0.1


system = System()
init = Init()
ekf = extended_kalman_filter(system, init)

trajectory = []
lidar_idx = 0

for k in range(1, len(imu_timestamp)):
    u = np.hstack((imu_lin_acc[k], imu_ang_vel[k]))
    dt = imu_timestamp[k] - imu_timestamp[k - 1]

    ekf.Q = process_noise_from_imu(imu_lin_acc_cov[k], imu_ang_vel_cov[k], dt)
    ekf.prediction(u, dt)

    while lidar_idx < n_lidar and lidar_timestamp[lidar_idx] <= imu_timestamp[k]:
        ekf.correction_lidar(lidar_in_odom[lidar_idx])
        lidar_idx += 1

    trajectory.append(ekf.x[0:3].copy())

trajectory = np.array(trajectory)

print(f"\nEKF final position (odom frame) = {ekf.x[0:3]}")
print(f"quat norm                      = {np.linalg.norm(ekf.x[6:10]):.6f}")
print(f"lidar corrections              = {lidar_idx} / {n_lidar}")

# move EKF to GT frame
ekf_in_gt = apply_rigid(trajectory, R_odom_to_gt, t_odom_to_gt)


# ---------------------------------------------------------------
# 6. Optional residual checks in GT frame
# ---------------------------------------------------------------
# odom vs gt over overlap
_, _, _, odom_gt_mean_res, odom_gt_max_res = rigid_align_over_overlap(
    odom_in_gt, odom_timestamp, gt_clean, gt_ts_clean
)

# lidar vs gt over overlap
_, _, _, lidar_gt_mean_res, lidar_gt_max_res = rigid_align_over_overlap(
    lidar_in_gt, lidar_timestamp, gt_clean, gt_ts_clean
)

# ekf vs gt over overlap
ekf_timestamp = imu_timestamp[1:]  # because trajectory starts from k=1
_, _, _, ekf_gt_mean_res, ekf_gt_max_res = rigid_align_over_overlap(
    ekf_in_gt, ekf_timestamp, gt_clean, gt_ts_clean
)

print(f"\nResiduals in GT frame:")
print(f"  odom  vs GT: mean={odom_gt_mean_res:.3f} m, max={odom_gt_max_res:.3f} m")
print(f"  lidar vs GT: mean={lidar_gt_mean_res:.3f} m, max={lidar_gt_max_res:.3f} m")
print(f"  ekf   vs GT: mean={ekf_gt_mean_res:.3f} m, max={ekf_gt_max_res:.3f} m")


# ---------------------------------------------------------------
# 7. Plot all four trajectories in GT frame
# ---------------------------------------------------------------
gt_plot = gt_clean - gt_clean[0]
odom_plot = odom_in_gt - odom_in_gt[0]
lidar_plot = lidar_in_gt - lidar_in_gt[0]
ekf_plot = ekf_in_gt - ekf_in_gt[0]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(
    ekf_plot[:, 0], ekf_plot[:, 1], ekf_plot[:, 2],
    label="EKF (IMU + LiDAR)", color='blue', linewidth=1.5
)
ax.plot(
    lidar_plot[:, 0], lidar_plot[:, 1], lidar_plot[:, 2],
    label=f"LiDAR KISS-ICP ({n_lidar} pts)", color='green', linewidth=1.2
)
ax.plot(
    odom_plot[:, 0], odom_plot[:, 1], odom_plot[:, 2],
    label=f"Odometry ({len(odom_pose_pos)} pts)", color='orange', linewidth=1.0, linestyle='--'
)
ax.plot(
    gt_plot[:, 0], gt_plot[:, 1], gt_plot[:, 2],
    label=f"Ground Truth ({len(gt_clean)} pts)", color='red', linewidth=1.5, linestyle='-.'
)

ax.scatter(*ekf_plot[0], s=80, marker='o', color='blue', zorder=5)
ax.scatter(*gt_plot[0], s=80, marker='*', color='red', zorder=5)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Trajectory Comparison in GT Frame')
ax.legend()
plt.tight_layout()
plt.show()