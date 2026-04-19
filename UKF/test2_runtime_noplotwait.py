import numpy as np
import open3d as o3d
import zarr
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import csv
import os
import time

# =========================================================
# 
# =========================================================
RUN_RADAR_ICP = True
RUN_UKF_FUSION = True

# =========================================================
# 
# =========================================================
project_dir = r"C:\Users\junxianw\Desktop\project1"

map_path = os.path.join(project_dir, "2024-11-02-17-10-25_dlio.ply")
radar_zarr_path = os.path.join(project_dir, "hesai_points_undistorted")
imu_path = os.path.join(project_dir, "stim320_imu")

# =========================================================
# 
# =========================================================
radar_output_dir = os.path.join(project_dir, "output_radar_full_only")
ukf_output_dir = os.path.join(project_dir, "output_ukf_full_only")

os.makedirs(radar_output_dir, exist_ok=True)
os.makedirs(ukf_output_dir, exist_ok=True)

# =========================================================
# 
# =========================================================
radar_full_npz = os.path.join(
    radar_output_dir, "radar_icp_tf_full_output_project1_v1.npz"
)
radar_full_csv = os.path.join(
    radar_output_dir, "radar_icp_tf_full_output_project1_v1.csv"
)

ukf_full_npz = os.path.join(
    ukf_output_dir, "ukf_fusion_full_output_project1_v1.npz"
)
ukf_full_csv = os.path.join(
    ukf_output_dir, "ukf_fusion_full_output_project1_v1.csv"
)

radar_plot_traj_csv = os.path.join(
    radar_output_dir, "radar_full_trajectory_coords_project1_v1.csv"
)
ukf_plot_traj_csv = os.path.join(
    ukf_output_dir, "ukf_full_trajectory_coords_project1_v1.csv"
)

full_overlay_fig_path = os.path.join(
    project_dir, "full_radar_ukf_overlay_project1_v1.png"
)

# =========================================================
# Runtime measurement
# =========================================================
total_start_time = time.time()
algorithm_start_time = None
processed_lidar_frames = 0

# =========================================================
# ICP 
# =========================================================
voxel_size = 0.3
max_corr_dist = 0.4
max_iteration = 50

# =========================================================
# UKF 
# 状态 x = [px, py, pz, vx, vy, vz]^T
# 测量 z = [px, py, pz]^T
# =========================================================
state_dim = 6
meas_dim = 3
kappa_f = 2.0
kappa_h = 2.0

Q = np.diag([
    1e-4, 1e-4, 1e-4,
    1e-2, 1e-2, 1e-2
])

R_meas = np.diag([
    0.05**2,
    0.05**2,
    0.08**2
])

Sigma0 = np.diag([
    1e-3, 1e-3, 1e-3,
    1e-2, 1e-2, 1e-2
])

# =========================================================
# 
# =========================================================
def make_T(tx, ty, tz, qx, qy, qz, qw):
    T = np.eye(4)
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = np.array([tx, ty, tz], dtype=float)
    return T

def T_to_pose(T):
    pos = T[:3, 3].copy()
    quat = R.from_matrix(T[:3, :3]).as_quat()
    return pos, quat

def sigma_points(mu, Sigma, kappa):
    n = mu.shape[0]
    L = np.sqrt(n + kappa) * np.linalg.cholesky(Sigma)
    Y = np.repeat(mu, n, axis=1)
    X = np.hstack((mu, Y + L, Y - L))

    w = np.zeros(2 * n + 1)
    w[0] = kappa / (n + kappa)
    w[1:] = 1.0 / (2.0 * (n + kappa))
    return X, w

def process_model(x, a_world, dt, stationary=False):
    x = x.reshape(6, 1)

    p = x[0:3, 0]
    v = x[3:6, 0]

    if stationary:
        v_new = np.zeros(3)
    else:
        v_new = v + a_world * dt

    p_new = p + v * dt + 0.5 * a_world * dt * dt

    out = np.zeros((6, 1))
    out[0:3, 0] = p_new
    out[3:6, 0] = v_new
    return out

def measurement_model(x):
    x = x.reshape(6, 1)
    return x[0:3].reshape(3, 1)

def ukf_prediction_step(mu_prev, Sigma_prev, a_world, dt, Q, stationary=False, kappa=2.0):
    X, w = sigma_points(mu_prev, Sigma_prev, kappa)

    Y_list = []
    for i in range(X.shape[1]):
        y_i = process_model(X[:, i].reshape(-1, 1), a_world, dt, stationary)
        Y_list.append(y_i)

    Y = np.hstack(Y_list)

    mu_pred = np.zeros((state_dim, 1))
    for i in range(Y.shape[1]):
        mu_pred += w[i] * Y[:, i].reshape(-1, 1)

    Sigma_pred = np.zeros((state_dim, state_dim))
    for i in range(Y.shape[1]):
        diff = Y[:, i].reshape(-1, 1) - mu_pred
        Sigma_pred += w[i] * (diff @ diff.T)

    Sigma_pred += Q
    return mu_pred, Sigma_pred

def ukf_correction_step(mu_pred, Sigma_pred, z_meas, R_meas, kappa=2.0):
    X, w = sigma_points(mu_pred, Sigma_pred, kappa)

    Z_list = []
    for i in range(X.shape[1]):
        z_i = measurement_model(X[:, i].reshape(-1, 1))
        Z_list.append(z_i)

    Z = np.hstack(Z_list)

    z_hat = np.zeros((meas_dim, 1))
    for i in range(Z.shape[1]):
        z_hat += w[i] * Z[:, i].reshape(-1, 1)

    S = np.zeros((meas_dim, meas_dim))
    for i in range(Z.shape[1]):
        dz = Z[:, i].reshape(-1, 1) - z_hat
        S += w[i] * (dz @ dz.T)
    S += R_meas

    Cov_xz = np.zeros((state_dim, meas_dim))
    for i in range(X.shape[1]):
        dx = X[:, i].reshape(-1, 1) - mu_pred
        dz = Z[:, i].reshape(-1, 1) - z_hat
        Cov_xz += w[i] * (dx @ dz.T)

    K = Cov_xz @ np.linalg.pinv(S)
    v = z_meas.reshape(meas_dim, 1) - z_hat

    mu_corr = mu_pred + K @ v
    Sigma_corr = Sigma_pred - K @ S @ K.T

    return mu_corr, Sigma_corr, z_hat, K, v, S

# =========================================================
# TF
# base_frame_id = box_base
# child_frame_id = hesai_lidar / stim320_imu
# =========================================================
T_box_hesai = make_T(
    tx=-0.04461184951630229,
    ty=0.3022381420105506,
    tz=-0.01253994548350209,
    qx=-0.7093301791430758,
    qy=0.7048516039326548,
    qz=0.003921407292290494,
    qw=-0.004419949690193552
)

T_box_stim = make_T(
    tx=-0.28505604358365294,
    ty=-0.07158336465343071,
    tz=0.15961588385259787,
    qx=-0.999969253444768,
    qy=-0.005117285966486087,
    qz=0.0009724399227390018,
    qw=-0.005861732683034935
)

T_hesai_stim = np.linalg.inv(T_box_hesai) @ T_box_stim

# =========================================================
# radar ICP + tf
# =========================================================
if algorithm_start_time is None:
    algorithm_start_time = time.time()

if RUN_RADAR_ICP:
    print("\n================ RADAR ICP + TF (FULL) ================\n")

    map_pcd = o3d.io.read_point_cloud(map_path)
    print(f"Map points: {len(map_pcd.points)}")

    map_pcd_down = map_pcd.voxel_down_sample(voxel_size)
    map_pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    print(f"Map: {len(map_pcd.points)} -> {len(map_pcd_down.points)} points")

    z = zarr.open(radar_zarr_path, mode='r')
    print("Radar keys:", list(z.keys()))
    num_frames = z["timestamp"].shape[0]
    processed_lidar_frames = num_frames
    print("Total radar frames:", num_frames)

    print("\nStatic transform T_hesai_stim:")
    print(T_hesai_stim)

    positions_hesai = []
    quaternions_hesai = []
    positions_stim = []
    quaternions_stim = []
    timestamps_out = []
    fitness_list = []
    rmse_list = []

    T_init = np.eye(4)

    for i in range(num_frames):
        n_valid = int(z['valid'][i, 0])
        scan_np = z['points'][i, :n_valid, :]

        scan_np = scan_np[np.isfinite(scan_np).all(axis=1)]
        scan_np = scan_np[np.linalg.norm(scan_np, axis=1) > 1e-6]

        scan_pcd = o3d.geometry.PointCloud()
        scan_pcd.points = o3d.utility.Vector3dVector(scan_np.astype(np.float64))

        result = o3d.pipelines.registration.registration_icp(
            source=scan_pcd,
            target=map_pcd_down,
            max_correspondence_distance=max_corr_dist,
            init=T_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
        )

        T_map_hesai = result.transformation
        T_init = T_map_hesai

        T_map_stim = T_map_hesai @ T_hesai_stim

        pos_h, quat_h = T_to_pose(T_map_hesai)
        pos_s, quat_s = T_to_pose(T_map_stim)

        positions_hesai.append(pos_h)
        quaternions_hesai.append(quat_h)
        positions_stim.append(pos_s)
        quaternions_stim.append(quat_s)
        timestamps_out.append(z['timestamp'][i])
        fitness_list.append(result.fitness)
        rmse_list.append(result.inlier_rmse)

        if i % 50 == 0:
            print(
                f"frame {i:4d} | valid={n_valid:5d} | "
                f"fitness={result.fitness:.4f} | rmse={result.inlier_rmse:.4f}"
            )

    positions_hesai = np.array(positions_hesai)
    quaternions_hesai = np.array(quaternions_hesai)
    positions_stim = np.array(positions_stim)
    quaternions_stim = np.array(quaternions_stim)
    timestamps_out = np.array(timestamps_out)
    fitness_list = np.array(fitness_list)
    rmse_list = np.array(rmse_list)

    positions_stim_rebased = positions_stim - positions_stim[0]

    np.savez(
        radar_full_npz,
        timestamp=timestamps_out,
        position_hesai=positions_hesai,
        quaternion_hesai=quaternions_hesai,
        position_stim320=positions_stim,
        quaternion_stim320=quaternions_stim,
        position_stim320_rebased=positions_stim_rebased,
        icp_fitness=fitness_list,
        icp_rmse=rmse_list
    )
    print(f"\nSaved NPZ to: {radar_full_npz}")

    with open(radar_full_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "x_stim", "y_stim", "z_stim",
            "x_stim_rebased", "y_stim_rebased", "z_stim_rebased",
            "qx_stim", "qy_stim", "qz_stim", "qw_stim",
            "fitness", "rmse"
        ])

        for i in range(len(timestamps_out)):
            writer.writerow([
                timestamps_out[i],
                positions_stim[i, 0], positions_stim[i, 1], positions_stim[i, 2],
                positions_stim_rebased[i, 0], positions_stim_rebased[i, 1], positions_stim_rebased[i, 2],
                quaternions_stim[i, 0], quaternions_stim[i, 1], quaternions_stim[i, 2], quaternions_stim[i, 3],
                fitness_list[i], rmse_list[i]
            ])
    print(f"Saved CSV to: {radar_full_csv}")

# =========================================================
# IMU UKF prediction + correction
# =========================================================
if RUN_UKF_FUSION:
    print("\n================ UKF FUSION (FULL) ================\n")

    radar_data = np.load(radar_full_npz)
    radar_timestamp = radar_data["timestamp"]

    if "position_stim320_rebased" in radar_data:
        radar_pos = radar_data["position_stim320_rebased"]
    elif "position_stim320" in radar_data:
        radar_pos = radar_data["position_stim320"] - radar_data["position_stim320"][0]
    else:
        raise KeyError("雷达 npz 中没有可用的位置字段")

    imu_root = zarr.open(imu_path, mode="r")
    print("IMU keys:", list(imu_root.keys()))

    imu_timestamp_all = np.array(imu_root["timestamp"]).squeeze()
    imu_lin_acc_all = np.array(imu_root["lin_acc"])
    imu_ang_vel_all = np.array(imu_root["ang_vel"])

    imu_start_idx = np.searchsorted(imu_timestamp_all, radar_timestamp[0], side="left")
    imu_end_idx = np.searchsorted(imu_timestamp_all, radar_timestamp[-1], side="right")

    imu_timestamp = imu_timestamp_all[imu_start_idx:imu_end_idx]
    imu_lin_acc = imu_lin_acc_all[imu_start_idx:imu_end_idx]
    imu_ang_vel = imu_ang_vel_all[imu_start_idx:imu_end_idx]

    N = len(imu_timestamp)

    print("\nMatched IMU start index:", imu_start_idx)
    print("Matched IMU end index:", imu_end_idx)
    print("Matched IMU frames:", N)

    if N < 2:
        raise ValueError("IMU 帧太少，无法做 UKF fusion。")

    dt_all = np.diff(imu_timestamp)
    dt_all = np.clip(dt_all, 1e-6, None)

    bias_n = min(200, N)

    gyro_bias = np.mean(imu_ang_vel[:bias_n], axis=0)
    acc_mean_body = np.mean(imu_lin_acc[:bias_n], axis=0)

    print("\nEstimated gyro bias:", gyro_bias)
    print("Initial mean accelerometer reading:", acc_mean_body)

    acc_norm = np.linalg.norm(acc_mean_body)
    if acc_norm < 1e-6:
        raise ValueError("初始加速度均值太小，无法估计重力方向。")

    world_gravity_dir = np.array([0.0, 0.0, -1.0])
    body_gravity_dir = acc_mean_body / acc_norm

    initial_rot, _ = R.align_vectors(
        world_gravity_dir.reshape(1, 3),
        body_gravity_dir.reshape(1, 3)
    )

    gravity_world = np.array([0.0, 0.0, -acc_norm])

    acc_world_init = np.zeros((bias_n, 3))
    for i in range(bias_n):
        acc_world_init[i] = initial_rot.apply(imu_lin_acc[i])

    acc_world_residual_bias = np.mean(acc_world_init - gravity_world, axis=0)
    print("Estimated world-frame accel residual bias:", acc_world_residual_bias)

    rot = initial_rot
    quat_hist = np.zeros((N, 4))
    quat_hist[0] = rot.as_quat()

    a_world_hist = np.zeros((N, 3))
    stationary_hist = np.zeros(N, dtype=bool)

    acc_stationary_th = 0.2
    gyro_stationary_th = 0.03

    for k in range(1, N):
        dt = dt_all[k - 1]

        omega = imu_ang_vel[k - 1] - gyro_bias
        rotvec = omega * dt
        dR = R.from_rotvec(rotvec)
        rot = rot * dR
        quat_hist[k] = rot.as_quat()

        acc_world = rot.apply(imu_lin_acc[k - 1])
        specific_acc_world = acc_world - gravity_world - acc_world_residual_bias
        a_world_hist[k - 1] = specific_acc_world

        if np.linalg.norm(specific_acc_world) < acc_stationary_th and np.linalg.norm(omega) < gyro_stationary_th:
            stationary_hist[k - 1] = True

    a_world_hist[-1] = a_world_hist[-2]
    stationary_hist[-1] = stationary_hist[-2]

    mu = np.zeros((state_dim, 1))
    Sigma = Sigma0.copy()

    mu_hist = []
    Sigma_hist = []

    radar_idx = 0
    correction_count = 0

    for k in range(N):
        if k == 0:
            mu_hist.append(mu.copy())
            Sigma_hist.append(Sigma.copy())
            continue

        dt = dt_all[k - 1]
        u_k = a_world_hist[k - 1]
        stationary = stationary_hist[k - 1]

        mu_pred, Sigma_pred = ukf_prediction_step(
            mu_prev=mu,
            Sigma_prev=Sigma,
            a_world=u_k,
            dt=dt,
            Q=Q,
            stationary=stationary,
            kappa=kappa_f
        )

        while radar_idx < len(radar_timestamp) and radar_timestamp[radar_idx] <= imu_timestamp[k]:
            z_meas = radar_pos[radar_idx].reshape(3, 1)
            mu_pred, Sigma_pred, z_hat, K, v, S = ukf_correction_step(
                mu_pred=mu_pred,
                Sigma_pred=Sigma_pred,
                z_meas=z_meas,
                R_meas=R_meas,
                kappa=kappa_h
            )
            radar_idx += 1
            correction_count += 1

        mu = mu_pred
        Sigma = Sigma_pred

        mu_hist.append(mu.copy())
        Sigma_hist.append(Sigma.copy())

        if k % 5000 == 0:
            print(f"UKF fusion frame {k}/{N}, radar corrections used: {correction_count}")

    mu_hist = np.hstack(mu_hist).T
    pos_corr = mu_hist[:, 0:3]
    vel_corr = mu_hist[:, 3:6]

    pos_corr = pos_corr - pos_corr[0]
    t = imu_timestamp - imu_timestamp[0]

    print("\nTotal radar corrections used:", correction_count)
    print("UKF corrected trajectory points:", len(pos_corr))

    P_diag_hist = np.array([np.diag(S) for S in Sigma_hist])

    np.savez(
        ukf_full_npz,
        timestamp=imu_timestamp,
        position=pos_corr,
        velocity=vel_corr,
        quaternion=quat_hist,
        covariance_diag=P_diag_hist,
        gyro_bias=gyro_bias,
        gravity_world=gravity_world,
        accel_residual_bias_world=acc_world_residual_bias,
        imu_start_idx=imu_start_idx,
        imu_end_idx=imu_end_idx,
        radar_corrections_used=correction_count
    )
    print(f"\nSaved NPZ to: {ukf_full_npz}")

    with open(ukf_full_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "x", "y", "z", "vx", "vy", "vz"])
        for i in range(N):
            writer.writerow([
                imu_timestamp[i],
                pos_corr[i, 0], pos_corr[i, 1], pos_corr[i, 2],
                vel_corr[i, 0], vel_corr[i, 1], vel_corr[i, 2]
            ])
    print(f"Saved CSV to: {ukf_full_csv}")

    # =====================================================
    # 
    # =====================================================
    with open(radar_plot_traj_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "x", "y", "z"])
        for i in range(len(radar_timestamp)):
            writer.writerow([
                radar_timestamp[i],
                radar_pos[i, 0],
                radar_pos[i, 1],
                radar_pos[i, 2]
            ])
    print(f"Saved radar full trajectory CSV to: {radar_plot_traj_csv}")

    with open(ukf_plot_traj_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "x", "y", "z"])
        for i in range(len(imu_timestamp)):
            writer.writerow([
                imu_timestamp[i],
                pos_corr[i, 0],
                pos_corr[i, 1],
                pos_corr[i, 2]
            ])
    print(f"Saved UKF full trajectory CSV to: {ukf_plot_traj_csv}")


    # =====================================================
    # Runtime summary (exclude interactive plot window time)
    # =====================================================
    algorithm_end_time = time.time()
    total_runtime_sec = algorithm_end_time - total_start_time

    if algorithm_start_time is not None:
        algorithm_runtime_sec = algorithm_end_time - algorithm_start_time
    else:
        algorithm_runtime_sec = np.nan

    if processed_lidar_frames > 0:
        avg_runtime_per_frame_sec = algorithm_runtime_sec / processed_lidar_frames
    else:
        avg_runtime_per_frame_sec = np.nan

    print("\n================ RUNTIME SUMMARY ================\n")
    print(f"total_runtime_sec: {total_runtime_sec:.6f}")
    print(f"algorithm_runtime_sec: {algorithm_runtime_sec:.6f}")
    print(f"avg_runtime_per_frame_sec: {avg_runtime_per_frame_sec:.6f}")
    print(f"processed_lidar_frames: {processed_lidar_frames}")

    runtime_txt_path = os.path.join(project_dir, "runtime_summary.txt")
    with open(runtime_txt_path, "w", encoding="utf-8") as f:
        f.write(f"total_runtime_sec: {total_runtime_sec:.6f}\n")
        f.write(f"algorithm_runtime_sec: {algorithm_runtime_sec:.6f}\n")
        f.write(f"avg_runtime_per_frame_sec: {avg_runtime_per_frame_sec:.6f}\n")
        f.write(f"processed_lidar_frames: {processed_lidar_frames}\n")

    print(f"Saved runtime summary to: {runtime_txt_path}")

    # =====================================================
    # 
    # =====================================================
    radar_t = radar_timestamp - radar_timestamp[0]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        radar_pos[:, 0], radar_pos[:, 1], radar_pos[:, 2],
        linestyle="--",
        linewidth=2.0,
        label="Radar full trajectory"
    )
    ax.plot(
        pos_corr[:, 0], pos_corr[:, 1], pos_corr[:, 2],
        linestyle="-",
        linewidth=1.5,
        label="UKF corrected full trajectory"
    )

    ax.scatter(*radar_pos[0], s=60, label="Radar start")
    ax.scatter(*radar_pos[-1], s=60, label="Radar end")
    ax.scatter(*pos_corr[0], s=60, label="UKF start")
    ax.scatter(*pos_corr[-1], s=60, label="UKF end")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Full Pipeline: Radar + IMU UKF (project1)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(full_overlay_fig_path, dpi=300)
    print(f"Saved overlay figure to: {full_overlay_fig_path}")
    plt.show()