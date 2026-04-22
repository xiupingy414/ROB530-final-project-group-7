import numpy as np
import open3d as o3d
import zarr
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as Rot


def skew(v):
    v = np.asarray(v, dtype=float).reshape(3,)
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0]
    ])


def gamma_1(phi):
    I = np.eye(3)
    angle = np.linalg.norm(phi)
    phi_skew = skew(phi)
    if angle < 1e-7:
        return I + 0.5 * phi_skew + (1.0 / 6.0) * (phi_skew @ phi_skew)
    return (
        I
        + ((1 - np.cos(angle)) / angle**2) * phi_skew
        + ((angle - np.sin(angle)) / angle**3) * (phi_skew @ phi_skew)
    )


def gamma_2(phi):
    I = np.eye(3)
    angle = np.linalg.norm(phi)
    phi_skew = skew(phi)
    if angle < 1e-7:
        return 0.5 * I + (1.0 / 6.0) * phi_skew + (1.0 / 24.0) * (phi_skew @ phi_skew)
    return (
        0.5 * I
        + ((angle - np.sin(angle)) / angle**3) * phi_skew
        + ((angle**2 + 2 * np.cos(angle) - 2) / (2 * angle**4)) * (phi_skew @ phi_skew)
    )


def log_so3(R):
    return Rot.from_matrix(R).as_rotvec()


def pose_to_matrix(qw, qx, qy, qz, tx, ty, tz):
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = np.array([tx, ty, tz], dtype=float)
    return T


def invert_T(T):
    T = np.asarray(T, dtype=float)
    T_inv = np.eye(4)
    R = T[:3, :3]
    p = T[:3, 3]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ p
    return T_inv


def get_tf_matrix(tf_entry):
    q = tf_entry["rotation"]
    t = tf_entry["translation"]
    return pose_to_matrix(
        q["w"], q["x"], q["y"], q["z"],
        t["x"], t["y"], t["z"]
    )


def pose_from_T(T):
    return T[:3, :3].copy(), T[:3, 3].copy()


def T_from_Rp(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def construct_A_matrix(R, v, p, g=np.array([0.0, 0.0, -9.81])):
    R = np.asarray(R, dtype=float)
    v = np.asarray(v, dtype=float).reshape(3,)
    p = np.asarray(p, dtype=float).reshape(3,)
    g = np.asarray(g, dtype=float).reshape(3,)

    A = np.zeros((15, 15), dtype=float)
    I3 = np.eye(3)

    th = slice(0, 3)
    dv = slice(3, 6)
    dp = slice(6, 9)
    dbg = slice(9, 12)
    dba = slice(12, 15)

    A[th, dbg] = -R
    A[dv, th] = skew(g)
    A[dv, dbg] = -skew(v) @ R
    A[dv, dba] = -R
    A[dp, dv] = I3
    A[dp, dbg] = -skew(p) @ R

    return A


def adjoint(R, v, p, augmented=True):
    R = np.asarray(R, dtype=float)
    v = np.asarray(v, dtype=float).reshape(3,)
    p = np.asarray(p, dtype=float).reshape(3,)

    Z = np.zeros((3, 3), dtype=float)
    I = np.eye(3, dtype=float)

    Ad_se23 = np.block([
        [R,            Z, Z],
        [skew(v) @ R,  R, Z],
        [skew(p) @ R,  Z, R],
    ])

    if not augmented:
        return Ad_se23

    Ad_aug = np.block([
        [Ad_se23,          np.zeros((9, 3)), np.zeros((9, 3))],
        [np.zeros((3, 9)), I,                Z               ],
        [np.zeros((3, 9)), Z,                I               ],
    ])
    return Ad_aug


def prediction(R, v, p, omega_raw, accel_raw, bg, ba, dt, P, Q):
    g = np.array([0.0, 0.0, -9.81])

    omega = omega_raw - bg
    accel = accel_raw - ba
    phi = omega * dt

    A = construct_A_matrix(R, v, p)
    Ad = adjoint(R, v, p, augmented=True)

    R_next = R @ expm(skew(phi))
    v_next = v + (R @ gamma_1(phi) @ accel) * dt + g * dt
    p_next = p + v * dt + (R @ gamma_2(phi) @ accel) * (dt ** 2) + 0.5 * g * (dt ** 2)

    bg_next = bg.copy()
    ba_next = ba.copy()

    Phi = expm(A * dt)
    Qd = (Ad @ Q @ Ad.T) * dt
    P_next = Phi @ P @ Phi.T + Qd

    return R_next, v_next, p_next, bg_next, ba_next, P_next


def correction(R, v, p, bg, ba, P, T_WL_icp, T_LI, N):
    T_WI_meas = T_WL_icp @ T_LI
    R_meas = T_WI_meas[:3, :3]
    p_meas = T_WI_meas[:3, 3]

    r_R = log_so3(R_meas @ R.T)
    r_p = p_meas - p
    r = np.hstack((r_R, r_p))

    H = np.zeros((6, 15), dtype=float)
    H[0:3, 0:3] = np.eye(3)
    H[3:6, 6:9] = np.eye(3)

    S = H @ P @ H.T + N
    K = P @ H.T @ np.linalg.inv(S)

    dx = K @ r
    dtheta = dx[0:3]
    dv = dx[3:6]
    dp = dx[6:9]
    dbg = dx[9:12]
    dba = dx[12:15]

    R_new = expm(skew(dtheta)) @ R
    v_new = v + dv
    p_new = p + dp
    bg_new = bg + dbg
    ba_new = ba + dba

    I15 = np.eye(15)
    P_new = (I15 - K @ H) @ P @ (I15 - K @ H).T + K @ N @ K.T

    return R_new, v_new, p_new, bg_new, ba_new, P_new


def run_icp(scan_np, map_pcd_down, T_init, max_corr_dist=0.4, max_iter=50):
    scan_pcd = o3d.geometry.PointCloud()
    scan_pcd.points = o3d.utility.Vector3dVector(scan_np.astype(np.float64))

    result = o3d.pipelines.registration.registration_icp(
        source=scan_pcd,
        target=map_pcd_down,
        max_correspondence_distance=max_corr_dist,
        init=T_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    return result


def main():
    dataset_root = "Dataset/2024-10-01-11-29-55"
    map_path = f"{dataset_root}/point_cloud_maps/2024-10-01-11-29-55_dlio.ply"

    imu_path = f"{dataset_root}/data/stim320_imu"
    lidar_path = f"{dataset_root}/data/hesai_points_undistorted"
    tf_path = f"{dataset_root}/data/tf"

    map_pcd = o3d.io.read_point_cloud(map_path)
    voxel_size = 0.3
    map_pcd_down = map_pcd.voxel_down_sample(voxel_size)
    map_pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    print(f"Map: {len(map_pcd.points)} -> {len(map_pcd_down.points)} points")


    z_imu = zarr.open(imu_path, mode="r")
    z_lidar = zarr.open(lidar_path, mode="r")
    z_tf = zarr.open(tf_path, mode="r")

    imu_t = z_imu["timestamp"][:]
    imu_w = z_imu["ang_vel"][:]
    imu_a = z_imu["lin_acc"][:]

    lidar_t = z_lidar["timestamp"][:]
    lidar_valid = z_lidar["valid"]
    lidar_points = z_lidar["points"]

    tf_dict = z_tf.attrs["tf"]

    print("imu samples:", len(imu_t))
    print("lidar frames:", len(lidar_t))

    # Build extrinsics
    T_BI = get_tf_matrix(tf_dict["stim320_imu"])   # box_base <- imu
    T_BL = get_tf_matrix(tf_dict["hesai_lidar"])   # box_base <- lidar

    T_IB = invert_T(T_BI)
    T_IL = T_IB @ T_BL      # imu <- lidar
    T_LI = invert_T(T_IL)   # lidar <- imu

    print("T_IL (imu <- lidar):\n", T_IL)
    print("T_LI (lidar <- imu):\n", T_LI)

    # Process noise Q
    Sigma_g = z_imu["ang_vel_cov"][0]
    Sigma_a = z_imu["lin_acc_cov"][0]

    Sigma_bg = 1e-6 * np.eye(3)
    Sigma_ba = 1e-4 * np.eye(3)

    Q = np.zeros((15, 15), dtype=float)
    Q[0:3, 0:3] = Sigma_g
    Q[3:6, 3:6] = Sigma_a
    Q[9:12, 9:12] = Sigma_bg
    Q[12:15, 12:15] = Sigma_ba

    # ICP measurement covariance
    sigma_rot = np.deg2rad(3.0)
    sigma_pos = 0.10
    N_icp = np.diag([
        sigma_rot**2, sigma_rot**2, sigma_rot**2,
        sigma_pos**2, sigma_pos**2, sigma_pos**2
    ])

    # First frame ICP for initialization
    k0 = 0
    n_valid0 = int(lidar_valid[k0, 0])
    scan0 = lidar_points[k0, :n_valid0, :]

    result0 = run_icp(scan0, map_pcd_down, np.eye(4))
    T_WL_0 = result0.transformation
    T_WI_0 = T_WL_0 @ T_LI

    R = T_WI_0[:3, :3].copy()
    p = T_WI_0[:3, 3].copy()
    v = np.zeros(3)
    bg = np.zeros(3)
    ba = np.zeros(3)

    # Initial covariance
    P = np.zeros((15, 15), dtype=float)
    P[0:3, 0:3] = (np.deg2rad(5.0) ** 2) * np.eye(3)
    P[3:6, 3:6] = (0.5 ** 2) * np.eye(3)
    P[6:9, 6:9] = (0.2 ** 2) * np.eye(3)
    P[9:12, 9:12] = (1e-3 ** 2) * np.eye(3)
    P[12:15, 12:15] = (1e-2 ** 2) * np.eye(3)

    print(f"Init ICP fitness = {result0.fitness:.4f}, rmse = {result0.inlier_rmse:.4f}")
    print("Initialized T_WI_0:\n", T_WI_0)

    # Storage
    imu_positions = [p.copy()]
    lidar_positions = [T_WL_0[:3, 3].copy()]

    T_WI_est_0 = T_from_Rp(R, p)
    T_WL_from_imu_0 = T_WI_est_0 @ T_IL
    lidar_from_imu_positions = [T_WL_from_imu_0[:3, 3].copy()]

    times_used = [lidar_t[0]]

    # Main loop over LiDAR frames
    max_k = len(lidar_t)

    for k in range(1, max_k):
        t_prev = lidar_t[k - 1]
        t_curr = lidar_t[k]

        imu_idx_start = np.searchsorted(imu_t, t_prev, side="right")
        imu_idx_end = np.searchsorted(imu_t, t_curr, side="right")

        # 1) Prediction with all IMU samples
        for j in range(imu_idx_start, imu_idx_end):
            if j == 0:
                continue

            dt = imu_t[j] - imu_t[j - 1]
            omega_raw = imu_w[j]
            accel_raw = imu_a[j]

            R, v, p, bg, ba, P = prediction(
                R, v, p,
                omega_raw, accel_raw,
                bg, ba,
                dt, P, Q
            )

        # 2) Current LiDAR scan
        n_valid = int(lidar_valid[k, 0])
        scan_np = lidar_points[k, :n_valid, :]

        # 3) ICP init = prediction transformed to LiDAR frame
        T_WI_pred = T_from_Rp(R, p)
        T_WL_pred = T_WI_pred @ T_IL

        result = run_icp(scan_np, map_pcd_down, T_WL_pred)
        T_WL_icp = result.transformation

        # 4) Correction
        R, v, p, bg, ba, P = correction(
            R, v, p, bg, ba, P,
            T_WL_icp=T_WL_icp,
            T_LI=T_LI,
            N=N_icp
        )

        # 5) Save trajectories
        imu_positions.append(p.copy())
        lidar_positions.append(T_WL_icp[:3, 3].copy())

        T_WI_est = T_from_Rp(R, p)
        T_WL_from_imu = T_WI_est @ T_IL
        lidar_from_imu_positions.append(T_WL_from_imu[:3, 3].copy())

        times_used.append(t_curr)

        if k % 50 == 0:
            print(
                f"frame {k:4d} | "
                f"fitness = {result.fitness:.4f} | "
                f"rmse = {result.inlier_rmse:.4f} | "
                f"imu_pos = {p}"
            )


    # Save results
    imu_positions = np.array(imu_positions)
    lidar_positions = np.array(lidar_positions)
    lidar_from_imu_positions = np.array(lidar_from_imu_positions)
    times_used = np.array(times_used)

    np.save("imu_positions.npy", imu_positions)
    np.save("lidar_positions.npy", lidar_positions)
    np.save("lidar_from_imu_positions.npy", lidar_from_imu_positions)
    np.save("times_used.npy", times_used)

    print("Done.")
    print("Saved:")
    print("  imu_positions.npy")
    print("  lidar_positions.npy")
    print("  lidar_from_imu_positions.npy")
    print("  times_used.npy")


if __name__ == "__main__":
    main()