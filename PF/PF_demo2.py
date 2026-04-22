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


def T_from_Rp(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def imu_predict(R, v, p, omega_raw, accel_raw, bg, ba, dt):
    g = np.array([0.0, 0.0, -9.81])

    omega = omega_raw - bg
    accel = accel_raw - ba
    phi = omega * dt

    R_next = R @ expm(skew(phi))
    v_next = v + (R @ gamma_1(phi) @ accel) * dt + g * dt
    p_next = p + v * dt + (R @ gamma_2(phi) @ accel) * dt**2 + 0.5 * g * dt**2

    return R_next, v_next, p_next


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


def low_variance_resample(weights, rng):
    N = len(weights)
    indices = np.zeros(N, dtype=int)

    r = rng.uniform(0.0, 1.0 / N)
    c = weights[0]
    i = 0

    for m in range(N):
        u = r + m / N
        while u > c:
            i += 1
            c += weights[i]
        indices[m] = i

    return indices


def effective_sample_size(weights):
    return 1.0 / np.sum(weights**2)


def weighted_rotation_mean(Rs, weights):
    return Rot.from_matrix(Rs).mean(weights=weights).as_matrix()


def estimate_from_particles(Rs, vs, ps, weights):
    R_est = weighted_rotation_mean(Rs, weights)
    v_est = np.average(vs, axis=0, weights=weights)
    p_est = np.average(ps, axis=0, weights=weights)
    return R_est, v_est, p_est


def apply_small_noise(R, v, p, rng, rot_std, vel_std, pos_std):
    dtheta = rng.normal(0.0, rot_std, size=3)
    dv = rng.normal(0.0, vel_std, size=3)
    dp = rng.normal(0.0, pos_std, size=3)

    R_new = expm(skew(dtheta)) @ R
    v_new = v + dv
    p_new = p + dp
    return R_new, v_new, p_new


def select_representative_particles(Rs, vs, ps, weights, K=4, rng=None):
    N = len(weights)
    assert K <= N

    # best particle
    idx_best = int(np.argmax(weights))

    # mean-nearest particle
    _, _, p_mean = estimate_from_particles(Rs, vs, ps, weights)
    dists = np.linalg.norm(ps - p_mean[None, :], axis=1)
    idx_mean = int(np.argmin(dists))

    chosen = {idx_best, idx_mean}

    # random fill
    all_idx = list(range(N))
    if rng is None:
        rng = np.random.default_rng()

    while len(chosen) < K:
        chosen.add(int(rng.choice(all_idx)))

    return list(chosen)


def run_multi_icp_candidates(scan_np, map_pcd_down, Rs, ps, T_IL, indices):
    best_result = None
    best_score = -np.inf
    best_idx = None

    for idx in indices:
        T_WI = T_from_Rp(Rs[idx], ps[idx])
        T_WL_init = T_WI @ T_IL

        result = run_icp(scan_np, map_pcd_down, T_WL_init)

        fitness = result.fitness
        rmse = result.inlier_rmse if np.isfinite(result.inlier_rmse) else 1e6

        score = fitness - 0.5 * rmse

        if score > best_score:
            best_score = score
            best_result = result
            best_idx = idx

    return best_result, best_idx


def compute_particle_log_weights_from_single_icp(
    Rs, ps, T_IL, T_WL_icp, sigma_pos, sigma_rot, fitness, rmse, sigma_rmse
):
    Np = len(ps)
    log_weights = np.zeros(Np)

    R_icp = T_WL_icp[:3, :3]
    p_icp = T_WL_icp[:3, 3]

    for i in range(Np):
        T_WI_i = T_from_Rp(Rs[i], ps[i])
        T_WL_i = T_WI_i @ T_IL

        p_i = T_WL_i[:3, 3]
        R_i = T_WL_i[:3, :3]

        pos_err = np.linalg.norm(p_i - p_icp)
        rot_err = np.linalg.norm(log_so3(R_icp @ R_i.T))

        log_weights[i] = (
            np.log(max(fitness, 1e-8))
            - 0.5 * (rmse / sigma_rmse) ** 2
            - 0.5 * (pos_err / sigma_pos) ** 2
            - 0.5 * (rot_err / sigma_rot) ** 2
        )

    return log_weights


def main():
    rng = np.random.default_rng(0)

    dataset_root = "Dataset/2024-10-01-11-29-55"
    map_path = f"{dataset_root}/point_cloud_maps/2024-10-01-11-29-55_dlio.ply"

    imu_path = f"{dataset_root}/data/stim320_imu"
    lidar_path = f"{dataset_root}/data/hesai_points_undistorted"
    tf_path = f"{dataset_root}/data/tf"

    N_particles = 50
    N_icp_candidates = 4
    resample_threshold = N_particles / 3.0

    # fixed bias
    bg = np.zeros(3)
    ba = np.zeros(3)

    # process noise injected into particle propagation
    gyro_noise_std = 0.005      # rad/s
    accel_noise_std = 0.10      # m/s^2

    # roughening after resampling
    rough_rot_std = np.deg2rad(0.8)
    rough_vel_std = 0.08
    rough_pos_std = 0.05

    # likelihood tuning
    sigma_rmse = 0.10
    sigma_pos = 0.15
    sigma_rot = np.deg2rad(5.0)

    # gating
    fitness_threshold = 0.80
    rmse_threshold = 0.14


    # Load map
    map_pcd = o3d.io.read_point_cloud(map_path)
    voxel_size = 0.3
    map_pcd_down = map_pcd.voxel_down_sample(voxel_size)
    map_pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    print(f"Map: {len(map_pcd.points)} -> {len(map_pcd_down.points)} points")

    # Load IMU / LiDAR / TF data
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


    # First frame ICP for initialization
    k0 = 0
    n_valid0 = int(lidar_valid[k0, 0])
    scan0 = lidar_points[k0, :n_valid0, :]

    result0 = run_icp(scan0, map_pcd_down, np.eye(4))
    T_WL_0 = result0.transformation
    T_WI_0 = T_WL_0 @ T_LI

    R0 = T_WI_0[:3, :3].copy()
    p0 = T_WI_0[:3, 3].copy()
    v0 = np.zeros(3)

    print(f"Init ICP fitness = {result0.fitness:.4f}, rmse = {result0.inlier_rmse:.4f}")
    print("Initialized T_WI_0:\n", T_WI_0)


    # Initialize particles
    Rs = np.zeros((N_particles, 3, 3))
    vs = np.zeros((N_particles, 3))
    ps = np.zeros((N_particles, 3))

    init_rot_std = np.deg2rad(2.0)
    init_pos_std = 0.20
    init_vel_std = 0.20

    for i in range(N_particles):
        dtheta = rng.normal(0.0, init_rot_std, size=3)
        Rs[i] = expm(skew(dtheta)) @ R0
        ps[i] = p0 + rng.normal(0.0, init_pos_std, size=3)
        vs[i] = v0 + rng.normal(0.0, init_vel_std, size=3)

    weights = np.ones(N_particles) / N_particles


    # Storage
    pf_imu_positions = []
    pf_lidar_positions = []
    pf_best_imu_positions = []
    pf_best_lidar_positions = []
    pf_neff = []
    pf_times_used = []

    R_est, v_est, p_est = estimate_from_particles(Rs, vs, ps, weights)
    T_WI_est = T_from_Rp(R_est, p_est)
    T_WL_est = T_WI_est @ T_IL

    best_idx = np.argmax(weights)
    T_WL_best = T_from_Rp(Rs[best_idx], ps[best_idx]) @ T_IL

    pf_imu_positions.append(p_est.copy())
    pf_lidar_positions.append(T_WL_est[:3, 3].copy())
    pf_best_imu_positions.append(ps[best_idx].copy())
    pf_best_lidar_positions.append(T_WL_best[:3, 3].copy())
    pf_neff.append(effective_sample_size(weights))
    pf_times_used.append(lidar_t[0])


    # Main loop over LiDAR frames
    max_k = len(lidar_t)
    max_k = min(len(lidar_t), 300)   # 先跑300帧测试

    for k in range(1, max_k):
        t_prev = lidar_t[k - 1]
        t_curr = lidar_t[k]

        imu_idx_start = np.searchsorted(imu_t, t_prev, side="right")
        imu_idx_end = np.searchsorted(imu_t, t_curr, side="right")


        # 1) Prediction for all particles
        for j in range(imu_idx_start, imu_idx_end):
            if j == 0:
                continue

            dt = imu_t[j] - imu_t[j - 1]

            for i in range(N_particles):
                omega_noisy = imu_w[j] + rng.normal(0.0, gyro_noise_std, size=3)
                accel_noisy = imu_a[j] + rng.normal(0.0, accel_noise_std, size=3)

                Rs[i], vs[i], ps[i] = imu_predict(
                    Rs[i], vs[i], ps[i],
                    omega_noisy, accel_noisy,
                    bg, ba,
                    dt
                )

        # 2) Current LiDAR scan
        n_valid = int(lidar_valid[k, 0])
        scan_np = lidar_points[k, :n_valid, :]

        # 3) Choose representative particles and run several ICPs
        rep_indices = select_representative_particles(
            Rs, vs, ps, weights, K=N_icp_candidates, rng=rng
        )

        result, icp_src_idx = run_multi_icp_candidates(
            scan_np, map_pcd_down,
            Rs, ps, T_IL,
            rep_indices
        )

        T_WL_icp = result.transformation
        rmse = result.inlier_rmse if np.isfinite(result.inlier_rmse) else 1e6
        fitness = result.fitness

 
        # 4) Measurement gating
        use_measurement = (fitness >= fitness_threshold) and (rmse <= rmse_threshold)

        if use_measurement:
            log_like = compute_particle_log_weights_from_single_icp(
                Rs, ps, T_IL, T_WL_icp,
                sigma_pos=sigma_pos,
                sigma_rot=sigma_rot,
                fitness=fitness,
                rmse=rmse,
                sigma_rmse=sigma_rmse
            )

            log_weights = np.log(weights + 1e-300) + log_like
            log_weights -= np.max(log_weights)

            weights = np.exp(log_weights)
            wsum = np.sum(weights)

            if (not np.isfinite(wsum)) or wsum < 1e-300:
                print("Warning: weight collapse, reset to uniform.")
                weights = np.ones(N_particles) / N_particles
            else:
                weights /= wsum
        else:
            print(
                f"frame {k:4d} | ICP rejected "
                f"(fitness={fitness:.4f}, rmse={rmse:.4f})"
            )


        # 5) State estimate
        R_est, v_est, p_est = estimate_from_particles(Rs, vs, ps, weights)
        T_WI_est = T_from_Rp(R_est, p_est)
        T_WL_est = T_WI_est @ T_IL

        best_idx = np.argmax(weights)
        T_WL_best = T_from_Rp(Rs[best_idx], ps[best_idx]) @ T_IL

        pf_imu_positions.append(p_est.copy())
        pf_lidar_positions.append(T_WL_est[:3, 3].copy())
        pf_best_imu_positions.append(ps[best_idx].copy())
        pf_best_lidar_positions.append(T_WL_best[:3, 3].copy())

        neff = effective_sample_size(weights)
        pf_neff.append(neff)
        pf_times_used.append(t_curr)


        # 6) Resample if needed
        if neff < resample_threshold:
            idx = low_variance_resample(weights, rng)

            Rs = Rs[idx].copy()
            vs = vs[idx].copy()
            ps = ps[idx].copy()

            weights = np.ones(N_particles) / N_particles

            for i in range(N_particles):
                Rs[i], vs[i], ps[i] = apply_small_noise(
                    Rs[i], vs[i], ps[i], rng,
                    rough_rot_std, rough_vel_std, rough_pos_std
                )

        if k % 10 == 0:
            print(
                f"frame {k:4d} | "
                f"use_meas = {use_measurement} | "
                f"icp_src_idx = {icp_src_idx:2d} | "
                f"neff = {neff:6.2f} | "
                f"best_w = {np.max(weights):.4f} | "
                f"rmse = {rmse:.4f} | "
                f"fitness = {fitness:.4f} | "
                f"pf_pos = {p_est}"
            )


    # Save results
    np.save("pf_imu_positions.npy", np.array(pf_imu_positions))
    np.save("pf_lidar_positions.npy", np.array(pf_lidar_positions))
    np.save("pf_best_imu_positions.npy", np.array(pf_best_imu_positions))
    np.save("pf_best_lidar_positions.npy", np.array(pf_best_lidar_positions))
    np.save("pf_neff.npy", np.array(pf_neff))
    np.save("pf_times_used.npy", np.array(pf_times_used))

    print("Done PF (multi-candidate ICP).")
    print("Saved:")
    print("  pf_imu_positions.npy")
    print("  pf_lidar_positions.npy")
    print("  pf_best_imu_positions.npy")
    print("  pf_best_lidar_positions.npy")
    print("  pf_neff.npy")
    print("  pf_times_used.npy")


if __name__ == "__main__":
    main()