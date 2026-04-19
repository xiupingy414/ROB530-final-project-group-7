import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================================================
# 1. 工具函数
# =========================================================
def load_traj_csv(csv_path):
    """
    CSV format expected:
    timestamp, x, y, z
    """
    df = pd.read_csv(csv_path)
    required_cols = ["timestamp", "x", "y", "z"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"{csv_path} is missing required column: {c}")

    ts = df["timestamp"].to_numpy()
    pos = df[["x", "y", "z"]].to_numpy()
    return ts, pos


def interp_traj_to_time(src_pos, src_ts, dst_ts):
    """
    Interpolate source trajectory positions to destination timestamps
    """
    return np.stack([
        np.interp(dst_ts, src_ts, src_pos[:, 0]),
        np.interp(dst_ts, src_ts, src_pos[:, 1]),
        np.interp(dst_ts, src_ts, src_pos[:, 2]),
    ], axis=1)


def calc_overlap_residual(est_pos, est_ts, ref_pos, ref_ts):
    """
    Same idea as your classmate's residual logic:
    1) find overlap time range
    2) interpolate est trajectory to ref timestamps
    3) compute Euclidean residual

    Here:
    - est = UKF trajectory
    - ref = Radar ICP trajectory
    """
    t_start = max(est_ts[0], ref_ts[0])
    t_end   = min(est_ts[-1], ref_ts[-1])

    if t_end <= t_start:
        raise ValueError("No overlapping time range between UKF and Radar trajectories.")

    mask_ref = (ref_ts >= t_start) & (ref_ts <= t_end)
    ref_overlap = ref_pos[mask_ref]
    ref_ts_overlap = ref_ts[mask_ref]

    est_interp = interp_traj_to_time(est_pos, est_ts, ref_ts_overlap)

    residual = np.linalg.norm(est_interp - ref_overlap, axis=1)

    mean_res = residual.mean()
    max_res  = residual.max()
    rmse_res = np.sqrt(np.mean(residual**2))

    return {
        "t_overlap": ref_ts_overlap,
        "ref_overlap": ref_overlap,
        "est_interp": est_interp,
        "residual": residual,
        "mean": mean_res,
        "max": max_res,
        "rmse": rmse_res
    }


def save_residual_csv(save_path, t, ref_pos, est_pos, residual):
    df = pd.DataFrame({
        "timestamp": t,
        "radar_x": ref_pos[:, 0],
        "radar_y": ref_pos[:, 1],
        "radar_z": ref_pos[:, 2],
        "ukf_x": est_pos[:, 0],
        "ukf_y": est_pos[:, 1],
        "ukf_z": est_pos[:, 2],
        "residual_norm": residual
    })
    df.to_csv(save_path, index=False)


# =========================================================
# 2. 三个场景路径
#    你只需要确认这些文件名和你的实际保存结果一致
# =========================================================
SCENES = {
    "scene_1": {
    "radar_csv": r"C:\Users\junxianw\Desktop\project\output_radar\final_plot_radar_trajectory_coords.csv",
    "ukf_csv":   r"C:\Users\junxianw\Desktop\project\output_ukf\final_plot_ukf_trajectory_coords.csv",
    "output_dir": r"C:\Users\junxianw\Desktop\project\residual_results"
},
"scene_2": {
    "radar_csv": r"C:\Users\junxianw\Desktop\project1\output_radar_full_only\radar_full_trajectory_coords_project1_v1.csv",
    "ukf_csv":   r"C:\Users\junxianw\Desktop\project1\output_ukf_full_only\ukf_full_trajectory_coords_project1_v1.csv",
    "output_dir": r"C:\Users\junxianw\Desktop\project1\residual_results"
},
"scene_3": {
    "radar_csv": r"C:\Users\junxianw\Desktop\project2\output_radar_full_only\radar_full_trajectory_coords_project2_v1.csv",
    "ukf_csv":   r"C:\Users\junxianw\Desktop\project2\output_ukf_full_only\ukf_full_trajectory_coords_project2_v1.csv",
    "output_dir": r"C:\Users\junxianw\Desktop\project2\residual_results"
}
}

# =========================================================
# 3. 主程序
# =========================================================
summary_rows = []

for scene_name, cfg in SCENES.items():
    print("\n" + "=" * 60)
    print(f"Processing {scene_name}")
    print("=" * 60)

    radar_csv = cfg["radar_csv"]
    ukf_csv   = cfg["ukf_csv"]
    output_dir = cfg["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(radar_csv):
        print(f"[WARNING] Radar CSV not found: {radar_csv}")
        continue
    if not os.path.exists(ukf_csv):
        print(f"[WARNING] UKF CSV not found: {ukf_csv}")
        continue

    radar_ts, radar_pos = load_traj_csv(radar_csv)
    ukf_ts, ukf_pos     = load_traj_csv(ukf_csv)

    result = calc_overlap_residual(
        est_pos=ukf_pos,
        est_ts=ukf_ts,
        ref_pos=radar_pos,
        ref_ts=radar_ts
    )

    print(f"Mean residual : {result['mean']:.6f} m")
    print(f"Max residual  : {result['max']:.6f} m")
    print(f"RMSE residual : {result['rmse']:.6f} m")
    print(f"Overlap points: {len(result['residual'])}")

    # -----------------------------------------------------
    # 保存每个场景的 residual csv
    # -----------------------------------------------------
    residual_csv_path = os.path.join(output_dir, f"{scene_name}_ukf_vs_radar_residual.csv")
    save_residual_csv(
        residual_csv_path,
        result["t_overlap"],
        result["ref_overlap"],
        result["est_interp"],
        result["residual"]
    )
    print(f"Saved residual CSV: {residual_csv_path}")

    # -----------------------------------------------------
    # 保存残差曲线图
    # -----------------------------------------------------
    t_rel = result["t_overlap"] - result["t_overlap"][0]

    plt.figure(figsize=(10, 4))
    plt.plot(t_rel, result["residual"], linewidth=1.2)
    plt.xlabel("time (s)")
    plt.ylabel("Residual norm (m)")
    plt.title(f"{scene_name}: UKF vs Radar ICP Residual")
    plt.grid(True)
    plt.tight_layout()

    residual_fig_path = os.path.join(output_dir, f"{scene_name}_ukf_vs_radar_residual.png")
    plt.savefig(residual_fig_path, dpi=200)
    plt.close()
    print(f"Saved residual plot: {residual_fig_path}")

    # -----------------------------------------------------
    # 保存 summary
    # -----------------------------------------------------
    summary_rows.append({
        "scene": scene_name,
        "mean_residual_m": result["mean"],
        "max_residual_m": result["max"],
        "rmse_residual_m": result["rmse"],
        "overlap_points": len(result["residual"])
    })

# =========================================================
# 4. 保存三个场景总表
# =========================================================
if len(summary_rows) > 0:
    summary_df = pd.DataFrame(summary_rows)

    summary_csv = r"C:\Users\junxianw\Desktop\ukf_radar_residual_summary_3scenes.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 60)
    print("Summary for all scenes")
    print("=" * 60)
    print(summary_df)
    print(f"\nSaved summary CSV: {summary_csv}")
else:
    print("\nNo valid scenes were processed.")