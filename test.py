import numpy as np
import zarr
import matplotlib.pyplot as plt
from extended_kalman_filter import extended_kalman_filter


class System:
    def __init__(self):
        self.Q = np.eye(10) * 0.01

    def A(self, x, u, dt):
        from extended_kalman_filter import quat_to_rot
        A = np.eye(10)

        # dp/dv
        A[0, 3] = dt
        A[1, 4] = dt
        A[2, 5] = dt

        # dv/dq : v_new = v + R(q)*a_body*dt, 所以 dv/dq = d(R(q)*a_body)/dq * dt
        qw, qx, qy, qz = x[6], x[7], x[8], x[9]
        ax, ay, az = u[0], u[1], u[2]

        # d(R*a)/dqw
        A[3, 6] = 2*( qw*ax - qz*ay + qy*az) * dt
        A[4, 6] = 2*( qz*ax + qw*ay - qx*az) * dt
        A[5, 6] = 2*(-qy*ax + qx*ay + qw*az) * dt

        # d(R*a)/dqx
        A[3, 7] = 2*( qx*ax + qy*ay + qz*az) * dt
        A[4, 7] = 2*( qy*ax - qx*ay - qw*az) * dt
        A[5, 7] = 2*( qz*ax + qw*ay - qx*az) * dt

        # d(R*a)/dqy
        A[3, 8] = 2*(-qy*ax + qx*ay + qw*az) * dt
        A[4, 8] = 2*( qx*ax + qy*ay + qz*az) * dt
        A[5, 8] = 2*(-qw*ax + qz*ay - qy*az) * dt

        # d(R*a)/dqz
        A[3, 9] = 2*(-qz*ax - qw*ay + qx*az) * dt
        A[4, 9] = 2*( qw*ax - qz*ay + qy*az) * dt
        A[5, 9] = 2*( qx*ax + qy*ay + qz*az) * dt

        return A


# -----------------------------
# 1. 读取数据
# -----------------------------
imu_data = zarr.open(
    r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/stim320_imu/stim320_imu",
    mode='r'
)

imu_lin_acc = imu_data['lin_acc'][:]
imu_ang_vel = imu_data['ang_vel'][:]
imu_ang_vel_cov = imu_data['ang_vel_cov'][:]
imu_lin_acc_cov = imu_data['lin_acc_cov'][:]
imu_timestamp = imu_data['timestamp'][:]

odom_data = zarr.open(
    r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/anymal_state_odometry/anymal_state_odometry",
    mode='r'
)

odom_pose_pos = odom_data['pose_pos'][:]
odom_pose_cov = odom_data['pose_cov'][:]
odom_timestamp = odom_data['timestamp'][:]

# -----------------------------
# 2. 初始条件
# -----------------------------
class Init:
    def __init__(self):
        # 用 odom 第一个位置作为初始位置
        v0 = (odom_pose_pos[1] - odom_pose_pos[0]) / (odom_timestamp[1] - odom_timestamp[0])
        self.x = np.array([
            odom_pose_pos[0][0], odom_pose_pos[0][1], odom_pose_pos[0][2],
            v0[0], v0[1], v0[2],
            1.0, 0.0, 0.0, 0.0
        ])

        self.Sigma = np.eye(10) * 0.1


def process_noise_from_imu(acc_cov, gyro_cov, dt):
    Q = np.zeros((10, 10))

    # position noise
    Q[0:3, 0:3] = 0.25 * acc_cov * dt**4

    # velocity noise
    Q[3:6, 3:6] = acc_cov * dt**2

    # quaternion noise
    gyro_level = np.mean(np.diag(gyro_cov))
    Q[6:10, 6:10] = np.eye(4) * gyro_level * dt**2

    return Q


# -----------------------------
# 3. 建立 EKF
# -----------------------------
system = System()
init = Init()
ekf = extended_kalman_filter(system, init)

lidar_data = np.load(r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/results/latest/kiss_icp_input_poses.npy")
print("lidar shape =", lidar_data.shape)
print("first pose =\n", lidar_data[0])
print("second pose =\n", lidar_data[1])

# 看LiDAR轨迹范围
lidar_traj = np.array([lidar_data[i][:3, 3] for i in range(len(lidar_data))])
print("lidar traj range =", np.max(lidar_traj, axis=0) - np.min(lidar_traj, axis=0))
# -----------------------------
# 4. 主循环：只用 odometry correction
# -----------------------------
odom_idx = 0
trajectory = []
odom_used = []

N_imu = len(imu_timestamp)

for k in range(1, N_imu):
    u = np.hstack((imu_lin_acc[k], imu_ang_vel[k]))
    dt = imu_timestamp[k] - imu_timestamp[k - 1]

    # prediction
    ekf.Q = process_noise_from_imu(
        imu_lin_acc_cov[k],
        imu_ang_vel_cov[k],
        dt
    )
    ekf.prediction(u, dt)

    # odometry correction
    while odom_idx < len(odom_timestamp) and odom_timestamp[odom_idx] <= imu_timestamp[k]:
        z_odom = odom_pose_pos[odom_idx]
        R_odom = odom_pose_cov[odom_idx][0:3, 0:3]

        ekf.correction_odometry(z_odom, R_odom)

        odom_used.append(z_odom.copy())
        odom_idx += 1

    trajectory.append(ekf.x[0:3].copy())


# -----------------------------
# 5. 输出检查
# -----------------------------
trajectory = np.array(trajectory)
odom_used = np.array(odom_used)

print("final position =", ekf.x[0:3])
print("final velocity =", ekf.x[3:6])
print("final quaternion =", ekf.x[6:10])
print("quat norm =", np.linalg.norm(ekf.x[6:10]))
print("used odom updates =", odom_idx)
print("trajectory shape =", trajectory.shape)

print("trajectory first 5:\n", trajectory[:5])
print("trajectory last 5:\n", trajectory[-5:])
print("EKF min =", np.min(trajectory, axis=0))
print("EKF max =", np.max(trajectory, axis=0))
print("EKF range =", np.max(trajectory, axis=0) - np.min(trajectory, axis=0))

if len(odom_used) > 0:
    print("odom_used shape =", odom_used.shape)
    print("odom min =", np.min(odom_used, axis=0))
    print("odom max =", np.max(odom_used, axis=0))
    print("odom range =", np.max(odom_used, axis=0) - np.min(odom_used, axis=0))


# -----------------------------
# 6. 统一起点后画图
# -----------------------------
trajectory_plot = trajectory - trajectory[0]

if len(odom_used) > 0:
    odom_plot = odom_used - odom_used[0]
else:
    odom_plot = np.empty((0, 3))

# 先输出原始起点坐标
print("EKF original start =", trajectory[0])

if len(odom_used) > 0:
    print("Odometry original start =", odom_used[0])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(
    trajectory_plot[:, 0],
    trajectory_plot[:, 1],
    trajectory_plot[:, 2],
    label="EKF"
)

# 标记 EKF 统一后的起点
ax.scatter(
    trajectory_plot[0, 0],
    trajectory_plot[0, 1],
    trajectory_plot[0, 2],
    s=80,
    marker='o'
)
ax.text(
    trajectory_plot[0, 0],
    trajectory_plot[0, 1],
    trajectory_plot[0, 2],
    ' EKF start'
)

if len(odom_plot) > 0:
    ax.plot(
        odom_plot[:, 0],
        odom_plot[:, 1],
        odom_plot[:, 2],
        label="Odometry"
    )

    # 标记 Odometry 统一后的起点
    ax.scatter(
        odom_plot[0, 0],
        odom_plot[0, 1],
        odom_plot[0, 2],
        s=80,
        marker='^'
    )
    ax.text(
        odom_plot[0, 0],
        odom_plot[0, 1],
        odom_plot[0, 2],
        ' Odom start'
    )

ax.legend()
plt.show()