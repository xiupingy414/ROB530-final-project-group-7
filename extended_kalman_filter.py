
import numpy as np


# This function is used to wrap angles in radians to the interval [-pi, pi]
# pi maps to pi and -pi maps to -pi
def warpToPi(phase):
    x_wrap = np.remainder(phase, 2 * np.pi)
    while np.abs(x_wrap) > np.pi:
        x_wrap -= 2 * np.pi * np.sign(x_wrap)
    return x_wrap

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_normalize(q):
    return q / np.linalg.norm(q)

def quat_to_rot(q):        #transform a quaternion to a rotation matrix
    qw, qx, qy, qz = q

    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R

def quat_update(q, omega, dt):
    wx, wy, wz = omega

    dq = np.array([
        1.0,
        0.5 * wx * dt,
        0.5 * wy * dt,
        0.5 * wz * dt
    ])

    q_new = quat_multiply(q, dq)
    q_new = quat_normalize(q_new)
    return q_new

# Extended Kalman filter class for state estimation of a nonlinear system
class extended_kalman_filter:

    def __init__(self, system, init):
        # EKF Construct an instance of this class
        #
        # Inputs:
        #   system: system and noise models
        #   init:   initial state mean and covariance
        self.A = system.A
        self.Q = system.Q
        self.x = init.x
        self.Sigma = init.Sigma

    def f(self, x, u, dt):
        px, py, pz, vx, vy, vz, qw, qx, qy, qz = x
        ax, ay, az, wx, wy, wz = u

        # current state
        p = np.array([px, py, pz])
        v = np.array([vx, vy, vz])
        q = np.array([qw, qx, qy, qz])
        a_body = np.array([ax, ay, az])
        omega = np.array([wx, wy, wz])

        # 1. update orientation using gyro
        q_new = quat_update(q, omega, dt)

        # 2. rotate acceleration from body frame to world frame
        R = quat_to_rot(q)
        a_world = R @ a_body

        # 3. subtract gravity
        g = np.array([0.0, 0.0, 9.81])
        a_world = a_world - g

        # 4. update velocity and position
        v_new = v + a_world * dt
        p_new = p + v * dt + 0.5 * a_world * dt**2

        x_new = np.hstack((p_new, v_new, q_new))
        return x_new

    def prediction(self,u,dt):  # u is IMU measurement: [ax, ay, az, wx, wy, wz], 500Hz
        A = self.A(self.x, u, dt)

        # state prediction
        self.x = self.f(self.x, u, dt)

        # covariance prediction
        self.Sigma = A @ self.Sigma @ A.T + self.Q

    def correction_odometry(self, z, pose_cov):
        # z is odometry position measurement: [x, y, z]

        # measurement model: h(x) = position
        z_hat = self.x[0:3]

        # measurement Jacobian H
        H = np.zeros((3, 10))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        # measurement noise covariance
        R = pose_cov

        # innovation
        self.v = z - z_hat

        # innovation covariance
        self.S = H @ self.Sigma @ H.T + R

        # Kalman gain
        self.K = self.Sigma @ H.T @ np.linalg.inv(self.S)

        # state update
        self.x = self.x + self.K @ self.v

        # quaternion normalization after update
        self.x[6:10] = quat_normalize(self.x[6:10])

        # covariance update (Joseph form)
        I = np.eye(len(self.x))
        temp = I - self.K @ H
        self.Sigma = temp @ self.Sigma @ temp.T + self.K @ R @ self.K.T

    def correction_lidar(self, z):
        z_hat = self.x[0:3]

        H = np.zeros((3, 10))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        R = np.diag([0.01, 0.01, 0.01])**2

        v = z - z_hat
        S = H @ self.Sigma @ H.T + R
        K = self.Sigma @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ v
        self.x[6:10] = quat_normalize(self.x[6:10])

        I = np.eye(len(self.x))
        temp = I - K @ H
        self.Sigma = temp @ self.Sigma @ temp.T + K @ R @ K.T