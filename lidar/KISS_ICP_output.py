import os
import zarr
import numpy as np

z = zarr.open(
    r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/lidar/hesai_points_undistorted/hesai_points_undistorted",
    mode='r'
)

points = z['points']
intensity = z['intensity']

num_frames = points.shape[0]

save_dir = r"C:/Users/Lenovo/Desktop/Master/ROB530/final project/lidar/kiss_icp_input"
os.makedirs(save_dir, exist_ok=True)

for i in range(num_frames):
    xyz = np.asarray(points[i], dtype=np.float32)      # [N,3]
    inten = np.asarray(intensity[i], dtype=np.float32) # [N]

    inten = inten.reshape(-1, 1)  # → [N,1]

    pts4 = np.hstack([xyz, inten])  # → [N,4]

    filename = os.path.join(save_dir, f"{i:06d}.bin")
    pts4.astype(np.float32).tofile(filename)

print("Export finished with REAL intensity.")