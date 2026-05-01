import pyzed.sl as sl
import numpy as np
import open3d as o3d
from datetime import datetime
import os

def main():
    # 1. Initialize the ZED camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    
    # MAX CAPABILITY SETTINGS
    init_params.camera_resolution = sl.RESOLUTION.HD2K 
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL 
    init_params.coordinate_units = sl.UNIT.METER 
    
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED.")
        return

    # 2. Setup Runtime Parameters (Confidence & Fill)
    runtime_params = sl.RuntimeParameters()
    runtime_params.confidence_threshold = 95 # High detail
    runtime_params.enable_fill_mode = True   # Smoother surfaces

    point_cloud_sl = sl.Mat()
    
    # 3. Capture a frame
    print("Capturing High-Res Point Cloud...")
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_measure(point_cloud_sl, sl.MEASURE.XYZRGBA)
        point_cloud_np = point_cloud_sl.get_data()
        
        # 4. Clean Data
        pts = point_cloud_np.reshape(-1, 4)
        mask = np.all(np.isfinite(pts[:, :3]), axis=1)
        pts_clean = pts[mask].copy() 

        if len(pts_clean) == 0:
            print("No points captured.")
            return

        # 5. Extract XYZ and RGB
        xyz = pts_clean[:, :3]
        colors_raw = pts_clean[:, 3].copy()
        colors_packed = colors_raw.view(np.uint8).reshape(-1, 4)
        rgb = colors_packed[:, :3] / 255.0
        rgb = np.flip(rgb, axis=1) # BGR to RGB

        # 6. Create Open3D Object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        # 7. SAVE THE FILE
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"zed_cloud_{timestamp}.ply"
        
        print(f"Saving point cloud to: {filename}")
        # .ply is the standard format for 3D point clouds
        o3d.io.write_point_cloud(filename, pcd)

        # 8. Visualize
        print("Opening Viewer...")
        o3d.visualization.draw_geometries([pcd], window_name=filename)

    zed.close()

if __name__ == "__main__":
    main()