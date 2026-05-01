import pyzed.sl as sl
import numpy as np
import open3d as o3d

def main():
    # 1. Initialize the ZED camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    
    init_params = sl.InitParameters()

    # 1. Highest possible resolution
    init_params.camera_resolution = sl.RESOLUTION.HD2K 

    # 2. Maximum Depth Accuracy
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL 

    # 3. Coordinate Units (Meters is best for robot math)
    init_params.coordinate_units = sl.UNIT.METER 

    # 4. Fill in the "holes" in the mesh
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD

    runtime_params = sl.RuntimeParameters()

    # Range: 1 to 100. 
    # 100 = Allow everything (noisy). 50 = Balanced. 
    # For testing, try 95 to see every possible point the camera can find.
    runtime_params.confidence_threshold = 99

    # This fills in small holes in the point cloud for a "solid" look
    runtime_params.enable_fill_mode = True 

    # Apply the parameters during the grab
    zed.grab(runtime_params)

    # NEURAL mode provides the best depth accuracy and cleaner edges
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL 
    init_params.coordinate_units = sl.UNIT.METER 
    
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED. Ensure it is plugged into a USB 3.0 port.")
        return

    # 2. Prepare containers
    point_cloud_sl = sl.Mat()
    
    # 3. Capture a frame
    print("Capturing NEURAL point cloud... (This may take a moment to initialize GPU)")
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # Retrieve the point cloud (XYZ + Color)
        zed.retrieve_measure(point_cloud_sl, sl.MEASURE.XYZRGBA)
        
        # Convert to numpy array [Height, Width, 4]
        # Channels: [0]=X, [1]=Y, [2]=Z, [3]=Color (packed float)
        point_cloud_np = point_cloud_sl.get_data()
        
        # 4. Reshape and clean data
        pts = point_cloud_np.reshape(-1, 4)
        
        # Remove "NaN" or "Inf" points (areas with no depth data)
        mask = np.all(np.isfinite(pts[:, :3]), axis=1)
        pts_clean = pts[mask].copy() 

        if len(pts_clean) == 0:
            print("Error: No valid depth points detected.")
            return

        # 5. Extract Geometry (XYZ)
        xyz = pts_clean[:, :3]
        
        # 6. Extract and Normalize Colors
        # We must copy the 4th column specifically to make it contiguous 
        # before viewing it as 4 individual bytes (uint8)
        colors_raw = pts_clean[:, 3].copy()
        colors_packed = colors_raw.view(np.uint8).reshape(-1, 4)
        
        # ZED stores color as BGRA; Open3D wants RGB normalized to [0, 1]
        rgb = colors_packed[:, :3] / 255.0
        rgb = np.flip(rgb, axis=1) 

        # 7. Create Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        print(f"Success! Visualizing {len(xyz)} points.")
        print("-" * 30)
        print("CONTROLS:")
        print("  - Left-Click + Drag: Rotate")
        print("  - Right-Click + Drag: Pan")
        print("  - Mouse Wheel: Zoom")
        print("  - Close window to exit.")
        print("-" * 30)
        
        # 8. Launch the 3D Viewer
        o3d.visualization.draw_geometries([pcd], 
                                          window_name="ZED 3D Point Cloud Inspection",
                                          width=1280, 
                                          height=720)

    else:
        print("Error: Camera grab failed.")

    zed.close()

if __name__ == "__main__":
    main()