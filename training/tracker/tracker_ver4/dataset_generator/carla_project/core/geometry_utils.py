import numpy as np
import carla

def build_projection_matrix(w, h, fov):
    """
    Builds the intrinsic camera matrix (K) given width, height, and horizontal FOV (degrees).
    """
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_3d_world_from_pixel(x, y, depth_image_meters, K, camera_transform):
    """
    Unprojects a 2D pixel (x,y) to a 3D absolute world coordinate using the CARLA depth buffer.
    
    Args:
        x, y: Pixel coordinates.
        depth_image_meters: A 2D numpy array containing absolute depth in meters.
        K: 3x3 intrinsic projection matrix.
        camera_transform: carla.Transform of the camera capturing the image.
        
    Returns:
        np.array([X, Y, Z]) world coordinates, or None if invalid depth (sky).
    """
    height, width = depth_image_meters.shape
    if not (0 <= x < width and 0 <= y < height):
        return None

    Z = depth_image_meters[int(y), int(x)]
    if Z == 0 or Z > 999.0:
        # 0 or 1000m usually means sky/void in CARLA depending on parsing
        return None
        
    # Unproject to camera coordinates (3D)
    X = (x - K[0, 2]) * Z / K[0, 0]
    Y = (y - K[1, 2]) * Z / K[1, 1]
    
    # CARLA camera coordinate system differs from standard computer vision.
    # Standard CV: +Z is forward, +X is right, +Y is down.
    # CARLA UE4 World: +X is forward, +Y is right, +Z is up.
    # Converting Camera (X, Y, Z) to UE4 Local (X_ue, Y_ue, Z_ue):
    # Forward = Z, Right = X, Up = -Y
    p_ue4_local = np.array([Z, X, -Y, 1.0])
    
    # Get Extrinsic matrix (Camera Local -> Absolute World)
    extrinsic = np.array(camera_transform.get_matrix())
    
    # Multiply Extrinsic Matrix by local vector
    p_world = np.dot(extrinsic, p_ue4_local)
    return p_world[:3]

def get_3d_world_to_pixel(world_point, K, camera_transform):
    """
    Projects a 3D absolute world coordinate back to a 2D pixel (x,y) on the camera screen.
    Returns (x, y) if the point is in front of the camera, else None.
    """
    if world_point is None:
        return None
        
    p_world = np.array([world_point[0], world_point[1], world_point[2], 1.0])
    
    # 1. World to Camera Local (inverse of extrinsic)
    extrinsic_inv = np.array(camera_transform.get_inverse_matrix())
    p_ue4_local = np.dot(extrinsic_inv, p_world)
    
    # 2. UE4 Local to Camera Standard CV space
    # UE4: X is forward, Y is right, Z is up
    # CV: Z is forward, X is right, Y is down
    Z_cam = p_ue4_local[0]
    X_cam = p_ue4_local[1]
    Y_cam = -p_ue4_local[2]
    
    if Z_cam <= 0.0:
        # Point is behind the camera plane
        return None
        
    p_cam = np.array([X_cam, Y_cam, Z_cam])
    
    # 3. Project to 2D pixel coordinates using Intrinsic Matrix K
    p_img = np.dot(K, p_cam)
    
    # Normalize by Z
    u = p_img[0] / p_img[2]
    v = p_img[1] / p_img[2]
    
    return (int(u), int(v))

