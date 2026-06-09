import numpy as np

def project_3d_to_pixel(point_3d, camera_params, width=None, height=None):
    """
    Projects a 3D world coordinate (X, Y, Z) onto the 2D camera image plane (x, y)
    using the camera parameters (view and projection matrices) returned by Replicator.
    """
    view_matrix = camera_params.get("cameraViewTransform")
    proj_matrix = camera_params.get("cameraProjection")

    if view_matrix is None or proj_matrix is None:
        return None

    resolution = camera_params.get("renderProductResolution")
    if resolution is not None:
        width, height = int(resolution[0]), int(resolution[1])
    if width is None or height is None:
        return None

    view_matrix = np.asarray(view_matrix, dtype=np.float64).reshape((4, 4))
    proj_matrix = np.asarray(proj_matrix, dtype=np.float64).reshape((4, 4))
    point_world = np.array(
        [float(point_3d[0]), float(point_3d[1]), float(point_3d[2]), 1.0],
        dtype=np.float64,
    )

    # Replicator camera matrices use row-vector multiplication.
    point_camera = point_world @ view_matrix
    if point_camera[2] >= 0.0:
        return None

    point_clip = point_camera @ proj_matrix
    if not np.all(np.isfinite(point_clip)) or abs(point_clip[3]) < 1e-8:
        return None

    point_ndc = point_clip / point_clip[3]
    if not np.all(np.isfinite(point_ndc[:2])):
        return None

    screen_x = (point_ndc[0] + 1.0) * 0.5 * width
    screen_y = (1.0 - point_ndc[1]) * 0.5 * height
    return (int(round(screen_x)), int(round(screen_y)))
