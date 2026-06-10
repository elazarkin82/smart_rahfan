import numpy as np


def pixel_depth_to_world(pixel, distance, camera_params, width=None, height=None):
    """Back-projects a pixel and Euclidean camera distance into world space."""
    view_matrix = camera_params.get("cameraViewTransform")
    proj_matrix = camera_params.get("cameraProjection")
    if view_matrix is None or proj_matrix is None:
        return None

    resolution = camera_params.get("renderProductResolution")
    if resolution is not None:
        width, height = int(resolution[0]), int(resolution[1])
    if width is None or height is None or width <= 0 or height <= 0:
        return None
    if not np.isfinite(distance) or distance <= 0.0:
        return None

    view_matrix = np.asarray(view_matrix, dtype=np.float64).reshape((4, 4))
    proj_matrix = np.asarray(proj_matrix, dtype=np.float64).reshape((4, 4))
    try:
        inv_view = np.linalg.inv(view_matrix)
        inv_proj = np.linalg.inv(proj_matrix)
    except np.linalg.LinAlgError:
        return None

    ndc_x = 2.0 * float(pixel[0]) / float(width) - 1.0
    ndc_y = 1.0 - 2.0 * float(pixel[1]) / float(height)
    clip = np.array([ndc_x, ndc_y, 0.0, 1.0], dtype=np.float64)
    camera_point = clip @ inv_proj
    if abs(camera_point[3]) < 1e-8:
        return None
    camera_point /= camera_point[3]

    camera_origin = np.array([0.0, 0.0, 0.0, 1.0]) @ inv_view
    world_point = camera_point @ inv_view
    ray = world_point[:3] - camera_origin[:3]
    ray_norm = np.linalg.norm(ray)
    if not np.isfinite(ray_norm) or ray_norm < 1e-8:
        return None

    target = camera_origin[:3] + ray * (float(distance) / ray_norm)
    return target if np.all(np.isfinite(target)) else None


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
