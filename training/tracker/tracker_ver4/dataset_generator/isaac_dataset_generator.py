#!/usr/bin/env python3
import argparse
import sys
import os
import json
import random
import math
import pickle
import traceback
from collections import Counter
import numpy as np
import cv2

# Set up paths to load local packages
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.dirname(script_dir))

from isaac.core.isaac_client import IsaacClientManager
from isaac.core.sensor_manager import IsaacSensorManager
from isaac.core.geometry_utils import pixel_depth_to_world, project_3d_to_pixel
from isaac.core.viewpoint_bank import (
    build_viewpoint_bank,
    load_viewpoint_bank,
    prepare_bank_context,
)

def load_config(path="pipeline_config.json"):
    if not os.path.exists(path) and path == "pipeline_config.json":
        path = os.path.join(script_dir, path)
    with open(path, 'r') as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TargetTrackerVer4 flights with NVIDIA Isaac Sim."
    )
    parser.add_argument(
        "--single-flight",
        "--single_flight",
        action="store_true",
        help="Generate exactly one new flight and save it to the existing cache.",
    )
    parser.add_argument(
        "--disable-all-features",
        "--disable_all_features",
        action="store_true",
        help="Disable all optional flight augmentation and debug features.",
    )
    parser.add_argument("--disable-random-fov", action="store_true")
    parser.add_argument("--disable-look-offset", action="store_true")
    parser.add_argument("--disable-step-noise", action="store_true")
    parser.add_argument("--disable-drift", action="store_true")
    parser.add_argument("--disable-jitter", action="store_true")
    parser.add_argument("--disable-false-negatives", action="store_true")
    parser.add_argument("--disable-texture-filter", action="store_true")
    parser.add_argument("--disable-debug-output", action="store_true")
    return parser.parse_args()


def resolve_disabled_features(args):
    feature_names = (
        "random_fov",
        "look_offset",
        "step_noise",
        "drift",
        "jitter",
        "false_negatives",
        "texture_filter",
        "debug_output",
    )
    if args.disable_all_features:
        return {name: True for name in feature_names}
    return {
        name: bool(getattr(args, f"disable_{name}"))
        for name in feature_names
    }


def get_viewpoint_bank_config(isaac_config):
    defaults = {
        "enabled": True,
        "cache_dir": "viewpoint_cache",
        "rebuild": False,
        "scout_resolution": [320, 240],
        "grid_size": [3, 3],
        "yaw_count": 4,
        "pitch_degrees": [-45.0],
        "altitude_m": 35.0,
        "edge_margin_ratio": 0.1,
        "max_scene_span_m": 500.0,
        "central_margin_ratio": 0.15,
        "min_valid_depth_ratio": 0.05,
        "pixel_stride": 12,
        "max_targets_per_viewpoint": 64,
        "max_bank_entries": 4096,
        "min_entries": 32,
        "reprojection_tolerance_px": 2.0,
    }
    defaults.update(isaac_config.get("viewpoint_bank", {}))
    return defaults


def lerp(a, b, t):
    return a + (b - a) * t

def nlerp_quat(q1, q2, t):
    """Normalized Linear Interpolation (NLerp) for smooth quaternion rotation."""
    # Ensure shortest path
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
    q = q1 * (1.0 - t) + q2 * t
    return q / np.linalg.norm(q)

def quaternion_multiply(q1, q2):
    """Multiplies two quaternions in [w, x, y, z] order."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    q = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    return q / np.linalg.norm(q)

def quaternion_from_euler(roll, pitch, yaw):
    """Creates a local rotation quaternion from Euler angles in radians."""
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])

def rotate_vector(vector, axis, angle):
    """Rotates a vector around an axis using Rodrigues' formula."""
    axis = axis / np.linalg.norm(axis)
    return (
        vector * math.cos(angle)
        + np.cross(axis, vector) * math.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - math.cos(angle))
    )

def rotation_matrix_to_quaternion(R):
    """Converts a 3x3 rotation matrix to a normalized quaternion [w, x, y, z]."""
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)

def get_lookat_quaternion(eye, target):
    """Computes a quaternion so that the camera looks directly at the target coordinate."""
    forward = target - eye
    dist = np.linalg.norm(forward)
    if dist > 0:
        forward_unit = forward / dist
    else:
        forward_unit = np.array([1.0, 0.0, 0.0])
        
    temp_up = np.array([0.0, 0.0, 1.0])
    # Prevent singularities
    if abs(np.dot(forward_unit, temp_up)) > 0.999:
        temp_up = np.array([0.0, 1.0, 0.0])
        
    right = np.cross(forward_unit, temp_up)
    right /= np.linalg.norm(right)
    
    up = np.cross(right, forward_unit)
    up /= np.linalg.norm(up)
    
    # In USD, camera default looks along -Z axis. 
    # Therefore, Column 2 of the camera matrix is -forward
    R = np.stack([right, up, -forward_unit], axis=-1)
    return rotation_matrix_to_quaternion(R)

def get_map_for_flight(maps, flight_index, num_flights_target):
    """Matches CARLA's ordered, contiguous flight allocation per map."""
    if not maps:
        raise ValueError("At least one Isaac map must be configured.")

    flights_per_map = max(1, num_flights_target // len(maps))
    map_idx = min(flight_index // flights_per_map, len(maps) - 1)
    return maps[map_idx]


def select_available_map(maps, scheduled_map, failed_maps):
    """Uses the scheduled map, or the next healthy map after a failure."""
    if scheduled_map not in failed_maps:
        return scheduled_map

    scheduled_idx = maps.index(scheduled_map)
    for offset in range(1, len(maps) + 1):
        candidate = maps[(scheduled_idx + offset) % len(maps)]
        if candidate not in failed_maps:
            return candidate
    return None


def get_depth_at_pixel(depth_arr, pixel):
    if depth_arr is None:
        return None
    x, y = int(pixel[0]), int(pixel[1])
    if y < 0 or y >= depth_arr.shape[0] or x < 0 or x >= depth_arr.shape[1]:
        return None
    distance = float(depth_arr[y, x])
    if not np.isfinite(distance) or distance <= 0.0:
        return None
    return distance


def find_visible_target(
    rgb_arr,
    depth_arr,
    params,
    width,
    height,
    min_texture_std,
    min_distance,
    max_distance,
    max_samples,
):
    """Selects a textured, visible RGB/depth pixel like the CARLA generator."""
    failures = Counter()
    if rgb_arr is None:
        failures["scout_missing_rgb"] += 1
        return None, failures
    if depth_arr is None:
        failures["scout_missing_depth"] += 1
        return None, failures
    if params is None:
        failures["scout_missing_camera_params"] += 1
        return None, failures

    margin_x = max(1, int(width * 0.25))
    margin_y = max(1, int(height * 0.25))
    for _ in range(max_samples):
        px = random.randint(margin_x, width - margin_x - 1)
        py = random.randint(margin_y, height - margin_y - 1)

        patch = rgb_arr[
            max(0, py - 7):min(height, py + 8),
            max(0, px - 7):min(width, px + 8),
        ]
        if patch.size == 0:
            failures["scout_empty_patch"] += 1
            continue
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        if float(np.std(patch_gray)) < min_texture_std:
            failures["scout_low_texture"] += 1
            continue

        distance = get_depth_at_pixel(depth_arr, (px, py))
        if distance is None or not (min_distance <= distance <= max_distance):
            failures["scout_invalid_distance"] += 1
            continue

        target_3d = pixel_depth_to_world(
            (px, py),
            distance,
            params,
            width,
            height,
        )
        if target_3d is None:
            failures["scout_backprojection_failed"] += 1
            continue

        reprojection = project_3d_to_pixel(target_3d, params, width, height)
        if (
            reprojection is None
            or abs(reprojection[0] - px) > 2
            or abs(reprojection[1] - py) > 2
        ):
            failures["scout_reprojection_mismatch"] += 1
            continue
        return (target_3d, (px, py), distance), failures

    failures["scout_no_valid_target"] += 1
    return None, failures


def report_failed_attempt(
    target_map,
    attempts_for_flight,
    max_attempts_per_flight,
    rejection_counts,
    failed_maps,
):
    if attempts_for_flight < max_attempts_per_flight:
        print("[!] Flight generation failed. Retrying on the loaded stage...")
        return

    summary = ", ".join(
        f"{reason}={count}"
        for reason, count in rejection_counts.most_common()
    )
    print(
        f"[!] Map exhausted after {max_attempts_per_flight} attempts: "
        f"{target_map}"
    )
    if summary:
        print(f"[!] Rejection summary: {summary}")
    failed_maps.add(target_map)


def main():
    args = parse_args()
    disabled_features = resolve_disabled_features(args)
    config = load_config()
    
    cache_dir = config['generation'].get('cache_dir', 'cache')
    debug_dir = config['generation'].get('debug_dir', 'debug')
    
    # Resolve relative directories relative to script
    if not os.path.isabs(cache_dir):
        cache_dir = os.path.join(script_dir, cache_dir)
    if not os.path.isabs(debug_dir):
        debug_dir = os.path.join(script_dir, debug_dir)
        
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    
    configured_flights_target = config['generation']['num_flights_to_cache']
    
    existing_flights = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
    flights_generated = len(existing_flights)

    if args.single_flight:
        num_flights_target = flights_generated + 1
        print(
            f"[*] Single-flight mode: generating one new flight "
            f"(cache target: {num_flights_target})."
        )
    else:
        num_flights_target = configured_flights_target

    if not args.single_flight and flights_generated >= num_flights_target:
        print(f"[+] Cache already has {flights_generated} flights. Skipping generation.")
        return

    disabled_feature_names = [
        name.replace("_", "-")
        for name, disabled in disabled_features.items()
        if disabled
    ]
    if disabled_feature_names:
        print(f"[*] Disabled features: {', '.join(disabled_feature_names)}")
        
    width = config['sensor']['width']
    height = config['sensor']['height']
    fov = config['sensor']['fov']
    fov_range = config['sensor'].get('fov_range', [fov, fov])
    
    # Read Isaac Sim configs
    isaac_cfg = config.get("isaac", {})
    maps = isaac_cfg.get("maps", ["/Isaac/Environments/Simple_Warehouse/warehouse.usd"])
    renderer = isaac_cfg.get("renderer", "RayTracedLighting")
    headless = isaac_cfg.get("headless", True)
    rt_subframes = isaac_cfg.get("rt_subframes", 1)
    warmup_frames = isaac_cfg.get("warmup_frames", 1)
    stage_load_timeout = isaac_cfg.get("stage_load_timeout_sec", 180.0)
    repair_asset_paths = isaac_cfg.get("repair_missing_asset_paths", True)
    max_attempts_per_flight = isaac_cfg.get("max_attempts_per_flight", 25)
    target_search_samples = isaac_cfg.get("target_search_samples", 100)
    target_distance_range = isaac_cfg.get("target_distance_range", [8.0, 80.0])
    stop_distance_range = isaac_cfg.get("stop_distance_range_m", [3.0, 6.0])
    camera_clipping_range = isaac_cfg.get(
        "camera_clipping_range_m",
        [0.1, 1000.0],
    )
    occlusion_tolerance = isaac_cfg.get("occlusion_tolerance_m", 1.5)
    disable_async_rendering = isaac_cfg.get("disable_async_rendering", True)
    viewpoint_bank_cfg = get_viewpoint_bank_config(isaac_cfg)
    force_process_exit = isaac_cfg.get("force_process_exit_on_shutdown", True)

    viewpoint_cache_dir = viewpoint_bank_cfg["cache_dir"]
    if not os.path.isabs(viewpoint_cache_dir):
        viewpoint_cache_dir = os.path.join(script_dir, viewpoint_cache_dir)

    if not maps:
        raise ValueError("isaac.maps must contain at least one USD map.")
    
    frames_per_flight = config['generation']['frames_per_flight']
    num_false_negatives = config['generation']['num_false_negatives']
    debug_interval = config['generation']['debug_interval']
    min_texture_std = config['generation'].get('min_texture_std', 2.0)

    if disabled_features["false_negatives"]:
        num_false_negatives = 0
    if disabled_features["texture_filter"]:
        min_texture_std = 0.0
    
    # Noise parameters
    noise_cfg = config['generation']['noise_params']
    pos_xy_amp = noise_cfg.get('pos_xy_amp', 0.3)
    pos_z_amp = noise_cfg.get('pos_z_amp', 0.1)
    rot_pitch_amp = noise_cfg.get('rot_pitch_amp', 0.5)
    rot_yaw_amp = noise_cfg.get('rot_yaw_amp', 0.5)
    rot_roll_amp = noise_cfg.get('rot_roll_amp', 2.5)
    
    # 1. Initialize Simulator
    sys.argv = [sys.argv[0]]
    client_mgr = IsaacClientManager(headless=headless, renderer=renderer)
    simulation_app, stage_context = client_mgr.connect()

    if disable_async_rendering:
        import carb

        carb.settings.get_settings().set(
            "/exts/isaacsim.core.throttling/enable_async",
            False,
        )
    
    sensor_mgr = IsaacSensorManager(camera_path="/World/Camera", width=width, height=height, fov=fov)
    current_map = None
    viewpoint_entries = None
    viewpoint_pool = []
    attempts_for_flight = 0
    failed_maps = set()
    rejection_counts = Counter()
    exit_code = 0
    pending_exception = None
    
    try:
        while flights_generated < num_flights_target:
            # Use the same contiguous per-map allocation as the CARLA generator.
            scheduled_map = get_map_for_flight(
                maps,
                flights_generated,
                max(configured_flights_target, num_flights_target),
            )
            target_map = select_available_map(maps, scheduled_map, failed_maps)
            if target_map is None:
                raise RuntimeError(
                    f"All {len(maps)} Isaac maps failed while generating "
                    f"flight {flights_generated}."
                )
            if target_map != scheduled_map and current_map != target_map:
                print(
                    f"[!] Scheduled map is unavailable; continuing with "
                    f"{target_map}"
                )

            # Keep a loaded stage alive across retries to avoid recompiling RTX
            # shaders and rebuilding the Replicator graph on every failed flight.
            if current_map != target_map:
                flights_per_map = max(1, num_flights_target // len(maps))
                map_idx = maps.index(target_map)
                first_flight = map_idx * flights_per_map
                last_flight = (
                    num_flights_target - 1
                    if map_idx == len(maps) - 1
                    else min(num_flights_target - 1, first_flight + flights_per_map - 1)
                )
                print(
                    f"[*] Map {map_idx + 1}/{len(maps)} will generate flights "
                    f"{first_flight + 1}-{last_flight + 1}."
                )
                sensor_mgr.destroy()
                stage = client_mgr.load_map(
                    target_map,
                    timeout_sec=stage_load_timeout,
                    repair_asset_paths=repair_asset_paths,
                )

                # Retrieve dynamic scale factor (meters to USD stage units)
                scale = client_mgr.scale_to_stage
                scaled_distance_range = [r * scale for r in target_distance_range]
                scaled_occlusion_tolerance = occlusion_tolerance * scale
                scaled_pos_xy_amp = pos_xy_amp * scale
                scaled_pos_z_amp = pos_z_amp * scale
                clipping_range_stage = [
                    float(camera_clipping_range[0]) * scale,
                    float(camera_clipping_range[1]) * scale,
                ]

                viewpoint_entries = None
                viewpoint_pool = []
                if viewpoint_bank_cfg["enabled"]:
                    scout_width, scout_height = [
                        int(value)
                        for value in viewpoint_bank_cfg["scout_resolution"]
                    ]
                    bank_context = prepare_bank_context(
                        stage,
                        target_map,
                        scale,
                        viewpoint_bank_cfg,
                        (scout_width, scout_height),
                        float(fov),
                        target_distance_range,
                        min_texture_std,
                        viewpoint_cache_dir,
                    )
                    viewpoint_entries = load_viewpoint_bank(
                        bank_context,
                        rebuild=bool(viewpoint_bank_cfg["rebuild"]),
                    )

                    if viewpoint_entries is None:
                        scout_sensor_mgr = IsaacSensorManager(
                            camera_path="/World/Camera",
                            width=scout_width,
                            height=scout_height,
                            fov=fov,
                        )
                        try:
                            scout_sensor_mgr.create_camera(
                                clipping_range=clipping_range_stage
                            )
                            scout_sensor_mgr.initialize_replicator(
                                simulation_app,
                                warmup_frames=warmup_frames,
                                rt_subframes=rt_subframes,
                            )
                            viewpoint_entries = build_viewpoint_bank(
                                stage,
                                scout_sensor_mgr,
                                scale,
                                viewpoint_bank_cfg,
                                bank_context,
                                target_distance_range,
                                min_texture_std,
                                rt_subframes,
                            )
                        finally:
                            scout_sensor_mgr.destroy()

                    print(
                        f"[+] Viewpoint bank ready with "
                        f"{len(viewpoint_entries)} target(s)."
                    )
                else:
                    client_mgr.prepare_target_candidates()

                sensor_mgr = IsaacSensorManager(
                    camera_path="/World/Camera",
                    width=width,
                    height=height,
                    fov=fov,
                )
                sensor_mgr.create_camera(
                    clipping_range=clipping_range_stage
                )
                sensor_mgr.initialize_replicator(
                    simulation_app,
                    warmup_frames=warmup_frames,
                    rt_subframes=rt_subframes,
                )

                current_map = target_map
                attempts_for_flight = 0
                rejection_counts.clear()

            attempts_for_flight += 1

            if disabled_features["random_fov"]:
                flight_fov = float(fov)
            else:
                flight_fov = random.uniform(
                    float(fov_range[0]),
                    float(fov_range[1]),
                )
            sensor_mgr.set_fov(flight_fov)
            
            if viewpoint_entries is not None:
                if not viewpoint_pool:
                    viewpoint_pool = list(viewpoint_entries)
                    random.shuffle(viewpoint_pool)
                viewpoint = viewpoint_pool.pop()
                start_pos = np.asarray(
                    viewpoint["start_pos"],
                    dtype=np.float64,
                )
                target_3d = np.asarray(
                    viewpoint["target_3d"],
                    dtype=np.float64,
                )
                selected_distance = float(viewpoint["distance"])
            else:
                # Legacy fallback when the viewpoint bank is disabled.
                camera_anchor = client_mgr.get_random_target()
                start_dist = random.uniform(20.0 * scale, 45.0 * scale)
                theta = random.uniform(0, 2 * math.pi)
                phi = random.uniform(
                    math.radians(15),
                    math.radians(45),
                )
                start_pos = camera_anchor + start_dist * np.array([
                    math.sin(phi) * math.cos(theta),
                    math.sin(phi) * math.sin(theta),
                    math.cos(phi),
                ])

                scout_quat = get_lookat_quaternion(start_pos, camera_anchor)
                sensor_mgr.move_to(start_pos, scout_quat)
                scout_rgb, scout_depth, scout_params = sensor_mgr.get_sync_data(
                    rt_subframes=rt_subframes
                )
                target_result, scout_failures = find_visible_target(
                    scout_rgb,
                    scout_depth,
                    scout_params,
                    width,
                    height,
                    min_texture_std,
                    float(scaled_distance_range[0]),
                    float(scaled_distance_range[1]),
                    int(target_search_samples),
                )
                rejection_counts.update(scout_failures)
                if target_result is None:
                    report_failed_attempt(
                        target_map,
                        attempts_for_flight,
                        max_attempts_per_flight,
                        rejection_counts,
                        failed_maps,
                    )
                    continue
                target_3d, _, selected_distance = target_result

            # Flight approaches closely to target
            stop_dist = random.uniform(
                float(stop_distance_range[0]) * scale,
                float(stop_distance_range[1]) * scale,
            )
            vec = target_3d - start_pos
            start_dist = np.linalg.norm(vec)
            if start_dist <= stop_dist:
                rejection_counts["target_too_close_for_flight"] += 1
                report_failed_attempt(
                    target_map,
                    attempts_for_flight,
                    max_attempts_per_flight,
                    rejection_counts,
                    failed_maps,
                )
                continue
            dir_vec = vec / start_dist
            end_pos = start_pos + dir_vec * (start_dist - stop_dist)
            
            # Compute camera orientations looking at the target
            start_quat = get_lookat_quaternion(start_pos, target_3d)
            
            # Look slightly off-center at the end of the flight
            if disabled_features["look_offset"]:
                pitch_offset = 0.0
                yaw_offset = 0.0
            else:
                pitch_offset = random.uniform(-10.0, 10.0)
                yaw_offset = random.uniform(-10.0, 10.0)
            
            # Create end rotation matrix with offset
            forward_unit = dir_vec
            temp_up = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(forward_unit, temp_up)) > 0.999:
                temp_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward_unit, temp_up)
            right /= np.linalg.norm(right)
            up = np.cross(right, forward_unit)
            up /= np.linalg.norm(up)
            
            # Apply offset to vectors
            pitch_rad = math.radians(pitch_offset)
            yaw_rad = math.radians(yaw_offset)
            
            # Perturb forward unit vector
            perturbed_forward = forward_unit + right * math.sin(yaw_rad) + up * math.sin(pitch_rad)
            perturbed_forward /= np.linalg.norm(perturbed_forward)
            
            end_quat = get_lookat_quaternion(end_pos, end_pos + perturbed_forward * (10.0 * scale))
            
            # Compute orthogonal vectors for strafing/jitter
            right_vec = right
            up_vec = up
            if disabled_features["drift"]:
                drift_right_amp = 0.0
                drift_up_amp = 0.0
                drift_right_freq = 0.0
                drift_up_freq = 0.0
            else:
                drift_right_amp = random.uniform(-2.5 * scale, 2.5 * scale)
                drift_up_amp = random.uniform(-1.5 * scale, 1.5 * scale)
                drift_right_freq = random.uniform(1.0, 3.0)
                drift_up_freq = random.uniform(1.0, 3.0)
            
            print(
                f"[{flights_generated+1}/{num_flights_target}] Generating flight "
                f"(attempt {attempts_for_flight}/{max_attempts_per_flight}, "
                f"target distance {selected_distance / scale:.1f}m)"
            )
            
            flight_data = []
            valid_flight = True
            failure_reason = None
            
            is_debug = (
                not disabled_features["debug_output"]
                and flights_generated % debug_interval == 0
            )
            flight_debug_dir = os.path.join(debug_dir, f"flight_{flights_generated:04d}")
            if is_debug:
                os.makedirs(flight_debug_dir, exist_ok=True)
                
            for frame_idx in range(frames_per_flight):
                t = frame_idx / float(frames_per_flight - 1)
                
                # Perturb step size slightly for non-linearity
                if (
                    not disabled_features["step_noise"]
                    and 0 < frame_idx < frames_per_flight - 1
                ):
                    t_noisy = t + random.uniform(-0.04, 0.04)
                    t_noisy = max(0.01, min(0.99, t_noisy))
                else:
                    t_noisy = t
                    
                smooth_t = t_noisy * t_noisy * (3 - 2 * t_noisy)
                
                # Interpolate base transform
                clean_pos = lerp(start_pos, end_pos, smooth_t)
                clean_quat = nlerp_quat(start_quat, end_quat, smooth_t)
                
                # Apply lateral strafe drift
                drift_factor = math.sin(smooth_t * math.pi)
                drift_offset = (right_vec * drift_right_amp * math.sin(smooth_t * drift_right_freq * math.pi) +
                                up_vec * drift_up_amp * math.sin(smooth_t * drift_up_freq * math.pi)) * drift_factor
                
                clean_pos += drift_offset
                
                frame_success = False
                for attempt in range(5):
                    pos = clean_pos.copy()
                    quat = clean_quat.copy()
                    
                    if not disabled_features["jitter"]:
                        # Scale down jitter noise as camera gets closer.
                        current_dist = (
                            start_dist
                            - smooth_t * (start_dist - stop_dist)
                        )
                        noise_scale = min(
                            1.0,
                            current_dist / (30.0 * scale),
                        )

                        phase = (
                            flights_generated * 10
                            + frame_idx * 0.2
                            + attempt * 5.0
                        )
                        pos[0] += (
                            math.sin(phase * 1.3)
                            * scaled_pos_xy_amp
                            * noise_scale
                        )
                        pos[1] += (
                            math.cos(phase * 1.7)
                            * scaled_pos_xy_amp
                            * noise_scale
                        )
                        pos[2] += (
                            math.sin(phase * 0.9)
                            * scaled_pos_z_amp
                            * noise_scale
                        )

                        roll_wobble = (
                            math.sin(phase * 2.5)
                            * math.radians(rot_roll_amp)
                            * noise_scale
                        )
                        pitch_wobble = (
                            math.cos(phase * 2.1)
                            * math.radians(rot_pitch_amp)
                            * noise_scale
                        )
                        yaw_wobble = (
                            math.cos(phase * 1.5)
                            * math.radians(rot_yaw_amp)
                            * noise_scale
                        )

                        rot_perturbed = get_lookat_quaternion(pos, target_3d)
                        quat = nlerp_quat(quat, rot_perturbed, 0.5)
                        jitter_quat = quaternion_from_euler(
                            roll_wobble,
                            pitch_wobble,
                            yaw_wobble,
                        )
                        quat = quaternion_multiply(quat, jitter_quat)
                    
                    if (
                        disabled_features["look_offset"]
                        and disabled_features["jitter"]
                    ):
                        sensor_mgr.look_at(pos, target_3d)
                    else:
                        sensor_mgr.move_to(pos, quat)
                    
                    # Get frame data from replicator
                    rgb_arr, depth_arr, params = sensor_mgr.get_sync_data(
                        rt_subframes=rt_subframes
                    )
                    
                    if rgb_arr is None:
                        rejection_counts["positive_missing_rgb"] += 1
                        continue
                    if depth_arr is None:
                        rejection_counts["positive_missing_depth"] += 1
                        continue
                    if params is None:
                        rejection_counts["positive_missing_camera_params"] += 1
                        continue
                        
                    # Project target onto 2D image coordinates
                    px_draw = project_3d_to_pixel(target_3d, params, width, height)
                    # Target must remain in screen bounds during positive frames
                    if px_draw is None or not (0 <= px_draw[0] < width and 0 <= px_draw[1] < height):
                        rejection_counts["positive_target_offscreen"] += 1
                        continue

                    actual_dist = np.linalg.norm(target_3d - pos)
                    visible_dist = get_depth_at_pixel(depth_arr, px_draw)
                    if visible_dist is None:
                        rejection_counts["positive_invalid_depth"] += 1
                        continue
                    if visible_dist < actual_dist - scaled_occlusion_tolerance:
                        rejection_counts["positive_target_occluded"] += 1
                        continue
                        
                    frame_success = True
                    break
                    
                if not frame_success:
                    valid_flight = False
                    failure_reason = "positive_frame_exhausted"
                    break
                    
                gray_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
                patch_radius = 24
                patch_x0 = max(0, px_draw[0] - patch_radius)
                patch_x1 = min(width, px_draw[0] + patch_radius + 1)
                patch_y0 = max(0, px_draw[1] - patch_radius)
                patch_y1 = min(height, px_draw[1] + patch_radius + 1)
                target_patch = gray_arr[patch_y0:patch_y1, patch_x0:patch_x1]
                if target_patch.size == 0 or float(np.std(target_patch)) < min_texture_std:
                    valid_flight = False
                    failure_reason = "positive_low_texture"
                    break

                flight_data.append({
                    "frame_index": frame_idx,
                    "image_gray": gray_arr,
                    "target_2d": px_draw,
                    "target_3d": target_3d,
                    "distance_to_target": actual_dist
                })
                
                if is_debug:
                    bgr = cv2.cvtColor(gray_arr, cv2.COLOR_GRAY2BGR)
                    cv2.circle(bgr, px_draw, 20, (0, 0, 255), 2)
                    cv2.circle(bgr, px_draw, 3, (0, 0, 255), -1)
                    cv2.imwrite(os.path.join(flight_debug_dir, f"{frame_idx:02d}_frame.png"), bgr)
                    
            # 5. Handle False Negatives (target completely out of frame)
            if valid_flight:
                for fn_idx in range(num_false_negatives):
                    target_vector = target_3d - pos
                    rotation_axis = np.array([0.0, 0.0, 1.0])
                    target_direction = target_vector / np.linalg.norm(target_vector)
                    if abs(np.dot(target_direction, rotation_axis)) > 0.95:
                        rotation_axis = np.array([0.0, 1.0, 0.0])
                    away_vector = rotate_vector(
                        target_vector,
                        rotation_axis,
                        math.radians(110.0 + fn_idx * 20.0),
                    )
                    fn_quat = get_lookat_quaternion(pos, pos + away_vector)
                    
                    sensor_mgr.move_to(pos, fn_quat)
                    rgb_arr, _, params = sensor_mgr.get_sync_data(
                        rt_subframes=rt_subframes
                    )
                    
                    if rgb_arr is None:
                        valid_flight = False
                        failure_reason = "negative_missing_rgb"
                        break
                    if params is None:
                        valid_flight = False
                        failure_reason = "negative_missing_camera_params"
                        break
                        
                    # Project target to ensure it is indeed out of bounds
                    px_draw = project_3d_to_pixel(target_3d, params, width, height)
                    if px_draw is not None and (0 <= px_draw[0] < width and 0 <= px_draw[1] < height):
                        valid_flight = False
                        failure_reason = "negative_target_still_visible"
                        break
                        
                    gray_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
                    actual_dist = np.linalg.norm(target_3d - pos)
                    
                    flight_data.append({
                        "frame_index": frames_per_flight + fn_idx,
                        "image_gray": gray_arr,
                        "target_2d": None, # Target is out of frame
                        "target_3d": target_3d,
                        "distance_to_target": actual_dist
                    })
                    
                    if is_debug:
                        bgr = cv2.cvtColor(gray_arr, cv2.COLOR_GRAY2BGR)
                        cv2.putText(bgr, "False Negative (Out of Frame)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        cv2.imwrite(os.path.join(flight_debug_dir, f"{frames_per_flight + fn_idx:02d}_frame.png"), bgr)
                        
            # 6. Save Pickle file
            if valid_flight and len(flight_data) == (frames_per_flight + num_false_negatives):
                pkl_path = os.path.join(cache_dir, f"flight_{flights_generated:04d}.pkl")
                
                # Metadata on frame 0
                flight_data[0]["map_name"] = os.path.basename(target_map).replace(".usd", "")
                flight_data[0]["crop_max_size"] = config['compiler'].get('crop_max_size', 512)
                flight_data[0]["crop_min_size"] = config['compiler'].get('crop_min_size', 16)
                flight_data[0]["fov"] = flight_fov
                
                with open(pkl_path, 'wb') as f:
                    pickle.dump(flight_data, f)
                    
                flights_generated += 1
                attempts_for_flight = 0
                rejection_counts.clear()
                print(f"[SUCCESS] Flight successfully saved to cache: flight_{flights_generated-1:04d}.pkl")
            else:
                rejection_counts[failure_reason or "incomplete_flight"] += 1
                report_failed_attempt(
                    target_map,
                    attempts_for_flight,
                    max_attempts_per_flight,
                    rejection_counts,
                    failed_maps,
                )
            
    except KeyboardInterrupt:
        print("[*] Interrupted by user.")
        exit_code = 130
    except Exception as exc:
        exit_code = 1
        pending_exception = exc
        traceback.print_exc()
    finally:
        sensor_mgr.destroy()
        print("[*] Dataset generation finished.")
        client_mgr.shutdown(
            force_process_exit=force_process_exit,
            exit_code=exit_code,
        )

    if pending_exception is not None:
        raise pending_exception

if __name__ == '__main__':
    main()
