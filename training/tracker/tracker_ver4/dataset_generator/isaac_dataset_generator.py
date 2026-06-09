#!/usr/bin/env python3
import sys
import os
import json
import random
import math
import pickle
import traceback
import numpy as np
import cv2

# Set up paths to load local packages
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.dirname(script_dir))

from isaac.core.isaac_client import IsaacClientManager
from isaac.core.sensor_manager import IsaacSensorManager
from isaac.core.geometry_utils import project_3d_to_pixel

def load_config(path="pipeline_config.json"):
    if not os.path.exists(path) and path == "pipeline_config.json":
        path = os.path.join(script_dir, path)
    with open(path, 'r') as f:
        return json.load(f)

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

def main():
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
    
    num_flights_target = config['generation']['num_flights_to_cache']
    
    existing_flights = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
    flights_generated = len(existing_flights)
    
    if flights_generated >= num_flights_target:
        print(f"[+] Cache already has {flights_generated} flights. Skipping generation.")
        return
        
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
    force_process_exit = isaac_cfg.get("force_process_exit_on_shutdown", True)

    if not maps:
        raise ValueError("isaac.maps must contain at least one USD map.")
    
    frames_per_flight = config['generation']['frames_per_flight']
    num_false_negatives = config['generation']['num_false_negatives']
    debug_interval = config['generation']['debug_interval']
    min_texture_std = config['generation'].get('min_texture_std', 2.0)
    
    # Noise parameters
    noise_cfg = config['generation']['noise_params']
    pos_xy_amp = noise_cfg.get('pos_xy_amp', 0.3)
    pos_z_amp = noise_cfg.get('pos_z_amp', 0.1)
    rot_pitch_amp = noise_cfg.get('rot_pitch_amp', 0.5)
    rot_yaw_amp = noise_cfg.get('rot_yaw_amp', 0.5)
    rot_roll_amp = noise_cfg.get('rot_roll_amp', 2.5)
    
    # 1. Initialize Simulator
    client_mgr = IsaacClientManager(headless=headless, renderer=renderer)
    simulation_app, stage_context = client_mgr.connect()
    
    sensor_mgr = IsaacSensorManager(camera_path="/World/Camera", width=width, height=height, fov=fov)
    current_map = None
    attempts_for_flight = 0
    exit_code = 0
    pending_exception = None
    
    try:
        while flights_generated < num_flights_target:
            # Use the same contiguous per-map allocation as the CARLA generator.
            target_map = get_map_for_flight(
                maps,
                flights_generated,
                num_flights_target,
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
                client_mgr.load_map(
                    target_map,
                    timeout_sec=stage_load_timeout,
                    repair_asset_paths=repair_asset_paths,
                )
                sensor_mgr.create_camera()
                sensor_mgr.initialize_replicator(
                    simulation_app,
                    warmup_frames=warmup_frames,
                    rt_subframes=rt_subframes,
                )
                current_map = target_map

            attempts_for_flight += 1
            if attempts_for_flight > max_attempts_per_flight:
                raise RuntimeError(
                    f"Unable to generate flight {flights_generated} after "
                    f"{max_attempts_per_flight} attempts on map {target_map}"
                )

            flight_fov = random.uniform(float(fov_range[0]), float(fov_range[1]))
            sensor_mgr.set_fov(flight_fov)
            
            # 3. Locate Target
            target_3d = client_mgr.get_random_target()
            
            # 4. Compute Flight Start and End
            start_dist = random.uniform(20.0, 45.0)
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(math.radians(15), math.radians(45)) # Angle from vertical
            
            start_pos = target_3d + start_dist * np.array([
                math.sin(phi)*math.cos(theta),
                math.sin(phi)*math.sin(theta),
                math.cos(phi)
            ])
            
            # Flight approaches closely to target
            stop_dist = random.uniform(3.0, 6.0)
            vec = target_3d - start_pos
            start_dist = np.linalg.norm(vec)
            dir_vec = vec / start_dist
            end_pos = start_pos + dir_vec * (start_dist - stop_dist)
            
            # Compute camera orientations looking at the target
            start_quat = get_lookat_quaternion(start_pos, target_3d)
            
            # Look slightly off-center at the end of the flight
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
            
            end_quat = get_lookat_quaternion(end_pos, end_pos + perturbed_forward * 10.0)
            
            # Compute orthogonal vectors for strafing/jitter
            right_vec = right
            up_vec = up
            drift_right_amp = random.uniform(-2.5, 2.5)
            drift_up_amp = random.uniform(-1.5, 1.5)
            drift_right_freq = random.uniform(1.0, 3.0)
            drift_up_freq = random.uniform(1.0, 3.0)
            
            print(
                f"[{flights_generated+1}/{num_flights_target}] Generating flight "
                f"(attempt {attempts_for_flight}/{max_attempts_per_flight}, "
                f"target distance {start_dist:.1f}m)"
            )
            
            flight_data = []
            valid_flight = True
            
            is_debug = (flights_generated % debug_interval == 0)
            flight_debug_dir = os.path.join(debug_dir, f"flight_{flights_generated:04d}")
            if is_debug:
                os.makedirs(flight_debug_dir, exist_ok=True)
                
            for frame_idx in range(frames_per_flight):
                t = frame_idx / float(frames_per_flight - 1)
                
                # Perturb step size slightly for non-linearity
                if 0 < frame_idx < frames_per_flight - 1:
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
                    
                    # Scale down jitter noise as camera gets closer
                    current_dist = start_dist - smooth_t * (start_dist - stop_dist)
                    noise_scale = min(1.0, current_dist / 30.0)
                    
                    # Add mechanical wind noise to position
                    phase = flights_generated * 10 + frame_idx * 0.2 + attempt * 5.0
                    pos[0] += math.sin(phase * 1.3) * pos_xy_amp * noise_scale
                    pos[1] += math.cos(phase * 1.7) * pos_xy_amp * noise_scale
                    pos[2] += math.sin(phase * 0.9) * pos_z_amp * noise_scale
                    
                    # Apply small rotations to quat for mechanical wobble
                    roll_wobble = math.sin(phase * 2.5) * math.radians(rot_roll_amp) * noise_scale
                    pitch_wobble = math.cos(phase * 2.1) * math.radians(rot_pitch_amp) * noise_scale
                    yaw_wobble = math.cos(phase * 1.5) * math.radians(rot_yaw_amp) * noise_scale
                    
                    # Compute perturbed lookat orientation
                    rot_perturbed = get_lookat_quaternion(pos, target_3d)
                    
                    quat = nlerp_quat(quat, rot_perturbed, 0.5)
                    jitter_quat = quaternion_from_euler(
                        roll_wobble,
                        pitch_wobble,
                        yaw_wobble,
                    )
                    quat = quaternion_multiply(quat, jitter_quat)
                    
                    sensor_mgr.move_to(pos, quat)
                    
                    # Get frame data from replicator
                    rgb_arr, params = sensor_mgr.get_sync_data(rt_subframes=rt_subframes)
                    
                    if rgb_arr is None or params is None:
                        continue
                        
                    # Project target onto 2D image coordinates
                    px_draw = project_3d_to_pixel(target_3d, params, width, height)
                    # Target must remain in screen bounds during positive frames
                    if px_draw is None or not (0 <= px_draw[0] < width and 0 <= px_draw[1] < height):
                        continue
                        
                    frame_success = True
                    break
                    
                if not frame_success:
                    valid_flight = False
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
                    break

                actual_dist = np.linalg.norm(target_3d - pos)
                
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
                    rgb_arr, params = sensor_mgr.get_sync_data(rt_subframes=rt_subframes)
                    
                    if rgb_arr is None or params is None:
                        valid_flight = False
                        break
                        
                    # Project target to ensure it is indeed out of bounds
                    px_draw = project_3d_to_pixel(target_3d, params, width, height)
                    if px_draw is not None and (0 <= px_draw[0] < width and 0 <= px_draw[1] < height):
                        valid_flight = False
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
                print(f"[SUCCESS] Flight successfully saved to cache: flight_{flights_generated-1:04d}.pkl")
            else:
                print("[!] Flight generation failed or target was lost. Retrying on the loaded stage...")
            
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
