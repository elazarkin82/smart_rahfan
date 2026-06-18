import carla
import json
import time
import os
import math
import random
import pickle
import numpy as np
import cv2
import queue
import argparse
from carla_project.core.carla_client import CarlaClientManager
from carla_project.core.sensor_manager import SensorManager
from carla_project.core.geometry_utils import build_projection_matrix, get_3d_world_from_pixel, get_3d_world_to_pixel

def load_config(path="pipeline_config.json"):
    with open(path, 'r') as f:
        return json.load(f)

def lerp(a, b, t):
    return a + (b - a) * t

def shortest_angle_diff(source, target):
    diff = (target - source) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff

def lerp_angle(a, b, t):
    return a + shortest_angle_diff(a, b) * t

def lerp_transform(t1, t2, t):
    loc = carla.Location(
        lerp(t1.location.x, t2.location.x, t),
        lerp(t1.location.y, t2.location.y, t),
        lerp(t1.location.z, t2.location.z, t)
    )
    rot = carla.Rotation(
        lerp_angle(t1.rotation.pitch, t2.rotation.pitch, t),
        lerp_angle(t1.rotation.yaw, t2.rotation.yaw, t),
        lerp_angle(t1.rotation.roll, t2.rotation.roll, t)
    )
    return carla.Transform(loc, rot)

def clear_queues(sensor_mgr):
    while not sensor_mgr.image_queue.empty():
        try: sensor_mgr.image_queue.get_nowait()
        except queue.Empty: break
    while not sensor_mgr.depth_queue.empty():
        try: sensor_mgr.depth_queue.get_nowait()
        except queue.Empty: break

def sample_range(range_values, default_value):
    if not isinstance(range_values, (list, tuple)) or len(range_values) != 2:
        return default_value
    lo, hi = float(range_values[0]), float(range_values[1])
    if lo > hi:
        lo, hi = hi, lo
    if lo == hi:
        return lo
    return random.uniform(lo, hi)

def normalize_range(range_values):
    if not isinstance(range_values, (list, tuple)) or len(range_values) != 2:
        return None
    lo, hi = float(range_values[0]), float(range_values[1])
    return (min(lo, hi), max(lo, hi))

def is_value_in_ranges(value, ranges):
    if not isinstance(ranges, list):
        return False
    for range_values in ranges:
        normalized = normalize_range(range_values)
        if normalized is None:
            continue
        lo, hi = normalized
        if lo <= value <= hi:
            return True
    return False

def get_target_distance_limits(target_distance_cfg, flight_altitude):
    allowed_range = normalize_range(target_distance_cfg.get("allowed_range_m"))
    if allowed_range is not None:
        return allowed_range
    if target_distance_cfg.get("use_altitude_scaled_defaults", True):
        return (
            0.96 * flight_altitude + 1.0,
            min(200.0, 6.6 * flight_altitude + 2.0)
        )
    return None, None

def is_target_distance_allowed(distance_m, target_distance_cfg, flight_altitude):
    min_allowed_dist, max_allowed_dist = get_target_distance_limits(target_distance_cfg, flight_altitude)
    if min_allowed_dist is not None and distance_m < min_allowed_dist:
        return False
    if max_allowed_dist is not None and distance_m > max_allowed_dist:
        return False
    return not is_value_in_ranges(distance_m, target_distance_cfg.get("exclude_ranges_m", []))

def build_lateral_dominant_indices(frames_per_flight, count):
    if count <= 0 or frames_per_flight <= 2:
        return set()
    intermediate = list(range(1, frames_per_flight - 1))
    count = min(count, len(intermediate))
    center = (frames_per_flight - 1) / 2.0
    return set(sorted(intermediate, key=lambda idx: abs(idx - center))[:count])

def copy_transform(transform):
    return carla.Transform(
        carla.Location(
            x=transform.location.x,
            y=transform.location.y,
            z=transform.location.z
        ),
        carla.Rotation(
            pitch=transform.rotation.pitch,
            yaw=transform.rotation.yaw,
            roll=transform.rotation.roll
        )
    )

def look_at_rotation(from_location_np, target_3d, pitch_offset_deg=0.0, yaw_offset_deg=0.0, roll_deg=0.0):
    vec = target_3d - from_location_np
    dist = np.linalg.norm(vec)
    if dist < 1e-6:
        return carla.Rotation(pitch=0.0, yaw=0.0, roll=roll_deg)
    dir_vec = vec / dist
    pitch_rad = math.asin(dir_vec[2])
    yaw_rad = math.atan2(dir_vec[1], dir_vec[0])
    return carla.Rotation(
        pitch=math.degrees(pitch_rad) + pitch_offset_deg,
        yaw=math.degrees(yaw_rad) + yaw_offset_deg,
        roll=roll_deg
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--balance_existing", action="store_true", help="Balance generated flights across maps based on existing cache files")
    args = parser.parse_args()

    config = load_config()
    
    cache_dir = config['generation'].get('cache_dir', 'cache')
    debug_dir = config['generation'].get('debug_dir', 'debug')
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    
    num_flights_target = config['generation']['num_flights_to_cache']
    existing_flights = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
    
    if len(existing_flights) >= num_flights_target:
        print(f"Cache already has {len(existing_flights)} flights. Skipping generation.")
        return

    width = config['sensor']['width']
    height = config['sensor']['height']
    fov = config['sensor']['fov']
    fov_range = config['sensor'].get('fov_range', [fov, fov])
    fps = config['sensor']['fps']
    
    client_mgr = CarlaClientManager(host=config['carla']['host'], port=config['carla']['port'], timeout=config['carla']['timeout'])
    client, world = client_mgr.connect()
    
    maps = config['carla']['maps']
    weathers = config['carla']['weather_presets']
    
    # Read existing map counts if balance_existing is enabled
    map_counts = {m: 0 for m in maps}
    if args.balance_existing:
        print("Scanning existing flights to count per-map occurrences...")
        for pkl_file in existing_flights:
            pkl_path = os.path.join(cache_dir, pkl_file)
            try:
                with open(pkl_path, 'rb') as f:
                    flight_data = pickle.load(f)
                if flight_data and len(flight_data) > 0:
                    saved_map = flight_data[0].get("map_name")
                    if saved_map in map_counts:
                        map_counts[saved_map] += 1
            except Exception:
                pass
        print(f"Current map counts: {map_counts}")

    flights_generated = len(existing_flights)
    frames_per_flight = config['generation']['frames_per_flight']
    num_false_negatives = config['generation'].get('num_false_negatives', 0)
    min_texture_std = config['generation'].get('min_texture_std', 0.0)
    noise_cfg = config['generation'].get('noise_params', {})
    target_distance_cfg = config['generation'].get('target_distance', {})
    flight_motion_cfg = config['generation'].get('flight_motion', {})
    lateral_motion_cfg = config['generation'].get('lateral_motion', {})
    roll_motion_cfg = config['generation'].get('roll_motion', {})
    pos_xy_amp = noise_cfg.get('pos_xy_amp', 0.3)
    pos_z_amp = noise_cfg.get('pos_z_amp', 0.1)
    rot_pitch_amp = noise_cfg.get('rot_pitch_amp', 0.5)
    rot_yaw_amp = noise_cfg.get('rot_yaw_amp', 0.5)
    rot_roll_amp = noise_cfg.get('rot_roll_amp', 1.0)
    altitude_range = config['generation'].get('altitude_range', [20.0, 60.0])
    debug_interval = config['generation']['debug_interval']
    
    sensor_mgr = None
    current_loaded_map = None
    
    print(f"Starting Dataset Generation. Goal: {num_flights_target} flights.")
    
    try:
        while flights_generated < num_flights_target:
            # Determine target map
            if args.balance_existing:
                num_maps = len(maps)
                target_per_map = num_flights_target // num_maps
                target_map = None
                for m in maps:
                    if map_counts[m] < target_per_map:
                        target_map = m
                        break
                if target_map is None:
                    target_map = min(maps, key=lambda m: map_counts[m])
            else:
                num_maps = len(maps)
                flights_per_map = max(1, num_flights_target // num_maps)
                map_idx = min(flights_generated // flights_per_map, num_maps - 1)
                target_map = maps[map_idx]
            
            if current_loaded_map != target_map or sensor_mgr is None:
                if isinstance(sensor_mgr, SensorManager):
                    sensor_mgr.destroy()
                    
                while True:
                    if not maps:
                        print("Error: No valid maps left to load!")
                        return
                        
                    # Re-calculate in case the maps list shrank due to dynamic removal of invalid maps
                    if args.balance_existing:
                        num_maps = len(maps)
                        target_per_map = num_flights_target // num_maps
                        target_map = None
                        for m in maps:
                            if map_counts[m] < target_per_map:
                                target_map = m
                                break
                        if target_map is None:
                            target_map = min(maps, key=lambda m: map_counts[m])
                    else:
                        num_maps = len(maps)
                        flights_per_map = max(1, num_flights_target // num_maps)
                        map_idx = min(flights_generated // flights_per_map, num_maps - 1)
                        target_map = maps[map_idx]
                    
                    print(f"--- Loading Map: {target_map} ---")
                    try:
                        world = client_mgr.load_world(target_map)
                        current_loaded_map = target_map
                        break
                    except RuntimeError as e:
                        print(f"Map '{target_map}' not found or crashed. Removing from rotation.")
                        maps.remove(target_map)
                
                # ENABLE SYNCHRONOUS MODE!
                settings = world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 1.0 / fps
                world.apply_settings(settings)
                
                # Placeholder indicating map is loaded
                sensor_mgr = True
                
            client_mgr.set_weather(random.choice(weathers))
            world.tick()
            
            # 1. Search for a valid target
            spawn_points = world.get_map().get_spawn_points()
            start_point = random.choice(spawn_points)
            
            # Sample random altitude and pitch dynamically
            flight_altitude = random.uniform(altitude_range[0], altitude_range[1])
            alt_min, alt_max = altitude_range[0], altitude_range[1]
            alt_diff = (alt_max - alt_min) if (alt_max > alt_min) else 1.0
            norm_alt = (flight_altitude - alt_min) / alt_diff
            
            pitch_min = lerp(-15.0, -60.0, norm_alt)
            pitch_max = lerp(-5.0, -20.0, norm_alt)
            flight_pitch = random.uniform(pitch_min, pitch_max)
            
            start_point.location.z += flight_altitude
            start_point.rotation.pitch = flight_pitch
            start_point.rotation.yaw += random.uniform(-180.0, 180.0)
            start_point.rotation.roll = sample_range(roll_motion_cfg.get('base_roll_range_deg'), 0.0)
            
            # Sample random FOV and compute camera intrinsic matrix K
            flight_fov = random.uniform(fov_range[0], fov_range[1])
            K = build_projection_matrix(width, height, flight_fov)
            
            # Recreate sensor manager with the randomized FOV for this flight
            if isinstance(sensor_mgr, SensorManager):
                sensor_mgr.destroy()
            sensor_mgr = SensorManager(world, width, height, flight_fov)
            sensor_mgr.spawn_cameras(start_point)
            
            # First tick moves/spawns the camera actor on the server
            world.tick()
            
            # Clear stale frames, then tick a second time to render while stationary
            clear_queues(sensor_mgr)
            fid_start = world.tick()
            
            rgb_array, depth_array, rgb_transform = sensor_mgr.get_sync_data(frame_id=fid_start, timeout=5.0)
            if rgb_array is None or depth_array is None:
                continue
                
            # Pick a random pixel in the middle 50% of the screen and try up to 100 times in the same frame
            margin_x = int(width * 0.25)
            margin_y = int(height * 0.25)
            
            found_target = False
            for _ in range(100):
                px = random.randint(margin_x, width - margin_x)
                py = random.randint(margin_y, height - margin_y)
                
                # Check texture diversity around the selected pixel to avoid flat/homogeneous areas (e.g. sky or smooth road)
                if min_texture_std > 0:
                    patch = rgb_array[max(0, py-7):min(height, py+8), max(0, px-7):min(width, px+8)]
                    patch_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                    texture_std = np.std(patch_gray)
                    if texture_std < min_texture_std:
                        continue # Try another pixel in this frame
                
                target_3d = get_3d_world_from_pixel(px, py, depth_array, K, rgb_transform)
                if target_3d is None:
                    continue # Try another pixel in this frame
                    
                dist = np.linalg.norm(target_3d - np.array([start_point.location.x, start_point.location.y, start_point.location.z]))
                if not is_target_distance_allowed(dist, target_distance_cfg, flight_altitude):
                    continue # Try another pixel in this frame
                    
                found_target = True
                break
                
            if not found_target:
                continue # Skip this camera spawn entirely if no pixel was valid in 100 attempts
                
            print(f"[{flights_generated+1}/{num_flights_target}] Generating Flight... Target Dist: {dist:.1f}m")
            
            # 2. Calculate flight path
            start_loc_np = np.array([start_point.location.x, start_point.location.y, start_point.location.z])
            vec = target_3d - start_loc_np
            dir_vec = vec / dist
            
            longitudinal_mode = flight_motion_cfg.get("longitudinal_mode", "approach_stop_distance")
            if longitudinal_mode == "approach_stop_distance":
                stop_dist = sample_range(flight_motion_cfg.get("stop_distance_range_m"), random.uniform(2.0, 5.0))
                longitudinal_delta_m = max(0.0, dist - stop_dist)
            else:
                delta_range = normalize_range(flight_motion_cfg.get("longitudinal_delta_range_m", [-35.0, 35.0]))
                if delta_range is None:
                    delta_range = (-35.0, 35.0)
                delta_min, delta_max = delta_range
                if not flight_motion_cfg.get("allow_recede", True):
                    delta_min = max(delta_min, 0.0)
                if not flight_motion_cfg.get("allow_approach", True):
                    delta_max = min(delta_max, 0.0)
                if delta_min > delta_max:
                    longitudinal_delta_m = 0.0
                else:
                    longitudinal_delta_m = random.uniform(delta_min, delta_max)
                stop_dist = max(0.0, dist - longitudinal_delta_m)
            
            end_loc = start_loc_np + dir_vec * longitudinal_delta_m
            
            # Compute vectors perpendicular to the camera flight vector (up and right)
            p_start = start_loc_np
            flight_vector = end_loc - p_start
            flight_dist = np.linalg.norm(flight_vector)
            if flight_dist > 0:
                flight_dir_unit = flight_vector / flight_dist
            else:
                flight_dir_unit = dir_vec
                
            right_vec = np.cross(flight_dir_unit, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(right_vec) < 1e-4:
                right_vec = np.cross(flight_dir_unit, np.array([0.0, 1.0, 0.0]))
            right_vec /= np.linalg.norm(right_vec)
            
            up_vec = np.cross(right_vec, flight_dir_unit)
            up_vec /= np.linalg.norm(up_vec)
            
            # Sample per-flight random amplitudes and frequencies for lateral/vertical strafing
            drift_right_amp = sample_range(lateral_motion_cfg.get('right_amplitude_range_m'), random.uniform(-3.0, 3.0))
            drift_up_amp = sample_range(lateral_motion_cfg.get('up_amplitude_range_m'), random.uniform(-2.0, 2.0))
            drift_right_freq = sample_range(lateral_motion_cfg.get('frequency_range'), random.uniform(1.0, 3.0))
            drift_up_freq = sample_range(lateral_motion_cfg.get('frequency_range'), random.uniform(1.0, 3.0))
            lateral_dominant_indices = build_lateral_dominant_indices(
                frames_per_flight,
                int(lateral_motion_cfg.get('lateral_dominant_frames_per_flight', 0))
            )
            lateral_dominant_longitudinal_scale = float(lateral_motion_cfg.get('lateral_dominant_longitudinal_scale', 0.15))
            
            # Look slightly off-center so the target wanders around the screen
            pitch_offset = random.uniform(-15.0, 15.0) # Look slightly above or below
            yaw_offset = random.uniform(-15.0, 15.0)   # Look slightly left or right
            roll_delta = sample_range(roll_motion_cfg.get('roll_delta_range_deg'), 0.0)
            roll_osc_amp = sample_range(roll_motion_cfg.get('oscillation_amplitude_range_deg'), 0.0)
            roll_osc_freq = sample_range(roll_motion_cfg.get('oscillation_frequency_range'), 1.0)
            roll_osc_phase = random.uniform(0.0, 2.0 * math.pi)
            
            end_transform = carla.Transform(
                carla.Location(x=end_loc[0], y=end_loc[1], z=end_loc[2]),
                look_at_rotation(
                    end_loc,
                    target_3d,
                    pitch_offset_deg=pitch_offset,
                    yaw_offset_deg=yaw_offset,
                    roll_deg=start_point.rotation.roll + roll_delta
                )
            )
            
            # 3. Execute flight and record frames
            flight_data = []
            is_debug = (flights_generated % debug_interval == 0)
            flight_debug_dir = os.path.join(debug_dir, f"flight_{flights_generated:04d}")
            if is_debug:
                os.makedirs(flight_debug_dir, exist_ok=True)
                
            valid_flight = True
            
            # Thorough flush to clear any late-arriving TCP packets from previous ticks
            for _ in range(4):
                fid_flush = world.tick()
                sensor_mgr.get_sync_data(frame_id=fid_flush, timeout=1.0)
            
            for frame_idx in range(frames_per_flight):
                t = frame_idx / float(frames_per_flight - 1)
                
                # Perturb step size t to make it non-uniform (non-linear step size)
                if 0 < frame_idx < frames_per_flight - 1:
                    t_noisy = t + random.uniform(-0.05, 0.05)
                    t_noisy = max(0.01, min(0.99, t_noisy))
                else:
                    t_noisy = t
                    
                smooth_t = t_noisy * t_noisy * (3 - 2 * t_noisy)
                longitudinal_t = smooth_t
                if frame_idx in lateral_dominant_indices:
                    longitudinal_t *= lateral_dominant_longitudinal_scale
                clean_base_t = lerp_transform(start_point, end_transform, longitudinal_t)
                
                roll_envelope = math.sin(smooth_t * math.pi)
                clean_base_t.rotation.roll += (
                    roll_osc_amp *
                    roll_envelope *
                    math.sin(smooth_t * roll_osc_freq * 2.0 * math.pi + roll_osc_phase)
                )
                
                # Apply lateral and vertical strafing offset (smoothly zero at t=0 and t=1)
                drift_factor = math.sin(smooth_t * math.pi)
                drift_offset = (right_vec * drift_right_amp * math.sin(smooth_t * drift_right_freq * math.pi) +
                                up_vec * drift_up_amp * math.sin(smooth_t * drift_up_freq * math.pi)) * drift_factor
                                
                clean_base_t.location.x += drift_offset[0]
                clean_base_t.location.y += drift_offset[1]
                clean_base_t.location.z += drift_offset[2]
                
                frame_success = False
                for attempt in range(10):
                    base_t = copy_transform(clean_base_t)
                    
                    # Dynamic Noise Scaling: works for both approach and recede paths.
                    current_loc_np = np.array([base_t.location.x, base_t.location.y, base_t.location.z])
                    current_dist = np.linalg.norm(target_3d - current_loc_np)
                    noise_scale = min(1.0, current_dist / 30.0)
                    
                    # Add smooth mechanical/wind noise (Pitch, Roll, X, Y)
                    phase = flights_generated * 10 + frame_idx * 0.2 + attempt * 5.0
                    base_t.location.x += math.sin(phase * 1.3) * pos_xy_amp * noise_scale
                    base_t.location.y += math.cos(phase * 1.7) * pos_xy_amp * noise_scale
                    base_t.location.z += math.sin(phase * 0.9) * pos_z_amp * noise_scale
                    base_t.rotation.pitch += math.cos(phase * 2.1) * rot_pitch_amp * noise_scale
                    base_t.rotation.roll += math.sin(phase * 2.5) * rot_roll_amp * noise_scale
                    base_t.rotation.yaw += math.cos(phase * 1.5) * rot_yaw_amp * noise_scale
                    
                    sensor_mgr.move_to(base_t)
                    
                    # First tick moves the camera actor on the server
                    world.tick()
                    
                    # Second tick renders the frame while the camera is stationary at base_t
                    fid_step = world.tick()
                    rgb_arr, d_arr, trans = sensor_mgr.get_sync_data(frame_id=fid_step, timeout=2.0)
                    
                    if rgb_arr is None or trans is None or d_arr is None:
                        continue
                        
                    # Project target
                    px_draw = get_3d_world_to_pixel(target_3d, K, trans)
                    
                    # If target goes out of frame, this flight is ruined
                    if px_draw is None or not (0 <= px_draw[0] < width and 0 <= px_draw[1] < height):
                        continue
                        
                    # Check occlusion
                    actual_dist = np.linalg.norm(target_3d - np.array([trans.location.x, trans.location.y, trans.location.z]))
                    depth_val = d_arr[px_draw[1], px_draw[0]]
                    if depth_val < actual_dist - 1.5:
                        continue # Occluded!
                    
                    frame_success = True
                    break
                    
                if not frame_success:
                    valid_flight = False
                    break
                    
                gray_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
                    
                flight_data.append({
                    "frame_index": frame_idx,
                    "image_gray": gray_arr,
                    "target_2d": px_draw,
                    "target_3d": target_3d,
                    "distance_to_target": actual_dist
                })
                
                # Debug Drawing
                if is_debug:
                    bgr = cv2.cvtColor(gray_arr, cv2.COLOR_GRAY2BGR)
                    cv2.circle(bgr, px_draw, 20, (0, 0, 255), 2)
                    cv2.circle(bgr, px_draw, 3, (0, 0, 255), -1)
                    cv2.imwrite(os.path.join(flight_debug_dir, f"{frame_idx:02d}_frame.png"), bgr)

            if valid_flight:
                # Add False Negative (Target out-of-frame) frames
                for fn_idx in range(num_false_negatives):
                    # Start from last transform, but rotate away so target is not visible
                    fn_t = copy_transform(clean_base_t)
                    fn_t.rotation.yaw += 45.0 + fn_idx * 10.0
                    fn_t.rotation.pitch += 15.0
                    
                    sensor_mgr.move_to(fn_t)
                    
                    world.tick()
                    fid_fn = world.tick()
                    rgb_arr, d_arr, trans = sensor_mgr.get_sync_data(frame_id=fid_fn, timeout=2.0)
                    
                    if rgb_arr is None or trans is None:
                        valid_flight = False
                        break
                        
                    # Project target to double check it is indeed out of frame
                    px_draw = get_3d_world_to_pixel(target_3d, K, trans)
                    if px_draw is not None and (0 <= px_draw[0] < width and 0 <= px_draw[1] < height):
                        # Target is still in frame, rotate it more
                        fn_t.rotation.yaw += 45.0
                        sensor_mgr.move_to(fn_t)
                        world.tick()
                        fid_fn = world.tick()
                        rgb_arr, d_arr, trans = sensor_mgr.get_sync_data(frame_id=fid_fn, timeout=2.0)
                        px_draw = get_3d_world_to_pixel(target_3d, K, trans)
                        if px_draw is not None and (0 <= px_draw[0] < width and 0 <= px_draw[1] < height):
                            valid_flight = False
                            break
                            
                    gray_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
                    actual_dist = np.linalg.norm(target_3d - np.array([trans.location.x, trans.location.y, trans.location.z]))
                    
                    flight_data.append({
                        "frame_index": frames_per_flight + fn_idx,
                        "image_gray": gray_arr,
                        "target_2d": None,  # Out of frame
                        "target_3d": target_3d,
                        "distance_to_target": actual_dist
                    })
                    
                    # Debug drawing for False Negatives
                    if is_debug:
                        bgr = cv2.cvtColor(gray_arr, cv2.COLOR_GRAY2BGR)
                        cv2.putText(bgr, "False Negative (Out of Frame)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        cv2.imwrite(os.path.join(flight_debug_dir, f"{frames_per_flight + fn_idx:02d}_frame.png"), bgr)

            if valid_flight and len(flight_data) == (frames_per_flight + num_false_negatives):
                pkl_path = os.path.join(cache_dir, f"flight_{flights_generated:04d}.pkl")
                
                # Inject map name, crop sizes, and FOV into frame 0 metadata
                flight_data[0]["map_name"] = target_map
                flight_data[0]["crop_max_size"] = config['compiler'].get('crop_max_size', 128)
                flight_data[0]["crop_min_size"] = config['compiler'].get('crop_min_size', 4)
                flight_data[0]["fov"] = flight_fov
                flight_data[0]["longitudinal_delta_m"] = float(longitudinal_delta_m)
                flight_data[0]["lateral_dominant_frame_indices"] = sorted(lateral_dominant_indices)
                
                with open(pkl_path, 'wb') as f:
                    pickle.dump(flight_data, f)
                flights_generated += 1
                if args.balance_existing:
                    map_counts[target_map] += 1
            else:
                print("Flight invalid (Target lost from frame or false negative check failed). Retrying...")
 
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # ALWAYS restore async mode, otherwise the server freezes forever!
        if 'world' in locals():
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            
        if isinstance(sensor_mgr, SensorManager):
            sensor_mgr.destroy()
        print("Dataset generation cleaned up!")

if __name__ == '__main__':
    main()
