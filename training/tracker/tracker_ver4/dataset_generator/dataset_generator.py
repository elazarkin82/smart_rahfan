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
from core.carla_client import CarlaClientManager
from core.sensor_manager import SensorManager
from core.geometry_utils import build_projection_matrix, get_3d_world_from_pixel, get_3d_world_to_pixel

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

def main():
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
    fps = config['sensor']['fps']
    K = build_projection_matrix(width, height, fov)
    
    client_mgr = CarlaClientManager(host=config['carla']['host'], port=config['carla']['port'], timeout=config['carla']['timeout'])
    client, world = client_mgr.connect()
    
    maps = config['carla']['maps']
    weathers = config['carla']['weather_presets']
    
    flights_generated = len(existing_flights)
    frames_per_flight = config['generation']['frames_per_flight']
    debug_interval = config['generation']['debug_interval']
    
    sensor_mgr = None
    current_loaded_map = None
    
    print(f"Starting Dataset Generation. Goal: {num_flights_target} flights.")
    
    try:
        while flights_generated < num_flights_target:
            # Determine target map based on sequential chunking
            num_maps = len(maps)
            flights_per_map = max(1, num_flights_target // num_maps)
            map_idx = min(flights_generated // flights_per_map, num_maps - 1)
            target_map = maps[map_idx]
            
            if current_loaded_map != target_map or sensor_mgr is None:
                if sensor_mgr:
                    sensor_mgr.destroy()
                    
                while True:
                    if not maps:
                        print("Error: No valid maps left to load!")
                        return
                        
                    # Re-calculate in case the maps list shrank due to dynamic removal of invalid maps
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
                        
                sensor_mgr = SensorManager(world, width, height, fov)
                sp = world.get_map().get_spawn_points()
                if not sp:
                    print("No spawn points found. Retrying map.")
                    continue
                sensor_mgr.spawn_cameras(sp[0])
                world.tick() # Initial tick to spawn
                
            client_mgr.set_weather(random.choice(weathers))
            world.tick()
            
            # 1. Search for a valid target
            spawn_points = world.get_map().get_spawn_points()
            start_point = random.choice(spawn_points)
            
            # Elevate and pitch down randomly
            start_point.location.z += random.uniform(20.0, 60.0)
            start_point.rotation.pitch = random.uniform(-60.0, -10.0)
            start_point.rotation.yaw += random.uniform(-180.0, 180.0)
            start_point.rotation.roll = 0.0
            
            sensor_mgr.move_to(start_point)
            
            # First tick moves the camera actor to start_point on the server
            world.tick()
            
            # Clear stale frames, then tick a second time to render while stationary
            clear_queues(sensor_mgr)
            fid_start = world.tick()
            
            rgb_array, depth_array, rgb_transform = sensor_mgr.get_sync_data(frame_id=fid_start, timeout=5.0)
            if rgb_array is None or depth_array is None:
                continue
                
            # Pick a random pixel in the middle 50% of the screen
            margin_x = int(width * 0.25)
            margin_y = int(height * 0.25)
            px = random.randint(margin_x, width - margin_x)
            py = random.randint(margin_y, height - margin_y)
            
            target_3d = get_3d_world_from_pixel(px, py, depth_array, K, rgb_transform)
            if target_3d is None:
                continue # Sky or invalid
                
            # Check distance
            dist = np.linalg.norm(target_3d - np.array([start_point.location.x, start_point.location.y, start_point.location.z]))
            if dist < 30.0 or dist > 150.0:
                continue
                
            print(f"[{flights_generated+1}/{num_flights_target}] Generating Flight... Target Dist: {dist:.1f}m")
            
            # 2. Calculate flight path
            # Fly towards the target, getting very close
            stop_dist = random.uniform(2.0, 5.0)
            vec = target_3d - np.array([start_point.location.x, start_point.location.y, start_point.location.z])
            dir_vec = vec / dist
            end_loc = np.array([start_point.location.x, start_point.location.y, start_point.location.z]) + dir_vec * (dist - stop_dist)
            
            # Look slightly off-center so the target wanders around the screen
            pitch_rad = math.asin(dir_vec[2])
            yaw_rad = math.atan2(dir_vec[1], dir_vec[0])
            pitch_offset = random.uniform(-15.0, 15.0) # Look slightly above or below
            yaw_offset = random.uniform(-15.0, 15.0)   # Look slightly left or right
            
            end_transform = carla.Transform(
                carla.Location(x=end_loc[0], y=end_loc[1], z=end_loc[2]),
                carla.Rotation(pitch=math.degrees(pitch_rad) + pitch_offset, yaw=math.degrees(yaw_rad) + yaw_offset, roll=0.0)
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
                smooth_t = t * t * (3 - 2 * t)
                clean_base_t = lerp_transform(start_point, end_transform, smooth_t)
                
                frame_success = False
                for attempt in range(10):
                    base_t = carla.Transform(clean_base_t.location, clean_base_t.rotation)
                    
                    # Dynamic Noise Scaling: scale down wind/mechanical noise as the drone gets closer to the target
                    # (since pixel projection sensitivity rises exponentially at close range).
                    current_dist = dist - smooth_t * (dist - stop_dist)
                    noise_scale = min(1.0, current_dist / 30.0)
                    
                    # Add smooth mechanical/wind noise (Pitch, Roll, X, Y)
                    phase = flights_generated * 10 + frame_idx * 0.2 + attempt * 5.0
                    base_t.location.x += math.sin(phase * 1.3) * 0.3 * noise_scale
                    base_t.location.y += math.cos(phase * 1.7) * 0.3 * noise_scale
                    base_t.location.z += math.sin(phase * 0.9) * 0.1 * noise_scale
                    base_t.rotation.pitch += math.cos(phase * 2.1) * 0.5 * noise_scale
                    base_t.rotation.roll += math.sin(phase * 2.5) * 1.0 * noise_scale
                    base_t.rotation.yaw += math.cos(phase * 1.5) * 0.5 * noise_scale
                    
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

            if valid_flight and len(flight_data) == frames_per_flight:
                pkl_path = os.path.join(cache_dir, f"flight_{flights_generated:04d}.pkl")
                with open(pkl_path, 'wb') as f:
                    pickle.dump(flight_data, f)
                flights_generated += 1
            else:
                print("Flight invalid (Target lost from frame). Retrying...")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # ALWAYS restore async mode, otherwise the server freezes forever!
        if 'world' in locals():
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            
        if sensor_mgr:
            sensor_mgr.destroy()
        print("Dataset generation cleaned up!")

if __name__ == '__main__':
    main()
