import carla
import pygame
import json
import time
import os
import math
import numpy as np
from carla_project.core.carla_client import CarlaClientManager
from carla_project.core.sensor_manager import SensorManager
from carla_project.core.geometry_utils import build_projection_matrix, get_3d_world_from_pixel, get_3d_world_to_pixel
from carla_project.core.data_exporter import DataExporter

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
    """Linearly interpolates between two carla.Transforms."""
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

def main():
    config = load_config()
    pygame.init()
    width = config['sensor']['width']
    height = config['sensor']['height']
    fov = config['sensor']['fov']
    display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA Tracker V4 - Manual Tester")
    
    client_mgr = CarlaClientManager(host=config['carla']['host'], port=config['carla']['port'], timeout=config['carla']['timeout'])
    client, world = client_mgr.connect()
    client_mgr.load_world(config['carla']['town'])
    client_mgr.set_weather(config['carla']['weather'])
    
    sensor_mgr = SensorManager(world, width, height, fov)
    spawn_points = world.get_map().get_spawn_points()
    current_transform = spawn_points[0]
    current_transform.location.z += config['generation']['drone_altitude_meters']
    current_transform.rotation.pitch = config['generation']['drone_pitch_degrees']
    sensor_mgr.spawn_cameras(current_transform)
    
    K = build_projection_matrix(width, height, fov)
    exporter = DataExporter(config['generation']['output_dir'])
    
    clock = pygame.time.Clock()
    fps = config['sensor']['fps']
    
    print("\n=========================================")
    print("Controls:")
    print(" - Click: Fly UAV towards that point smoothly.")
    print(" - Arrows: Forward/Backward, Yaw Left/Right.")
    print(" - CTRL + Arrows: Altitude Up/Down, Strafe Left/Right.")
    print(" - 's' Key: Save current frame to disk.")
    print(" - ESC: Exit.")
    print("=========================================\n")
    
    target_3d_world = None
    
    # Flight interpolation state
    is_auto_flying = False
    flight_start_time = 0
    flight_duration = 3.0 # seconds to reach target
    flight_start_transform = None
    flight_target_transform = None
    
    # Delayed capture state
    pending_capture = False
    capture_timer = 0
    
    running = True
    while running:
        dt = clock.tick(fps) / 1000.0 # Delta time in seconds
        
        # 1. Update Movement (Manual or Auto)
        if is_auto_flying:
            t = (time.time() - flight_start_time) / flight_duration
            if t >= 1.0:
                t = 1.0
                is_auto_flying = False
            
            # Smooth step lerp
            smooth_t = t * t * (3 - 2 * t)
            base_transform = lerp_transform(flight_start_transform, flight_target_transform, smooth_t)
            
            # Add Wind Simulation (Subtler Perlin/Sine noise)
            wind_x = math.sin(time.time() * 2.5) * 0.05
            wind_y = math.cos(time.time() * 3.1) * 0.05
            wind_pitch = math.sin(time.time() * 4.0) * 0.1
            wind_roll = math.cos(time.time() * 2.7) * 0.1
            
            base_transform.location.x += wind_x
            base_transform.location.y += wind_y
            base_transform.rotation.pitch += wind_pitch
            base_transform.rotation.roll += wind_roll
            
            current_transform = base_transform
            sensor_mgr.move_to(current_transform)
        else:
            # Manual Arrow Key Control
            keys = pygame.key.get_pressed()
            mods = pygame.key.get_mods()
            move_speed = 15.0 * dt
            rot_speed = 45.0 * dt
            moved = False
            
            forward_vector = current_transform.get_forward_vector()
            right_vector = current_transform.get_right_vector()
            
            # Flatten vectors to move parallel to ground
            forward_vector.z = 0
            right_vector.z = 0
            
            if mods & pygame.KMOD_CTRL:
                # Strafe and Altitude
                if keys[pygame.K_UP]:
                    current_transform.location.z += move_speed
                    moved = True
                if keys[pygame.K_DOWN]:
                    current_transform.location.z -= move_speed
                    moved = True
                if keys[pygame.K_RIGHT]:
                    current_transform.location += right_vector * move_speed
                    moved = True
                if keys[pygame.K_LEFT]:
                    current_transform.location -= right_vector * move_speed
                    moved = True
            else:
                # Forward/Back and Yaw
                if keys[pygame.K_UP]:
                    current_transform.location += forward_vector * move_speed
                    moved = True
                if keys[pygame.K_DOWN]:
                    current_transform.location -= forward_vector * move_speed
                    moved = True
                if keys[pygame.K_RIGHT]:
                    current_transform.rotation.yaw += rot_speed
                    moved = True
                if keys[pygame.K_LEFT]:
                    current_transform.rotation.yaw -= rot_speed
                    moved = True
                
            if moved:
                sensor_mgr.move_to(current_transform)
        
        # 2. Sync CARLA Sensors (now using exact frame transform for zero drift)
        rgb_array, depth_array, rgb_transform = sensor_mgr.get_sync_data(timeout=2.0)
        
        if rgb_array is not None and rgb_transform is not None:
            # Pygame surface requires (Width, Height, Channels)
            surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
            
            # 3. Draw Target Tracking Circle if active
            pixel_drawn = None
            if target_3d_world is not None:
                # Use rgb_transform instead of sensor_mgr.get_transform() to eliminate lag/drift!
                px = get_3d_world_to_pixel(target_3d_world, K, rgb_transform)
                if px is not None:
                    # Draw circle on pygame surface (Red)
                    pygame.draw.circle(surface, (255, 0, 0), px, 20, 3)
                    pygame.draw.circle(surface, (255, 0, 0), px, 3, 0)
                    pixel_drawn = px
            
            display.blit(surface, (0, 0))
            pygame.display.flip()
            
            # Handle delayed auto capture
            if pending_capture and time.time() > capture_timer:
                print("[Tester] Executing delayed secondary capture...")
                exporter.save_capture(rgb_array, target_pixel=pixel_drawn, suffix="secondary")
                pending_capture = False

            # 4. Handle Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_s:
                        print("[Tester] Manual save triggered.")
                        exporter.save_capture(rgb_array, target_pixel=pixel_drawn, suffix="manual")
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if depth_array is None:
                        continue
                        
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    world_3d = get_3d_world_from_pixel(mouse_x, mouse_y, depth_array, K, rgb_transform)
                    
                    if world_3d is None:
                        print("[Tester] Sky/Void clicked.")
                        continue
                        
                    target_3d_world = world_3d
                    print(f"[Tester] Target Locked at X={world_3d[0]:.2f}, Y={world_3d[1]:.2f}, Z={world_3d[2]:.2f}")
                    
                    # Save initial click
                    exporter.save_capture(rgb_array, target_pixel=(mouse_x, mouse_y), suffix="initial")
                    
                    # Calculate Auto Flight Target: stop 15m away, looking directly at the target
                    cam_loc = current_transform.location
                    vec_x = world_3d[0] - cam_loc.x
                    vec_y = world_3d[1] - cam_loc.y
                    vec_z = world_3d[2] - cam_loc.z
                    
                    dist = math.sqrt(vec_x**2 + vec_y**2 + vec_z**2)
                    stop_dist = max(0.0, dist - 15.0) # Stop 15 meters away from target
                    
                    new_x = cam_loc.x + (vec_x / dist) * stop_dist
                    new_y = cam_loc.y + (vec_y / dist) * stop_dist
                    new_z = cam_loc.z + (vec_z / dist) * stop_dist
                    
                    pitch_rad = math.asin(vec_z / dist) if dist > 0 else 0.0
                    yaw_rad = math.atan2(vec_y, vec_x)
                    
                    flight_start_transform = current_transform
                    flight_target_transform = carla.Transform(
                        carla.Location(x=new_x, y=new_y, z=new_z), 
                        carla.Rotation(pitch=math.degrees(pitch_rad), yaw=math.degrees(yaw_rad), roll=0.0)
                    )
                    is_auto_flying = True
                    flight_start_time = time.time()
                    # Calculate duration based on distance (approx 5 meters per second, min 3 seconds)
                    flight_duration = max(3.0, (dist - stop_dist) / 5.0)
                    
                    pending_capture = True
                    capture_timer = time.time() + config['generation']['secondary_capture_delay_sec']

    print("\nCleaning up CARLA sensors...")
    sensor_mgr.destroy()
    pygame.quit()

if __name__ == '__main__':
    main()
