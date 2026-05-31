import carla
import pygame
import json
import time
import os
import numpy as np
from core.carla_client import CarlaClientManager
from core.sensor_manager import SensorManager
from core.geometry_utils import build_projection_matrix, get_3d_world_from_pixel
from core.data_exporter import DataExporter

def load_config(path="pipeline_config.json"):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    # Ensure dependencies are loaded
    config = load_config()
    
    # 1. Initialize Pygame
    pygame.init()
    width = config['sensor']['width']
    height = config['sensor']['height']
    fov = config['sensor']['fov']
    display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA Tracker V4 - Manual Tester")
    
    # 2. Connect to CARLA
    client_mgr = CarlaClientManager(
        host=config['carla']['host'], 
        port=config['carla']['port'], 
        timeout=config['carla']['timeout']
    )
    client, world = client_mgr.connect()
    client_mgr.load_world(config['carla']['town'])
    client_mgr.set_weather(config['carla']['weather'])
    
    # 3. Setup Sensors
    sensor_mgr = SensorManager(world, width, height, fov)
    
    # Find a spawn point
    spawn_points = world.get_map().get_spawn_points()
    start_transform = spawn_points[0]
    start_transform.location.z += config['generation']['drone_altitude_meters']
    start_transform.rotation.pitch = config['generation']['drone_pitch_degrees']
    
    sensor_mgr.spawn_cameras(start_transform)
    
    # 4. Math and Exporter setup
    K = build_projection_matrix(width, height, fov)
    exporter = DataExporter(config['generation']['output_dir'])
    
    clock = pygame.time.Clock()
    fps = config['sensor']['fps']
    
    print("\n=========================================")
    print("Instructions:")
    print(" - Click anywhere on the pygame screen to focus the UAV on that point.")
    print(" - Press ESC to exit.")
    print("=========================================\n")
    
    # State tracking for the delayed capture
    pending_capture = False
    capture_timer = 0
    
    running = True
    while running:
        # Sync CARLA data (RGB + Depth)
        rgb_array, depth_array = sensor_mgr.get_sync_data()
        
        if rgb_array is not None:
            # Pygame requires (Width, Height, Channels)
            surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
            display.blit(surface, (0, 0))
            pygame.display.flip()
            
        # Handle secondary delayed capture
        if pending_capture and time.time() > capture_timer:
            if rgb_array is not None:
                print("[Tester] Executing delayed secondary capture...")
                exporter.save_capture(rgb_array, target_pixel=None, suffix="secondary")
            pending_capture = False
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # Left click
                if rgb_array is None or depth_array is None:
                    continue
                    
                mouse_x, mouse_y = pygame.mouse.get_pos()
                print(f"\n[Tester] Clicked at Pixel: ({mouse_x}, {mouse_y})")
                
                # 1. Unproject 2D to 3D
                cam_transform = sensor_mgr.get_transform()
                world_3d = get_3d_world_from_pixel(mouse_x, mouse_y, depth_array, K, cam_transform)
                
                if world_3d is None:
                    print("[Tester] Invalid target (Sky or void clicked).")
                    continue
                    
                print(f"[Tester] Target 3D World Coordinates: X={world_3d[0]:.2f}, Y={world_3d[1]:.2f}, Z={world_3d[2]:.2f}")
                
                # 2. Save Initial Capture
                exporter.save_capture(rgb_array, target_pixel=(mouse_x, mouse_y), suffix="initial")
                
                # 3. Fly the UAV closer/above the target (Hover 15 meters exactly above the click)
                new_transform = carla.Transform(
                    carla.Location(x=world_3d[0], y=world_3d[1], z=world_3d[2] + 15.0), 
                    carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0) # Look straight down
                )
                sensor_mgr.move_to(new_transform)
                print("[Tester] UAV Moved to new hover position.")
                
                # 4. Schedule Secondary Capture
                pending_capture = True
                capture_timer = time.time() + config['generation']['secondary_capture_delay_sec']

        clock.tick(fps)

    print("\nCleaning up CARLA sensors...")
    sensor_mgr.destroy()
    pygame.quit()

if __name__ == '__main__':
    main()
