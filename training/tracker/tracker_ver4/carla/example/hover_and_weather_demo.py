"""
CARLA Hover & Weather Demo
==========================
This script connects to a running CARLA Simulator (started via Docker),
moves the spectator camera (acting as a hovering UAV), changes weather 
conditions, and switches between Urban and Rural maps to demonstrate realism.

Requirements:
    pip install carla==0.9.15
"""

import carla
import time
import random

def set_weather(world, preset_name):
    print(f"--> Changing weather to: {preset_name}")
    presets = {
        "Clear": carla.WeatherParameters.ClearNoon,
        "HeavyRain": carla.WeatherParameters.HardRainNoon,
        "Sunset": carla.WeatherParameters.ClearSunset,
        "Foggy": carla.WeatherParameters.MidFoggyNoon
    }
    world.set_weather(presets.get(preset_name, carla.WeatherParameters.Default))
    time.sleep(5) # Let the user enjoy the view for 5 seconds

def hover_demo(client, map_name, description):
    print(f"\n========================================")
    print(f"Loading Map: {map_name} ({description})")
    print(f"This might take 10-20 seconds...")
    print(f"========================================")
    
    # Load the world
    world = client.load_world(map_name)
    
    # Get the spectator (the free-flying camera we can see through)
    spectator = world.get_spectator()
    
    # Get spawn points to find a logical location (e.g. above a road)
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No spawn points found!")
        return

    # Pick a random spawn point and fly the spectator 30 meters above it
    start_point = random.choice(spawn_points)
    transform = carla.Transform(
        start_point.location + carla.Location(z=30.0), # 30m altitude
        carla.Rotation(pitch=-30.0, yaw=start_point.rotation.yaw, roll=0.0) # Look slightly down
    )
    
    spectator.set_transform(transform)
    print("\n[!] UAV is hovering. Look at the CARLA simulator window!")
    time.sleep(3)

    # Cycle through weathers
    set_weather(world, "Clear")
    set_weather(world, "HeavyRain")
    set_weather(world, "Sunset")
    set_weather(world, "Foggy")

def main():
    try:
        print("Connecting to CARLA server at localhost:2000...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(30.0) # Longer timeout in case server is still starting
        
        # 1. Urban Environment
        # Town10HD is a high-definition urban environment with skyscrapers and complex roads
        hover_demo(client, 'Town10HD_Opt', "High-Definition Urban")
        
        # 2. Rural/Forested Environment
        # Town07 is a rural environment with narrow roads, vegetation, and barns
        hover_demo(client, 'Town07_Opt', "Rural / Forested Environment")
        
        print("\nDemo completed successfully!")
        
    except RuntimeError as e:
        print(f"\n[ERROR] Failed to connect to CARLA. Is the Docker container running? Details: {e}")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
