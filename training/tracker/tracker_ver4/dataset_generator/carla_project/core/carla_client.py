import carla

class CarlaClientManager:
    """
    Manages the connection to the CARLA server, world loading, and weather manipulation.
    """
    def __init__(self, host='127.0.0.1', port=2000, timeout=30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client = None
        self.world = None

    def connect(self):
        print(f"[CarlaClient] Connecting to CARLA at {self.host}:{self.port}...")
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        print("[CarlaClient] Connected successfully.")
        return self.client, self.world

    def load_world(self, map_name):
        print(f"[CarlaClient] Loading map: {map_name} (This may take a moment)...")
        self.world = self.client.load_world(map_name)
        return self.world

    def set_weather(self, weather_str):
        if not self.world:
            return
        print(f"[CarlaClient] Setting weather to: {weather_str}")
        weather_presets = {
            "ClearNoon": carla.WeatherParameters.ClearNoon,
            "HeavyRain": carla.WeatherParameters.HardRainNoon,
            "Sunset": carla.WeatherParameters.ClearSunset,
            "Foggy": carla.WeatherParameters.CloudyNoon
        }
        preset = weather_presets.get(weather_str, carla.WeatherParameters.Default)
        self.world.set_weather(preset)
