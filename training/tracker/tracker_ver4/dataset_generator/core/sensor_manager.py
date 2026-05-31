import carla
import numpy as np
import weakref
import queue

class SensorManager:
    """
    Spawns and manages sensors (RGB and Depth), synchronizing their data.
    """
    def __init__(self, world, width, height, fov):
        self.world = world
        self.width = width
        self.height = height
        self.fov = fov
        
        self.rgb_sensor = None
        self.depth_sensor = None
        
        self.image_queue = queue.Queue()
        self.depth_queue = queue.Queue()
 
    def spawn_cameras(self, transform):
        blueprint_library = self.world.get_blueprint_library()
        
        # RGB
        rgb_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', str(self.width))
        rgb_bp.set_attribute('image_size_y', str(self.height))
        rgb_bp.set_attribute('fov', str(self.fov))
        self.rgb_sensor = self.world.spawn_actor(rgb_bp, transform)
        
        # Depth - Rigidly attach to RGB sensor to eliminate synchronization jitter and network set_transform delays
        depth_bp = blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', str(self.width))
        depth_bp.set_attribute('image_size_y', str(self.height))
        depth_bp.set_attribute('fov', str(self.fov))
        self.depth_sensor = self.world.spawn_actor(
            depth_bp, 
            carla.Transform(), 
            attach_to=self.rgb_sensor, 
            attachment_type=carla.AttachmentType.Rigid
        )
        
        # Listeners
        weak_self = weakref.ref(self)
        self.rgb_sensor.listen(lambda image: SensorManager._rgb_callback(weak_self, image))
        self.depth_sensor.listen(lambda image: SensorManager._depth_callback(weak_self, image))
 
    def move_to(self, transform):
        if self.rgb_sensor:
            self.rgb_sensor.set_transform(transform)
 
    def get_transform(self):
        if self.rgb_sensor:
            return self.rgb_sensor.get_transform()
        return None
 
    def destroy(self):
        if self.rgb_sensor:
            self.rgb_sensor.destroy()
        if self.depth_sensor:
            self.depth_sensor.destroy()

    @staticmethod
    def _rgb_callback(weak_self, image):
        self = weak_self()
        if not self:
            return
        # Convert raw data to numpy BGRA
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        # Keep RGB only for easier pygame conversion
        array = array[:, :, :3]
        # BGR to RGB
        array = array[:, :, ::-1]
        self.image_queue.put((image.frame, array, image.transform))

    @staticmethod
    def _depth_callback(weak_self, image):
        self = weak_self()
        if not self:
            return
        # Decode depth using raw 24-bit representation (millimetric precision)
        # Avoid converting to ColorConverter.Depth which degrades precision to 8-bit (4-meter quantization errors).
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        
        # Correct BGRA mapping to 24-bit depth formula:
        # channel 2 (Red) is the least significant byte (LSB / R in CARLA formula).
        # channel 0 (Blue) is the most significant byte (MSB / B in CARLA formula).
        R = array[:, :, 2].astype(np.float32)
        G = array[:, :, 1].astype(np.float32)
        B = array[:, :, 0].astype(np.float32)
        normalized = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
        depth_meters = normalized * 1000.0
        
        self.depth_queue.put((image.frame, depth_meters, image.transform))

    def get_sync_data(self, frame_id=None, timeout=2.0):
        """
        Attempts to fetch a synchronized pair of (RGB, Depth) frames.
        If frame_id is provided, discards any frames older than frame_id to prevent queue backlog.
        """
        try:
            # 1. Fetch RGB frame, discarding stale frames
            while True:
                rgb_frame, rgb_array, rgb_transform = self.image_queue.get(timeout=timeout)
                if frame_id is None or rgb_frame >= frame_id:
                    break
                # Discard older frames
                
            # 2. Fetch Depth frame, discarding stale frames
            while True:
                depth_frame, depth_array, depth_transform = self.depth_queue.get(timeout=timeout)
                if frame_id is None or depth_frame >= frame_id:
                    break
                # Discard older frames
            
            # 3. Synchronize frame IDs between RGB and Depth
            while rgb_frame != depth_frame:
                if rgb_frame < depth_frame:
                    while True:
                        rgb_frame, rgb_array, rgb_transform = self.image_queue.get(timeout=timeout)
                        if frame_id is None or rgb_frame >= frame_id:
                            break
                else:
                    while True:
                        depth_frame, depth_array, depth_transform = self.depth_queue.get(timeout=timeout)
                        if frame_id is None or depth_frame >= frame_id:
                            break
            
            if frame_id is not None and rgb_frame > frame_id:
                print(f"[SensorManager] Warning: Missed exact frame {frame_id}, caught up to {rgb_frame}")
                
            return rgb_array, depth_array, rgb_transform
        except queue.Empty:
            return None, None, None
