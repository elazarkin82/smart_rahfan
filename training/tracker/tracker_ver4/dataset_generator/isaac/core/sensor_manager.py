import math
import numpy as np

class IsaacSensorManager:
    """Manages the Isaac Sim camera prim, movement, and frame data acquisition."""
    def __init__(self, camera_path="/World/Camera", width=800, height=600, fov=90.0):
        self.camera_path = camera_path
        self.width = width
        self.height = height
        self.fov = fov
        
        self.render_product = None
        self.rgb_annotator = None
        self.depth_annotator = None
        self.params_annotator = None
        self.camera_prim = None
        self.rep = None
        
    def create_camera(self, clipping_range=None):
        """Creates a USD Camera prim with matching resolution and FOV attributes."""
        from isaacsim.core.utils.prims import create_prim, get_prim_at_path
        from pxr import Gf
        
        print(f"[*] Spawning Camera prim at: {self.camera_path}")
        self.camera_prim = create_prim(
            prim_path=self.camera_path,
            prim_type="Camera",
            position=np.array([0.0, 0.0, 10.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]) # w, x, y, z
        )
        
        # Calculate focal length based on FOV (horizontal aperture default is 21.0mm)
        horiz_aperture = 21.0
        vert_aperture = horiz_aperture * (self.height / float(self.width))
        focal_length = horiz_aperture / (2.0 * math.tan(math.radians(self.fov) / 2.0))
        
        # Set camera attributes
        self.camera_prim.GetAttribute("focalLength").Set(focal_length)
        self.camera_prim.GetAttribute("horizontalAperture").Set(horiz_aperture)
        self.camera_prim.GetAttribute("verticalAperture").Set(vert_aperture)
        if clipping_range is None:
            clipping_range = (0.1, 1000.0)
        self.set_clipping_range(*clipping_range)
        return self.camera_prim

    def set_clipping_range(self, near_distance, far_distance):
        """Updates camera clipping distances in USD stage units."""
        from pxr import Gf

        if not self.camera_prim:
            return
        near_distance = max(float(near_distance), 1e-6)
        far_distance = max(float(far_distance), near_distance + 1e-6)
        self.camera_prim.GetAttribute("clippingRange").Set(
            Gf.Vec2f(near_distance, far_distance)
        )

    def set_fov(self, fov):
        """Updates the camera horizontal field of view in degrees."""
        self.fov = float(fov)
        if not self.camera_prim:
            return

        horiz_aperture = float(self.camera_prim.GetAttribute("horizontalAperture").Get())
        focal_length = horiz_aperture / (2.0 * math.tan(math.radians(self.fov) / 2.0))
        self.camera_prim.GetAttribute("focalLength").Set(focal_length)

    def move_to(self, position, orientation_q):
        """
        Moves the camera to the specified 3D position and orientation quaternion.
        orientation_q must be format (w, x, y, z).
        """
        from pxr import Gf
        if not self.camera_prim:
            return
            
        # Convert position/orientation to Gf types
        pos_gf = Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
        
        # USD uses Gf.Quatd as (w, Gf.Vec3d(x, y, z))
        quat_gf = Gf.Quatd(
            float(orientation_q[0]), # w
            Gf.Vec3d(float(orientation_q[1]), float(orientation_q[2]), float(orientation_q[3])) # x, y, z
        )
        
        # Set translation and orientation attributes directly
        self.camera_prim.GetAttribute("xformOp:translate").Set(pos_gf)
        self.camera_prim.GetAttribute("xformOp:orient").Set(quat_gf)

    def look_at(self, position, target):
        """Moves the camera using USD's native look-at transform."""
        from pxr import Gf

        if not self.camera_prim:
            return

        position_gf = Gf.Vec3d(
            float(position[0]),
            float(position[1]),
            float(position[2]),
        )
        target_gf = Gf.Vec3d(
            float(target[0]),
            float(target[1]),
            float(target[2]),
        )
        view_matrix = Gf.Matrix4d(1.0)
        view_matrix.SetLookAt(
            position_gf,
            target_gf,
            Gf.Vec3d(0.0, 0.0, 1.0),
        )
        camera_matrix = view_matrix.GetInverse()
        orientation = camera_matrix.ExtractRotationQuat()

        self.camera_prim.GetAttribute("xformOp:translate").Set(position_gf)
        self.camera_prim.GetAttribute("xformOp:orient").Set(orientation)
        
    def initialize_replicator(self, simulation_app, warmup_frames=2, rt_subframes=4):
        """Attaches Replicator annotators for RGB, depth, and camera parameters."""
        import omni.replicator.core as rep

        self.rep = rep
        simulation_app.update()

        self.render_product = rep.create.render_product(self.camera_path, (self.width, self.height))
        self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb_annotator.attach(self.render_product)
        self.depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        self.depth_annotator.attach(self.render_product)
        self.params_annotator = rep.AnnotatorRegistry.get_annotator("camera_params")
        self.params_annotator.attach(self.render_product)

        rep.orchestrator.set_capture_on_play(False)
        for _ in range(warmup_frames):
            rep.orchestrator.step(
                rt_subframes=rt_subframes,
                delta_time=0.0,
                pause_timeline=False,
            )

    def get_sync_data(self, simulation_app=None, rt_subframes=4):
        """
        Updates the simulator step and retrieves the captured rgb array 
        and camera parameters (view and projection matrices).
        """
        if (
            not self.rgb_annotator
            or not self.depth_annotator
            or not self.params_annotator
            or self.rep is None
        ):
            raise RuntimeError("Replicator annotators not initialized.")

        self.rep.orchestrator.step(
            rt_subframes=rt_subframes,
            delta_time=0.0,
            pause_timeline=False,
        )

        rgb_data = self.rgb_annotator.get_data()
        depth_data = self.depth_annotator.get_data()
        params_data = self.params_annotator.get_data()

        if rgb_data is not None and getattr(rgb_data, "ndim", 0) == 3 and rgb_data.size:
            rgb_arr = rgb_data[:, :, :3].copy()
        else:
            rgb_arr = None

        depth_arr = np.asarray(depth_data, dtype=np.float32)
        if depth_arr.ndim == 3 and depth_arr.shape[2] == 1:
            depth_arr = depth_arr[:, :, 0]
        if depth_arr.ndim != 2 or not depth_arr.size:
            depth_arr = None

        if not isinstance(params_data, dict):
            params_data = None
        return rgb_arr, depth_arr, params_data

    def destroy(self):
        """Removes the camera and detaches annotators."""
        if self.rgb_annotator and self.render_product:
            self.rgb_annotator.detach(self.render_product)
        if self.depth_annotator and self.render_product:
            self.depth_annotator.detach(self.render_product)
        if self.params_annotator and self.render_product:
            self.params_annotator.detach(self.render_product)
        if self.render_product:
            self.render_product.destroy()
            self.render_product = None
        self.rgb_annotator = None
        self.depth_annotator = None
        self.params_annotator = None
        self.rep = None
        if self.camera_prim:
            from isaacsim.core.utils.prims import delete_prim
            delete_prim(self.camera_path)
            self.camera_prim = None
