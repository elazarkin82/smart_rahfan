import os
import random
import sys
import time
import numpy as np

# Note: SimulationApp must be imported before any other omni/pxr modules
from isaacsim import SimulationApp

class IsaacClientManager:
    """Manages the Isaac Sim SimulationApp lifecycle and map loading."""
    def __init__(self, headless=True, renderer="RayTracedLighting"):
        self.headless = headless
        self.renderer = renderer
        self.simulation_app = None
        self.stage = None
        
    def connect(self):
        """Launches the Isaac Sim simulator application."""
        print("[*] Launching NVIDIA Isaac Sim Application...")

        portable_root = os.environ.get("ISAAC_SIM_PORTABLE_ROOT")
        if portable_root and "--portable-root" not in sys.argv:
            os.makedirs(portable_root, exist_ok=True)
            sys.argv.extend(["--portable-root", portable_root])

        self.simulation_app = SimulationApp({
            "headless": self.headless,
            "renderer": self.renderer
        })
        
        # Import Omniverse/USD modules after app is running
        import omni.usd
        self.stage_context = omni.usd.get_context()
        return self.simulation_app, self.stage_context
        
    def load_map(self, usd_path, timeout_sec=180.0):
        """Loads a USD map stage and waits for it to compile assets."""
        if not os.path.isabs(usd_path) and not usd_path.startswith(("http://", "https://", "omniverse://")):
            dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            local_path = os.path.join(dataset_dir, usd_path)
            if os.path.exists(local_path):
                usd_path = local_path

        print(f"[*] Opening USD Stage: {usd_path} ...")

        if not usd_path.startswith(("http://", "https://", "omniverse://")) and not os.path.exists(usd_path):
            assets_root = "/isaac-sim/data/assets"
            asset_path = os.path.join(assets_root, usd_path.lstrip("/"))
            if os.path.exists(asset_path):
                usd_path = asset_path
            else:
                raise FileNotFoundError(f"USD map not found: {usd_path}")

        if not self.stage_context.open_stage(usd_path):
            raise RuntimeError(f"Isaac Sim failed to open USD stage: {usd_path}")

        deadline = time.monotonic() + timeout_sec
        while self.stage_context.get_stage_loading_status()[2] > 0:
            self.simulation_app.update()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out while loading USD stage after {timeout_sec:.0f}s: {usd_path}")

        for _ in range(5):
            self.simulation_app.update()

        self.stage = self.stage_context.get_stage()
        if self.stage is None:
            raise RuntimeError(f"USD stage is unavailable after loading: {usd_path}")

        print("[+] USD Stage loaded successfully.")
        return self.stage
        
    def get_random_target(self):
        """
        Traverses the USD stage to find an existing mesh prop to target.
        If no meshes are found, spawns a default target cube.
        """
        from pxr import Usd, UsdGeom

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        targets = []
        excluded_names = (
            "camera",
            "sensor",
            "ground",
            "floor",
            "ceiling",
            "sky",
            "environment",
            "navmesh",
            "light",
        )

        for prim in self.stage.TraverseAll():
            if not prim.IsValid() or not prim.IsActive() or not prim.IsA(UsdGeom.Gprim):
                continue

            path = prim.GetPath().pathString
            if any(name in path.lower() for name in excluded_names):
                continue

            try:
                world_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
                center = np.asarray(world_range.GetMidpoint(), dtype=np.float64)
                size = np.asarray(world_range.GetSize(), dtype=np.float64)
            except Exception:
                continue

            if not np.all(np.isfinite(center)) or not np.all(np.isfinite(size)):
                continue
            if np.max(size) < 0.25 or np.max(size) > 50.0:
                continue
            targets.append((prim, center))

        if targets:
            target_prim, center = random.choice(targets)
            target_path = target_prim.GetPath().pathString
            print(f"[+] Targeting existing USD geometry: {target_path}")
            return center

        print("[!] No suitable stage geometry found. Creating a procedural fallback scene...")
        from isaacsim.core.api.objects import VisualCuboid
        from pxr import Gf, UsdLux

        fallback_root = "/World/SyntheticFallback"
        prop_specs = [
            ((-14.0, -10.0, 1.5), (3.0, 3.0, 3.0), (0.85, 0.15, 0.12)),
            ((-6.0, 8.0, 2.5), (4.0, 3.0, 5.0), (0.12, 0.55, 0.85)),
            ((5.0, -9.0, 2.0), (3.0, 5.0, 4.0), (0.20, 0.75, 0.30)),
            ((12.0, 6.0, 3.0), (5.0, 4.0, 6.0), (0.85, 0.65, 0.12)),
            ((0.0, 0.0, 1.5), (2.5, 2.5, 3.0), (0.70, 0.20, 0.75)),
            ((-15.0, 12.0, 4.0), (3.0, 3.0, 8.0), (0.15, 0.75, 0.70)),
            ((15.0, -12.0, 2.5), (6.0, 3.0, 5.0), (0.80, 0.35, 0.15)),
            ((8.0, 15.0, 1.5), (4.0, 4.0, 3.0), (0.35, 0.40, 0.85)),
        ]
        if not self.stage.GetPrimAtPath(fallback_root).IsValid():
            UsdGeom.Xform.Define(self.stage, fallback_root)

            VisualCuboid(
                prim_path=f"{fallback_root}/Ground",
                name="fallback_ground",
                position=np.array([0.0, 0.0, -0.25]),
                scale=np.array([60.0, 60.0, 0.5]),
                size=1.0,
                color=np.array([0.18, 0.20, 0.24]),
            )

            for index, (position, scale, color) in enumerate(prop_specs):
                VisualCuboid(
                    prim_path=f"{fallback_root}/Prop_{index:02d}",
                    name=f"fallback_prop_{index:02d}",
                    position=np.asarray(position),
                    scale=np.asarray(scale),
                    size=1.0,
                    color=np.asarray(color),
                )

            distant_light = UsdLux.DistantLight.Define(
                self.stage,
                f"{fallback_root}/KeyLight",
            )
            distant_light.CreateIntensityAttr(3000.0)
            distant_light.CreateAngleAttr(1.0)

            fill_light = UsdLux.SphereLight.Define(
                self.stage,
                f"{fallback_root}/FillLight",
            )
            fill_light.CreateIntensityAttr(50000.0)
            fill_light.CreateRadiusAttr(5.0)
            UsdGeom.Xformable(fill_light.GetPrim()).AddTranslateOp().Set(
                Gf.Vec3d(0.0, 0.0, 20.0)
            )

            for _ in range(10):
                self.simulation_app.update()

        return np.asarray(random.choice(prop_specs)[0], dtype=np.float64)

    def shutdown(self):
        """Closes the simulation application cleanly."""
        if self.simulation_app:
            print("[*] Shutting down NVIDIA Isaac Sim application...")
            self.simulation_app.close()
            self.simulation_app = None
