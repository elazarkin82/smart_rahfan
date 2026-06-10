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
        self.original_working_dir = os.getcwd()
        self.stage_working_dir = None
        self.target_candidates = None
        self.meters_per_unit = 1.0
        self.scale_to_stage = 1.0
        
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
        
    def _repair_missing_asset_paths(self, asset_root):
        """Overrides broken legacy USD asset paths with local absolute paths."""
        from pxr import Sdf

        file_index = None
        repaired = 0
        unresolved = set()

        def build_file_index():
            index = {}
            for root, _, filenames in os.walk(asset_root):
                for filename in filenames:
                    index.setdefault(filename.lower(), []).append(
                        os.path.join(root, filename)
                    )
            return index

        def common_suffix_length(left, right):
            length = 0
            for left_part, right_part in zip(reversed(left), reversed(right)):
                if left_part.lower() != right_part.lower():
                    break
                length += 1
            return length

        def find_candidate(attr, authored_path):
            nonlocal file_index

            normalized_path = authored_path.replace("\\", "/")
            relative_parts = [
                part
                for part in normalized_path.split("/")
                if part not in ("", ".", "..")
            ]
            candidates = []

            for spec in attr.GetPropertyStack():
                layer_path = spec.layer.realPath or spec.layer.identifier
                if layer_path and os.path.isabs(layer_path):
                    candidates.append(
                        os.path.normpath(
                            os.path.join(os.path.dirname(layer_path), authored_path)
                        )
                    )

            candidates.append(
                os.path.normpath(os.path.join(asset_root, authored_path))
            )

            for anchor in ("Assets", "Library", "Materials", "Props"):
                if anchor in relative_parts:
                    suffix = relative_parts[relative_parts.index(anchor):]
                    candidates.append(os.path.join(asset_root, *suffix))
                    if anchor == "Library":
                        candidates.append(
                            os.path.join(asset_root, "ov-content", *suffix)
                        )

            for candidate in candidates:
                if os.path.isfile(candidate):
                    return os.path.abspath(candidate)

            if not relative_parts:
                return None
            if file_index is None:
                file_index = build_file_index()

            matches = file_index.get(relative_parts[-1].lower(), [])
            if len(matches) == 1:
                return os.path.abspath(matches[0])
            if matches:
                ranked_matches = sorted(
                    matches,
                    key=lambda path: common_suffix_length(
                        relative_parts,
                        os.path.relpath(path, asset_root).split(os.sep),
                    ),
                    reverse=True,
                )
                best_match = ranked_matches[0]
                best_score = common_suffix_length(
                    relative_parts,
                    os.path.relpath(best_match, asset_root).split(os.sep),
                )
                if best_score >= 2:
                    return os.path.abspath(best_match)
            return None

        for prim in self.stage.TraverseAll():
            for attr in prim.GetAttributes():
                try:
                    value = attr.Get()
                except Exception:
                    continue
                if not isinstance(value, Sdf.AssetPath):
                    continue

                authored_path = value.path
                if (
                    not authored_path
                    or value.resolvedPath
                    or os.path.isabs(authored_path)
                    or authored_path.startswith(("http://", "https://", "omniverse://"))
                ):
                    continue

                candidate = find_candidate(attr, authored_path)
                if candidate is None:
                    unresolved.add(authored_path)
                    continue

                try:
                    attr.Set(Sdf.AssetPath(candidate))
                    repaired += 1
                except Exception:
                    unresolved.add(authored_path)

        if repaired:
            print(f"[+] Repaired {repaired} missing local USD asset path(s).")
        if unresolved:
            preview = ", ".join(sorted(unresolved)[:5])
            suffix = "" if len(unresolved) <= 5 else ", ..."
            print(
                f"[!] {len(unresolved)} local asset path(s) remain unresolved: "
                f"{preview}{suffix}"
            )

    def load_map(self, usd_path, timeout_sec=180.0, repair_asset_paths=True):
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

        if not usd_path.startswith(("http://", "https://", "omniverse://")):
            # Some legacy AEC materials resolve texture paths relative to the
            # process working directory instead of the authoring USD layer.
            self.stage_working_dir = os.path.dirname(os.path.abspath(usd_path))
            os.chdir(self.stage_working_dir)
            print(f"[*] USD asset working directory: {self.stage_working_dir}")

        if not self.stage_context.open_stage(usd_path):
            raise RuntimeError(f"Isaac Sim failed to open USD stage: {usd_path}")

        deadline = time.monotonic() + timeout_sec
        while self.stage_context.get_stage_loading_status()[2] > 0:
            self.simulation_app.update()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out while loading USD stage after {timeout_sec:.0f}s: {usd_path}")

        self.stage = self.stage_context.get_stage()
        if self.stage is None:
            raise RuntimeError(f"USD stage is unavailable after loading: {usd_path}")
        self.target_candidates = None

        from pxr import Usd, UsdGeom
        self.meters_per_unit = UsdGeom.GetStageMetersPerUnit(self.stage)
        self.scale_to_stage = 1.0 / self.meters_per_unit
        print(f"[*] Stage Meters Per Unit: {self.meters_per_unit} (scale factor: {self.scale_to_stage})")

        if repair_asset_paths and self.stage_working_dir:
            # Detect global asset root if "isaac/assets" is in the path
            global_asset_root = self.stage_working_dir
            marker = os.path.join("isaac", "assets")
            idx = self.stage_working_dir.find(marker)
            if idx != -1:
                global_asset_root = self.stage_working_dir[:idx + len(marker)]
                print(f"[*] Detected global asset root: {global_asset_root}")
            self._repair_missing_asset_paths(global_asset_root)

        for _ in range(5):
            self.simulation_app.update()

        print("[+] USD Stage loaded successfully.")
        return self.stage
        
    def prepare_target_candidates(self):
        """Caches coarse scene anchors once per loaded map."""
        if self.target_candidates is not None:
            return self.target_candidates

        from pxr import Usd, UsdGeom

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        targets = []
        excluded_names = (
            "camera", "sensor", "ground", "floor", "ceiling", "sky", "environment",
            "navmesh", "light", "sprinkler", "receptacle", "outlet", "faucet",
            "shower", "toilet", "cabinet", "dishwasher", "mullion", "plumbing", "fixture",
            "specialty_equipment", "mechanical_equipment", "structural_framing", "casework",
            "ducts", "walls", "windows", "doors", "stair", "interior", "roof", "panel",
            "insulation", "stud", "framing", "pipe", "conduit", "curtain", "furniture",
            "appliance", "range", "fridge", "refrigerator", "oven", "cooktop", "hood",
            "dryer", "washer", "sink", "tub", "mirror", "bed", "chair", "table", "desk",
            "sofa", "couch", "shelf", "wardrobe", "dresser", "counter", "air_terminal",
            "generic_model", "support",
            "runs", "landings", "railing", "electrical", "switches", "disconnect",
            "grass", "trimmed", "instancer", "pointinstancer", "staircase", "tread",
            "riser", "stringer", "hvac", "wire", "cable", "junction", "switchboard",
            "panelboard", "transformer", "meter", "pump", "valve", "structural", "frame",
            "tree", "trees", "leaves", "leaf", "trunk", "branch", "branches", "bark",
            "topography", "rails", "rail", "hawthorn", "planter", "plant", "plants",
            "foliage", "vegetation", "shrub", "bush", "bushes"
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
            if np.max(size) < 2.0 * self.scale_to_stage or np.max(size) > 100.0 * self.scale_to_stage:
                continue
            targets.append((prim, center))

        self.target_candidates = targets
        print(f"[+] Cached {len(targets)} coarse target anchor(s) for this map.")
        return targets

    def get_random_target(self):
        """
        Returns a cached coarse scene anchor. The generator resolves the final
        target from a visible RGB/depth pixel.
        """
        targets = self.prepare_target_candidates()
        if targets:
            target_prim, center = random.choice(targets)
            target_path = target_prim.GetPath().pathString
            print(f"[+] Using USD geometry as camera anchor: {target_path}")
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

    def shutdown(self, force_process_exit=False, exit_code=0):
        """Closes the simulation application cleanly."""
        os.chdir(self.original_working_dir)
        self.stage_working_dir = None

        if self.simulation_app:
            print("[*] Shutting down NVIDIA Isaac Sim application...")
            if force_process_exit:
                # Isaac Sim 6.0 on this ARM/GB10 host aborts in TaskGroup
                # teardown with fast shutdown and takes several minutes with
                # full cleanup. The generator has already detached Replicator
                # and flushed its pickle files before reaching this point.
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(exit_code)
            self.simulation_app.close()
            self.simulation_app = None
