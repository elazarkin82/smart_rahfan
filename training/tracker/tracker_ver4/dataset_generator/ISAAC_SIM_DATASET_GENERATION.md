# NVIDIA Isaac Sim Dataset Generation

This flow runs `isaac_dataset_generator.py` in NVIDIA Isaac Sim 6.0 with the
host user's UID/GID. CARLA is not required.

## Prerequisites

- NVIDIA driver and `nvidia-container-toolkit`
- Docker access for the current host user
- Access to `nvcr.io/nvidia/isaac-sim:6.0.0`
- NVIDIA Isaac Sim EULA acceptance

For the initial image pull:

```bash
docker login nvcr.io
```

Use `$oauthtoken` as the username and an NVIDIA NGC API key as the password.

## Build

From the repository root:

```bash
cd training/tracker/tracker_ver4/dataset_generator/dockers/isaac
./build.sh
```

The Dockerfile keeps the Isaac installation readable and prepares
`/isaac-sim/kit/cache`, `/var/cache/hub`, and
`/isaac-sim/.nvidia-omniverse` for runtime users.

## Start And Connect

Start the background container:

```bash
./run.sh
```

The container stays idle after startup. It does not launch the Isaac streaming
application, because the dataset generator creates its own `SimulationApp`.
This prevents two Kit processes from competing for the same GPU and CPU.

Open a shell as the host user:

```bash
./connect.sh
```

`connect.sh` runs a short setup command as root, creates a container user with
the host UID/GID, and then enters the container as that user. It also sets:

- `HOME` to the mounted host home directory
- `XDG_CACHE_HOME`, `XDG_CONFIG_HOME`, and `XDG_DATA_HOME`
- `HUB__CACHE__PATH` to a writable cache under the host home
- `ISAAC_SIM_PORTABLE_ROOT` to a writable Isaac portable directory
- `__GL_SHADER_DISK_CACHE_PATH` to the persistent `nv_shadercache` directory
- `DISPLAY` to the host value, or `:1` when it is not already set

Do not use `docker exec -u 1001:1001` directly. A numeric UID without a passwd
entry causes `HOME=/` and Isaac attempts to create `/.cache`, which fails.

## Generate Cache

`connect.sh` opens the shell in the dataset generator directory. Run:

```bash
/isaac-sim/python.sh isaac_dataset_generator.py
```

The same command can be launched non-interactively from the host:

```bash
./connect.sh /isaac-sim/python.sh isaac_dataset_generator.py
```

The generator reads `pipeline_config.json`. Isaac-specific settings include:

```json
{
  "isaac": {
    "maps": [
      "isaac/assets/Demos/AEC/BrownstoneDemo/World_BrownstoneDemopack_Brownstone(8Gb).usd",
      "isaac/assets/Demos/AEC/BrownstoneDemo/World_BrownstoneDemopack_Lite(11Gb).usd",
      "isaac/assets/Demos/AEC/BrownstoneDemo/World_BrownstoneDemopack_Morning(20Gb).usd",
      "isaac/assets/Demos/AEC/BrownstoneDemo/World_BrownstoneDemopack_Night(20Gb).usd",
      "isaac/assets/Demos/AEC/BrownstoneDemo/World_BrownstoneDemopack_Park(8Gb).usd"
    ],
    "renderer": "RayTracedLighting",
    "headless": true,
    "rt_subframes": 1,
    "warmup_frames": 1,
    "stage_load_timeout_sec": 180.0,
    "repair_missing_asset_paths": true,
    "max_attempts_per_flight": 25,
    "force_process_exit_on_shutdown": true
  }
}
```

Map paths are relative to `dataset_generator`. Flights are assigned in
contiguous map blocks using the same rule as `carla_dataset_generator.py`:

```text
flights_per_map = max(1, num_flights_to_cache // number_of_maps)
```

Maps are loaded in their configured order. Any remainder is assigned to the
last map. The loaded stage and Replicator graph remain alive for all flights in
that map block and for failed-flight retries.

The first use of each map can still take several minutes while Isaac resolves
materials and compiles RTX shaders. Later flights on the same map reuse that
work. The generator keeps the Brownstone map directory as its working directory
while the stage is active because some legacy AEC materials resolve texture
paths relative to the process working directory.

With `repair_missing_asset_paths=true`, missing relative texture and MDL asset
attributes are resolved against their authoring layer, the Brownstone root, and
the extracted local files. Repairs are applied only to the in-memory stage; the
downloaded USD files are not modified. Assets that are genuinely absent from
the archive are reported and left unchanged.

`rt_subframes=1` is appropriate for the static camera captures used here and
avoids rendering every sample four times. Increase it only if movement-related
RTX artifacts are visible in debug images.

On this ARM/GB10 host, Isaac Sim 6.0 aborts in `TaskGroup` cleanup when fast
shutdown is used, while full extension teardown takes several minutes.
`force_process_exit_on_shutdown=true` exits the standalone generator process
after Replicator is detached and all completed pickle files are closed. Runtime
errors still produce exit code 1, and Ctrl+C produces exit code 130.

Successful output is written to:

```text
dataset_generator/cache/flight_0000.pkl
dataset_generator/debug/flight_0000/
```

The generator stops with an explicit error after
`max_attempts_per_flight` instead of retrying forever.

## Verify

Inside the container:

```bash
ls -lh cache/flight_*.pkl
```

Each pickle contains grayscale positive frames followed by the configured
number of false-negative frames. Frame zero also contains the map, crop, and
FOV metadata. Positive frames whose target patch is nearly uniform are rejected
using `generation.min_texture_std`.

## Compile

After cache generation:

```bash
python3 dataset_compiler.py
```

The compiled dataset is written under `dataset_generator/compiled`.

## Troubleshooting

- `PermissionError: '/.cache'`: enter through `./connect.sh`; verify `echo $HOME`.
- Hub cache permission errors: rebuild the image and restart the container.
- Two active Kit processes: restart with the current `run.sh`; `docker top
  isaac-sim-instance` should show `sleep infinity` before generation starts.
- Immediate headless segfault: verify `echo $DISPLAY`; this host requires
  `DISPLAY=:1` even when `isaac.headless` is enabled.
- `Destroying busy TaskGroup` during shutdown: keep
  `force_process_exit_on_shutdown=true` on this host.
- Repeated `OmniHub ... child exited` warnings can occur on the ARM/GB10 Isaac
  image. They are non-fatal for local USD generation if `Simulation App Startup
  Complete` appears and cache files are produced.
- Brownstone texture or MDL warnings: verify the complete `AECDemo` archive was
  extracted under `isaac/assets/Demos/AEC/BrownstoneDemo`. Some warnings come
  from legacy asset references, but local `./Assets/...` paths are resolved by
  keeping the map directory as the working directory.
- No RGB data: verify that Replicator starts successfully and that the GPU is
  visible with `nvidia-smi` inside the container.
- A loaded map with only `GroundPlane/CollisionMesh` indicates missing USD
  references. The generator will report that it is creating the procedural
  fallback scene.
- USD load timeout: verify the configured map path and increase
  `stage_load_timeout_sec` for large stages.
- Repeated target-loss retries: inspect the generated debug frames and the
  selected target path printed by the generator.
