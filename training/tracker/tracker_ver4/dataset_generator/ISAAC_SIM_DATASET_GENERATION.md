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
      "isaac/assets/warehouse.usd",
      "isaac/assets/office.usd"
    ],
    "renderer": "RayTracedLighting",
    "headless": true,
    "rt_subframes": 4,
    "stage_load_timeout_sec": 180.0,
    "max_attempts_per_flight": 25
  }
}
```

Map paths are relative to `dataset_generator`. The first run can take several
minutes while Isaac compiles shaders. Later retries on the same map reuse the
loaded stage and render product.

The checked-in `warehouse.usd` and `office.usd` files may load without their
external visual references. The generator detects a stage with no usable
render geometry and creates a local procedural scene with a floor, props, and
lights. This keeps generation functional without CARLA, network assets, or a
separate Isaac asset pack.

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
- Repeated `OmniHub ... child exited` warnings can occur on the ARM/GB10 Isaac
  image. They are non-fatal for local USD generation if `Simulation App Startup
  Complete` appears and cache files are produced.
- No RGB data: verify that Replicator starts successfully and that the GPU is
  visible with `nvidia-smi` inside the container.
- A loaded map with only `GroundPlane/CollisionMesh` indicates missing USD
  references. The generator will report that it is creating the procedural
  fallback scene.
- USD load timeout: verify the configured map path and increase
  `stage_load_timeout_sec` for large stages.
- Repeated target-loss retries: inspect the generated debug frames and the
  selected target path printed by the generator.
