# Bring-Up Guide: First Demo on Radxa Zero 3E

This document outlines the dependencies, environment setup, and installation steps required to build and run the target tracking demo application on the **Radxa Zero 3E** board.

---

## 📦 System Dependencies

To compile and run the application, the following dependencies must be present in the target root filesystem (or Yocto sysroot during cross-compilation):

### 1. Toolchain & Build Tools
*   **CMake** (version 3.19 or higher)
*   **GCC / G++** supporting C++11 (standard for thread/mutex support)
*   **pkg-config** (for dependency resolution)

### 2. Runtime & Development Libraries
*   **V4L2 (Video4Linux2)**: Standard Linux kernel header (`linux/videodev2.h`). No external package is needed, but `libv4l-dev` can optionally be used for color conversions if needed.
*   **libjpeg-turbo**: Used for high-speed JPEG compression of camera frames before streaming them over HTTP/MJPEG.
    *   *Debian/Ubuntu target:* `sudo apt-get install libjpeg-dev`
    *   *Yocto recipe:* Add `libjpeg-turbo` to `DEPENDS` in the recipe.
*   **CivetWeb**: Embedded web server library.
    *   *Note:* The CMake configuration can fetch and compile CivetWeb statically inline to avoid requiring a pre-installed host package.
*   **RKNN Runtime (librknnrt)**: The Rockchip NPU user-space library (`librknnrt.so`) and API header (`rknn_api.h`) from Rockchip's `rknpu2` repository.
    *   Must match the NPU driver version on the Radxa Zero 3E kernel.

---

## 🛠️ Yocto Integration (`meta-ksg`)

When building the final image via Yocto, the recipe for this application (e.g., `first-demo_1.0.bb`) should include the following:

```bitbake
SUMMARY = "RKNN Target Tracking Demo Application"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

# Dependencies
DEPENDS = "libjpeg-turbo rknpu2"

SRC_URI = "file://first_demo"

S = "${WORKDIR}/first_demo"

inherit cmake
```

---

## 💻 Local Compilation & Cross-Compilation

### 1. Preparing the Cross-Compilation SDK (Host Machine)
If compiling on an x86_64 host for the Radxa Zero 3E target:
1. Source the Yocto SDK environment script:
   ```bash
   source /opt/poky/3.x.y/environment-setup-aarch64-poky-linux
   ```
2. Ensure the sysroot contains the `libjpeg-turbo` and `rknpu2` headers and libraries.

### 2. Building the Application
Run the following commands in the application root directory:
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

---

## 🚀 Deployment & Execution on Radxa Zero 3E

1. **Copy Files to Target**:
   Copy the compiled binary (`first_demo`), the configuration file (`params.conf`), and your compiled RKNN model file (`.rknn`) to the board.
   ```bash
   scp first_demo params.conf tracker_model.rknn root@<radxa-ip>:/usr/bin/
   ```

2. **Verify Camera Dev Node**:
   Ensure your USB camera is recognized by the kernel:
   ```bash
   v4l2-ctl --list-devices
   ```
   Note the device path (e.g., `/dev/video0`).

3. **Configure parameters**:
   Edit `params.conf` to specify your model path, camera device, and desired resolution:
   ```text
   cam_dev=/dev/video0
   capture_width=640
   capture_height=480
   rknn_model_path=/usr/bin/tracker_model.rknn
   ```

4. **Run the Application**:
   ```bash
   first_demo params.conf
   ```

5. **Access the Web Interface**:
   Open a web browser and navigate to:
   ```text
   http://<radxa-ip>:8080
   ```
