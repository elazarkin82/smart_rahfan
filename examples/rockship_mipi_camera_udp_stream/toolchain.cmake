# CMake Toolchain file for Radxa Zero 3E (Rockchip RK3566)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Dynamically compute the Yocto build directory path relative to this file
set(YOCTO_BUILD_DIR "${CMAKE_CURRENT_LIST_DIR}/../../yocto/build")

# Cross-compiler paths derived from the dynamic build directory
set(CMAKE_C_COMPILER "${YOCTO_BUILD_DIR}/tmp/work/radxa_zero_3e-poky-linux/linux-torvalds-next/6.11-rc6+git/recipe-sysroot-native/usr/bin/aarch64-poky-linux/aarch64-poky-linux-gcc")
set(CMAKE_CXX_COMPILER "${YOCTO_BUILD_DIR}/tmp/work/radxa_zero_3e-poky-linux/linux-torvalds-next/6.11-rc6+git/recipe-sysroot-native/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++")

# Target Sysroot containing libraries and headers (including turbojpeg)
set(CMAKE_SYSROOT "${YOCTO_BUILD_DIR}/tmp/work/cortexa55-poky-linux/v4l-utils/1.26.1+git/recipe-sysroot")

# Search configurations
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
