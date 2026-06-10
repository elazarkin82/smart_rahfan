cmake_minimum_required(VERSION 3.19)
project(first_demo C CXX)

# Set standard C++ dialect (C++11 is sufficient for std::thread, mutex, condition_variable)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Fetch CivetWeb embedded server library dynamically
include(FetchContent)
FetchContent_Declare(
    civetweb
    GIT_REPOSITORY https://github.com/civetweb/civetweb.git
    GIT_TAG        v1.16
)

# Configure CivetWeb options before making it available
set(CIVETWEB_ENABLE_CXX ON CACHE BOOL "Enable CivetWeb C++ wrapper" FORCE)
set(CIVETWEB_ENABLE_WEBSOCKETS ON CACHE BOOL "Enable WebSockets support" FORCE)
set(CIVETWEB_BUILD_TESTING OFF CACHE BOOL "Disable testing build for civetweb" FORCE)
set(CIVETWEB_ENABLE_ASAN OFF CACHE BOOL "Disable ASAN for civetweb" FORCE)

FetchContent_MakeAvailable(civetweb)

# Find libjpeg (libjpeg-turbo) for grayscale image compression
find_package(JPEG REQUIRED)

# Look for RKNN Runtime library (Rockchip NPU User-Space Library)
# We search in the local rknpu2 repository first, and fall back to system library.
find_library(RKNNRT_LIB rknnrt
    HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../../docker/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64
)
if(NOT RKNNRT_LIB)
    message(WARNING "librknnrt not found by find_library. Will fallback to linking '-lrknnrt' dynamically.")
    set(RKNNRT_LIB "rknnrt")
endif()

# Thread support
find_package(Threads REQUIRED)

# Define target tracking demo executable
add_executable(first_demo
    src/main.cpp
    src/MainService.cpp
    src/core/CameraCapture.cpp
    src/core/TrackerService.cpp
    src/core/WebServer.cpp
)

# Header search directories
target_include_directories(first_demo PRIVATE
    include
    ${JPEG_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../docker/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/include
)

# Link dependencies
target_link_libraries(first_demo PRIVATE
    civetweb-cpp
    civetweb-c-library
    ${JPEG_LIBRARIES}
    ${RKNNRT_LIB}
    Threads::Threads
    dl
)
