cmake_minimum_required(VERSION 3.19)
project(first_demo C CXX)

# Set standard C++ dialect (C++11 is sufficient for std::thread, mutex, condition_variable)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure output directories
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

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
find_library(JPEG_STATIC_LIB NAMES libjpeg.a PATHS /usr/lib/aarch64-linux-gnu)
if(JPEG_STATIC_LIB)
    message(STATUS "Found static libjpeg: ${JPEG_STATIC_LIB}")
    find_package(JPEG REQUIRED)
    set(JPEG_LIBRARIES ${JPEG_STATIC_LIB})
else()
    find_package(JPEG REQUIRED)
endif()


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

# Options for Resize Mode selection (RGA, OMP, CPU)
set(RESIZE_MODE "RGA" CACHE STRING "Resize mode: RGA, OMP, CPU")
set_property(CACHE RESIZE_MODE PROPERTY STRINGS "RGA" "OMP" "CPU")

set(EXTRA_LIBS "")

if(RESIZE_MODE STREQUAL "RGA")
    message(STATUS "[CMake] Using RGA Hardware acceleration for resizing")
    find_library(RGA_LIB rga
        HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../../docker/rknn-toolkit2/rknpu2/examples/3rdparty/rga/libs/Linux/gcc-aarch64
    )
    if(NOT RGA_LIB)
        message(FATAL_ERROR "[CMake] librga not found! Install it or select another RESIZE_MODE (e.g. OMP or CPU).")
    endif()
    message(STATUS "[CMake] Found librga: ${RGA_LIB}")
    list(APPEND EXTRA_LIBS ${RGA_LIB})
    set(RGA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../docker/rknn-toolkit2/rknpu2/examples/3rdparty/rga/include)
elseif(RESIZE_MODE STREQUAL "OMP")
    message(STATUS "[CMake] Using OpenMP Parallel CPU acceleration for resizing")
    find_package(OpenMP REQUIRED)
else()
    message(STATUS "[CMake] Using simple CPU single-threaded resizing")
endif()

# Max limits definitions for MISRA-compliant static buffer allocation
set(MAX_HEATMAP_PXL_SIZE 256 CACHE STRING "Maximum heatmap width and height for pre-allocation")
set(MAX_STACK_LAYERS 64 CACHE STRING "Maximum reference stack layers for pre-allocation")
set(MAX_STACK_TARGET_SIZE 256 CACHE STRING "Maximum reference stack width and height for pre-allocation")

# Define target tracking demo executable
add_executable(first_demo
    src/main.cpp
    src/MainService.cpp
    src/core/CameraCapture.cpp
    src/core/TrackerService.cpp
    src/core/WebServer.cpp
)

target_compile_definitions(first_demo PRIVATE
    MAX_HEATMAP_PXL_SIZE=${MAX_HEATMAP_PXL_SIZE}
    MAX_STACK_LAYERS=${MAX_STACK_LAYERS}
    MAX_STACK_TARGET_SIZE=${MAX_STACK_TARGET_SIZE}
)

# Apply compile definitions based on selected Resize Mode
if(RESIZE_MODE STREQUAL "RGA")
    target_compile_definitions(first_demo PRIVATE USE_RGA)
elseif(RESIZE_MODE STREQUAL "OMP")
    target_compile_definitions(first_demo PRIVATE USE_OMP)
endif()

# Header search directories
target_include_directories(first_demo PRIVATE
    include
    ${JPEG_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../docker/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/include
    ${RGA_INCLUDE_DIR}
)

# Link dependencies
target_link_libraries(first_demo PRIVATE
    civetweb-cpp
    civetweb-c-library
    ${JPEG_LIBRARIES}
    ${RKNNRT_LIB}
    ${EXTRA_LIBS}
    Threads::Threads
    dl
)

# Link OpenMP if selected
if(RESIZE_MODE STREQUAL "OMP" AND OpenMP_CXX_FOUND)
    target_link_libraries(first_demo PRIVATE OpenMP::OpenMP_CXX)
endif()

# Standalone Subgraph RKNN Tester target
add_executable(rknn_tester
    src/tester/rknn_tester.cpp
)

target_compile_definitions(rknn_tester PRIVATE
    MAX_HEATMAP_PXL_SIZE=${MAX_HEATMAP_PXL_SIZE}
    MAX_STACK_LAYERS=${MAX_STACK_LAYERS}
    MAX_STACK_TARGET_SIZE=${MAX_STACK_TARGET_SIZE}
)

target_include_directories(rknn_tester PRIVATE
    include
    ${JPEG_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../docker/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/include
)

target_link_libraries(rknn_tester PRIVATE
    ${RKNNRT_LIB}
    Threads::Threads
    dl
)
