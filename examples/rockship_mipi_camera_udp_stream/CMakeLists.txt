cmake_minimum_required(VERSION 3.10)
project(rockship_mipi_camera_udp_stream C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O2")

add_executable(rockship_mipi_camera_udp_stream main.c)

# Link standard libraries and TurboJPEG
target_link_libraries(rockship_mipi_camera_udp_stream turbojpeg pthread)
