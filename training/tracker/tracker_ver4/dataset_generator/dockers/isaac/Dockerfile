# Dockerfile for NVIDIA Isaac Sim Server
FROM nvcr.io/nvidia/isaac-sim:6.0.0

USER root

# The runtime user is created by connect.sh and its UID/GID are only known on
# the host. Keep shared Isaac/Hub caches writable without changing ownership of
# the whole Isaac Sim installation at container startup.
RUN mkdir -p /isaac-sim/kit/cache /var/cache/hub /isaac-sim/.nvidia-omniverse \
    && chmod -R a+rX /isaac-sim \
    && chmod -R a+rwX /isaac-sim/.nvidia-omniverse \
    && chmod 1777 /isaac-sim/kit/cache /var/cache/hub

USER isaac-sim
