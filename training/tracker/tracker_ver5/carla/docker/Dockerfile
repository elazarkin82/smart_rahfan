FROM carlasim/carla:0.9.16

USER root

# Install basic utilities if needed
RUN apt-get update && apt-get install -y \
    sudo \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Add carla user to sudoers without password for convenience in development
RUN echo "carla ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER carla
