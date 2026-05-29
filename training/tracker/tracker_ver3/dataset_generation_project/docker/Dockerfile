FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install -y \
        vim \
        git \
        gitk \
        python3 \
        python3-pip \
        gawk \
        wget \
        git-core \
        diffstat \
        unzip \
        texinfo \
        gcc-multilib \
        build-essential \
        chrpath \
        zstd
        
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    locales \
    ca-certificates \
    cpio \
    debianutils \
    file \
    gcc \
    g++ \
    iputils-ping \
    libacl1 \
    python3-git \
    python3-jinja2 \
    python3-pexpect \
    python3-subunit \
    python3-setuptools \
    python3-venv \
    python-is-python3 \
    socat \
    xz-utils \
    curl \
    rsync \
    bc \
    bison \
    flex \
    make \
    patch \
    tar \
    gzip \
    bzip2 \
    xz-utils \
    lz4 \
    liblz4-tool \
    libssl-dev \
    libncurses-dev \
    libelf-dev \
    ccache \
    openssh-client \
    dosfstools \
    mtools \
    parted \
    e2fsprogs \
    bmap-tools \
    kmod \
    u-boot-tools \
    device-tree-compiler \
    nano \
    less \
    tree \
    tmux \
    screen \
    bash-completion \
    procps \
    iproute2 \
    net-tools \
    psmisc \
    findutils \
    xterm \
    geany \
    && sed -i 's/^# *en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && locale-gen \
    && update-ca-certificates
    
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    gcc-arm-linux-gnueabihf \
    g++-arm-linux-gnueabihf
    
RUN apt update && apt install -y locales && locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
RUN apt install -y cpio file lz4
    
RUN apt-get install -y sudo

# Install OpenCV build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    python3-dev \
    python3-numpy \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran

# Build OpenCV from source with opencv_contrib and nonfree features (SURF, etc.)
RUN git clone --depth 1 -b 4.9.0 https://github.com/opencv/opencv.git /tmp/opencv && \
    git clone --depth 1 -b 4.9.0 https://github.com/opencv/opencv_contrib.git /tmp/opencv_contrib && \
    mkdir -p /tmp/opencv/build && \
    cd /tmp/opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
          -D OPENCV_ENABLE_NONFREE=ON \
          -D BUILD_opencv_python3=ON \
          -D BUILD_DOCS=OFF \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
          .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /tmp/opencv /tmp/opencv_contrib

RUN rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    if id ubuntu >/dev/null 2>&1; then \
        userdel -r ubuntu || true; \
    fi

RUN set -eux; \
    if getent group ubuntu >/dev/null; then \
        groupdel ubuntu || true; \
    fi
    
