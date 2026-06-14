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

RUN rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    if id ubuntu >/dev/null 2>&1; then \
        userdel -r ubuntu || true; \
    fi

RUN set -eux; \
    if getent group ubuntu >/dev/null; then \
        groupdel ubuntu || true; \
    fi
    
