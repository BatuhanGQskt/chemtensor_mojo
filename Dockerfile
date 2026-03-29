# RunPod-compatible Dockerfile with SSH support
# Supports both interactive (SSH) and non-interactive (batch job) modes

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MOJO_VERSION=25.1
ARG PUBLIC_KEY=""

ENV SHELL=/bin/bash \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    git \
    curl \
    wget \
    # SSH server
    openssh-server \
    # Python
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    # Linear algebra libraries
    libopenblas-dev \
    liblapack-dev \
    liblapacke-dev \
    # HDF5
    libhdf5-dev \
    # Utilities
    htop \
    tmux \
    vim \
    nano \
    rsync \
    unzip \
    ca-certificates \
    locales \
    sudo \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Configure locale
RUN locale-gen en_US.UTF-8

# Install Python packages
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    h5py \
    jupyter \
    ipython

# Install pixi (for Mojo/MAX)
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

# Configure SSH
RUN mkdir -p /var/run/sshd /root/.ssh && \
    chmod 700 /root/.ssh && \
    # Allow root login and password authentication (RunPod sets password)
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    # Keep connection alive
    echo "ClientAliveInterval 30" >> /etc/ssh/sshd_config && \
    echo "ClientAliveCountMax 5" >> /etc/ssh/sshd_config && \
    # Generate host keys
    ssh-keygen -A

# Create workspace directory
WORKDIR /workspace

# Copy project files (adjust paths as needed)
COPY . /workspace/chemtensor_mojo

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose SSH port
EXPOSE 22

# RunPod specific: expose Jupyter port (optional)
EXPOSE 8888

# Set up pixi environment (will be activated in entrypoint)
WORKDIR /workspace/chemtensor_mojo
RUN pixi install || true

WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["start"]
