# Use NVIDIA's base image for GPU support
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and other dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --config python3

# Create a virtual environment
RUN python3 -m venv /opt/benchmarks_env

# Activate the virtual environment and install packages
ENV PATH="/opt/benchmarks_env/bin:$PATH"

# Copy requirements files
COPY requirements.txt /tmp/requirements.txt

# Install Python packages
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt

# Expose ports for Jupyter Notebook
EXPOSE 8888

# Set the default command
CMD ["bash"]