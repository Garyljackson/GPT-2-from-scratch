# Install PyTorch 2.11 with CUDA 12.8 (supports Blackwell/sm_120 GPUs like RTX 5090)
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

# Install Ubuntu packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y rsync git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Install Python packages
COPY requirements.txt requirements.txt
RUN uv pip install --system --break-system-packages --no-cache -r requirements.txt
