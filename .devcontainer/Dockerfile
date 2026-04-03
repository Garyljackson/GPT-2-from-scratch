# Install PyTorch 2.11 with CUDA 12.8 (supports Blackwell/sm_120 GPUs like RTX 5090)
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

# Install Ubuntu packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y rsync git curl ca-certificates wget && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p -m 755 /etc/apt/keyrings && \
    wget -nv -O /tmp/githubcli-archive-keyring.gpg https://cli.github.com/packages/githubcli-archive-keyring.gpg && \
    mv /tmp/githubcli-archive-keyring.gpg /etc/apt/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list && \
    apt-get update && apt-get install -y gh && \
    rm -rf /var/lib/apt/lists/*

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Install Python packages
COPY requirements.txt requirements.txt
RUN uv pip install --system --break-system-packages --no-cache -r requirements.txt
