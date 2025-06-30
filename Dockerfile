# sudo docker build -t speedrun_plm .
# sudo docker run --gpus all --shm-size=128g -v ${PWD}:/workspace speedrun_plm torchrun --standalone --nproc_per_node=4 train.py

# 1️⃣  CUDA / cuDNN base with no Python
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04

# 2️⃣  System prerequisites + Python 3.12
ENV        DEBIAN_FRONTEND=noninteractive \
           PYTHON_VERSION=3.12.7 \
           PATH=/usr/local/bin:$PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential curl git ca-certificates \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev libncursesw5-dev xz-utils tk-dev \
        libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -fsSLO https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j"$(nproc)" && \
    make altinstall && \
    cd .. && rm -rf Python-${PYTHON_VERSION}* && \
    ln -s /usr/local/bin/python3.12 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.12    /usr/local/bin/pip

# 3️⃣  Location of project code (inside image) – NOT shared with host
WORKDIR /app

# 4️⃣  Copy requirements first for layer caching
COPY requirements.txt .

RUN pip install --upgrade pip setuptools && \
    # force-install torch built for CUDA 12.6
    pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126 && \
    pip install -r requirements.txt

# 5️⃣  Copy the rest of the source
COPY . .

# ──────────────────────────────────────────────────────────────────────────────
# 6️⃣  Single persistent host volume (/workspace) for *all* artefacts & caches
#     Bind-mount it when you run the container:  -v ${PWD}:/workspace
# ──────────────────────────────────────────────────────────────────────────────
ENV PROJECT_ROOT=/workspace \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    XDG_CACHE_HOME=/workspace/.cache \
    WANDB_DIR=/workspace/logs \
    TQDM_CACHE=/workspace/.cache/tqdm

RUN mkdir -p \
      /workspace/.cache/huggingface \
      /workspace/.cache/torch \
      /workspace/.cache/tqdm \
      /workspace/logs \
      /workspace/data \
      /workspace/results

# Declare the volume so other developers know it's intended to persist
VOLUME ["/workspace"]

# 7️⃣  Default command – override in `docker run … python train.py`
CMD ["bash"]
