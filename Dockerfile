# docker build -t speedrun_plm .
# docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results -v $(pwd)/logs:/app/logs speedrun_plm python train_plm.py
# docker run --gpus all -v ${PWD}/data:/app/data -v ${PWD}/results:/app/results -v ${PWD}/logs:/app/logs speedrun_plm python train_plm.py
# Use PyTorch official image with CUDA support
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.12.7
ENV PATH=/usr/local/bin:$PATH

RUN apt update && apt install -y --no-install-recommends build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl git libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev \
    && apt clean && rm -rf /var/lib/apt/lists/*

RUN curl -O https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz

RUN ln -s /usr/local/bin/python3.12 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.12 /usr/local/bin/pip

# Set working directory
WORKDIR /app

# Copy requirements file first for better Docker layer caching
COPY requirements.txt .

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools

# PyTorch is already installed in the base image, but upgrade if needed
RUN pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install requirements
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the default command
CMD ["bash"]