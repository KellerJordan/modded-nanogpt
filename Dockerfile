# docker build -t speedrun_plm .
# docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results -v $(pwd)/logs:/app/logs speedrun_plm python train_plm.py
# docker run --gpus all -v ${PWD}/data:/app/data -v ${PWD}/results:/app/results -v ${PWD}/logs:/app/logs speedrun_plm python train_plm.py
# Use PyTorch official image with CUDA support
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file first for better Docker layer caching
COPY requirements.txt .

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools

# PyTorch is already installed in the base image, but upgrade if needed
RUN pip install --upgrade torch torchvision

# Install requirements
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Set the default command
CMD ["bash"] 