#!/bin/bash

# chmod +x run.sh
# ./run.sh

# Build the Docker image
sudo docker build -t speedrun_plm .

# Run the Docker container with GPU support and volume mounts
sudo docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results -v $(pwd)/logs:/app/logs speedrun_plm python train_plm.py
