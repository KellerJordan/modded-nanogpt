#!/bin/bash

# Build and test the Newton-Schulz CUDA kernel

echo "Building Newton-Schulz CUDA extension..."
cd /Users/jaso1024/Documents/code/modded-nanogpt/cuda

# Clean previous builds
rm -rf build/ dist/ *.egg-info __pycache__

# Build the extension
python setup.py install --user

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo ""
    echo "Running tests..."
    python test_newtonschulz5.py
else
    echo "Build failed!"
    exit 1
fi