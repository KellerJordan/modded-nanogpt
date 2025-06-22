#!/bin/bash

# chmod +x setup_speedrun.sh
# ./setup_speedrun.sh

# Set up error handling
set -e  # Exit immediately if a command exits with a non-zero status

echo "Setting up Python virtual environment for Speedrun..."

# Create virtual environment
python3 -m venv ~/speedrun_venv

# Activate virtual environment
source ~/speedrun_venv/bin/activate

# Update pip and setuptools
echo "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools

# Install torch and torchvision
echo "Installing torch and torchvision..."
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install requirements with force reinstall
echo "Installing requirements"
pip install -r requirements.txt

# List installed packages for verification
echo -e "\nInstalled packages:"
pip list

# Instructions for future use
echo -e "\n======================="
echo "Setup complete!"
echo "======================="
echo "To activate this environment in the future, run:"
echo "    source ~/speedrun_venv/bin/activate"
echo ""
echo "To deactivate the environment, simply run:"
echo "    deactivate"
echo ""
echo "Your virtual environment is located at: ~/speedrun_venv"
echo "======================="

