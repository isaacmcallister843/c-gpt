#!/bin/bash
set -e

echo "Installing system packages..."
sudo apt-get update
sudo apt-get install -y stockfish

echo "Installing Python dependencies..."
pip install -e .
pip install -e ".[cloud]"

# Renting cloud instances for DL generally come up with a pytorch + CUDA image, no need to resinstall CUDA
# echo "Installing CUDA" 
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "Done."
