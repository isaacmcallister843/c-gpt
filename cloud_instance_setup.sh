#!/bin/bash
set -e

echo "Installing system packages..."
sudo apt-get update
sudo apt-get install -y stockfish

echo "Installing Python dependencies..."
pip install -e .
pip install -e ".[cloud]"

echo "Done."
