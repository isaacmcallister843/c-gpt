#!/bin/bash
set -e

echo "Installing system packages..."
sudo apt-get update
sudo apt-get install -y stockfish pip python3.11 python3.11-venv

echo "Setting up venv..."
python3.11 -m venv venv
source venv/bin/activate

echo "Installing Python dependencies..."
pip install -U pip
pip install -e .
pip install -e ".[cloud]"

echo "Done."