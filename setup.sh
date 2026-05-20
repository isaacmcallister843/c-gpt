#!/bin/bash
set -e

echo "=== System packages ==="
sudo apt-get update
sudo apt-get install -y stockfish git

echo "=== Python 3.12 ==="
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv

echo "=== Venv setup ==="
python3.12 -m venv venv
source venv/bin/activate
pip install -U pip

echo "=== Project dependencies ==="
pip install -e .
pip install -e ".[cloud]"

echo "=== Verify ==="
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from cgpt.model_base import GPT; print('Package imports work')"
nvidia-smi

echo "=== Done ==="