from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]  # up from src/cgpt/ to project root
DATA_DIR = ROOT / 'data'
MODEL_DIR = ROOT / 'models'

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
