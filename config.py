from dataclasses import dataclass
import torch
from pathlib import Path

@dataclass
class Config:
    batch_size: int = 4
    block_size: int = 120
    max_iters: int = 10000
    learning_rate: float = 3e-4
    n_embd: int = 120
    n_head: int = 3
    n_layer: int = 2
    dropout: float = 0.2
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_rows: int = 14000000 # get all rows 
    min_elo : int = 2100 
    save_name = 'bishop'
    ROOT = Path(__file__).resolve().parent

    DATA_DIR = ROOT / 'data'
    MODEL_DIR = ROOT / 'models'
    MODEL_SAVE_DIR = MODEL_DIR / save_name 
    MODEL_CHK_DIR = MODEL_SAVE_DIR / 'check_points' 


config = Config()
