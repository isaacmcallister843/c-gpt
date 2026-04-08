from dataclasses import dataclass
import torch
from pathlib import Path

# ----------- global config class 
@dataclass
class Config:
    batch_size: int
    max_iters: int 
    learning_rate: float = 3e-4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_rows: int = 14000000 # get all rows 
    min_elo : int = 2100 
    
    ROOT = Path(__file__).resolve().parent
    DATA_DIR = ROOT / 'data'
    MODEL_DIR = ROOT / 'models'
    STOCK_FISH_DIR = ROOT / "misc/stockfish/stockfish-windows-armv8"

    continue_training = False

config_small_run = Config(
    batch_size = 4,
    max_iters = 12000,
)

config_large_run = Config(
    batch_size = 64,
    max_iters = 50000,
)

config = config_large_run

# --------- model setups 
@dataclass
class ModelParams(): 
    n_embd: int 
    n_head: int 
    n_layer: int 
    block_size: int
    save_name : str 
    dropout: float = 0.2

bishop_params =  ModelParams(
    n_embd = 120, 
    n_head = 3, 
    n_layer =2, 
    block_size=120, 
    save_name='bishop'
)

knight_params =  ModelParams(
    n_embd = 256, 
    n_head = 8, 
    n_layer = 8, 
    block_size = 120, 
    save_name='knight'
)

model_params = knight_params

