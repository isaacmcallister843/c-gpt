from dataclasses import dataclass
import torch
from pathlib import Path

# ----------- global config class 
# file paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
MODEL_DIR = ROOT / 'models'
device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class HF_DATA_CONFIG: 
    n_rows: int = 14000000 # get all rows 
    min_elo : int = 2100 

@dataclass
class TrainConfig:
    batch_size: int = 64
    max_iters: int = 50000
    learning_rate: float = 3e-4
    continue_training: bool = False
    print_loss_interval : int = 100
    train_test_split : float = .9
    loss_eval_iter: int = 20
    save_and_eval_interval: int = 1000

@dataclass
class TestConfig:
    eval_num_games: int = 1
    eval_lvl_start: int = 5
    eval_lvl_end: int = 25
    eval_lvl_jump: int = 5
    stockfish_path: Path = ROOT / "misc/stockfish/stockfish-windows-armv8"

@dataclass
class ModelParams(): 
    n_embd: int 
    n_head: int 
    n_layer: int 
    block_size : int 
    save_name : str 
    dropout: float = 0.2


# --------- Instances  
bishop_params =  ModelParams(
    n_embd = 80, 
    n_head = 4, 
    n_layer = 2, 
    block_size = 120,
    save_name='bishop'
)

knight_params =  ModelParams(
    n_embd = 256, 
    n_head = 8, 
    n_layer = 8,
    block_size = 180,
    save_name='knight'
)

small_training_run = TrainConfig(
    batch_size=4, 
    max_iters = 1000,
    print_loss_interval = 10,  
    save_and_eval_interval = 100
)

large_training_run = TrainConfig(
    batch_size= 64, 
    max_iters = 70000, 
    save_and_eval_interval = 5000
)

# ------- End points
model_params = bishop_params
training_params = small_training_run
hf_data_config = HF_DATA_CONFIG()
test_config = TestConfig(eval_num_games = 1)
