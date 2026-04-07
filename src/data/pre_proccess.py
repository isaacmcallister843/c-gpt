# ------ Libraries
# System 
from pathlib import Path

# Add SRC_DIR to path, and import config
from config import config

# Data handling
from datasets import load_dataset
import json
import numpy as np 
import torch

# ----- Setup 
block_size = config.block_size
n_rows = config.n_rows
DATA_DIR = config.DATA_DIR
min_elo = config.min_elo

# ------ create mapping
stoi = {'<PAD>': 0}
current_idx_stoi = 1

with open(config.ROOT / 'misc' / 'san_strings' / 'san_string.txt', 'r') as file:
    for line in file:
        stoi[str(line.strip())] = current_idx_stoi
        current_idx_stoi +=1 

# get decoding map 
itos = {i: ch for ch, i in stoi.items()}

# save encoding / decoding map
with open(DATA_DIR / "processed" / "stoi.json", "w") as f:
    json.dump(stoi, f, indent=4)

with open(DATA_DIR / "processed" / "itos.json", "w") as f:
    json.dump(itos, f, indent=4)

# ------ create x and y datasets 
dataset = load_dataset(str(DATA_DIR / 'raw'), data_files = 'dataset.parquet' )['train']
dataset = dataset.filter(lambda ex: ex['white_elo'] is not None and ex['white_elo'] >= min_elo)

x = np.zeros((n_rows, block_size))
y = np.zeros((n_rows, block_size))

current_idx = 0

for ex in dataset:
    if (not ex['white_elo']) or (ex['white_elo'] < min_elo):
        continue

    game = ex['moves_san'][0:block_size]
    game_row = np.zeros(block_size)

    for j, move in enumerate(game): 
        game_row[j]= stoi[move] 
    
    x[current_idx, 0:block_size] = game_row
    y[current_idx, 0:(block_size-1)] = game_row[1::]

    if current_idx % 1000 == 0:
        print(f"Processed {current_idx} games")

    current_idx += 1

    if current_idx == n_rows:
        break

# clip to current_idx 
x = x[0:current_idx, :]
y = y[0:current_idx, :]

# convert to pytorch objects
x_torch = torch.from_numpy(x).long()
y_torch = torch.from_numpy(y).long()

# save to data directory "data/processed/"
torch.save(x_torch, DATA_DIR / "processed" / "x.pt")
torch.save(y_torch, DATA_DIR / "processed" / "y.pt")