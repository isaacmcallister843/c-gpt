# ------ Libraries
from datasets import load_dataset
import numpy as np 
import torch
import sys
from cgpt.storage import LocalStorage, CloudStorage
import tomllib

# ----------- Setup 
config_path = sys.argv[1]
with open(config_path, 'rb') as f:
    config = tomllib.load(f)

# ----- Unpack Parameters 
data_params = config['data'] 
model_params = config['model'] 

# ----- Setup storage 
save_cloud = config['storage']['save_cloud']
if save_cloud: 
    storage_manager = CloudStorage.from_config(config)
else:
    storage_manager = LocalStorage.from_config(config)

if __name__ == '__main__': 
    # ----- Setup 
    block_size = model_params['block_size']
    n_rows = data_params['n_rows']
    min_elo = data_params['min_elo']

    # ------ create mapping
    stoi = {'<PAD>': 0}
    current_idx_stoi = 1

    with open(data_params['san_string_path'], 'r') as file:
        for line in file:
            stoi[str(line.strip())] = current_idx_stoi
            current_idx_stoi +=1 

    # get decoding map 
    itos = {i: ch for ch, i in stoi.items()}

    # ------ create x and y datasets 
    dataset = load_dataset('parquet', data_files=data_params['dataset_path'], streaming=True, split='train')
    dataset = dataset.take(n_rows)

    x = np.zeros((n_rows, block_size))
    y = np.zeros((n_rows, block_size))

    current_idx = 0

    for ex in dataset:
        if (not ex['white_elo']) or (ex['white_elo'] < min_elo):
            continue

        game = ex['moves_san'][0:block_size]
        game_row = np.zeros(block_size)

        try: 
            for j, move in enumerate(game): 
                game_row[j]= stoi[move] 
        except KeyError: 
            print("Unknown character found : ", move)
            continue 

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

    storage_manager.save_dataset(encoder = stoi, decoder = itos, x = x_torch , y  = y_torch)