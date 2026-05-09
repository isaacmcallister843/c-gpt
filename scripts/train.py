# ----------- Libraries 
import torch
from torch.nn import functional as F 
import numpy as np 
import pandas as pd 
import sys
from stockfish import Stockfish
from cgpt.model_base import GPT 
from cgpt.storage import LocalStorage, CloudStorage
from cgpt.evaluate import play_game_test
import tomllib

# ----------- Setup 
config_path = sys.argv[1]
with open(config_path, 'rb') as f:
    config = tomllib.load(f)

# ----- Unpack Parameters 
training_params = config['training']
evaluation_params = config['evaluation'] 
data_params = config['data'] 
model_params = config['model'] 
block_size = model_params['block_size']

# ----- Setup storage manager  
save_cloud = config['storage']['save_cloud']
if save_cloud: 
    storage_manager = CloudStorage.from_config(config)
else:
    storage_manager = LocalStorage.from_config(config)

# ----- set seed 
torch.manual_seed(42)

# ----------- Load in datasets 
# # ---- Load in x and y datasets 
encoder, decoder, x, y  = storage_manager.load_dataset()
vocab_size = len(encoder)

assert block_size <= x.shape[1] 

n_rows = x.shape[0]

# define train / test split 
n = int(training_params['train_test_split'] * n_rows)
train_data_x = x[:n]
train_data_y = y[:n]
val_data_x = x[n:]
val_data_y = y[n:]


# ----------- Helper functions
def get_batch(split : str): 
    if split == 'train': 
        split_data_x, split_data_y = train_data_x, train_data_y
    else:
        split_data_x, split_data_y = val_data_x, val_data_y

    idx = np.random.randint(0, split_data_x.shape[0], size = training_params['batch_size'], dtype=int)

    x = split_data_x[idx, 0:block_size]
    y = split_data_y[idx, 0:block_size]

    return x.to(training_params['device']), y.to(training_params['device'])
 
@torch.no_grad()
def estimate_loss(model : GPT) -> dict:
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(training_params['loss_eval_iter'])
        for k in range(training_params['loss_eval_iter']):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out

@torch.no_grad()
def on_save(
        model : GPT, 
        optimizer : torch.optim, 
        step : int, 
        stockfish : Stockfish
    ) -> None:

    storage_manager.save_checkpoint(
        save_dict  = model.state_dict(), 
        optim_dict = optimizer.state_dict(), 
        save_name = f"checkpoint_{step}", 
        step = step 
    )

    model.eval()
    losses = estimate_loss(model)
    test_results = []
    for _ in range(evaluation_params['eval_num_games']): 
        for lvl in range(evaluation_params['eval_lvl_start'], evaluation_params['eval_lvl_end'], evaluation_params['eval_lvl_jump']):
            output_dict = play_game_test(
                model = model, 
                stock_lvl = lvl, 
                stockfish = stockfish,
                encoder = encoder, 
                decoder = decoder, 
                device = training_params['device'] 
            )

            output_dict['step'] = step 
            output_dict['train_loss'] = losses['train'].item()
            output_dict['val_loss'] = losses['val'].item()
            test_results.append(output_dict)
    
    model.train()
    print(test_results[0], test_results[-1])
    df_test = pd.DataFrame(test_results)
    storage_manager.save_results(data = df_test)

# --------- Main
if __name__ == '__main__': 
    # --------- Load in stockfish model 
    stockfish = Stockfish(path = evaluation_params['stockfish_path'])

    # --------- Define Model setup  
    model = GPT(
        vocab_size = vocab_size,
        n_embd = model_params['n_embd'],
        n_head = model_params['n_head'],
        n_layer = model_params['n_layer'],
        block_size = block_size,
        dropout = model_params['dropout'],
        device = training_params['device'],
    ).to(training_params['device'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_params['learning_rate'])
    start_iter = 0 


    if training_params['continue_training']:
        files = storage_manager.list_checkpoints()
        
        if len(files) == 0: 
            print('no valid checkpoints to load, start a new training run')
            sys.exit(0)

        last_chk_pt = files[-1]
        print('loading chk_pt : ',  last_chk_pt)

        checkpoint = storage_manager.load_checkpoint(last_chk_pt)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['step']
        model.train()

    # -------- Training run 
    for step in range(start_iter, training_params['max_iters']+1):

        if (step % training_params['save_and_eval_interval'] == 0) and (step != 0): 
            on_save(model, optimizer, step, stockfish)
        elif step % training_params['print_loss_interval'] == 0:
            model.eval()
            losses = estimate_loss(model)
            model.train()
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    on_save(model, optimizer, step, stockfish)
