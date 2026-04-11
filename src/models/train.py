# ----------- Libraries 
import torch
from torch.nn import functional as F 
import numpy as np 
import pandas as pd 
import json  
from pathlib import Path
import sys
from stockfish import Stockfish

from config import model_params, DATA_DIR, MODEL_DIR, device, training_params, test_config
from .model_base import GPT 
from .test import play_game_test


# ----------- Setup 
# ----- Unpack training_params 
batch_size = training_params.batch_size
learning_rate = training_params.learning_rate
max_iters = training_params.max_iters
train_test_split = training_params.train_test_split
continue_training = training_params.continue_training
save_and_eval_interval = training_params.save_and_eval_interval
print_loss_interval = training_params.print_loss_interval
loss_eval_iter = training_params.loss_eval_iter

# ------ Unpack test_params 
eval_lvl_end = test_config.eval_lvl_end
eval_lvl_jump = test_config.eval_lvl_jump
eval_lvl_start = test_config.eval_lvl_start
eval_num_games = test_config.eval_num_games
stockfish = Stockfish(path= test_config.stockfish_path)

# ----- Configure save directories 
MODEL_SAVE_DIR = MODEL_DIR / model_params.save_name 
MODEL_CHK_DIR = MODEL_DIR / model_params.save_name  / 'check_points' 
TEST_SAVE_DIR = MODEL_DIR / model_params.save_name  / f"{model_params.save_name}.csv"  

# ----- set seed 
torch.manual_seed(42)

# --------- Setup DIR if need
MODEL_CHK_DIR.mkdir(parents=True, exist_ok=True)

# ----- Check our vocab length 
with open(DATA_DIR / 'processed' / 'itos.json', 'r') as file:
    itos = json.load(file)
vocab_size = len(itos)

# ---- Loading in x and y datasets 
x = torch.load(DATA_DIR / 'processed' / 'x.pt').to(device)
y = torch.load(DATA_DIR / 'processed' / 'y.pt').to(device)

n_rows = x.shape[0]

# define train / test split 
n = int(train_test_split * n_rows)
train_data_x = x[:n]
train_data_y = y[:n]
val_data_x = x[n:]
val_data_y = y[n:]

# ---- Define batch retrieval method 
def get_batch(split : str): 
    if split == 'train': 
        split_data_x, split_data_y = train_data_x, train_data_y
    else:
        split_data_x, split_data_y = val_data_x, val_data_y

    idx = np.random.randint(0, split_data_x.shape[0], size=batch_size, dtype=int)

    x = split_data_x[idx, : ]
    y = split_data_y[idx, : ]

    return x.to(device), y.to(device)

# -------- Helpers 
@torch.no_grad()
def estimate_loss(model : GPT) -> dict:
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(loss_eval_iter)
        for k in range(loss_eval_iter):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def on_eval(model : GPT, step : int) -> None:
    losses = estimate_loss(model)
    print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

def on_save(model : GPT, optimizer : torch.optim, step : int) -> None: 
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
        }, 
        MODEL_CHK_DIR / f'checkpoint_{step}.pt'
    )
    losses = estimate_loss(model)
    model.eval()
    test_results = []
    for _ in range(eval_num_games): 
        for lvl in range(eval_lvl_start, eval_lvl_end, eval_lvl_jump):
            output_dict = play_game_test(
                model = model, 
                stock_lvl = lvl, 
                stockfish= stockfish
            )
            output_dict['step'] = step 
            output_dict['stock_lvl'] = lvl 
            output_dict['train_loss'] = losses['train'].item()
            output_dict['val_loss'] = losses['val'].item()
            test_results.append(output_dict)
    
    model.train()
    print(test_results[0], test_results[-2])
    df_test = pd.DataFrame(test_results)
    
    if not Path(TEST_SAVE_DIR).is_file():
        df_test.to_csv(TEST_SAVE_DIR, index=False)
    else: 
        df_prev= pd.read_csv(TEST_SAVE_DIR)
        result = pd.concat([df_prev, df_test])
        result.to_csv(TEST_SAVE_DIR, index=False)

# --------- Main
if __name__ == '__main__': 
    # --------- Define Model setup  
    model = GPT(
        vocab_size=vocab_size,
        n_embd=model_params.n_embd,
        n_head=model_params.n_head,
        n_layer=model_params.n_layer,
        block_size=model_params.block_size,
        dropout=model_params.dropout,
        device=device,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start_iter = 0 

    if continue_training:
        files = sorted(
            MODEL_CHK_DIR.iterdir(), 
            key=lambda f: int(f.stem.split('_')[-1])
        )
        
        if len(files) == 0: 
            print('no valid checkpoints to load, start a new training run')
            sys.exit(0)

        last_chk_pt = files[-1]
        print('loading chk_pt : ',  last_chk_pt)

        checkpoint = torch.load(last_chk_pt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['step']
        model.train()

    # -------- Training run 
    for step in range(start_iter, max_iters+1):

        if (step % save_and_eval_interval == 0) and (step != 0): 
            on_save(model, optimizer, step)
        elif step % print_loss_interval == 0:
            on_eval(model, step)

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    on_save(model, optimizer, step)
    torch.save(model.state_dict(), MODEL_SAVE_DIR / f"{model_params.save_name}.pt")
