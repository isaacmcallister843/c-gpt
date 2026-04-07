# ----------- Libraries 
import torch
from torch.nn import functional as F 
import numpy as np 
import pandas as pd 
import json  
from pathlib import Path
from config import config
from .model_base import GPT 
from .test import play_game_test

# ----------- Setup 
# ----- Parameters
DATA_DIR = config.DATA_DIR
MODEL_DIR = config.MODEL_DIR
MODEL_CHK_DIR = config.MODEL_CHK_DIR
MODEL_SAVE_DIR = config.MODEL_SAVE_DIR

batch_size = config.batch_size
device = config.device
learning_rate = config.learning_rate
max_iters = config.max_iters
save_name = config.save_name 

TEST_SAVE_DIR = MODEL_SAVE_DIR / f"{save_name}.csv"  

loss_check_interval =  100
eval_iters = 20
save_and_eval_interval = 1000

eval_num_games = 1
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
n = int(0.9 * n_rows)
train_data_x = x[:n]
train_data_y = y[:n]
val_data_x = x[n:]
val_data_y = y[n:]

# ---- Define batch retrieval method 
def get_batch(split): 
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
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def on_eval():
    losses = estimate_loss()
    print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

def on_save(): 
    output_path = MODEL_CHK_DIR / f'checkpoint_{step}.pt'
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
        }, 
        output_path
    )
    losses = estimate_loss()
    
    test_results = []
    for _ in range(eval_num_games): 
        for lvl in range(5,25,5):
            output_dict = play_game_test(model_path = output_path, stock_lvl=lvl)
            output_dict['step'] = step 
            output_dict['stock_lvl'] = lvl 
            output_dict['train_loss'] = losses['train'].item()
            output_dict['val_loss'] = losses['val'].item()
            test_results.append(output_dict)
    
    print(test_results[0], test_results[-2])
    df_test = pd.DataFrame(test_results)
    
    if not Path(TEST_SAVE_DIR).is_file():
        df_test.to_csv(TEST_SAVE_DIR, index=False)
    else: 
        df_prev= pd.read_csv(TEST_SAVE_DIR)
        result = pd.concat([df_prev, df_test])
        result.to_csv(TEST_SAVE_DIR, index=False)

# --------- Training run 
model = GPT(
    vocab_size=vocab_size,
    n_embd=config.n_embd,
    n_head=config.n_head,
    n_layer=config.n_layer,
    block_size=config.block_size,
    dropout=config.dropout,
    device=config.device,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    if step % loss_check_interval == 0:
        on_eval()
    
    if step % save_and_eval_interval == 0: 
        on_save()
    
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

on_save()
torch.save(model.state_dict(), MODEL_SAVE_DIR / f"{save_name}.pt")
