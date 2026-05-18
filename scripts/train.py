# ----------- Libraries 
import torch
import sys
from cgpt.model_base import GPT 
from cgpt.storage import LocalStorage, CloudStorage
from cgpt.trainer import Trainer
from cgpt.callbacks import EstimateLossCallback, SaveCheckPointCallback, PlayGameStockCallback
from cgpt.datasets import ChessDataset
from torch.utils.data import DataLoader
import tomllib
import logging 
# ----------- Setup 

config_path = sys.argv[1]
with open(config_path, 'rb') as f:
    config = tomllib.load(f)

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s %(levelname)s %(message)s'
)
# ----- set seed 
torch.manual_seed(42)

# --------- Main
if __name__ == '__main__': 
        
    # ----- Setup storage manager  
    save_cloud = config['storage']['save_cloud']
    if save_cloud: 
        storage_manager = CloudStorage.from_config(config)
    else:
        storage_manager = LocalStorage.from_config(config)

    # ----------- Load in datasets 
    # ------ Load in x and y datasets 
    stoi, itos, x, y  = storage_manager.load_dataset()
    vocab_size = len(stoi)

    # ------ Create train_data and val_data
    n_rows = x.shape[0]
    n = int(config['training']['train_test_split'] * n_rows)
    train_data = DataLoader(
        ChessDataset(
            x = x[:n], 
            y = y[:n], 
            block_size = config['model']['block_size']
        ),
        batch_size= config['training']['batch_size'], 
        shuffle=True
    )

    val_data = DataLoader(
        ChessDataset(
            x = x[n:], 
            y = y[n:], 
            block_size = config['model']['block_size']
        ),
        batch_size= config['training']['batch_size'], 
        shuffle=True
    )
    # --------- Define Model setup  
    model = GPT.from_config(vocab_size, config)
    optimzier = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate']
    )
    callbacks = [
        EstimateLossCallback(), 
        SaveCheckPointCallback(),
        PlayGameStockCallback.from_config(
            stoi=stoi, 
            itos=itos, 
            config=config
        )
    ]
    trainer = Trainer.from_config(
        model = model, 
        optimzier=optimzier, 
        callbacks=callbacks, 
        train_loader=train_data, 
        val_loader=val_data, 
        storage_manager=storage_manager, 
        config=config
    )

    trainer.train()


    # if training_params['continue_training']:
    #     files = storage_manager.list_checkpoints()
        
    #     if len(files) == 0: 
    #         print('no valid checkpoints to load, start a new training run')
    #         sys.exit(0)

    #     last_chk_pt = files[-1]
    #     print('loading chk_pt : ',  last_chk_pt)

    #     checkpoint = storage_manager.load_checkpoint(last_chk_pt)

    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_iter = checkpoint['step']
    #     model.train()