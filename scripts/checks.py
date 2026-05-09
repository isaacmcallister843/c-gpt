import sys 
import tomllib
from cgpt.paths import get_device
from cgpt.storage import CloudStorage, LocalStorage
from cgpt.tests import test_load_model_GPT, test_block_size_to_model_align, test_load_stockfish
device = get_device() 

config_path = sys.argv[1]
with open(config_path, 'rb') as f:
    config = tomllib.load(f)

model_config = config['model']
train_config = config['training']
eval_config = config['evaluation']
save_cloud = config['storage']['save_cloud']

if save_cloud: 
    storage_manager = CloudStorage.from_config(config)
else:
    storage_manager = LocalStorage.from_config(config)


# ---- Loading in x and y datasets 
encoder, decoder, x, y  = storage_manager.load_dataset()
vocab_size = len(encoder)

# ----- Testing before training run  
if __name__ == '__main__': 
    print(
        f"""
        Training Config:
        device:     {device}
        model:      {model_config["save_name"]}
        batch_size: {train_config["batch_size"]}
        max_iters:  {train_config["max_iters"]}
        block_size : {model_config["block_size"]}
        """
    )    
    test_load_model_GPT(model_config = model_config, device = device, vocab_size = 10)
    test_load_stockfish(STOCK_FISH_DIR = eval_config["stockfish_path"])

    test_block_size_to_model_align(block_size =model_config["block_size"] , x = x)

