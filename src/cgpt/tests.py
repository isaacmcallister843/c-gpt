from . import model_base 
import torch 

# ----- Simple load test 
def test_load_model_GPT(model_config, device, vocab_size) -> None:
    model = model_base.GPT(
        vocab_size=vocab_size,
        n_embd=model_config["n_embd"],
        n_head=model_config["n_head"],
        n_layer=model_config["n_layer"],
        block_size=model_config["block_size"],
        dropout=model_config["dropout"],
        device= device
    )
    print("number of parameters in model : ", sum(p.numel() for p in model.parameters() if p.requires_grad)) 

def test_load_stockfish(STOCK_FISH_DIR):
    from stockfish import Stockfish
    stockfish = Stockfish(path= STOCK_FISH_DIR)
    print("Loaded stockfish correctly")

def test_block_size_to_model_align(block_size, x):
    print(
        f"Context and dataset aligned?:  {x.shape[1] == block_size}  x_shape: , {x.shape[1]}, block_size:  {block_size}"
    )
