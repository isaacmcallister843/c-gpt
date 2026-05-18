from . import model_base 
import logging  
import torch 

logger = logging.getLogger(__name__)
# ----- Simple load test 
def test_load_model_GPT(model_config : dict, device : str, vocab_size : int) -> None:
    model = model_base.GPT(
        vocab_size=vocab_size,
        n_embd=model_config["n_embd"],
        n_head=model_config["n_head"],
        n_layer=model_config["n_layer"],
        block_size=model_config["block_size"],
        dropout=model_config["dropout"],
        device= device
    )
    logger.info("Model loaded successfully, with num parameters : %d", model.num_parameters)

def test_load_stockfish(STOCK_FISH_DIR: str) -> None:
    from stockfish import Stockfish
    stockfish = Stockfish(path= STOCK_FISH_DIR)
    logger.info("Stockfish loaded successfully")

def test_block_size_to_model_align(block_size : int , x : torch.Tensor) -> None:
    assert x.shape[1] == block_size
    logger.info("Block size of %d aligns with x dataset shape of %s", block_size, x.shape)
