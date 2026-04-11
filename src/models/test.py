# ------ Libraries
import json 
from pathlib import Path
import time 
from stockfish import Stockfish

import chess 
import chess.svg
from config import ModelParams, ROOT, DATA_DIR, device, test_config
import torch 
from .model_base import GPT

# ----------- Setup 
# ----- unpack test params
STOCK_FISH_DIR = test_config.stockfish_path

with open(DATA_DIR / 'processed' / 'itos.json', 'r') as file:
    itos = json.load(file)
itos = {int(i) : ch for i,ch in itos.items()} 
with open(DATA_DIR / 'processed' / 'stoi.json', 'r') as file:
    stoi = json.load(file)

vocab_size = len(itos)

decode = lambda x : [itos[val] for val in x]
encode = lambda x : [stoi[val] for val in x]

# ----- Simple load test 
def test_load_model_GPT(model_params : ModelParams) -> None:
    model = GPT(
        vocab_size=vocab_size,
        n_embd=model_params.n_embd,
        n_head=model_params.n_head,
        n_layer=model_params.n_layer,
        block_size=model_params.block_size,
        dropout=model_params.dropout,
        device= device
    )
    print("number of parameters in model : ", sum(p.numel() for p in model.parameters() if p.requires_grad)) 

def test_load_stockfish():
    from stockfish import Stockfish
    stockfish = Stockfish(path= STOCK_FISH_DIR)
    print("Loaded stockfish correctly")

def test_block_size_to_model_align(model_params):
    x = torch.load(DATA_DIR / 'processed' / 'x.pt')
    print(f"Context and dataset aligned?:  {x.shape[1] == model_params.block_size} x_shape: , {x.shape[1]}, block_size:  {model_params.block_size}")

# ----- Evaluation Tests  
# methods to display the board 
def show_board(board, display_game, size=250, delay=0):
    if not display_game:
        return
    if delay:
        time.sleep(delay)
    from IPython.display import display, clear_output
    clear_output(wait=True)
    display(chess.svg.board(board, size=size))

def play_game_test(
        model, 
        stockfish ,
        stock_lvl : int = 1, 
        display_game : bool = False, 
        delay : float= .5
    ) -> dict:
    
    # set board state 
    board = chess.Board()

    # set variables 
    illegal_move_count = 0
    total_moves = 0
    size=250
    stockfish.set_skill_level(stock_lvl)

    # methods to play a chess game 
    def make_GPT_move(board):
        nonlocal illegal_move_count
        
        san_moves = []
        temp_board = chess.Board()
        for move in board.move_stack:
            san_moves.append(temp_board.san(move))
            temp_board.push(move)
        
        enc_game = encode(san_moves)
        idx = torch.tensor([enc_game]).to(device)
        
        # try the top move first
        move_idx = model.generate_single_step(idx)
        san_move = itos[move_idx]
        
        legal_san = [board.san(m) for m in board.legal_moves]
        
        if san_move in legal_san:
            board.push_san(san_move)
            return san_move
        
        # illegal move -- mask and resample
        illegal_move_count += 1
        probs = model.generate_probs(idx)[0]
        
        # build mask: set all illegal moves to 0
        legal_indices = [stoi[m] for m in legal_san if m in stoi]
        mask = torch.zeros_like(probs)
        mask[legal_indices] = 1

        probs = probs * mask

        # Check if sum is 0, if so make a random move - this shouldnt really call 
        if probs.sum() == 0:
            san_move = legal_san[torch.randint(len(legal_san), (1,)).item()]
            board.push_san(san_move)
            return san_move
        
        probs = probs / probs.sum()
        
        move_idx = torch.multinomial(probs, num_samples=1).item()
        san_move = itos[move_idx]
        board.push_san(san_move)
        return san_move

    def make_stock_move(board):
        uci_moves = [move.uci() for move in board.move_stack]
        stockfish.make_moves_from_start(uci_moves)
        uci_move = stockfish.get_best_move()
        board.push_uci(uci_move)
      
    # run the game 
    while True:
        make_stock_move(board)
        total_moves += 1 
        show_board(board, display_game, size = size, delay=delay)

        if board.is_game_over():
            break

        make_GPT_move(board)
        total_moves += 1 
        show_board(board, display_game, size = size, delay=delay)

        if board.is_game_over():
            break

    outcome = board.outcome()
    show_board(board, display_game, size = size, delay=delay)

    if outcome.winner is None: 
        winner = "draw"
    elif outcome.winner: 
        winner = 'stock'
    else:
        winner = 'model'

    output = {
        'illegal_moves' : illegal_move_count,
        'illegal_rate' : illegal_move_count/total_moves, 
        'stock_lvl' : stock_lvl,
        'winner' : winner,
        'game_length' : total_moves
    }
    return output 

# ----- Testing before training run  
if __name__ == '__main__': 
    from config import model_params, training_params
    print(
        f"""
        Training Config:
        device:     {device}
        model:      {model_params.save_name}
        batch_size: {training_params.batch_size}
        max_iters:  {training_params.max_iters}
        block_size : {model_params.block_size}
        """
    )    
    test_load_model_GPT(model_params=model_params)
    test_load_stockfish()
    test_block_size_to_model_align(model_params=model_params)



