# ------ Libraries
import json 
from pathlib import Path
import time 
import chess 
from IPython.display import display, clear_output
import chess.svg

from config import config

import torch 
from stockfish import Stockfish
from .model_base import GPT

# ------ Setup  
ROOT_DIR = config.ROOT
DATA_DIR = config.DATA_DIR
device = config.device
stockfish = Stockfish(path= config.STOCK_FISH_DIR)

with open(DATA_DIR / 'processed' / 'itos.json', 'r') as file:
    itos = json.load(file)
itos = {int(i) : ch for i,ch in itos.items()} 
with open(DATA_DIR / 'processed' / 'stoi.json', 'r') as file:
    stoi = json.load(file)

vocab_size = len(itos)

decode = lambda x : [itos[val] for val in x]
encode = lambda x : [stoi[val] for val in x]

# ----- Tests 
def play_game_test(
        model_path : str, 
        stock_lvl : int = 1, 
        display_game : bool = False, 
        delay : float= .5
    ) -> dict:

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
        move_idx = model.generate(idx, max_new_tokens=1)[0][-1].item()
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
        probs = probs / probs.sum()
        
        move_idx = torch.multinomial(probs, num_samples=1).item()
        san_move = itos[move_idx]
        board.push_san(san_move)

    def make_stock_move(board):
        uci_moves = [move.uci() for move in board.move_stack]
        stockfish.make_moves_from_start(uci_moves)
        uci_move = stockfish.get_best_move()
        board.push_uci(uci_move)
    
    # load in model 
    model = GPT(
        vocab_size=vocab_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        block_size=config.block_size,
        dropout=config.dropout,
        device=config.device,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # set board state 
    board = chess.Board()

    # set variables 
    illegal_move_count = 0
    total_moves = 0
    size=250
    stockfish.set_skill_level(stock_lvl)

    # run the game 
    while True:
        make_stock_move(board)
        total_moves += 1 
        if display_game: 
            clear_output(wait=True)
            display(chess.svg.board(board, size = size))

        if board.is_game_over():
            break

        make_GPT_move(board)
        total_moves += 1 
        if display_game: 
            time.sleep(delay)
            clear_output(wait=True)
            display(chess.svg.board(board,  size = size))

        if board.is_game_over():
            break

    outcome = board.outcome()
    if display_game: 
        print(outcome)
        print(f"Illegal moves: {illegal_move_count}, illegal rate {illegal_move_count/total_moves}")
    
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

