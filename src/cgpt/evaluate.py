# ------ Libraries
import json 
from pathlib import Path
import time 
from stockfish import Stockfish

import chess 
import chess.svg
import torch 
from .model_base import GPT

# ----------- Setup 

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
        encoder,
        decoder,
        device, 
        stock_lvl : int = 1, 
        display_game : bool = False, 
        delay : float= .5,
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
        
        enc_game = [encoder[t] for t in san_moves]
        idx = torch.tensor([enc_game]).to(device)
        
        # try the top move first
        move_idx = model.generate_single_step(idx)
        san_move = decoder[move_idx]
        
        legal_san = [board.san(m) for m in board.legal_moves]
        
        if san_move in legal_san:
            board.push_san(san_move)
            return san_move
        
        # illegal move -- mask and resample
        illegal_move_count += 1
        probs = model.generate_probs(idx)[0]
        
        # build mask: set all illegal moves to 0
        legal_indices = [encoder[m] for m in legal_san if m in encoder.keys()]
        mask = torch.zeros_like(probs)
        mask[legal_indices] = 1

        probs = probs * mask

        # Check if sum is 0, if so make a random move - this shouldnt really call 
        if probs.sum() < 1e-9:
            san_move = legal_san[torch.randint(len(legal_san), (1,)).item()]
            board.push_san(san_move)
            return san_move
        
        probs = probs / probs.sum()
        
        move_idx = torch.multinomial(probs, num_samples=1).item()
        san_move = decoder[move_idx]
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
