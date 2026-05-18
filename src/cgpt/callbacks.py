# Lazy type hints 
from __future__ import annotations
from typing import List 
import logging 
from cgpt.evaluate import play_game_test
import pandas as pd 
from cgpt.trainer import Trainer

class TrainerCallbacks: 
    def on_save(self, trainer : Trainer): 
        pass 

    def on_monitor(self, trainer : Trainer):
        pass 

class EstimateLossCallback(TrainerCallbacks): 
    def __init__(self): 
        self.logger = logging.getLogger(__name__ + "_EstimateLossCallback_")

    def on_monitor(self, trainer : Trainer): 
        self.logger.info("Step %s: Train Loss = %s, Val Loss = %s", trainer.cur_step, trainer.cur_losses['train'], trainer.cur_losses['val'])
    
    def on_save(self, trainer : Trainer): 
        self.on_monitor(trainer)


class SaveCheckPointCallback(TrainerCallbacks): 
    def __init__(self): 
        self.logger = logging.getLogger(__name__ + "_SaveCheckpointCallback_")

    def on_save(self, trainer : Trainer): 
        trainer.storage_manager.save_checkpoint(
            save_dict = trainer.model.state_dict(), 
            optim_dict = trainer.optimizer.state_dict(), 
            step = trainer.cur_step
        )
        self.logger.info("Saved checkpoint at step %s", trainer.cur_step)


class PlayGameStockCallback(TrainerCallbacks): 
    def __init__(
            self, 
            stockfish_path : int, 
            eval_num_games : int, 
            eval_lvls: List[int], 
            stoi : dict, 
            itos : dict,         
        ): 
        from stockfish import Stockfish
        self.stockfish = Stockfish(path = stockfish_path)
        self.eval_num_games = eval_num_games
        self.eval_lvls = eval_lvls
        self.stoi = stoi 
        self.itos = itos 
        self.logger = logging.getLogger(__name__ + "_PlayGameCallback_")

    @classmethod
    def from_config(cls, config, stoi, itos): 
        return cls(
            stockfish_path = config['evaluation']['stockfish_path'],
            eval_num_games = config['evaluation']['eval_num_games'],
            eval_lvls = config['evaluation']['eval_lvls'],
            stoi = stoi, 
            itos = itos
        )

    def on_save(self, trainer : Trainer): 
        self.logger.info("Starting game evaluation")
        test_results = []
        for _ in range(self.eval_num_games): 
            for lvl in self.eval_lvls:
                output_dict = play_game_test(
                    model = trainer.model, 
                    stock_lvl = lvl, 
                    stockfish = self.stockfish,
                    stoi = self.stoi, 
                    itos = self.itos, 
                    device = trainer.device
                )

                output_dict['step'] = trainer.cur_step 
                losses = trainer.cur_losses
                output_dict['train_loss'] = losses['train'].item()
                output_dict['val_loss'] = losses['val'].item()
                test_results.append(output_dict)

        for result in [test_results[0], test_results[-1]]:
            self.logger.info(
                f"lvl={result['stock_lvl']} winner={result['winner']} "
                f"illegal_rate={result['illegal_rate']:.3f} length={result['game_length']}"
            )
        df_test = pd.DataFrame(test_results)
        trainer.storage_manager.save_results(data = df_test)
