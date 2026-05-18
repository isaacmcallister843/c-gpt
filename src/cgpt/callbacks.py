# Lazy type hints 
from __future__ import annotations

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
        self.logger = logging.getLogger("EstimateLossCallback")

    def on_monitor(self, trainer : Trainer): 
        self.logger.info("Step %s: Train Loss = %s, Val Loss = %s", trainer.cur_step, trainer.cur_losses['train'], trainer.cur_losses['val'])
    
    def on_save(self, trainer : Trainer): 
        self.on_monitor(trainer)


class SaveCheckPointCallback(TrainerCallbacks): 
    def __init__(self): 
        self.logger = logging.getLogger("SaveCheckpointCallback")

    def on_save(self, trainer : Trainer): 
        trainer.storage_manager.save_checkpoint(
            save_dict = trainer.model.state_dict(), 
            optim_dict = trainer.optimizer.state_dict(), 
            save_name = f"checkpoint_{trainer.cur_step}", 
            step = trainer.cur_step
        )
        self.logger.info("Saved checkpoint at step %s", trainer.cur_step)


class PlayGameStockCallback(TrainerCallbacks): 
    def __init__(
            self, 
            stockfish_path, 
            eval_num_games, 
            eval_lvl_start, 
            eval_lvl_end, 
            eval_lvl_jump,
            encoder, 
            decoder,         
        ): 
        from stockfish import Stockfish
        self.stockfish = Stockfish(path = stockfish_path)
        self.eval_num_games = eval_num_games
        self.eval_lvl_start = eval_lvl_start
        self.eval_lvl_end = eval_lvl_end
        self.eval_lvl_jump = eval_lvl_jump
        self.encoder = encoder 
        self.decoder = decoder 
        self.logger = logging.getLogger("PlayGameCallback")

    @classmethod
    def from_config(cls, config, encoder, decoder): 
        return cls(
            stockfish_path = config['evaluation']['stockfish_path'],
            eval_num_games = config['evaluation']['eval_num_games'],
            eval_lvl_start = config['evaluation']['eval_lvl_start'],
            eval_lvl_end = config['evaluation']['eval_lvl_end'],
            eval_lvl_jump = config['evaluation']['eval_lvl_jump'],
            encoder = encoder, 
            decoder = decoder
        )

    def on_save(self, trainer : Trainer): 
        test_results = []
        for _ in range(self.eval_num_games): 
            for lvl in range(self.eval_lvl_start, self.eval_lvl_end, self.eval_lvl_jump):
                output_dict = play_game_test(
                    model = trainer.model, 
                    stock_lvl = lvl, 
                    stockfish = self.stockfish,
                    encoder = self.encoder, 
                    decoder = self.decoder, 
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
