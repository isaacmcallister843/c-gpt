import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from . import storage
from . import datasets

class Trainer: 
    def __init__(
            self, 
            model : nn.Module,
            optimzier : torch.optim.Optimizer, 
            train_loader : DataLoader, 
            val_loader : DataLoader, 
            storage_manager : storage.ModelStorage, 
            learning_rate : float, 
            max_iters : int, 
            save_event_inter : int, 
            monitor_event_inter : int, 
            loss_eval_iter : int, 
            device : str, 
            callbacks 
    ): 
        self.model= model
        self.optimizer = optimzier
        self.train_iter = datasets.infinite_loader(train_loader)
        self.val_iter = datasets.infinite_loader(val_loader)
        self.storage_manager = storage_manager
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.save_event_inter = save_event_inter
        self.loss_eval_iter = loss_eval_iter
        self.monitor_event_inter = monitor_event_inter
        self.device = device 
        self.callbacks = callbacks 
        self.start_iter = 0 
        self.cur_step = 0 
        self.cur_losses = None 


    @torch.no_grad()
    def estimate_loss(self) -> dict:
        out = {}
        for name, loader in [('train', self.train_iter), ('val', self.val_iter)]:
            losses = torch.zeros(self.loss_eval_iter)
            for k in range(self.loss_eval_iter):
                X, Y = next(loader)
                X, Y = X.to(self.device), Y.to(self.device)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[name] = losses.mean()
        self.cur_losses = out 
    
    def train(self) -> None:
        for step in range(self.start_iter, self.max_iters+1):
            self.cur_step = step 

            if (step % self.save_event_inter == 0) and (step != 0): 
                self.estimate_loss()
                self.model.eval()
                for func in self.callbacks: func.on_save(self)
                self.model.train()
            elif step % self.monitor_event_inter == 0:
                self.estimate_loss()
                self.model.eval()
                for func in self.callbacks: func.on_monitor(self)
                self.model.train() 

            xb, yb = next(self.train_iter)
            xb, yb = xb.to(self.device), yb.to(self.device)

            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
    
    @classmethod
    def from_config(cls, model, optimzier, callbacks, train_loader, val_loader, storage_manager, config): 
        return cls(
            model = model,
            optimzier = optimzier, 
            callbacks = callbacks, 
            train_loader = train_loader, 
            val_loader = val_loader, 
            storage_manager = storage_manager, 
            learning_rate = config['training']['learning_rate'],
            max_iters = config['training']['max_iters'], 
            save_event_inter = config['training']['save_event_inter'],
            loss_eval_iter = config['training']['loss_eval_iter'],
            monitor_event_inter = config['training']['monitor_event_inter'],
            device= config['training']['device']
        )

