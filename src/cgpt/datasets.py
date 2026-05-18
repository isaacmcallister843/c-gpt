import torch 
from torch.utils.data import DataLoader

class ChessDataset(torch.utils.data.Dataset): 
    def __init__(self, x : torch.Tensor, y  : torch.Tensor, block_size : int): 
        assert block_size <= x.shape[0],  f"block_size {block_size} > data width {x.shape[1]}"
        self.x = x 
        self.y = y 
        self.block_size = block_size


    def __len__(self): 
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx, :self.block_size], self.y[idx, :self.block_size]

def infinite_loader(loader): 
    while True:
        for batch in loader: 
            yield batch 

def create_loader_generator(x, y, block_size, batch_size):
    return infinite_loader(
        DataLoader(
            ChessDataset(
                x = x, 
                y = y, 
                block_size=block_size
            ), 
            batch_size = batch_size, 
            shuffle=True
        )
    )
