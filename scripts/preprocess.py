# ------ Libraries
import pyarrow.parquet as pq
import numpy as np 
import torch
import sys
from cgpt.storage import LocalStorage, CloudStorage
import tomllib
import logging
import pyarrow.fs as fs

# ----------- Setup 
config_path = sys.argv[1]
with open(config_path, 'rb') as f:
    config = tomllib.load(f)

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == '__main__': 
    # ----- Setup storage 
    save_cloud = config['storage']['save_cloud']
    if save_cloud: 
        gcs = fs.GcsFileSystem()
        storage_manager = CloudStorage.from_config(config)
        path = config['data']['dataset_path'].replace('gs://', '')
        parquet_file = pq.ParquetFile(path, filesystem=gcs)
    else:
        storage_manager = LocalStorage.from_config(config)
        parquet_file = pq.ParquetFile(config['data']['dataset_path'])

    # ----- Setup 
    block_size = config['model']['block_size']
    n_rows = config['data']['n_rows']
    min_elo = config['data']['min_elo']
    chunk_size = 500000

    # ------ create mapping
    stoi = {'<PAD>': 0}
    current_idx_stoi = 1

    with open(config['data']['san_string_path'], 'r') as file:
        for line in file:
            stoi[str(line.strip())] = current_idx_stoi
            current_idx_stoi +=1 

    # get encoder for vectorization 
    def encode(x): 
        return stoi[x] 
    encode_vec = np.vectorize(encode)

    # get decoding map 
    itos = {i: ch for ch, i in stoi.items()}

    # ------ Proccess file 
    chunks = []

    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        df = batch.to_pandas()
        df = df[df['white_elo'] > min_elo]
        moves = df['moves_san'].apply(lambda x: x[0:block_size])
        add = moves.apply(
            lambda x: np.concatenate((x, np.repeat(['<PAD>'], block_size - len(x))))
        )
        stack = np.stack(add.values)
        chunks.append(encode_vec(stack))
        logger.info(f"Processed chunk, {len(chunks)} done")
        if len(chunks) * chunk_size > n_rows: 
            break 

    x = np.concatenate(chunks, axis=0)
    y = np.zeros(x.shape)
    y[:, 0:(block_size-1)] = x[:, 1::] 

    # convert to pytorch objects
    x_torch = torch.from_numpy(x).long()
    y_torch = torch.from_numpy(y).long()

    storage_manager.save_dataset(encoder = stoi, decoder = itos, x = x_torch , y = y_torch)