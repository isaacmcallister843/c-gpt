# src/cgpt/storage.py
from abc import ABC, abstractmethod
from pathlib import Path
import torch
import os 
import json 
import pandas as pd
import logging
from cgpt.paths import DATA_DIR, MODEL_DIR
import threading


class ModelStorage(ABC): 
    @abstractmethod
    def save_checkpoint(self, save_dict: dict, optim_dict: dict, step: int) -> None: 
        pass

    @abstractmethod
    def load_checkpoint(self, save_name: str): 
        pass

    @abstractmethod
    def list_checkpoints(self) -> list[str]:
        pass

    @abstractmethod
    def save_results(self, data: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def save_dataset(self, encoder: dict, decoder: dict, x: torch.Tensor, y: torch.Tensor) -> None: 
        pass 

    @abstractmethod
    def load_dataset(self): 
        pass 


class LocalStorage(ModelStorage): 
    @classmethod 
    def from_config(cls, config): 
        return cls(
            model_dir = MODEL_DIR / config['model']['save_name'],
            data_dir = DATA_DIR / config['data']['dataset_name']
        )

    def __init__(self, model_dir: Path, data_dir: Path):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.data_dir = data_dir 
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__ + "_LocalStorage_")
        self.logger.info("Initalized local storage")

    def save_checkpoint(self, save_dict, optim_dict, step : int):
        full_path = self.model_dir / f"checkpoint_{str(step)}.pt"
        torch.save(
            {
                'model_state_dict': save_dict,
                'optimizer_state_dict': optim_dict,
                'step': step,
            }, 
            full_path
        )
        self.logger.info(f"saved checkpoint to {full_path}")
    
    def load_checkpoint(self, load_name):
        # load_name ends with .pt 
        self.logger.info(f"loaded checkpoint {self.model_dir / load_name}")
        return torch.load(self.model_dir / load_name, map_location="cpu")

    def list_checkpoints(self) -> list[str]:
        sorted_paths = sorted(
            [f for f in self.model_dir.iterdir() if f.suffix == '.pt'],
            key=lambda f: int(f.stem.split('_')[-1])
        )
        return [p.name for p in sorted_paths]

    def save_results(self, data: pd.DataFrame) -> None:
        results_path = self.model_dir / "results.csv"
        if results_path.exists():
            existing = pd.read_csv(results_path)
            data = pd.concat([existing, data], ignore_index=True)
        data.to_csv(results_path, index=False)
        self.logger.info("saved results.csv file")

    def save_dataset(self, encoder: dict, decoder: dict, x: torch.Tensor, y: torch.Tensor): 
        self.data_dir.mkdir(parents=True, exist_ok=True)

        with open(self.data_dir / "stoi.json", "w") as f:
            json.dump(encoder, f, indent=4)

        with open(self.data_dir / "itos.json", "w") as f:
            json.dump(decoder, f, indent=4)

        torch.save(x, self.data_dir / "x.pt")
        torch.save(y, self.data_dir / "y.pt") 
        self.logger.info(f"saved dataset at {self.data_dir}")

    def load_dataset(self): 
        with open(self.data_dir / "stoi.json", "r") as f:
            stoi = json.load(f)

        with open(self.data_dir / "itos.json", "r") as f:
            itos = {int(k): v for k, v in json.load(f).items()}

        x = torch.load(self.data_dir / "x.pt")
        y = torch.load(self.data_dir / "y.pt")

        self.logger.info(f"loaded dataset from {self.data_dir}")
        return stoi, itos, x, y 


class CloudStorage(ModelStorage): 
    """
    Manages both local and cloud storage 
    """
    def __init__(self, bucket_name: str, local_model_dir : Path, local_data_dir: Path, cloud_model_dir : str, cloud_data_dir : str):
        from google.cloud import storage
        self.bucket = storage.Client().bucket(bucket_name)
        self.local_model_dir = local_model_dir
        self.local_data_dir = local_data_dir
        self.cloud_model_dir = cloud_model_dir
        self.cloud_data_dir = cloud_data_dir

        # ensure directories exist, if not create them 
        self.local_model_dir.mkdir(parents=True, exist_ok=True)
        self.local_data_dir.mkdir(parents=True, exist_ok=True)

        # logging methods 
        self.logger = logging.getLogger(__name__ + "_GCSStorage_")
        self.logger.info("Initalized cloud storage")
    
    @classmethod 
    def from_config(cls, config): 
        return cls(
            bucket_name = config['cloud']['bucket_name'],
            cloud_model_dir = config['model']['save_name'],
            cloud_data_dir = config['data']['dataset_name'],
            local_model_dir = MODEL_DIR / config['model']['save_name'],
            local_data_dir = DATA_DIR / config['data']['dataset_name'],
        )

    def _upload(self, blob_path: str, local_path: str, keep_local_save = True, wait = False) -> None:
        self.logger.info(f"uploading {local_path} -> {blob_path}")

        # send off upload task to a new thread 
        def _upload_helper(): 
            self.bucket.blob(blob_path).upload_from_filename(local_path)
            self.logger.info(f"upload complete from {local_path} -> {blob_path}")
            if not keep_local_save: 
                os.remove(local_path)
                self.logger.info(f"cleaned up: {local_path}")

        thread = threading.Thread(target = _upload_helper)
        thread.start()
        if wait: 
            thread.join()
        
    def _download(self, blob_path: str, local_path: str) -> None:
        self.logger.info(f"downloading {blob_path} -> {local_path}")
        self.bucket.blob(blob_path).download_to_filename(local_path)

    def _blob_exists(self, blob_path: str) -> bool:
        return self.bucket.blob(blob_path).exists()

    def save_checkpoint(self, save_dict, optim_dict, step : int, keep_local_save : bool = False):
        local_path = self.local_model_dir / f"checkpoint_{str(step)}.pt"
        blob_path = f"{self.cloud_model_dir}/checkpoint_{str(step)}.pt"
        torch.save(
            {
                'model_state_dict': save_dict,
                'optimizer_state_dict': optim_dict,
                'step': step,
            }, 
            local_path
        )
        self._upload(blob_path, local_path, keep_local_save = keep_local_save, wait = False)
    
    def load_checkpoint(self, load_name):
        local_path = self.local_model_dir / load_name
        blob_path = f"{self.cloud_model_dir}/{load_name}"
        self._download(blob_path, local_path)
        checkpoint = torch.load(local_path, map_location="cpu")
        os.remove(local_path)
        self.logger.info(f"loaded checkpoint {blob_path}")
        return checkpoint
    
    def list_checkpoints(self) -> list[str]:
        all_files = [x.name for x in list(self.bucket.list_blobs(prefix=f"{self.cloud_model_dir}/"))]
        sorted_paths = sorted(
            [os.path.basename(f) for f in all_files if f.endswith(".pt")],
            key=lambda f: int(f.replace('.pt', '').split('_')[-1])
        )
        return sorted_paths

    def save_results(self, data: pd.DataFrame) -> None:
        blob_path = f"{self.cloud_model_dir}/results.csv"
        local_path = self.local_data_dir / "results.csv"

        if self._blob_exists(blob_path):
            self._download(blob_path, local_path)
            existing = pd.read_csv(local_path)
            data = pd.concat([existing, data], ignore_index=True)

        data.to_csv(local_path, index=False)
        self._upload(blob_path, local_path, keep_local_save = False)

    def save_dataset(self, encoder: dict, decoder: dict, x: torch.Tensor, y: torch.Tensor, keep_local_save : bool = True):
        # Save locally to /tmp, upload, clean up
        def _save_json(data, path): 
            with open(path, 'w') as f: 
                json.dump(data, f, indent=4)

        tmp_files = {
            "stoi.json": lambda p: _save_json(data = encoder, path = p),
            "itos.json": lambda p: _save_json(data = decoder, path = p),
            "x.pt": lambda p: torch.save(x, p),
            "y.pt": lambda p: torch.save(y, p),
        }
        for filename, save_fn in tmp_files.items():
            local_path = self.local_data_dir / filename
            cloud_path = f"{self.cloud_data_dir}/{filename}"
            save_fn(local_path)
            self._upload(cloud_path, local_path, keep_local_save = keep_local_save, wait=True)
        
        self.logger.info(f"saved dataset at {self.cloud_data_dir}")

    def load_dataset(self, force_redownload : bool = False, keep_local_save = True):
        filenames = ["stoi.json", "itos.json", "x.pt", "y.pt"]
        for filename in filenames:
            local_path = self.local_data_dir / filename 
            blob_path = f"{self.cloud_data_dir}/{filename}"
            if not local_path.is_file() or force_redownload: 
                self._download(blob_path, local_path)

        with open(self.local_data_dir / "stoi.json", "r") as f:
            stoi = json.load(f)

        with open(self.local_data_dir / "itos.json", "r") as f:
            itos = {int(k): v for k, v in json.load(f).items()}

        x = torch.load(self.local_data_dir / "x.pt", map_location="cpu")
        y = torch.load(self.local_data_dir / "y.pt", map_location="cpu")
        self.logger.info(f"loaded dataset from {self.cloud_data_dir}")

        if not keep_local_save: 
            for filename in filenames:
                os.remove(self.local_data_dir / filename)
            self.logger.info("removed local builds")

        return stoi, itos, x, y 