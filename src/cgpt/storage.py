# src/cgpt/storage.py
from abc import ABC, abstractmethod
from pathlib import Path
import torch
import os 
import json 
import pandas as pd
import logging

class ModelStorage(ABC): 
    @abstractmethod
    def save_checkpoint(self, save_dict: dict, optim_dict: dict, save_name: str, step: int) -> None: 
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
        from cgpt.paths import DATA_DIR, MODEL_DIR
        return cls(
            model_dir = MODEL_DIR / config['model']['save_name'],
            data_dir = DATA_DIR / config['data']['dataset_name']
        )

    def __init__(self, model_dir: Path, data_dir: Path):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.data_dir = data_dir 
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("_LocalStorage_")
        self.logger.info("Initalized local storage")

    def save_checkpoint(self, save_dict, optim_dict, save_name, step : int):
        torch.save(
            {
                'model_state_dict': save_dict,
                'optimizer_state_dict': optim_dict,
                'step': step,
            }, 
            self.model_dir / f"{save_name}_{str(step)}.pt"
        )
    
    def load_checkpoint(self, load_name):
        # load_name ends with .pt 
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
            encoder = json.load(f)

        with open(self.data_dir / "itos.json", "r") as f:
            decoder = {int(k): v for k, v in json.load(f).items()}

        x = torch.load(self.data_dir / "x.pt")
        y = torch.load(self.data_dir / "y.pt")

        return encoder, decoder, x, y 


class CloudStorage(ModelStorage): 
    def __init__(self, bucket_name: str, model_dir: str, data_dir: str):
        from google.cloud import storage
        self.bucket = storage.Client().bucket(bucket_name)
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.logger = logging.getLogger("_GCSStorage_")
        self.logger.info("Initalized local storage")
    
    @classmethod 
    def from_config(cls, config): 
        return cls(
            bucket_name=config['cloud']['bucket_name'],
            model_dir=config['model']['save_name'],
            data_dir=config['data']['dataset_name']
        )

    def _upload(self, blob_path: str, local_path: str) -> None:
        self.bucket.blob(blob_path).upload_from_filename(local_path)

    def _download(self, blob_path: str, local_path: str) -> None:
        self.bucket.blob(blob_path).download_to_filename(local_path)

    def _blob_exists(self, blob_path: str) -> bool:
        return self.bucket.blob(blob_path).exists()

    def save_checkpoint(self, save_dict, optim_dict, save_name, step : int):
        local_path = f"/tmp/{save_name}_{str(step)}.pt"
        torch.save(
            {
                'model_state_dict': save_dict,
                'optimizer_state_dict': optim_dict,
                'step': step,
            }, 
            local_path
        )
        self._upload(f"{self.model_dir}/{save_name}_{str(step)}.pt", local_path)
        os.remove(local_path)
    
    def load_checkpoint(self, load_name):
        local_path = f"/tmp/{load_name}"
        self._download(f"{self.model_dir}/{load_name}", local_path)
        checkpoint = torch.load(local_path, map_location="cpu")
        os.remove(local_path)
        return checkpoint
    
    def list_checkpoints(self) -> list[str]:
        all_files = [x.name for x in list(self.bucket.list_blobs(prefix=f"{self.model_dir}/"))]
        sorted_paths = sorted(
            [os.path.basename(f) for f in all_files if f.suffix == '.pt'],
            key=lambda f: int(f.stem.split('_')[-1])
        )
        return sorted_paths

    def save_results(self, data: pd.DataFrame) -> None:
        blob_path = f"{self.model_dir}/results.csv"
        local_path = "/tmp/results.csv"

        if self._blob_exists(blob_path):
            self._download(blob_path, local_path)
            existing = pd.read_csv(local_path)
            data = pd.concat([existing, data], ignore_index=True)

        data.to_csv(local_path, index=False)
        self._upload(blob_path, local_path)
        os.remove(local_path)

    def save_dataset(self, encoder: dict, decoder: dict, x: torch.Tensor, y: torch.Tensor):
        # Save locally to /tmp, upload, clean up
        tmp_files = {
            "stoi.json": lambda p: json.dump(encoder, open(p, "w"), indent=4),
            "itos.json": lambda p: json.dump(decoder, open(p, "w"), indent=4),
            "x.pt": lambda p: torch.save(x, p),
            "y.pt": lambda p: torch.save(y, p),
        }

        for filename, save_fn in tmp_files.items():
            local_path = f"/tmp/{filename}"
            save_fn(local_path)
            self._upload(f"{self.data_dir}/{filename}", local_path)
            os.remove(local_path)

    def load_dataset(self):
        filenames = ["stoi.json", "itos.json", "x.pt", "y.pt"]
        for filename in filenames:
            self._download(f"{self.data_dir}/{filename}", f"/tmp/{filename}")

        with open("/tmp/stoi.json", "r") as f:
            encoder = json.load(f)

        with open("/tmp/itos.json", "r") as f:
            decoder = {int(k): v for k, v in json.load(f).items()}

        x = torch.load("/tmp/x.pt", map_location="cpu")
        y = torch.load("/tmp/y.pt", map_location="cpu")

        for filename in filenames:
            os.remove(f"/tmp/{filename}")

        return encoder, decoder, x, y 