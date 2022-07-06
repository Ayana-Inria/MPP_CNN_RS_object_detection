import glob
import os
import re
from abc import ABC, abstractmethod

import torch
from torch.nn import Module


class BaseModel(ABC):
    save_path: str

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def infer(self, subset: str, min_confidence: float, display_min_confidence: float, overwrite: bool):
        raise NotImplementedError

    @abstractmethod
    def eval(self):
        raise NotImplementedError

    @abstractmethod
    def data_preview(self):
        raise NotImplementedError


class TorchModel(ABC):
    save_path: str
    model: Module
    n_epochs: int

    def _load(self):
        model_path = os.path.join(self.save_path, 'model.pt')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.last_epoch = self.n_epochs
        else:
            print('no model.pt found, falling back to checkpoints')
            checkpoint = glob.glob(os.path.join(self.save_path, 'checkpoint_*.pt'))
            if len(checkpoint) > 0:
                checkpoint = checkpoint[-1]
                checkpoint_id = int(re.match(r'checkpoint_([0-9]+).pt', os.path.split(checkpoint)[1]).group(1))
                self.model.load_state_dict(torch.load(checkpoint))
                self.last_epoch = checkpoint_id
            else:
                raise FileNotFoundError
