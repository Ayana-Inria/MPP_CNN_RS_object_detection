import glob
import json
import os
from datetime import datetime
from typing import Dict, List
import numpy as np
import torch
from torch import nn

from utils.files import NumpyEncoder
from utils.misc import append_lists_in_dict


class Logger:
    def __init__(self, save_dir: str, checkpoint_interval: int = None, model: nn.Module = None):
        self.log = dict()
        self.save_dir = save_dir
        self.checkpoint_interval = checkpoint_interval
        self.model = model

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            log = json.load(f)
        loaded_logger = Logger(save_dir=os.path.split(path)[0])
        loaded_logger.log = log
        return loaded_logger

    def clear(self):
        self.log = dict()

    def log_model(self, checkpoint_interval: int = None, model: nn.Module = None):
        self.checkpoint_interval = checkpoint_interval
        self.model = model

    def update_train_val(self, epoch: int, train_metrics: Dict[str, List[float]], val_metrics: Dict[str, List[float]]):
        metrics = {**{'train_' + k: np.mean(v) for k, v in train_metrics.items()},
                   **{'val_' + k: np.mean(v) for k, v in val_metrics.items()}}
        self.update(epoch, metrics=metrics)

    def update(self, epoch: int, metrics: Dict[str, float], prefix=''):

        timestamp_str = datetime.now().strftime("%m/%d/%y-%H:%M:%S")

        append_lists_in_dict(self.log, {'epoch': epoch})
        append_lists_in_dict(self.log, {'timestamp': timestamp_str})
        append_lists_in_dict(self.log, {prefix + k: v for k, v in metrics.items()})

        if self.checkpoint_interval is not None and epoch % self.checkpoint_interval == 0:
            last_checkpoint_path = glob.glob(os.path.join(self.save_dir, 'checkpoint_*.pt'))
            if len(last_checkpoint_path) > 0:
                for p in last_checkpoint_path:
                    os.remove(p)
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'checkpoint_{epoch:04}.pt'))

        with open(os.path.join(self.save_dir, 'log.json'), 'w') as f:
            json.dump(self.log, f, cls=NumpyEncoder, indent=1)
