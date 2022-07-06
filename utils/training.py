import json
import logging
import os
import shutil
from abc import ABC
from typing import Dict, List, Tuple, Any

import numpy as np
from numpy.random import Generator
from torch import Tensor
from torch.utils.data import DataLoader

from data.augmentation import DataAugment
from utils.data import get_model_base_path
from data.image_dataset import ImageDataset, LabelProcessor
from data.patch_making import make_patch_dataset
from utils.files import make_if_not_exist
from utils.logger import Logger

MetricsDict = Dict[str, List[float]]
Config = Dict[str, Any]


def update_metrics(loss_dict: Dict[str, Tensor], metrics: MetricsDict) -> MetricsDict:
    if metrics is None:
        metrics = {k: [v.detach().cpu().numpy()] for k, v in loss_dict.items()}
    else:
        for k in loss_dict:
            metrics[k].append(loss_dict[k].detach().cpu().numpy())
    return metrics


def print_metrics(epoch: int, train_metrics: MetricsDict, val_metrics: MetricsDict):
    print(f"[{epoch:04}] Train ", end='')
    for k, v in train_metrics.items():
        print(f"{k}: {np.mean(v):.3f} ", end='')
    print(f"| Eval ", end='')
    for k, v in val_metrics.items():
        print(f"{k}: {np.mean(v):.3f} ", end='')
    print('')


def startup_config(config: Config, model_type: str, load_model=False, overwrite=False) -> Tuple[Dict, Logger, str]:
    """

    :return: config dict, save path
    """

    # get paths
    base_path_model = get_model_base_path()

    save_path = os.path.join(base_path_model, model_type, config['model_name'])

    if os.path.exists(save_path):
        if not load_model:
            if not overwrite:
                print(f'found model in {save_path}')
                raise FileExistsError
            else:
                print(f'found model in {save_path}, writing over')
                shutil.rmtree(save_path)
                make_if_not_exist(save_path, recursive=True)
    else:
        make_if_not_exist(save_path, recursive=True)

    local_config_file = os.path.join(save_path, 'config.json')
    if not os.path.exists(local_config_file):
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=1)

    log_file = os.path.join(save_path, 'log.json')
    if os.path.exists(log_file) and load_model:
        logger = Logger.load(log_file)
    else:
        logger = Logger(save_dir=save_path)

    logging.basicConfig(format='%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S',
                        level=logging.INFO)

    return config, logger, save_path


class PatchBasedTrainer(ABC):
    temp_dataset: str
    dataset: str
    config: Dict[str, Any]
    rng: Generator
    label_processor_train: LabelProcessor
    label_processor_val: LabelProcessor
    batch_size: int

    def __init_data__(self, reuse_data=False, collate_fn=None):
        if not reuse_data:
            make_patch_dataset(new_dataset=self.temp_dataset,
                               source_dataset=self.dataset,
                               config=self.config,
                               make_val=True,
                               rng=self.rng)
        self.dataset_update_interval = self.config["data_loader"]['dataset_update_interval']

        augmenter = DataAugment(
            rng=self.rng,
            dataset=self.dataset, subset='train',
            **self.config['data_loader'].get('augment_params')
        ) if 'augment_params' in self.config['data_loader'] else None

        self.data_train = ImageDataset(
            dataset=self.temp_dataset,
            subset='train',
            rgb=True,
            rng=self.rng,
            augmenter=augmenter,
            label_processor=self.label_processor_train
        )

        self.data_val = ImageDataset(
            dataset=self.temp_dataset,
            subset='val',
            rgb=True,
            rng=self.rng,
            label_processor=self.label_processor_val
        )

        self.train_loader = DataLoader(self.data_train, batch_size=self.batch_size, num_workers=8, prefetch_factor=16,
                                       collate_fn=collate_fn)
        self.val_loader = DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=8,
                                     prefetch_factor=16, collate_fn=collate_fn)
        self.figure_loader = DataLoader(self.data_val, batch_size=8, shuffle=True, collate_fn=collate_fn)

        self.images_figs, self.label_figs = self.figure_loader.__iter__().next()
