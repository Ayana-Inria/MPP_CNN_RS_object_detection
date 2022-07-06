import pickle as pkl
from abc import abstractmethod
from typing import Union, Tuple, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from data.augmentation import DataAugment
from utils.data import fetch_data_paths


class LabelProcessor:

    @abstractmethod
    def process(self, patch: np.ndarray, centers: np.ndarray, params: np.ndarray, idx: int) \
            -> Tuple[Tensor, Dict[str, Union[Tensor, List[Tensor]]]]:
        pass


class ImageDataset(Dataset):
    """
    loads images and processes labels
    """

    def __init__(self, dataset: str, subset: str,
                 rng: np.random.Generator,
                 rgb: bool,
                 label_processor: LabelProcessor,
                 augmenter: DataAugment = None):
        super().__init__()
        self.dataset = dataset
        self.subset = subset
        self.paths = fetch_data_paths(self.dataset, self.subset, metadata=False)
        self.rng = np.random.default_rng() if rng is None else rng
        self.augmenter = augmenter
        self.rgb = rgb
        self.label_processor = label_processor

    def update_files(self):
        self.paths = fetch_data_paths(self.dataset, self.subset, metadata=False)

    def __getitem__(self, item):
        if self.rgb:
            patch = plt.imread(self.paths['images'][item])[:, :, :3]
        else:
            patch = plt.imread(self.paths['images'][item])[:, :, :[0]]
        assert np.all(patch <= 1.0)

        with open(self.paths['annotations'][item], 'rb') as f:
            labels_dict = pkl.load(f)
        centers, params = labels_dict['centers'], labels_dict['parameters']

        if self.augmenter is not None:
            patch, centers, params, augment_dict = self.augmenter.transform(patch, centers, params)

        return self.label_processor.process(
            patch=patch,
            centers=centers,
            params=params,
            idx=item
        )

    def __len__(self):
        return len(self.paths['images'])
