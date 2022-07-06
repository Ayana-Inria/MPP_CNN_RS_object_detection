import pickle as pkl

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from data.augmentation import DataAugment
from utils.data import fetch_data_paths
from data.image_dataset import LabelProcessor
from data.patch_samplers import PatchSampler
from utils.images import extract_patch


class PatchDataset(Dataset):
    """
    extracts patches from whole images, processes labels
    """

    def __init__(self, patch_size: int,
                 dataset: str, subset: str,
                 rng: np.random.Generator,
                 rgb: bool,
                 label_processor: LabelProcessor,
                 patch_sampler: PatchSampler,
                 augmenter: DataAugment = None):
        super().__init__()
        self.label_processor = label_processor
        self.patch_size = patch_size

        self.paths = fetch_data_paths(dataset, subset)

        self.patch_sampler = patch_sampler

        self.patch_sampler.initialise(
            self.paths['images'], self.paths['annotations'], self.paths['metadata'])

        self.rng = np.random.default_rng() if rng is None else rng
        self.augmenter = augmenter
        self.rgb = rgb

    def __getitem__(self, item):

        image_id = self.patch_sampler.sample_image()

        if self.rgb:
            image = plt.imread(self.paths['images'][image_id])[:, :, :3]
        else:
            image = plt.imread(self.paths['images'][image_id])[:, :, :[0]]

        with open(self.paths['annotations'][image_id], 'rb') as f:
            labels_dict = pkl.load(f)
        centers, params = labels_dict['centers'], labels_dict['parameters']

        shape = np.array(image.shape[:2])

        anchor = self.patch_sampler.sample_patch_center(image_id=image_id, shape=shape, centers=centers)
        patch, tl_anchor, centers_offset = extract_patch(
            image=image, center_anchor=anchor, patch_size=self.patch_size
        )

        patch_centers = []
        patch_parameters = []
        for i, c in enumerate(centers):
            offset_c = c + centers_offset
            if np.all(tl_anchor <= offset_c) and np.all(
                    offset_c < (tl_anchor + self.patch_size)):  # check if object in image
                patch_centers.append(c - tl_anchor + centers_offset)
                patch_parameters.append(params[i])

        if len(patch_centers) == 0:
            patch_centers = np.array([])
            patch_parameters = np.array([])

        else:
            patch_centers = np.stack(patch_centers, axis=0)
            patch_parameters = np.stack(patch_parameters, axis=0)

        if self.augmenter is not None:
            patch, patch_centers, patch_parameters, augment_dict = self.augmenter.transform(
                patch, patch_centers, patch_parameters)

        assert patch.shape == (self.patch_size, self.patch_size, 3)

        return self.label_processor.process(
            patch=patch,
            centers=patch_centers,
            params=patch_parameters,
            idx=item
        )

    def __len__(self):
        return len(self.patch_sampler)
