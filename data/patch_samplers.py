import json
import os
import re
from abc import abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from utils.sampler2d import sample_point_2d


class PatchSampler:

    @abstractmethod
    def initialise(self, patch_files, label_files, meta_files):
        pass

    @property
    @abstractmethod
    def sample_density_per_image(self):
        pass

    @abstractmethod
    def sample_image(self) -> int:
        pass

    @abstractmethod
    def sample_patch_center(self, image_id, shape, centers):
        pass

    @abstractmethod
    def __len__(self):
        pass


@dataclass
class UniformSampler(PatchSampler):
    n_patches: int
    patch_size: int
    rng: np.random.Generator
    sample_density_per_image = None
    n_images = None

    def initialise(self, patch_files, label_files, meta_files):
        pixel_count = []
        self.n_images = len(meta_files)
        assert self.n_images <= self.n_patches
        for mf in meta_files:
            with open(mf, 'r') as f:
                meta = json.load(f)
            shape = meta['shape']
            pixel_count.append(shape[0] * shape[1])
        pixel_count = np.array(pixel_count)

        samples_per_image = (pixel_count / np.sum(pixel_count)) * (self.n_patches - self.n_images) + 1
        # so that every image has at least one sample
        self.sample_density_per_image = samples_per_image / np.sum(samples_per_image)

    def sample_image(self) -> int:
        return self.rng.choice(np.arange(self.n_images), p=self.sample_density_per_image)

    def sample_patch_center(self, image_id, shape, centers):
        anchor = self.rng.integers((0, 0), shape)
        return anchor

    def __len__(self):
        return self.n_patches


@dataclass
class ObjectSampler(PatchSampler):
    n_patches: int
    patch_size: int
    rng: np.random.Generator
    sigma: float = 0
    sample_density_per_image = None
    n_images = None

    def initialise(self, patch_files, label_files, meta_files):
        vehicles_count = []
        self.n_images = len(patch_files)
        for mf in meta_files:
            with open(mf, 'r') as f:
                meta = json.load(f)
            vehicles_count.append(meta['n_objects'])
        vehicles_count = np.array(vehicles_count)
        samples_per_image = (vehicles_count / np.sum(vehicles_count)) * (self.n_patches - self.n_images) + 1
        self.sample_density_per_image = samples_per_image / np.sum(samples_per_image)

    def sample_image(self) -> int:
        return self.rng.choice(np.arange(self.n_images), p=self.sample_density_per_image)

    def sample_patch_center(self, image_id, shape, centers):
        if len(centers) > 0:
            anchor = self.rng.choice(centers, axis=0).astype(int)
            if self.sigma != 0:
                anchor = self.rng.normal(anchor, self.sigma).astype(int)
            anchor = np.clip(anchor, (0, 0), shape)
        else:
            # default to uniform
            anchor = self.rng.integers((0, 0), shape)
        return anchor

    def __len__(self):
        return self.n_patches


@dataclass
class DensitySampler(PatchSampler):
    n_patches: int
    patch_size: int
    rng: np.random.Generator
    density_files: List[str]
    density_maps = None
    sample_density_per_image = None
    n_images = None
    rescale_fac: float = 1

    def __post_init__(self):
        self.density_files.sort()

    def initialise(self, patch_files, label_files, meta_files):
        self.n_images = len(patch_files)
        assert len(self.density_files) == len(patch_files)
        summed_density_per_image = []
        for df in self.density_files:
            density = plt.imread(df)[..., 0]
            summed_density_per_image.append(np.sum(density))
        summed_density_per_image = np.array(summed_density_per_image)
        self.sample_density_per_image = summed_density_per_image / np.sum(summed_density_per_image)
        id_re = re.compile(r'[^0-9]*([0-9]+).*.png')
        for df, pf in zip(self.density_files, patch_files):
            id_df = id_re.match(os.path.split(df)[1]).group(1)
            id_pf = id_re.match(os.path.split(pf)[1]).group(1)
            assert id_df == id_pf

    def sample_image(self) -> int:
        return self.rng.choice(np.arange(self.n_images), p=self.sample_density_per_image)

    def sample_patch_center(self, image_id, shape, centers):
        density = plt.imread(self.density_files[image_id])[..., 0]
        if self.rescale_fac == 1.0:
            assert np.all(shape[:2] == density.shape)
        if np.max(density) == 0:
            center = self.rng.integers((0, 0), shape)
        else:
            center = sample_point_2d(
                img_shape=density.shape[:2],
                density=density,
                skip_normalization=False,
                rng=self.rng,
            ).squeeze()
        center = (center / self.rescale_fac).astype(int)
        anchor = np.clip(center, (0, 0), shape)
        return anchor

    def __len__(self):
        return self.n_patches


@dataclass
class MixedSampler(PatchSampler):
    n_patches: int
    samplers: List[PatchSampler]
    weights: List[float]
    rng: np.random.Generator
    sample_density_per_image = None
    n_images = None

    def __post_init__(self):
        self.weights = np.array(self.weights) / np.sum(np.array(self.weights))

    def add_sampler(self, sampler: PatchSampler, weight: float):
        self.samplers.append(sampler)
        self.weights = [w * (1 - weight) for w in self.weights]
        self.weights.append(weight)
        self.__post_init__()

    def initialise(self, patch_files, label_files, meta_files):
        self.n_images = len(patch_files)
        for s in self.samplers:
            s.initialise(patch_files, label_files, meta_files)
        weighted_samplers_densities = [w * s.sample_density_per_image for s, w in zip(self.samplers, self.weights)]
        self.sample_density_per_image = np.sum(weighted_samplers_densities, axis=0)
        # just in case
        self.sample_density_per_image = self.sample_density_per_image / np.sum(self.sample_density_per_image)

    def sample_image(self) -> int:
        return self.rng.choice(np.arange(self.n_images), p=self.sample_density_per_image)

    def sample_patch_center(self, image_id, shape, centers):
        # choose a sampler at random
        sampler: PatchSampler = self.rng.choice(self.samplers, p=self.weights)
        return sampler.sample_patch_center(image_id=image_id, shape=shape, centers=centers)

    def __len__(self):
        return self.n_patches

# @dataclass
# class CoveringSampler(PatchSampler):
#     patch_size: int
#     sample_density_per_image = None
#     n_images = None
#
#     def initialise(self, patch_files, label_files, meta_files):
#         shapes = []
#         self.n_patches_per_image = []
#         for mf in meta_files:
#             with open(mf, 'r') as f:
#                 meta = json.load(f)
#
#             shape = meta['shape'][:2]
#             shapes.append(shape)
#
#             nx = shape[0] // self.patch_size
#             ny = shape[1] // self.patch_size
#             anchors_x = np.linspace(0, shape[0] - self.patch_size, max(1, nx - 1), dtype=int)
#             anchors_y = np.linspace(0, shape[1] - self.patch_size, max(1, ny - 1), dtype=int)
#
#             n_patches = len(anchors_x) * len(anchors_y)
#             self.n_patches_per_image
#
#     def sample_image(self) -> int:
#         pass
#
#     def sample_patch_center(self, image_id, shape, centers):
#         pass
#
#     def __len__(self):
#         pass
