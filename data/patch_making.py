import json
import os
import pickle
import time
from functools import partial
from multiprocessing import Pool
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.data import fetch_data_paths, check_data_match, get_dataset_base_path
from data.patch_samplers import MixedSampler, UniformSampler, ObjectSampler, DensitySampler, PatchSampler
from utils.files import make_if_not_exist, NumpyEncoder
from utils.images import extract_patch


def make_patch_dataset(new_dataset: str, source_dataset: str, config: Dict, rng: np.random.Generator,
                       make_val=False,
                       sampling_densities=None, d_sampler_weight=None, densities_rescale_fac=1):
    base_data_path = get_dataset_base_path()

    make_if_not_exist(os.path.join(base_data_path, new_dataset))

    n_patches = config["data_loader"]['patch_maker_params']['n_patches']
    patch_size = config["data_loader"]['patch_maker_params']['patch_size']
    sigma = config["data_loader"]['patch_maker_params']['obj_sampler_sigma']
    sigma = 0 if sigma is None else sigma
    for subset in (['train', 'val'] if make_val else ['train']):
        print(f"(re)generating {subset} patches data")

        sampler = MixedSampler(
            n_patches=n_patches,
            samplers=[
                UniformSampler(n_patches=n_patches, patch_size=patch_size, rng=rng),
                ObjectSampler(n_patches=n_patches, patch_size=patch_size, rng=rng, sigma=sigma)
            ],
            weights=[
                config["data_loader"]['patch_maker_params']['unf_sampler_weight'],
                config["data_loader"]['patch_maker_params']['obj_sampler_weight']
            ],
            rng=rng
        )

        if sampling_densities is not None:
            d_sampler = DensitySampler(
                n_patches=n_patches, patch_size=patch_size, rng=rng,
                density_files=sampling_densities, rescale_fac=densities_rescale_fac)
            sampler.add_sampler(d_sampler, d_sampler_weight)

        _make_patches(
            source_dataset=source_dataset,
            subset=subset,
            new_dataset=new_dataset,
            sampler=sampler,
            n_patches=n_patches if subset == 'train' else n_patches // 2,
            patch_size=patch_size,
            rng=rng,
            multiprocess=True,
            clear=True,
            verbose=0
        )


def _make_patches(source_dataset: str, subset: str, new_dataset: str, sampler: PatchSampler, n_patches: int,
                  patch_size: int,
                  rng: np.random.Generator, multiprocess=True, clear=False, verbose=1):
    paths = fetch_data_paths(source_dataset, subset)
    sampler.initialise(paths['images'], paths['annotations'], paths['metadata'])

    samples_per_image = rng.multinomial(n=n_patches, pvals=sampler.sample_density_per_image)

    new_dataset_path = os.path.join(get_dataset_base_path(), new_dataset, subset)
    make_if_not_exist(new_dataset_path, recursive=True)
    make_if_not_exist([os.path.join(new_dataset_path, d) for d in ['images', 'annotations', 'metadata']])

    if len(os.listdir(new_dataset_path)) > 0:
        if verbose > 0:
            print(f"{new_dataset_path} contains items")
        if clear:
            if verbose > 0:
                print("Clearing")
            for d in os.listdir(new_dataset_path):
                for f in os.listdir(os.path.join(new_dataset_path, d)):
                    os.remove(os.path.join(new_dataset_path, d, f))
        else:
            raise FileExistsError

    if verbose > 0:
        print("Making patches !")
    start = time.time()
    fun = partial(_make_one_patch, sampler=sampler, patch_size=patch_size, data_dest=new_dataset_path)
    it = zip(range(len(paths['images'])), samples_per_image, paths['images'], paths['annotations'], paths['metadata'])
    if multiprocess:
        with Pool() as p:
            p.starmap(fun, it)
    else:
        [fun(*arg) for arg in it]
    if verbose > 0:
        print(f'done ({time.time() - start:.2f}s)')


def _make_one_patch(i, n_local_patches, patch_path, label_path, meta_path, sampler, patch_size, data_dest):
    if n_local_patches == 0:
        return

    image = np.asarray(Image.open(patch_path)) / 255

    image_id = check_data_match([patch_path, label_path, meta_path])

    with open(label_path, 'rb') as f:
        labels_dict = pickle.load(f)
    centers = labels_dict['centers']
    params = labels_dict['parameters']
    cats = labels_dict['categories']
    difficulty = labels_dict['difficult']

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    shape = np.array(image.shape[:2])

    for k in range(n_local_patches):

        anchor = sampler.sample_patch_center(image_id=i, shape=shape, centers=centers)
        patch, tl_anchor, centers_offset = extract_patch(
            image=image, center_anchor=anchor, patch_size=patch_size
        )
        patch_centers = []
        patch_parameters = []
        patch_categories = []
        patch_difficulty = []
        for j, c in enumerate(centers):
            offset_c = c + centers_offset
            if np.all(tl_anchor <= offset_c) and np.all(
                    offset_c < (tl_anchor + patch_size)):  # check if object in image
                patch_centers.append(c - tl_anchor + centers_offset)
                patch_parameters.append(params[j])
                patch_categories.append(cats[j])
                patch_difficulty.append(difficulty[j])

        if len(patch_centers) == 0:
            patch_centers = np.array([])
            patch_parameters = np.array([])
            patch_categories = np.array([])
            patch_difficulty = np.array([])
        else:
            patch_centers = np.stack(patch_centers, axis=0)
            patch_parameters = np.stack(patch_parameters, axis=0)
            patch_categories = np.array(patch_categories)
            patch_difficulty = np.array(patch_difficulty)

        patch_name = f"{image_id:04}_{k:04}"

        plt.imsave(os.path.join(data_dest, 'images', f"{patch_name}.png"), patch)
        with open(os.path.join(data_dest, 'annotations', f'{patch_name}.pkl'), 'wb') as f:
            pickle.dump(
                {'centers': patch_centers, 'parameters': patch_parameters, 'categories': patch_categories,
                 'difficult': patch_difficulty},
                f)
        with open(os.path.join(data_dest, 'metadata', f"{patch_name}.json"), 'w') as f:
            json.dump({
                **meta,
                "source": os.path.split(patch_path)[1],
                "anchor": anchor
            }, f, cls=NumpyEncoder, indent=1)
