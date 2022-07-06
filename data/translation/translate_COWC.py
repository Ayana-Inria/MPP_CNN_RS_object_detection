import glob
import json
import os
import pickle
import re
import shutil
from multiprocessing import Pool
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.transform import rescale
from tqdm import tqdm

from utils.data import get_dataset_base_path
from utils.files import find_existing_path, make_if_not_exist, NumpyEncoder


def fetch_cowc_paths(data_path):
    png_files = glob.glob(os.path.join(data_path, '*/*.png'))

    p = re.compile(r'(.*)_Annotated_Cars.png')
    annotations = [s for s in png_files if p.match(s)]
    annotations.sort()
    p = re.compile(r'(.*)((?:_Annotated_Cars)|(?:_Annotated_Negatives)).png')
    images = [s for s in png_files if not p.match(s)]
    images.sort()

    dataset_df = pd.DataFrame(images, columns=['images'])
    dataset_df['annotations'] = annotations
    dataset_df['gsd'] = 0.15
    dataset_df['id'] = range(len(dataset_df))

    return dataset_df


def count_objects(annotation_file):
    annot = plt.imread(annotation_file)
    pos = np.any(annot > 0, axis=-1)
    num = np.sum(pos)
    return num


def prepare_one_image(image_id, path_image, path_label, save_folder, scale, n_objects, info):
    image = plt.imread(path_image)[..., :3]
    annotations = plt.imread(path_label)
    centers = np.array(np.where(np.any(annotations > 0, axis=-1))).T

    shutil.copy(path_image, os.path.join(save_folder, 'raw_images', f'{image_id:04}.png'))

    image = rescale(image, scale=scale, anti_aliasing=True, multichannel=True)
    centers = centers * scale
    centers = centers.astype(int)

    parameters = np.array([[4.0, 4.0, 0.0] for c in centers])
    categories = np.array(['vehicle' for _ in centers])
    difficult = np.zeros(len(centers))

    if len(centers) == 0:
        centers = np.array([])
        parameters = np.array([])

    assert image.shape[2] == 3
    if np.any(image < 0) or np.any(image > 1):
        print(f"image {path_image} has values outside range [0,1] : clipping !")
        image = np.clip(image, 0, 1)

    plt.imsave(os.path.join(save_folder, 'images', f"{image_id:04}.png"), image)
    with open(os.path.join(save_folder, 'annotations', f"{image_id:04}.pkl"), 'wb') as f:
        pickle.dump(
            {'centers': centers, 'parameters': parameters, 'categories': categories, 'difficult': difficult},
            f)
    with open(os.path.join(save_folder, 'metadata', f"{image_id:04}.json"), 'w') as f:
        json.dump({
            'shape': list(image.shape),
            'n_objects': n_objects,
            'scale': scale,
            **info
        }, f, cls=NumpyEncoder, indent=1)


def make_dataset(data_path: str, save_dir: str, target_gsd: float, prune_empty: bool, drop_rate: float, rng_seed: int):
    dataset_df = fetch_cowc_paths(data_path)

    n_images_prev = len(dataset_df)
    print(f"found {n_images_prev} images")
    # dataset_paths = dataset_paths[dataset_paths.id.isin(whitelist)]

    with Pool() as p:
        objects_per_image = p.map(count_objects, dataset_df['annotations'])
    objects_per_image = np.array(objects_per_image)

    dataset_df['scale'] = dataset_df.gsd / target_gsd

    dataset_df['n_objects'] = objects_per_image
    sample_density = objects_per_image / np.sum(objects_per_image)
    dataset_df['sample_density'] = sample_density
    if prune_empty:
        dataset_df = dataset_df[dataset_df['n_objects'] > 0]

    n_images = len(dataset_df)
    print(
        f'{"" if prune_empty else "not "}pruning images with no objects gives a total of '
        f'{n_images} images ({n_images / n_images_prev:.2%})'
    )

    rng = np.random.default_rng(rng_seed)
    # tentative de drop
    if drop_rate > 0:
        n_image_init = len(dataset_df)
        assert drop_rate < 1.0
        target_n_image = int(n_image_init * (1 - drop_rate))
        kept_images_index = rng.choice(range(n_image_init), size=target_n_image, replace=False)
        kept_images_index.sort()
        dataset_df = dataset_df.iloc[kept_images_index]

        print(
            f'dropping {drop_rate:.2%} images gives a total of '
            f'{len(dataset_df)} images ({len(dataset_df) / n_image_init:.2%})'
        )

    dataset_df.to_pickle(os.path.join(save_dir, 'df_paths_and_meta.pkl'))

    def prepare(row):
        prepare_one_image(
            image_id=row['id'],
            path_image=row['images'],
            path_label=row['annotations'],
            save_folder=save_dir,
            scale=row['scale'],
            n_objects=row['n_objects'],
            info={'original_gsd': row['gsd']}
        )

    tqdm.pandas(desc='extracting patches')

    dataset_df.progress_apply(prepare, axis=1)


def translate_cowc(config: Dict[str, Any]):
    cowc_base_path = config["cowc_base_path"]
    name = config["name"]
    target_gsd = config["target_gsd"]
    prune_empty = bool(config["prune_empty"])

    drop_rate = config['drop_rate']

    source_base = find_existing_path(cowc_base_path)

    dest_base = get_dataset_base_path()
    save_dir = os.path.join(dest_base, name)
    make_if_not_exist(save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=1)

    sub_folders = ['raw_images', 'images', 'raw_annotations', 'annotations', 'metadata', 'images_w_annotations']

    subset = 'val'
    subset_save_dir = os.path.join(save_dir, subset)
    make_if_not_exist(subset_save_dir)
    make_if_not_exist([os.path.join(subset_save_dir, s) for s in sub_folders])
    print(f'making {subset} patches')
    make_dataset(
        save_dir=subset_save_dir,
        data_path=source_base,
        target_gsd=target_gsd,
        prune_empty=prune_empty,
        drop_rate=drop_rate,
        rng_seed=0
    )
