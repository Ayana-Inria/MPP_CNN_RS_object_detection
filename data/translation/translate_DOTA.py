import glob
import json
import os
import re
import shutil
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from PIL import Image
from skimage.transform import rescale
from tqdm import tqdm
import pandas as pd
from dateutil.parser import ParserError

from base.shapes.rectangle import polygon_to_abw
from utils.files import NumpyEncoder, make_if_not_exist, find_existing_path
from utils.data import get_dataset_base_path

SCALE_ACCEPTABLE_DELTA = 1e-2

ALL_CATEGORIES = ['large-vehicle', 'roundabout', 'plane', 'tennis-court', 'helipad', 'airport', 'small-vehicle',
                  'baseball-diamond', 'harbor', 'bridge', 'swimming-pool', 'storage-tank', 'helicopter',
                  'container-crane', 'soccer-ball-field', 'basketball-court', 'ship', 'ground-track-field']


def fetch_dota_paths(base_path: str, subset: str):
    assert subset in ['train', 'val', 'test']

    images_df = pd.DataFrame(glob.glob(os.path.join(base_path, subset, 'images', 'P*.png')),
                             columns=['path'])

    images_df['id'] = images_df['path'].str.extract(r'P([0-9]+).png').astype(int)

    if subset != 'test':
        labels_df = pd.DataFrame(glob.glob(os.path.join(base_path, subset, f'DOTA-v2.0_{subset}', 'P*.txt')),
                                 columns=['path'])

        labels_df['id'] = labels_df['path'].str.extract(r'P([0-9]+).txt').astype(int)

        # seek meta :
        metas_df = pd.DataFrame(glob.glob(os.path.join(base_path, subset, 'meta', 'P*.txt')),
                                columns=['path_meta'])

        metas_df['id'] = metas_df['path_meta'].str.extract(r'P([0-9]+).txt').astype(int)

        dataset_df = pd.merge(images_df, labels_df, on='id', suffixes=("_image", "_label"))
        dataset_df = pd.merge(dataset_df, metas_df, on='id')

        # extract metadata

        date_parser = re.compile(r'acquisition dates?:([^\n]*)')
        source_parser = re.compile(r'imagesource:([^\n]*)')
        gsd_parser = re.compile(r'gsd:([^\n]*)')

        def _extract(path_meta: str):
            with open(path_meta, 'r') as f:
                text = f.readlines()
                # try:
                date = date_parser.match(text[0]).group(1)
                source = source_parser.match(text[1]).group(1)
                gsd = gsd_parser.match(text[2]).group(1)
                # except AttributeError:
                #     print(f'no match for file {path_meta}')
                #     date, source, gsd = None,None,'None'
            try:
                gsd = float(gsd)
            except ValueError:
                gsd = None
            try:
                date = pd.to_datetime(date)
            except ParserError:
                date = None
            source = None if source == 'None' else source
            return date, source, gsd

        dataset_df['date'], dataset_df['source'], dataset_df['gsd'] = zip(*dataset_df['path_meta'].map(_extract))

        return dataset_df

    else:
        images_df.rename(columns={"path": "path_image"})

        return images_df


def get_content_per_image(dataset_df: pd.DataFrame, categories: List[str] = None):
    """
    counts number of object per category in the dataset
    :param dataset_df:
    :param categories:
    :return:
    """

    if categories is None:
        categories = ALL_CATEGORIES.copy()

    for c in categories:
        assert c in ALL_CATEGORIES

    def _count_objs(path_label):
        labels = parse_label_file(path_label)
        value_counts = labels.category.value_counts()
        for c in categories:
            if c not in value_counts:
                value_counts[c] = 0

        return tuple(value_counts[categories].values)

    tqdm.pandas(desc='browsing contents')

    res = zip(*dataset_df.path_label.progress_map(_count_objs))

    for r, c in zip(res, categories):
        dataset_df[c] = r

    return dataset_df


def show_categories(images_df):
    categories = set()
    for i in tqdm(range(len(images_df)), desc="looking for categories"):
        labels = parse_label_file(images_df.path_label[i])
        categories = categories | set(labels.category.values)
    print('')
    return list(categories)


def parse_label_file(label_file: str):
    labels = pd.read_csv(label_file, sep=' ',
                         names=['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'category', 'difficult'])
    return labels


def extract_image_and_boxes(image_file: str, label_file: str, target_categories: List[str]):
    """
    return the image, boxes and centers for the image filtering with specified categories
    :param image_file: path of image file
    :param label_file: path of label file
    :param target_categories: target categories
    :return: image, array of posisions of boxes corners (shape (N,4,2) ), centers of objects (shape (N,2))
    """
    label = parse_label_file(label_file)
    image = np.asarray(Image.open(image_file)) / 255

    unique_categories = np.unique(label.category)
    at_least_one = False
    for category in target_categories:
        if category in unique_categories:
            at_least_one = True
    if not at_least_one:
        print(f"[warning] {target_categories} not in categories {unique_categories} in image {image_file}")

    label_subset = label[np.isin(label.category, target_categories)]

    all_boxes = np.stack((label_subset[['y1', 'y2', 'y3', 'y4']].values,
                          label_subset[['x1', 'x2', 'x3', 'x4']].values), axis=-1)

    centers = np.mean(all_boxes, axis=1)

    centers = centers.astype(int)

    obj_categories = label_subset.category.to_numpy()
    difficult = label_subset.difficult.to_numpy()

    return image, all_boxes, centers, obj_categories, difficult


def prepare_one_image(image_id: int, path_image: str, path_label: str, target_categories: List[str],
                      save_folder: str,
                      n_objects: int,
                      scale: float, info: dict):
    image, polygons, centers, categories, difficult = extract_image_and_boxes(path_image, path_label, target_categories)

    shutil.copy(path_label, os.path.join(save_folder, 'raw_annotations', f'{image_id:04}.txt'))
    shutil.copy(path_image, os.path.join(save_folder, 'raw_images', f'{image_id:04}.png'))

    if abs(1 - scale) > SCALE_ACCEPTABLE_DELTA:
        assert scale <= 1
        image = rescale(image, scale=scale, anti_aliasing=True, multichannel=True)
        polygons = polygons * scale
        centers = centers * scale
        centers = centers.astype(int)

    parameters = np.array(list(map(polygon_to_abw, polygons)))

    if len(centers) == 0:
        centers = np.array([])
        parameters = np.array([])
        categories = np.array([])
        difficult = np.array([])

    assert image.shape[2] == 3
    if np.any(image < 0) or np.any(image > 1):
        print(f"image {path_image} has values outside range [0,1] : clipping !")
        image = np.clip(image, 0, 1)

    plt.imsave(os.path.join(save_folder, 'images', f"{image_id:04}.png"), image)
    with open(os.path.join(save_folder, 'annotations', f"{image_id:04}.pkl"), 'wb') as f:
        pkl.dump(
            {'centers': centers, 'parameters': parameters, 'categories': categories, 'difficult': difficult},
            f)
    with open(os.path.join(save_folder, 'metadata', f"{image_id:04}.json"), 'w') as f:
        json.dump({
            'shape': list(image.shape),
            'n_objects': n_objects,
            'scale': scale,
            **info
        }, f, cls=NumpyEncoder, indent=1)


def make_dataset(subset: str, data_path: str, save_dir: str,
                 categories: List[str], target_gsd: float, prune_empty: bool, drop_rate: float, rng_seed: int,
                 banned_sources: List[str] = None):
    assert subset in ['train', 'val']  # 'test' not supported yet

    dataset_df = fetch_dota_paths(data_path, subset=subset)
    dataset_df = get_content_per_image(dataset_df, categories)
    n_images_prev = len(dataset_df)

    if banned_sources is not None:
        unique_sources = dataset_df.source.unique()
        for s in banned_sources:
            if s not in unique_sources:  # check if banning existing sources
                print(f'WARNING: source {s} does not exist ({unique_sources})')
        dataset_df = dataset_df[~dataset_df.source.isin(banned_sources)]

    dataset_df = dataset_df[dataset_df.gsd <= target_gsd]
    dataset_df['scale'] = dataset_df.gsd / target_gsd

    n_images = len(dataset_df)
    print('')
    print(
        f'Pruning gsd above {target_gsd} (and banned sources) gives a total of {n_images} images ({n_images / n_images_prev:.2%})')

    # dataset_paths = dataset_paths[dataset_paths.id.isin(whitelist)]
    objects_of_interest_per_image = dataset_df[categories].sum(axis=1).values
    dataset_df['n_objects'] = objects_of_interest_per_image
    sample_density = objects_of_interest_per_image / np.sum(objects_of_interest_per_image)
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
            path_image=row['path_image'],
            path_label=row['path_label'],
            target_categories=categories,
            save_folder=save_dir,
            scale=row['scale'],
            n_objects=row['n_objects'],
            info={'original_gsd': row['gsd'], 'source': row['source'], 'date': str(row['date'])}
        )

    tqdm.pandas(desc='extracting patches')

    dataset_df.progress_apply(prepare, axis=1)


def translate_dota(config: Dict[str, Any]):
    dota_base_path = config["dota_base_path"]
    subsets = config["subsets"]
    name = config["name"]
    categories = config["categories"]
    banned_sources = config["banned_sources"]
    target_gsd = config["target_gsd"]
    prune_empty = bool(config["prune_empty"])

    if 'drop_rate' not in config:
        drop_rate = {ss: 0.0 for ss in subsets}
    else:
        drop_rate = config['drop_rate']

    source_base = find_existing_path(dota_base_path)

    dest_base = get_dataset_base_path()

    save_dir = os.path.join(dest_base, name)
    make_if_not_exist(save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=1)

    sub_folders = ['raw_images', 'images', 'raw_annotations', 'annotations', 'metadata', 'images_w_annotations']

    for ss in subsets:
        subset_save_dir = os.path.join(save_dir, ss)
        make_if_not_exist(subset_save_dir)
        make_if_not_exist([os.path.join(subset_save_dir, s) for s in sub_folders])
        print(f'making {ss} patches')
        make_dataset(
            subset=ss,
            save_dir=subset_save_dir,
            data_path=source_base,
            categories=categories,
            target_gsd=target_gsd,
            banned_sources=banned_sources,
            prune_empty=prune_empty,
            drop_rate=drop_rate[ss],
            rng_seed=0
        )
