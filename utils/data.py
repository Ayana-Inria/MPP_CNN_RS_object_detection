import glob
import json
import logging
import os
import re
import sys
from typing import List

import numpy as np

from utils.files import find_existing_path


def fetch_data_paths(dataset: str, subset: str, images=True, annotations=True, metadata=True):
    """

    :param dataset: dataset name, eg DOTA_gsd50
    :param subset: subset name, usally train or val
    :param images: if true will fetch images
    :param annotations: if true will fetch annotations
    :param metadata: if true will fetch metadata
    :return:
    """

    data_path = os.path.join(get_dataset_base_path(), dataset, subset)

    res = {}
    if images:
        paths = glob.glob(os.path.join(data_path, 'images', '*.png'))
        paths.sort()
        res['images'] = paths
    if annotations:
        paths = glob.glob(os.path.join(data_path, 'annotations', '*.pkl'))
        paths.sort()
        res['annotations'] = paths
    if metadata:
        paths = glob.glob(os.path.join(data_path, 'metadata', '*.json'))
        paths.sort()
        res['metadata'] = paths

    base_len = len(res[list(res.keys())[0]])
    assert all([len(res[k]) == base_len for k in res.keys()])

    return res


def get_inference_path(model_name: str, dataset: str, subset: str):
    base_dataset_dir = get_dataset_base_path()
    results_dir = os.path.join(base_dataset_dir, 'inference', dataset, subset, model_name)
    return results_dir


def load_paths_config():
    try:
        with open('paths_config.json', "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        for p in sys.path:
            try:
                with open(os.path.join(p, 'paths_config.json'), "r") as f:
                    config = json.load(f)
            except (FileNotFoundError, NotADirectoryError):
                config = None
            if config is not None:
                break
    return config


def get_dataset_base_path():
    config = load_paths_config()
    return find_existing_path(config['dataset_path'])


def get_model_base_path():
    config = load_paths_config()
    return find_existing_path(config['model_path'])


def get_model_config_by_name(name: str, return_config_file=True):
    model_dir = os.path.join(get_model_base_path(), '*', name)
    if return_config_file:
        match_list = glob.glob(os.path.join(model_dir, 'config.json'))
    else:
        match_list = glob.glob(model_dir)
    if len(match_list) == 1:
        return match_list[-1]
    elif len(match_list) == 0:
        return None
    else:
        logging.warning(f"found more than one model for {name}: {match_list} \nreturning {match_list[-1]}")
        return match_list[-1]


def get_config_from_model_configs(name: str):
    if '.json' not in name:
        print(f"{name} does not look like a json file")
    base = None
    for p in sys.path:
        if os.path.exists(os.path.join(p, 'model_configs')):
            base = p
            break
    if base is None:
        raise FileNotFoundError
    match_list = glob.glob(os.path.join(base, 'model_configs', '*', name))
    if len(match_list) == 0:
        return None
    elif len(match_list) == 1:
        return match_list[-1]
    else:
        logging.warning(f"found more than one model for {name}: {match_list} \nreturning {match_list[-1]}")
        return match_list[-1]


def resolve_model_config_path(config_file_or_model_name: str):
    """
    :param config_file_or_model_name: a full path to a config .json file, a config.json file located in
    model_configs/*/ or a model name
    :return: the config file full path, or raises a FileNotFoundError if not found
    """
    # is it a full path ?
    if os.path.exists(config_file_or_model_name):
        config_file = config_file_or_model_name
    else:
        # is it a .json model config located in model_configs ?
        config_file = get_config_from_model_configs(config_file_or_model_name)
        if config_file is None:
            # is it a model name ?
            config_file = get_model_config_by_name(config_file_or_model_name, return_config_file=True)
            if config_file is None:
                print(f"no model with name (or config with path) {config_file_or_model_name}")
                raise FileNotFoundError
    return config_file


def check_data_match(paths: List[str]) -> int:
    ids = []
    for p in paths:
        object_id = re.match(r'([0-9]+)\.[a-zA-z]+', os.path.split(p)[1]).group(1)
        ids.append(object_id)

    for k, i in enumerate(ids):
        try:
            assert ids[0] == i
        except AssertionError as e:
            print(f"id does not match {ids[0]} != {i} for object {paths[0]} and {paths[k]} ")
            raise e

    return int(ids[0])


def split_image(image: np.ndarray, centers: np.ndarray, params: np.ndarray, target_size: int, min_overlap: int = 0):
    shape = image.shape[:2]
    n_x = int(np.ceil(shape[0] / (target_size - min_overlap)))
    n_y = int(np.ceil(shape[1] / (target_size - min_overlap)))

    if n_y > 1 or n_x > 1:
        x_anchors = np.linspace(0, shape[0] - target_size, num=n_x, dtype=int)
        y_anchors = np.linspace(0, shape[1] - target_size, num=n_y, dtype=int)

        overlap_x = (target_size - np.mean(np.diff(x_anchors))) / target_size
        overlap_y = (target_size - np.mean(np.diff(y_anchors))) / target_size

        patch_data_list = []
        for i, x in enumerate(x_anchors):
            for j, y in enumerate(y_anchors):
                slc = np.s_[x:x + target_size, y:y + target_size]

                patch = image[slc]
                new_labels = {'centers': [], 'parameters': []}
                for c, p in zip(centers, params):
                    if x <= c[0] < x + target_size and y <= c[1] < y + target_size:
                        new_labels['centers'].append(c - np.array([x, y]))
                        new_labels['parameters'].append(p)

                patch_data = {'image': patch, 'anchor': [x, y], **new_labels}

                patch_data_list.append(patch_data)

        return patch_data_list
    else:
        return [{'image': image, 'centers': centers, 'parameters': params, 'anchor': [0, 0]}]
