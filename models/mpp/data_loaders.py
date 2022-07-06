import glob
import json
import logging
import os
import pickle
import re
from copy import copy
from functools import partial
from multiprocessing import Pool
from typing import List, Union

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from base.shapes.rectangle import Rectangle, wla_to_sra
from data.patch_samplers import MixedSampler, UniformSampler, ObjectSampler
from models.mpp.custom_types.image_w_maps import ImageWMaps
from models.mpp.custom_types.energy import EnergyCombinationModel
from models.mpp.energies.energy_utils import EnergySetup
from models.mpp.point_set.energy_point_set import EPointsSet
from utils.data import get_dataset_base_path, get_model_config_by_name, fetch_data_paths, get_inference_path

PARAM_NAMES = ['size', 'ratio', 'angle']

OVERWRITE = False
FORCE_CHECK_INFER = False


def load_image_w_maps(patch_id: Union[str, int], dataset: str, subset: str, position_model: str,
                      shape_model: str) -> ImageWMaps:
    if type(patch_id) is str:
        patch_id = int(patch_id)

    basic_data_path = os.path.join(get_dataset_base_path(), dataset, subset)
    posnet_infer_path = get_inference_path(model_name=position_model, dataset=dataset, subset=subset)
    shapenet_infer_path = get_inference_path(model_name=shape_model, dataset=dataset, subset=subset)

    image_path = os.path.join(basic_data_path, 'images', f'{patch_id:04}.png')
    annot_path = os.path.join(basic_data_path, 'annotations', f'{patch_id:04}.pkl')
    posnet_inference_path = os.path.join(posnet_infer_path, f'{patch_id:04}_results.pkl')
    shapenet_inference_path = os.path.join(shapenet_infer_path, f'{patch_id:04}_results.pkl')

    image = plt.imread(image_path)[:, :, :3]
    with open(annot_path, 'rb') as f:
        label_dict = pickle.load(f)

    with open(posnet_inference_path, 'rb') as f:
        detection_map = pickle.load(f)['detection_map']

    with open(shapenet_inference_path, 'rb') as f:
        shapenet_inference = pickle.load(f)

    param_dist_maps = [np.moveaxis(p[0], 0, -1) for p in shapenet_inference['output']]
    mappings = shapenet_inference['mappings']

    shape = image.shape[:2]

    gt_config = labels_to_rectangles(label_dict, Rectangle.PARAMETERS)

    return ImageWMaps(
        image=image,
        name=f"{patch_id:04}",
        shape=shape,
        detection_map=detection_map,
        param_dist_maps=param_dist_maps,
        mappings=mappings,
        param_names=PARAM_NAMES,
        labels=label_dict,
        gt_config=gt_config
    )


def crop_image_w_maps(image_data: ImageWMaps, tl_anchor: np.ndarray, patch_size: int) -> ImageWMaps:
    s = np.s_[tl_anchor[0]:tl_anchor[0] + patch_size, tl_anchor[1]:tl_anchor[1] + patch_size]

    image_crop = image_data.image[s]
    detection_map_crop = image_data.detection_map[s]
    param_dist_maps_crop = [p[s] for p in image_data.param_dist_maps]
    assert image_crop.shape[:2] == detection_map_crop.shape[:2]
    assert all([image_crop.shape[:2] == p.shape[:2] for p in param_dist_maps_crop])
    centers_crop = []
    params_crop = []
    categories_crop = []
    difficulty_crop = []
    for j, c in enumerate(image_data.labels['centers']):
        new_c = c - tl_anchor
        if np.all(new_c >= 0) and np.all(new_c < np.array(image_crop.shape[:2])):
            centers_crop.append(new_c)
            params_crop.append(image_data.labels['parameters'][j])
            categories_crop.append(image_data.labels['categories'][j])
            difficulty_crop.append(image_data.labels['difficult'][j])

    centers_crop = np.array(centers_crop)
    params_crop = np.array(params_crop)
    categories_crop = np.array(categories_crop)
    difficulty_crop = np.array(difficulty_crop)

    new_label_dict = {
        'parameters': params_crop,
        'centers': centers_crop,
        'categories': categories_crop,
        'difficult': difficulty_crop
    }

    gt_config = labels_to_rectangles(new_label_dict, Rectangle.PARAMETERS)

    return ImageWMaps(
        image=image_crop,
        name=image_data.name,
        shape=image_crop.shape[:2],
        detection_map=detection_map_crop,
        param_dist_maps=param_dist_maps_crop,
        mappings=image_data.mappings,
        param_names=PARAM_NAMES,
        labels=new_label_dict,
        gt_config=gt_config,
        crop_data={'tl_anchor': tl_anchor}
    )


def merge_patches(patches: List[ImageWMaps], results: List[List[Rectangle]], original_image: ImageWMaps,
                  energy_model: EnergyCombinationModel, method: str,
                  energy_setup: EnergySetup, **kwargs):
    assert method in ['distance', 'rjmcmc']

    uec, pec = energy_setup.make_energies(image_data=original_image)
    aggregated_result = EPointsSet(
        points=[], support_shape=original_image.shape, unit_energies_constructors=uec, pair_energies_constructors=pec
    )

    for patch, result in zip(patches, results):
        for r in result:
            new_r = copy(r)
            anchor = patch.crop_data['tl_anchor']
            new_r.x = new_r.x + anchor[0]
            new_r.y = new_r.y + anchor[1]
            aggregated_result.add(new_r)

    if method == 'distance':
        distance = kwargs['distance']
        to_remove = set()
        for p in aggregated_result:
            if p in to_remove:
                continue
            neigh = aggregated_result.points.get_neighbors(p, radius=distance, exclude_itself=False)
            neigh = neigh - to_remove
            neigh = list(neigh)
            if len(neigh) == 0:
                continue
            scores = [aggregated_result.papangelou(p, energy_combinator=energy_model, remove_u_from_point_set=True) for
                      p in neigh]
            best = neigh[np.argmax(scores)]
            to_remove |= set(neigh)
            to_remove = to_remove - {best}

        logging.info(f"merge removing {len(to_remove)} point(s)")
        for p in to_remove:
            aggregated_result.remove(p)

    return aggregated_result.points


class MPPDataset(Dataset):
    def __init__(self, dataset: str, subset: str, position_model: str, shape_model: str, patch_size: int,
                 patch_ids: List[int] = None):
        self.dataset = dataset
        self.subset = subset
        self.patch_size = patch_size

        # fetch data paths
        self.basic_data_path = os.path.join(get_dataset_base_path(), dataset, subset)
        self.position_model = position_model
        self.posnet_infer_path = get_inference_path(model_name=position_model, dataset=dataset, subset=subset)
        if not os.path.exists(self.posnet_infer_path) or FORCE_CHECK_INFER:
            print(f"Found no inference files at {self.posnet_infer_path} (or force check) : "
                  f"starting inference procedure")
            # if no results for this dataset, infer it
            from models.position_net.pos_net_model import PosNetModel
            pos_config_file = get_model_config_by_name(position_model, return_config_file=True)
            with open(pos_config_file, 'r') as f:
                pos_config = json.load(f)
            pos_config['data_loader']['dataset'] = dataset
            pos_model = PosNetModel(pos_config, train=False, load=True)
            pos_model.infer(subset=subset, min_confidence=0.2, display_min_confidence=0.5, overwrite=OVERWRITE)
            assert os.path.exists(self.posnet_infer_path)
            del pos_model
        self.shape_model = shape_model
        self.shapenet_infer_path = get_inference_path(model_name=shape_model, dataset=dataset, subset=subset)
        if not os.path.exists(self.shapenet_infer_path) or FORCE_CHECK_INFER:
            print(f"Found no inference files at {self.shapenet_infer_path} (or force check) : "
                  f"starting inference procedure")
            # if no results for this dataset, infer it
            from models.shape_net.shape_net_model import ShapeNetModel
            shape_config_file = get_model_config_by_name(shape_model, return_config_file=True)
            with open(shape_config_file, 'r') as f:
                shape_config = json.load(f)
            shape_config['data_loader']['dataset'] = dataset
            shape_model = ShapeNetModel(shape_config, train=False, load=True)
            shape_model.infer(subset=subset, min_confidence=0.2, display_min_confidence=0.5, overwrite=OVERWRITE)
            assert os.path.exists(self.posnet_infer_path)
            del shape_model

        if patch_ids is not None:
            self.patches_index = [f'{i:04}' for i in patch_ids]
        else:
            image_paths = glob.glob(os.path.join(self.basic_data_path, 'images', '*.png'))
            assert len(image_paths) > 0
            re_pattern = re.compile(r'([0-9]+)\.[a-zA-z]+')
            self.patches_index = [re_pattern.match(os.path.split(p)[1]).group(1) for p in image_paths]



        self.rng = np.random.default_rng(0)

        self.patch_sampler = MixedSampler(
            n_patches=len(self.patches_index),
            samplers=[
                UniformSampler(n_patches=len(self.patches_index), patch_size=patch_size, rng=self.rng),
                ObjectSampler(n_patches=len(self.patches_index), patch_size=patch_size, rng=self.rng, sigma=10)
            ],
            weights=[1 / 10, 9 / 10],
            rng=self.rng
        )
        files_dict = fetch_data_paths(dataset, subset)
        self.patch_sampler.initialise(
            patch_files=files_dict['images'], label_files=files_dict['annotations'], meta_files=files_dict['metadata'])

    def __getitem__(self, index):

        sampler_image_id = self.patch_sampler.sample_image()
        patch_id = self.patches_index[sampler_image_id]

        image_data = load_image_w_maps(
            patch_id=patch_id,
            dataset=self.dataset, subset=self.subset,
            position_model=self.position_model, shape_model=self.shape_model
        )

        center_anchor = self.patch_sampler.sample_patch_center(
            image_id=sampler_image_id, shape=image_data.shape, centers=image_data.labels['centers'])
        tl_anchor = (center_anchor - (self.patch_size // 2)).astype(int)
        tl_anchor = np.clip(tl_anchor, (0, 0),
                            (image_data.shape[0] - self.patch_size, image_data.shape[1] - self.patch_size))

        return crop_image_w_maps(image_data=image_data, tl_anchor=tl_anchor, patch_size=self.patch_size)

    def __len__(self):
        return len(self.patches_index)


def labels_to_rectangles(labels, param_names: List[str]):
    centers, params = labels['centers'], labels['parameters']
    rectangles = []
    for c, p in zip(centers, params):
        s, r, a = wla_to_sra(p[0], p[1], p[2])
        rectangles.append(Rectangle(
            c[0], c[1], size=s, ratio=r, angle=a % np.pi
        ))
    return rectangles


def split_image(image_data: ImageWMaps, target_size: int, min_overlap: int) -> List[ImageWMaps]:
    shape = image_data.image.shape[:2]

    n_x = int(np.ceil(shape[0] / (target_size - min_overlap)))
    n_y = int(np.ceil(shape[1] / (target_size - min_overlap)))

    if n_y > 1 or n_x > 1:
        x_anchors = np.linspace(0, shape[0] - target_size, num=n_x, dtype=int)
        y_anchors = np.linspace(0, shape[1] - target_size, num=n_y, dtype=int)

        overlap_x = (target_size - np.mean(np.diff(x_anchors))) / target_size
        overlap_y = (target_size - np.mean(np.diff(y_anchors))) / target_size

        logging.info(f"split image {image_data.name} into {n_x}x{n_y} patches "
                     f"with overlap {overlap_x:.1%}(x) {overlap_y:.1%}(y)")

        patch_data_list = []
        for i, x in enumerate(x_anchors):
            for j, y in enumerate(y_anchors):
                slc = np.s_[x:x + target_size, y:y + target_size]

                patch = image_data.image[slc]
                new_labels = {'centers': [], 'parameters': []}
                for c, p in zip(image_data.labels['centers'], image_data.labels['parameters']):
                    if x <= c[0] < x + target_size and y <= c[1] < y + target_size:
                        new_labels['centers'].append(c - np.array([x, y]))
                        new_labels['parameters'].append(p)

                new_gt_config = []
                for p in image_data.gt_config:
                    if x <= p.x < x + target_size and y <= p.y < y + target_size:
                        new_p = copy(p)
                        new_p.x = new_p.x - x
                        new_p.y = new_p.y - y
                        new_gt_config.append(new_p)

                patch_data = ImageWMaps(
                    image=patch,
                    name=image_data.name + f"_p{i:02}-{j:02}",
                    shape=patch.shape,
                    detection_map=image_data.detection_map[slc],
                    param_dist_maps=[p[slc] for p in image_data.param_dist_maps],
                    mappings=image_data.mappings,
                    param_names=image_data.param_names,
                    labels=new_labels,
                    gt_config=new_gt_config,
                    crop_data={'x_anchor': x, 'y_anchor': y}
                )

                patch_data_list.append(patch_data)

        return patch_data_list
    else:
        logging.info(f"did not split since {image_data.name} is small enough {shape}")
        return [image_data]


def split_multiple_images(image_data_list: List[ImageWMaps],
                          target_size: int, min_overlap: int,
                          multiprocess=True) -> List[ImageWMaps]:
    fun = partial(split_image, target_size=target_size, min_overlap=min_overlap)
    if multiprocess:
        with Pool() as p:
            patch_data_list = p.map(fun, image_data_list)
    else:
        patch_data_list = [fun(i) for i in image_data_list]
    result = []
    for p in patch_data_list:
        result = result + p
    return result
