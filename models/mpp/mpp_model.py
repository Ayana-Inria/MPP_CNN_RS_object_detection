import logging
import logging
import os
import pickle
import re
import time
import warnings
from functools import partial
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from base.base_model import BaseModel
from base.shapes.rectangle import show_rectangles, Rectangle, sra_to_wla, rect_to_poly
from metrics.dota_results_translator import DOTAResultsTranslator
from models.mpp.custom_types.energy import EnergyCombinationModel
from models.mpp.custom_types.image_w_maps import ImageWMaps
from models.mpp.data_loaders import MPPDataset, load_image_w_maps, crop_image_w_maps, \
    merge_patches
from models.mpp.energies.combination.hierarchical import ManualHierarchicalEnergyCombinator
from models.mpp.energies.energy_utils import EnergySetup
from models.mpp.point_set.energy_point_set import EPointsSet
from models.mpp.rjmcmc_sampler.sample_rjmcmc import sample_rjmcmc
from models.mpp.train_energy_combination.train_utils import _map_to_images
from models.shape_net.display import _pred_to_image2
from utils.data import get_inference_path, fetch_data_paths
from utils.files import make_if_not_exist
from utils.math_utils import normalize
from utils.training import Config, startup_config

DEBUG = False
SAVE_PATCHES = True
ENERGY_SETUP = 1


def collate_fn(batch):
    return list(batch)


class MPPModel(BaseModel):
    TRAIN_MODES = ['manual', 'grad_descent', 'integral_criterion', 'ordering_criterion']

    def __init__(self, config: Config, phase: str, overwrite=False, load=False, dataset: str = None):
        self.config, self.logger, self.save_path = startup_config(config, 'mpp', overwrite=overwrite, load_model=load)

        if dataset is not None:
            self.config['dataset']['dataset'] = dataset

        self.rng = np.random.default_rng(0)
        self.dataset = self.config['dataset']['dataset']

        assert phase in ['val', 'train']

        self.__init_data__(subset=phase)

        self.energy_setup: EnergySetup
        self.energy_model: EnergyCombinationModel

        energy_setup_conf = self.config.get("energy_setup")
        if energy_setup_conf is None:
            energy_setup_conf = 'legacy'
        energy_setup_params = self.config.get("energy_setup_params")
        energy_setup_params = {} if energy_setup_params is None else energy_setup_params
        calibration_params = {} if 'params' not in self.config['calibration'] else self.config['calibration']['params']
        if energy_setup_conf == "legacy":
            from models.mpp.energies.energy_setups.energy_setup_legacy import LegacyEnergySetup
            self.energy_setup = LegacyEnergySetup(calibration_params=calibration_params)
            logging.info(f"using LEGACY energies {self.energy_setup.energy_names}")

        elif energy_setup_conf == "no-calibration":
            from models.mpp.energies.energy_setups.energy_setup_no_calibration import NoCalibrationEnergySetup
            self.energy_setup = NoCalibrationEnergySetup(**energy_setup_params)
            logging.info(f"using NO CALIBRATION energies {self.energy_setup.energy_names}")

        elif energy_setup_conf == 'contrast':
            from models.mpp.energies.energy_setups.energy_setup_contrast import ContrastMeasureEnergySetup
            self.energy_setup = ContrastMeasureEnergySetup(**energy_setup_params)
            logging.info(f"using CONTRAST energies {self.energy_setup.energy_names}")

        else:
            print("energy_setup must be one of : 'legacy', 'no-calibration', 'contrast'")
            raise ValueError

        if load:
            try :
                with open(os.path.join(self.save_path, 'energy_combination_model.pkl'), 'rb') as f:
                    self.energy_model: EnergyCombinationModel = pickle.load(f)
                self.energy_setup.load_calibration(self.save_path)
            except FileNotFoundError as e :
                if self._find_train_mode() == 'manual':
                    self.train()
        else:
            assert phase == 'train'
            self.calibrate()
            self.energy_model = None

    def __init_data__(self, subset: str):
        self.data = MPPDataset(
            **self.config['dataset'],
            subset=subset
        )

        self.data_loader = DataLoader(self.data, batch_size=self.config['data_loader']['batch_size'],
                                      num_workers=8, prefetch_factor=2,
                                      collate_fn=collate_fn)

    def calibrate(self):
        n_calibration_images = self.config['calibration']['n_images']
        if n_calibration_images > len(self.data):
            print(f"Cannot take {n_calibration_images} in a size {len(self.data)} dataset, "
                  f"taking only {len(self.data)} calibration images")
            n_calibration_images = len(self.data)
        calibration_indexes = self.rng.choice(range(len(self.data)),
                                              size=n_calibration_images,
                                              replace=False)
        calibration_data = [self.data.__getitem__(i) for i in calibration_indexes]

        self.energy_setup.calibrate(
            image_configs=calibration_data,
            rng=self.rng,
            save_path=self.save_path
        )

    def _find_train_mode(self):
        train_mode = None
        for t in self.TRAIN_MODES:
            if t in self.config:
                if train_mode is not None:
                    logging.error(f"found {train_mode} and {t} in model config : can only have one train mode")
                    raise ValueError
                train_mode = t
        return train_mode

    def train(self):
        self.data_preview()

        train_mode = self._find_train_mode()

        if train_mode == 'grad_descent' or train_mode == 'integral_criterion':
            if train_mode == 'grad_descent':
                warnings.warn(f"train_mode '{train_mode}' is deprecated, use 'integral_criterion' instead")
            # grad_descent ensure back-compatibility of previous config files
            from models.mpp.train_energy_combination.train_integral_criterion import train_integral_criterion
            config_train = self.config.get("grad_descent") if 'grad_descent' in self.config else self.config[
                'integral_criterion']
            self.energy_model = train_integral_criterion(
                train_loader=self.data_loader,
                rng=self.rng,
                save_dir=self.save_path,
                logger=self.logger,
                **config_train)
        elif train_mode == 'manual':
            from models.mpp.energies.combination.hierarchical import HierarchicalEnergyCombinator

            if self.config.get("energy_setup") == "legacy":
                weights_dict = self.config["manual"]

                data_prior = [weights_dict[k] for k in ['Data', 'Prior']]
                sub_data = [weights_dict[k] for k in ['PositionEnergy', 'ShapeEnergy']]
                sub_prior = [weights_dict[k] for k in ['RectangleOverlapEnergy', 'ShapeAlignmentEnergy', 'AreaPriorEnergy']]
                threshold = 0.0 if 'threshold' not in weights_dict else weights_dict['threshold']

                data_prior = normalize(data_prior)
                sub_data = normalize(sub_data)
                sub_prior = normalize(sub_prior)

                self.energy_model = HierarchicalEnergyCombinator(
                    weights_data=sub_data,
                    weights_prior=sub_prior,
                    data_prior_weights=data_prior,
                    detection_threshold=threshold
                )
                print(f"data/prior: {data_prior}\ndata {sub_data}\nprior {sub_prior}")

            else:
                self.energy_model = ManualHierarchicalEnergyCombinator(
                    weights_dict=self.config["manual"].get("weights"),
                    indicator_energy=self.config["manual"].get("indicator_energy"),
                    detection_threshold=self.config["manual"].get("threshold")
                )

        elif train_mode == 'ordering_criterion':
            from models.mpp.train_energy_combination.train_ordering_criterion import train_ordering_criterion
            self.energy_model = train_ordering_criterion(
                train_loader=self.data_loader,
                rng=self.rng,
                save_dir=self.save_path,
                logger=self.logger,
                energy_setup=self.energy_setup,
                **self.config["ordering_criterion"]
            )

        else:
            raise NotImplementedError

        with open(os.path.join(self.save_path, 'energy_combination_model.pkl'), 'wb') as f:
            pickle.dump(self.energy_model, f)

    def infer(self, subset: str, min_confidence: float = 0.1, display_min_confidence: float = 0.5,
              overwrite: bool = True):
        dataset = self.config['dataset']['dataset']

        results_dir = get_inference_path(
            model_name=os.path.split(self.save_path)[1], dataset=dataset, subset=subset)

        make_if_not_exist(results_dir, recursive=True)
        dota_trlt = DOTAResultsTranslator(dataset, subset, results_dir, det_type='obb', all_classes=['vehicle'])
        dota_trlt_2 = DOTAResultsTranslator(dataset, subset, results_dir, det_type='obb', all_classes=['vehicle'],
                                            postfix='-SV')
        # todo use a smart patch maker to cover all + regroup
        # todo use multiprocessing
        id_re = re.compile(r'([0-9]+).*.png')
        paths_dict = fetch_data_paths(dataset, subset=subset)
        image_paths = paths_dict['images']
        annot_paths = paths_dict['annotations']
        meta_paths = paths_dict['metadata']
        for pf, af, mf in zip(tqdm(image_paths, desc=f'inferring on {dataset}/{subset}'), annot_paths,
                              meta_paths):
            patch_id = int(id_re.match(os.path.split(pf)[1]).group(1))
            if os.path.exists(os.path.join(results_dir, f'{patch_id:04}_results.pkl')) and not overwrite:
                print(f"{patch_id:04}_results.pkl exists, skipping")
                continue

            image_data = load_image_w_maps(
                patch_id, dataset=dataset, subset=subset, position_model=self.data.position_model,
                shape_model=self.data.shape_model)

            patch_size = 256

            shape = image_data.shape[:2]

            nx = int(np.ceil(shape[0] / patch_size))
            ny = int(np.ceil(shape[1] / patch_size))
            anchors_x = np.linspace(0, shape[0] - patch_size, max(1, nx), dtype=int)
            anchors_y = np.linspace(0, shape[1] - patch_size, max(1, ny), dtype=int)
            image_patches = []
            for i_a, x_a in enumerate(anchors_x):
                for j_a, y_a in enumerate(anchors_y):
                    tl_anchor = np.array([x_a, y_a])

                    image_data_crop = crop_image_w_maps(
                        image_data=image_data, tl_anchor=tl_anchor, patch_size=patch_size
                    )

                    image_patches.append(image_data_crop)

            partial_func = partial(
                sample_rjmcmc,
                init_config='naive',
                rng=self.rng,
                num_samples=1,
                verbose=1 if DEBUG else False,
                energy_combinator=self.energy_model,
                energy_setup=self.energy_setup,
                **self.config['inference']['rjmcmc_params']
            )
            logging.info(f"forking into {len(image_patches)} rjmcmc runs")
            start = time.perf_counter()
            results = _map_to_images(partial_func, image_patches, multiprocess=not DEBUG)
            logging.info(f"ran {len(image_patches)} rjmcmc in parallel in {time.perf_counter() - start:.2f}")
            results = [r[-1] for r in results]
            if SAVE_PATCHES:
                make_if_not_exist(os.path.join(results_dir, 'patches'))
                for k, r in enumerate(results):
                    result: List[Rectangle] = r
                    pred_params = [sra_to_wla(p.size, p.ratio, p.angle) for p in result]
                    pred_centers = [[p.x, p.y] for p in result]
                    image_w_pred = _pred_to_image2(
                        centers=pred_centers,
                        params=pred_params,
                        scores=None,
                        image=image_patches[k].image,
                        color=(1.0, 0.1, 0.1))

                    image_w_gt = _pred_to_image2(
                        centers=image_patches[k].labels['centers'],
                        params=image_patches[k].labels['parameters'],
                        scores=None,
                        image=image_patches[k].image,
                        color=(0, 1.0, 0))

                    plt.imsave(os.path.join(results_dir, 'patches', f'{patch_id:04}-{k:02}_detection.png'),
                               image_w_pred)
                    plt.imsave(os.path.join(results_dir, 'patches', f'{patch_id:04}-{k:02}_gt.png'), image_w_gt)

            logging.info(f"merging {len(image_patches)} patches ...")
            start = time.perf_counter()
            all_rect = merge_patches(patches=image_patches, results=results, original_image=image_data,
                                     method='distance',
                                     energy_model=self.energy_model, distance=3, energy_setup=self.energy_setup)
            logging.info(f"merge done in {time.perf_counter() - start:.2f}")

            uec, pec = self.energy_setup.make_energies(image_data=image_data)
            result: EPointsSet = EPointsSet(
                points=all_rect, support_shape=image_data.shape, unit_energies_constructors=uec,
                pair_energies_constructors=pec
            )
            pred_params = [sra_to_wla(p.size, p.ratio, p.angle) for p in result]
            pred_centers = np.array([[p.x, p.y] for p in result])
            pred_scores = [result.papangelou(p, energy_combinator=self.energy_model, remove_u_from_point_set=True)
                           for p in result]

            image_w_pred = _pred_to_image2(
                centers=pred_centers,
                params=pred_params,
                scores=pred_scores,
                image=image_data.image,
                color='plasma',
                max_score=1.0 if len(pred_scores) == 0 else np.max(pred_scores),
            )

            image_w_gt = _pred_to_image2(
                centers=image_data.labels['centers'],
                params=image_data.labels['parameters'],
                scores=None,
                image=image_data.image,
                color=(0, 1.0, 0))

            plt.imsave(os.path.join(results_dir, f'{patch_id:04}_detection.png'), image_w_pred)
            plt.imsave(os.path.join(results_dir, f'{patch_id:04}_gt.png'), image_w_gt)

            centers, params = image_data.labels['centers'], image_data.labels['parameters']
            difficulty = image_data.labels['difficult']
            gt_as_poly = np.array(
                [rect_to_poly(c, short=p[0], long=p[1], angle=p[2]) for c, p in zip(centers, params)])

            detection_as_poly = np.array(
                [rect_to_poly(c, p[0], p[1], p[2]) for c, p in zip(pred_centers, pred_params)])

            dota_trlt.add_gt(image_id=patch_id, polygons=gt_as_poly, difficulty=difficulty,
                             categories=['vehicle' for _ in gt_as_poly])

            dota_trlt_2.add_gt(
                image_id=patch_id, polygons=gt_as_poly,
                difficulty=[d or c == 'large-vehicle' for d, c in
                            zip(image_data.labels['difficult'], image_data.labels['categories'])],
                categories=['vehicle' for _ in gt_as_poly])

            pred_score_01 = np.array(pred_scores)
            max_score = self.config['inference'].get('max_score')
            max_score = 4.0 if max_score is None else max_score
            pred_score_01 = pred_score_01 / max_score
            if len(pred_score_01) > 0 and np.max(pred_score_01) > 1.0:
                logging.warning(f"pred score higher than max, effective score is {np.max(pred_scores)} "
                                f"while param says {max_score}")

            dota_trlt.add_detections(image_id=patch_id, scores=pred_score_01, polygons=detection_as_poly,
                                     flip_coor=True, class_names=['vehicle' for _ in pred_scores])

            dota_trlt_2.add_detections(image_id=patch_id, scores=pred_score_01, polygons=detection_as_poly,
                                       flip_coor=True, class_names=['vehicle' for _ in pred_scores])

            results_dict = {
                'detection': detection_as_poly,
                'detection_points': all_rect,
                'detection_type': 'poly',
                'detection_center': pred_centers,
                'detection_score': pred_scores,
                'detection_params': pred_params,
                'mappings': image_data.mappings,
            }
            with open(os.path.join(results_dir, f'{patch_id:04}_results.pkl'), 'wb') as f:
                pickle.dump(results_dict, f)

        dota_trlt.save()
        dota_trlt_2.save()
        print('saved dota translation')

    def eval(self):
        from metrics.dota_eval import dota_eval
        dota_eval(
            model_dir=self.save_path,
            dataset=self.dataset,
            subset='val',
            det_type='obb'
        )

        dota_eval(
            model_dir=self.save_path,
            dataset=self.dataset,
            subset='val',
            det_type='obb',
            postfix='-SV'
        )

    def data_preview(self):
        plt.ioff()
        data_preview_path = os.path.join(self.save_path, 'data_preview')
        make_if_not_exist(data_preview_path)
        for i in range(8):
            x: ImageWMaps = self.data.__getitem__(i)
            fig, axs = plt.subplots(1, 1)
            axs.imshow(x.image)
            show_rectangles(axs, x.gt_config)
            plt.savefig(os.path.join(data_preview_path, f"preview_{x.name}_gt.png"))
            plt.close('all')
