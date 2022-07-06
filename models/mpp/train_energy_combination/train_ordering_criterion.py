import logging
import os
import time
from dataclasses import dataclass
from functools import partial
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.random import Generator
from torch.nn import Module
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from models.mpp.custom_types.energy import ConfigurationEnergyVector, EnergyCombinationModel
from models.mpp.custom_types.image_w_maps import ImageWMaps
from models.mpp.display import cross_plot
from models.mpp.energies.combination.base import WeightModel
from models.mpp.energies.energy_utils import compute_many_energy_vectors, EnergySetup
from models.mpp.perturbation_sampler import sample_multiple_kernel_perturbations
from models.mpp.point_set.energy_point_set import EPointsSet
from models.mpp.train_energy_combination.train_utils import _map_to_images, init_model
from utils.logger import Logger


@dataclass
class EnergyComputeTorch(EnergyCombinationModel):
    energy_names: List[str]
    weights_model: Union[WeightModel, Module]

    def _dict_to_tensor(self, vectors: ConfigurationEnergyVector):
        return torch.tensor(np.stack([vectors[k] for k in self.energy_names], axis=-1)).float()

    def compute(self, vectors: ConfigurationEnergyVector):
        t = self._dict_to_tensor(vectors)
        if len(t) == 0:
            return 0.0
        else:
            return self.weights_model.forward(t)


def train_ordering_criterion(train_loader: DataLoader, rng: Generator,
                             logger: Logger, samples_per_image: int, n_epochs: int, save_dir: str,
                             energy_setup: EnergySetup,
                             neg_sampling_method: str = 'rjmcmc', pos_sampling_method: str = 'single',
                             reg_weight=None, optim: str = 'adam', lr_scheduler: bool = False,
                             learning_rate=1e-1, weight_model_type='hierarchical',
                             multiprocess=True, **kwargs):
    energy_names = energy_setup.energy_names

    weights_model = init_model(weight_model_type, energy_setup=energy_setup, **kwargs)

    if optim == 'adam':
        from torch.optim import Adam
        optim = Adam(params=weights_model.parameters(), lr=learning_rate)
    elif optim == 'sgd':
        from torch.optim import SGD
        optim = SGD(params=weights_model.parameters(), lr=learning_rate)

    scheduler = None
    if lr_scheduler:
        scheduler = ExponentialLR(optim, **kwargs["lr_scheduler_params"])

    n_batches = len(train_loader)
    image_data = []
    perturbations_per_image = []
    for epoch_id in range(n_epochs):
        for batch_id, image_data in enumerate(train_loader):
            image_data: List[ImageWMaps]
            optim.zero_grad()

            # load gt configs as EPointsSet
            start = time.perf_counter()
            for d in image_data:  # todo pool ? see timings
                uec, pec = energy_setup.make_energies(image_data=d)

                points = EPointsSet(
                    points=d.gt_config,
                    support_shape=d.shape[:2],
                    unit_energies_constructors=uec,
                    pair_energies_constructors=pec,
                )

                d.gt_config_set = points
            logging.info(f"loading {len(image_data)} gt configs as EPointsSet in {time.perf_counter() - start:.2f}s")

            # make perturbations_per_image
            start = time.perf_counter()
            partial_func = partial(
                sample_multiple_kernel_perturbations,
                energy_setup=energy_setup,
                rng=rng,
                **kwargs['neg_pert_config'],
                n_samples=samples_per_image,
                return_perturbations=True,
                aggregate_pert=True
            )
            # WARNING : if multiprocess true then causes issues since it performs copies of everything
            perturbations_per_image = _map_to_images(partial_func, image_data, multiprocess=False)
            logging.info(f"samplng {len(perturbations_per_image) * samples_per_image} perturbations"
                         f" in {time.perf_counter() - start:.2f}s")
            # perturbations_per_image : List[List[Perturbation]] , one list of pert per image

            # needs dict to array + energy array to energy torch scalar

            # todo need a compute method on this one ?
            energy_combinator = EnergyComputeTorch(weights_model=weights_model, energy_names=energy_names)

            # compute deltas
            deltas = []
            for d, perturbations_list in zip(image_data, perturbations_per_image):

                for pert in perturbations_list:
                    points_set = d.gt_config_set
                    delta = points_set.energy_delta(p=pert, energy_combinator=energy_combinator)
                    if delta != 0.0:
                        deltas.append(delta)

            # aggregate deltas to loss
            loss = - torch.mean(torch.stack(deltas))

            # step optimiser
            loss.backward()
            optim.step()

            log = {
                'batch': batch_id,
                'loss': float(loss.cpu().detach()),
                'lr': learning_rate if scheduler is None else scheduler.get_last_lr()[0],
                **weights_model.as_dict()
            }
            logger.update(epoch=epoch_id, metrics=log)

            print(f"[epoch {epoch_id + 1}/{n_epochs}][batch {batch_id + 1}/{n_batches}] "
                  f"loss: {log['loss']:.4f} "
                  f"lr {log['lr']:.4f} ")

        # ----------------- end train epoch ---------------
        if scheduler is not None:
            scheduler.step()

        try:
            fig, axs = plt.subplots(1, 1, figsize=(8, 4))
            axs: plt.Axes
            axs.set_xlabel('epoch')
            xx = np.array(range(len(logger.log['loss']))) / n_batches
            axs.plot(xx, logger.log['loss'], label='loss')
            axs.legend()
            plt.savefig(os.path.join(save_dir, f'losses_plot.png'))
            plt.close('all')

            log_keys = logger.log.keys()
            has_biases = any(['bias' in k for k in log_keys])
            n_cols = 2 if has_biases else 1

            fig, axs = plt.subplots(1, n_cols, figsize=(n_cols * 8, 4), squeeze=False)
            ax_w: plt.Axes = axs[0, 0]
            ax_w.set_xlabel('epoch')
            for key in log_keys:
                if '_weight' in key:
                    ax_w.plot(xx, logger.log[key], label=key)
            ax_w.legend()
            if has_biases:
                ax_b: plt.Axes = axs[0, 1]
                ax_b.set_xlabel('epoch')
                for key in log_keys:
                    if 'bias' in key:
                        ax_b.plot(xx, logger.log[key], label=key)
                ax_b.legend()
            plt.savefig(os.path.join(save_dir, f'weights_plot.png'))
            plt.close('all')
        except Exception as e:
            print(f"Exception occurred while making figures, continuing ")
            print(e)

        print(weights_model.as_dict())

        all_v_vec = []
        all_nv_vec = []
        try:

            all_v_vec = []
            for d in image_data:
                uec, pec = energy_setup.make_energies(image_data=d)
                all_v_vec.append(
                    compute_many_energy_vectors(
                        configurations=[d.gt_config], image_config=d, ue=uec, pe=pec, energy_names=energy_names))

            all_nv_vec = []
            for d, perturbations_list in zip(image_data, perturbations_per_image):
                pts = []
                for p in perturbations_list:
                    new_points = d.gt_config_set.apply_perturbation(p, inplace=False)
                    pts.append(list(new_points.points))
                uec, pec = energy_setup.make_energies(image_data=d)
                all_nv_vec.append(
                    compute_many_energy_vectors(
                        configurations=pts, image_config=d, ue=uec, pe=pec, energy_names=energy_names))

            all_v_vec = np.concatenate(all_v_vec, axis=0)
            all_nv_vec = np.concatenate(all_nv_vec, axis=0)
            decision_function = weights_model.get_decision_function()
            if len(all_v_vec) > 1000:
                all_v_vec = rng.choice(all_v_vec, size=1000, replace=False, axis=0)
            if len(all_nv_vec) > 1000:
                all_nv_vec = rng.choice(all_nv_vec, size=1000, replace=False, axis=0)
            x = np.concatenate([all_nv_vec, all_v_vec], axis=0)
            y = [0] * len(all_nv_vec) + [1] * len(all_v_vec)
            cross_plot(x, energy_names, labels=y, colors=['tab:red', 'tab:blue'],
                       decision_function=decision_function, label_names=['x-', 'x+'])
            plt.savefig(os.path.join(save_dir, f'cross_{epoch_id:02}.png'))
            plt.close('all')
        except ValueError as e:
            print(f"error displaying cross plot : \n{e}")
            print(f"{len(all_v_vec)} valid vectors ")
            print(f"{len(all_nv_vec)} non_valid_energy_vectors vectors ")

    return weights_model.get_energy_combination_function()
