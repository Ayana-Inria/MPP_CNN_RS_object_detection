import os
from functools import partial
from typing import Generator, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from models.mpp.custom_types.image_w_maps import ImageWMaps
from models.mpp.display import cross_plot, show_some_configurations
from models.mpp.energies.energy_utils import compute_many_energy_vectors, EnergySetup
from models.mpp.train_energy_combination.train_utils import _map_to_images, init_model
from utils.logger import Logger


def train_integral_criterion(train_loader: DataLoader, rng: Generator,
                             logger: Logger,energy_setup: EnergySetup,
                             samples_per_image: int, n_epochs: int, save_dir: str,
                             neg_sampling_method: str = 'rjmcmc', pos_sampling_method: str = 'single',
                             reg_weight=None, optim: str = 'adam', lr_scheduler: bool = False,
                             learning_rate=1e-1, weight_model_type='hierarchical',
                             multiprocess=True, **kwargs):
    energy_names = energy_setup.energy_names

    weights_model = init_model(weight_model_type, **kwargs)

    print(f'initial weights: {weights_model.as_dict()}')

    if optim == 'adam':
        optim = Adam(params=weights_model.parameters(), lr=learning_rate)
    elif optim == 'sgd':
        optim = SGD(params=weights_model.parameters(), lr=learning_rate)

    scheduler = None
    if lr_scheduler:
        scheduler = ExponentialLR(optim, **kwargs["lr_scheduler_params"])

    n_batches = len(train_loader)

    for epoch_id in range(n_epochs):

        all_v_vec = []
        all_nv_vec = []
        # ------------------------ start train epoch
        for batch_id, image_data in enumerate(train_loader):
            image_data: List[ImageWMaps]
            optim.zero_grad()
            batch_len = len(image_data)

            # generate configs from current W
            if neg_sampling_method == 'rjmcmc':
                energy_combinator = weights_model.get_energy_combination_function()
                from models.mpp.rjmcmc_sampler.sample_rjmcmc import sample_rjmcmc
                partial_func = partial(
                    sample_rjmcmc,
                    init_config='gt',
                    rng=rng,
                    energy_combinator=energy_combinator,
                    num_samples=samples_per_image,
                    **kwargs['rjmcmc_params']
                )
                neg_samples_per_image = samples_per_image
                neg_configurations = _map_to_images(partial_func, image_data, multiprocess)
            elif neg_sampling_method == 'perturbation':
                from models.mpp.perturbation_sampler import sample_perturbations
                partial_func = partial(
                    sample_perturbations,
                    rng=rng,
                    **kwargs['neg_pert_config'],
                    n_samples=samples_per_image
                )
                neg_samples_per_image = samples_per_image
                neg_configurations = _map_to_images(partial_func, image_data, multiprocess)
            elif neg_sampling_method == 'kernel':
                from models.mpp.perturbation_sampler import sample_multiple_kernel_perturbations
                partial_func = partial(
                    sample_multiple_kernel_perturbations,
                    rng=rng,
                    **kwargs['neg_pert_config'],
                    n_samples=samples_per_image
                )
                neg_samples_per_image = samples_per_image
                neg_configurations = _map_to_images(partial_func, image_data, multiprocess)
            else:
                raise ValueError  # neg_samples_per_image must be 'rjmcmc' or 'perturbation'

            if pos_sampling_method == 'single':
                pos_configurations = [[d.gt_config] for d in image_data]
                pos_samples_per_image = 1
            elif pos_sampling_method == 'perturbation':
                from models.mpp.perturbation_sampler import sample_perturbations
                partial_func = partial(
                    sample_perturbations,
                    rng=rng,
                    **kwargs['pos_pert_config'],
                    n_samples=samples_per_image
                )
                pos_configurations = _map_to_images(partial_func, image_data, multiprocess)
                pos_samples_per_image = samples_per_image

            elif pos_sampling_method == 'kernel':
                from models.mpp.perturbation_sampler import sample_multiple_kernel_perturbations
                partial_func = partial(
                    sample_multiple_kernel_perturbations,
                    rng=rng,
                    **kwargs['pos_pert_config'],
                    n_samples=samples_per_image
                )
                pos_configurations = _map_to_images(partial_func, image_data, multiprocess)
                pos_samples_per_image = samples_per_image
            else:
                raise ValueError  # pos_sampling_method must be 'single' or 'perturbation'

            valid_energy_vectors = []
            non_valid_energy_vectors = []
            for i, d in enumerate(image_data):
                ue,pe = energy_setup.make_energies(image_data=d)
                v = compute_many_energy_vectors(
                    configurations=pos_configurations[i],
                    image_config=d,
                    multiprocess=True,
                    energy_names=energy_names,
                    ue=ue,pe=pe
                )
                nv = compute_many_energy_vectors(
                    configurations=neg_configurations[i],
                    image_config=d,
                    multiprocess=True,
                    energy_names=energy_names,
                    ue=ue, pe=pe
                )
                valid_energy_vectors.append(v)
                non_valid_energy_vectors.append(nv)

            valid_energy_vectors = np.concatenate(valid_energy_vectors, axis=0)
            non_valid_energy_vectors = np.concatenate(non_valid_energy_vectors, axis=0)
            all_v_vec.append(valid_energy_vectors)
            all_nv_vec.append(non_valid_energy_vectors)

            x_plus = torch.tensor(valid_energy_vectors, requires_grad=False)
            x_minus = torch.tensor(non_valid_energy_vectors, requires_grad=False)

            E_plus = torch.div(weights_model.forward(x_plus), pos_samples_per_image)
            E_minus = torch.div(weights_model.forward(x_minus), neg_samples_per_image)

            if reg_weight is not None and reg_weight != 0.0:
                reg = reg_weight * weights_model.regularisation_term(E_plus=E_plus, E_minus=E_minus)
            else:
                reg = 0

            loss = E_plus - E_minus + reg

            loss.backward()
            optim.step()

            if type(reg) is Tensor:
                reg = reg.detach().cpu()

            log = {
                'batch': batch_id,
                'loss': float(loss.cpu().detach()),
                'e_plus': float(E_plus.cpu().detach()),
                'n_e_plus': len(valid_energy_vectors) / pos_samples_per_image,
                'e_minus': float(E_minus.cpu().detach()),
                'n_e_minus': len(non_valid_energy_vectors) / neg_samples_per_image,
                'reg': float(reg),
                'lr': learning_rate if scheduler is None else scheduler.get_last_lr()[0],
                **weights_model.as_dict()
            }
            logger.update(epoch=epoch_id, metrics=log)

            print(f"[epoch {epoch_id + 1}/{n_epochs}][batch {batch_id + 1}/{n_batches}] "
                  f"loss: {log['loss']:.4f} "
                  f"e_plus: {log['e_plus']:.4f} "
                  f"e_minus: {log['e_minus']:.4f} "
                  f"lr {log['lr']:.4f} "
                  f"reg {log['reg']:.4f}")

        # ----------------- end train epoch
        if scheduler is not None:
            scheduler.step()
        try:
            show_some_configurations(image_data, [r[-1] for r in neg_configurations])
            plt.savefig(os.path.join(save_dir, f'nv_samples_{epoch_id:02}.png'))

            show_some_configurations(image_data, [r[-1] for r in pos_configurations])
            plt.savefig(os.path.join(save_dir, f'v_samples_{epoch_id:02}.png'))
        except Exception as e:
            print(f'printing of some training data failed because :{e}')
            print('skipping')

        try:
            fig, axs = plt.subplots(1, 1, figsize=(8, 4))
            axs: plt.Axes
            axs.set_xlabel('epoch')
            xx = np.array(range(len(logger.log['loss']))) / n_batches
            axs.plot(xx, logger.log['loss'], label='loss')
            axs.plot(xx, logger.log['e_plus'], label='e_plus')
            axs.plot(xx, logger.log['e_minus'], label='e_minus')
            if reg_weight is not None:
                axs.plot(xx, logger.log['reg'], label='reg')
            axs.legend()
            plt.savefig(os.path.join(save_dir, f'losses_plot.png'))
            plt.close('all')

            fig, axs = plt.subplots(1, 1, figsize=(8, 4))
            axs: plt.Axes
            axs.set_xlabel('epoch')
            xx = np.array(range(len(logger.log['loss']))) / n_batches
            n_plus = np.array(logger.log['n_e_plus'])
            n_minus = np.array(logger.log['n_e_minus'])
            axs.plot(xx, np.array(logger.log['loss']) / (n_plus + n_minus), label='loss')
            axs.plot(xx, np.array(logger.log['e_plus']) / n_plus, label='e_plus')
            axs.plot(xx, np.array(logger.log['e_minus']) / n_minus, label='e_minus')
            if reg_weight is not None:
                axs.plot(xx, logger.log['reg'] / (n_plus + n_minus), label='reg')
            axs.legend()
            plt.savefig(os.path.join(save_dir, f'losses_plot_norm.png'))
            plt.close('all')

            fig, axs = plt.subplots(1, 1, figsize=(8, 4))
            axs: plt.Axes
            axs.set_xlabel('epoch')
            for key in logger.log.keys():
                if '_weight' in key:
                    axs.plot(xx, logger.log[key], label=key)
            axs.legend()
            plt.savefig(os.path.join(save_dir, f'weights_plot.png'))
            plt.close('all')
        except Exception as e:
            print(f"Exception occurred while making figures, continuing ")
            print(e)

        print(weights_model.as_dict())
        try:
            all_v_vec = np.concatenate(all_v_vec, axis=0)
            all_nv_vec = np.concatenate(all_nv_vec, axis=0)
            decision_function = weights_model.get_decision_function()
            if len(all_v_vec) > 1000:
                all_v_vec = rng.choice(all_v_vec, size=1000, replace=False, axis=0)
            if len(all_nv_vec) > 1000:
                all_nv_vec = rng.choice(all_nv_vec, size=1000, replace=False, axis=0)
            x = np.concatenate([all_v_vec, all_nv_vec], axis=0)
            y = [1] * len(all_v_vec) + [0] * len(all_nv_vec)
            cross_plot(x, energy_names, labels=y, colors=['tab:red', 'tab:blue'],
                       decision_function=decision_function, label_names=['x-', 'x+'])
            plt.savefig(os.path.join(save_dir, f'cross_{epoch_id:02}.png'))
            plt.close('all')
        except ValueError as e:
            print(f"error displaying cross plot : \n{e}")
            print(f"{len(all_v_vec)} valid vectors ")
            print(f"{len(all_nv_vec)} non_valid_energy_vectors vectors ")

    return weights_model.get_energy_combination_function()
