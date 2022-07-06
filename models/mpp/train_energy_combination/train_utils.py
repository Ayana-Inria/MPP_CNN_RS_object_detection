from multiprocessing import Pool
from typing import Callable, List, Union

from torch.nn import Module

from models.mpp.custom_types.image_w_maps import ImageWMaps
from models.mpp.energies.combination.base import WeightModel
from models.mpp.energies.energy_utils import EnergySetup


def _map_to_images(func: Callable, image_data: List[ImageWMaps], multiprocess: bool):
    if multiprocess:
        with Pool() as p:
            results = p.map(func, image_data)
    else:
        results = [func(i) for i in image_data]

    return results


def init_model(weight_model_type: str, energy_setup: EnergySetup, **kwargs) -> Union[WeightModel, Module]:
    if weight_model_type == 'hierarchical':
        from models.mpp.energies.combination.hierarchical import HierarchicalEnergyModel
        weights_model = HierarchicalEnergyModel(
            threshold=0.0,
            **kwargs['weights_model_params'])  # since the detection threshold has been calibrated to have
    elif weight_model_type == 'mlp':
        from models.mpp.energies.combination.mlp import MLPEnergyModel
        weights_model = MLPEnergyModel(**kwargs['mlp_params'], energy_names=energy_setup.energy_names)
    elif weight_model_type == 'linear':
        from models.mpp.energies.combination.linear import LinearEnergyModel
        weights_model = LinearEnergyModel(**kwargs['weights_model_params'])
    elif weight_model_type == 'logistic':
        from models.mpp.energies.combination.logistic import LogisticEnergyModel
        weights_model = LogisticEnergyModel(use_bias=True, energy_names=energy_setup.energy_names)
    elif weight_model_type == 'loghrc':
        from models.mpp.energies.combination.logistic import HierarchicalLogisticEnergyModel
        weights_model = HierarchicalLogisticEnergyModel()
    else:
        raise ValueError  # weight_model_type is not a known model
    return weights_model
