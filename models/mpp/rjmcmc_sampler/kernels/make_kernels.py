from typing import List

import numpy as np
from numpy.random import Generator

from models.mpp.custom_types.image_w_maps import ImageWMaps
from models.mpp.rjmcmc_sampler.kernels.base_kernels import BirthKernel, DeathKernel, Kernel
from models.mpp.rjmcmc_sampler.kernels.shape_samplers import RectangleSampler, UniformRectangleSampler
from models.mpp.rjmcmc_sampler.kernels.transform_kernels import DataDrivenTranslationKernel, \
    GaussianShapeTransformKernel, GaussianTranslationKernel, DataDrivenShapeTransformKernel
from utils.math_utils import normalize

BASE_KERNEL_WEIGHTS = {
    "bd_weight": 1,
    "uniform_bd_weight": 1,
    "data_bd_weight": 2,
    "ms_weight": 1,
    "translation_weight": 1,
    "gaussian_translation_weight": 1,
    "data_translation_weight": 2,
    "transformation_weight": 1,
    "gaussian_transformation_weight": 1,
    "data_transformation_weight": 2
}

"""
kernel choice decision tree (probas set above are rescale to sum to 1)
├── B&D (birth & death) : p = bd_weight
|   ├── Uniform B&D : p = uniform_bd_weight
|   |   ├── Birth : p = 0.5
|   |   └── Death : p = 0.5
|   └── Data Driven B&D : p = data_bd_weight
|       ├── Birth : p = 0.5
|       └── Death : p = 0.5
├── Split and Merge : p = ms_weight
|   ├── Birth : p = 0.5
|   └── Death : p = 0.5
├── Translation : p = translation_weight
|   ├── Gaussian Translation : p = gaussian_translation_weight
|   └── Data Driven Translation : p = data_translation_weight
└── Transformation : p = transformation_weight
    ├── Gaussian Transformation : p = gaussian_transformation_weight
    └── Data Driven Transformation : p = data_transformation_weight



"""


def make_kernels(image_data: ImageWMaps, intensity: float, rng: Generator, use_split_merge: bool = False,
                 kernel_weights=None):
    if kernel_weights is None:
        kernel_weights = BASE_KERNEL_WEIGHTS
    shape = image_data.detection_map.shape[:2]

    birth_map_sampler = RectangleSampler(
        detection_map=image_data.detection_map,
        param_dist_maps=image_data.param_dist_maps,
        param_names=image_data.param_names,
        mappings=image_data.mappings,
        rng=rng
    )

    uniform_sampler = UniformRectangleSampler(
        shape=image_data.shape,
        param_names=image_data.param_names,
        mappings=image_data.mappings,
        rng=rng
    )

    # intensity = np.sum(detection_map > detection_threshold)
    # intensity = 1 * np.prod(shape) * (32**3)
    # actual_intensity = intensity / (np.prod(shape) * np.prod([m.n_classes for m in image_data.mappings]))
    actual_intensity = intensity
    if use_split_merge:
        p_bd, p_ms, p_trl, p_trf = normalize(
            [kernel_weights[k] for k in ['bd_weight', 'ms_weight', 'translation_weight', 'transformation_weight']])
    else:
        p_bd, p_trl, p_trf = normalize(
            [kernel_weights[k] for k in ['bd_weight', 'translation_weight', 'transformation_weight']])
        p_ms = None
    p_bd_unif, p_bd_data = normalize([kernel_weights[k] for k in ['uniform_bd_weight', 'data_bd_weight']])
    p_trl_gaus, p_trl_data = normalize(
        [kernel_weights[k] for k in ['gaussian_translation_weight', 'data_translation_weight']])
    p_trf_gaus, p_trf_data = normalize(
        [kernel_weights[k] for k in ['gaussian_transformation_weight', 'data_transformation_weight']])

    kernels: List[Kernel] = [
        # Uniform Birth Kernel
        BirthKernel(
            p_birth=0.5 * p_bd_unif * p_bd,
            p_death=0.5 * p_bd_unif * p_bd,
            sampler=uniform_sampler,
            intensity=actual_intensity
        ),
        # Uniform Death Kernel
        DeathKernel(
            p_birth=0.5 * p_bd_unif * p_bd,
            p_death=0.5 * p_bd_unif * p_bd,
            sampler=uniform_sampler,
            intensity=actual_intensity
        ),
        # Data Driven Birth Kernel
        BirthKernel(
            p_birth=0.5 * p_bd_data * p_bd,
            p_death=0.5 * p_bd_data * p_bd,
            sampler=birth_map_sampler,
            intensity=actual_intensity,
        ),
        # Data Driven Death Kernel
        DeathKernel(
            p_birth=0.5 * p_bd_data * p_bd,
            p_death=0.5 * p_bd_data * p_bd,
            sampler=birth_map_sampler,
            intensity=actual_intensity,
        ),
        # Gaussian Translation Kernel
        GaussianTranslationKernel(
            p_kernel=p_trl * p_trl_gaus,
            sigma=2,
            shape=shape
        ),
        # Data Driven Translation Kernel
        DataDrivenTranslationKernel(
            p_kernel=p_trl * p_trl_data,
            max_delta=8,
            normalised_density=birth_map_sampler.normalised_detection_map
        ),
        # Gaussian Shape Transform Kernel
        GaussianShapeTransformKernel(
            p_kernel=p_trf * p_trf_gaus,
            sigma=0.1,
            param_names=image_data.param_names,
            mappings=image_data.mappings
        ),
        # Data Driven Shape Transform Kernel
        DataDrivenShapeTransformKernel(
            p_kernel=p_trf * p_trf_data,
            params_density_maps=image_data.param_dist_maps,
            param_names=image_data.param_names,
            mappings=image_data.mappings,
            re_normalize=True
        )
    ]
    if use_split_merge:
        from models.mpp.rjmcmc_sampler.kernels.split_and_merge_kernels import SplitKernel, MergeKernel, SplitSampler
        radius = 16
        split_sampler = SplitSampler(
            pos_radius=radius, shape_sigmas=[0.1, 0.1, 0.1], mappings=image_data.mappings
        )
        
        kernels = kernels + [
            SplitKernel(
                p_split=p_ms * 0.5, p_merge=p_ms * 0.5, split_sampler=split_sampler,
                support_shape=image_data.shape[:2], intensity=intensity, merge_radius=radius
            ),
            MergeKernel(
                p_split=p_ms * 0.5, p_merge=p_ms * 0.5, split_sampler=split_sampler,
                support_shape=image_data.shape[:2], intensity=intensity, merge_radius=radius
            )
        ]

    p_kernels = [k.p_kernel for k in kernels]

    if abs(1 - np.sum(p_kernels)) < 1e-8:  # expect some numerical errors
        p_kernels = np.array(p_kernels) / np.sum(p_kernels)

    assert len(kernels) == len(p_kernels)
    try:
        assert abs(1 - np.sum(p_kernels)) < 1e-8
    except AssertionError as e:
        print(f"probabilities do no sum to 1")
        print(p_kernels)
        print(f"sum = {np.sum(p_kernels)}")
        raise e

    return kernels, p_kernels
