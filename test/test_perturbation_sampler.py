from functools import partial

from tqdm import tqdm

from base.shapes.rectangle import Rectangle
from models.mpp.calibration.energy_calibration import EnergiesCalibration
from models.mpp.custom_types.image_w_maps import ImageWMaps
from models.mpp.data_loaders import labels_to_rectangles
from models.mpp.energies.energy_setups.energy_setup_legacy import make_energies
from models.mpp.perturbation_sampler import sample_kernel_perturbations, sample_multiple_kernel_perturbations
from models.mpp.point_set.energy_point_set import EPointsSet
from models.mpp.rjmcmc_sampler.kernels.make_kernels import make_kernels
import numpy as np

from models.mpp.train_energy_combination.train_utils import _map_to_images
from models.shape_net.mappings import ValueMapping
from utils.math_utils import normalize


def test_aggregate_perturbations():
    rng = np.random.default_rng(0)

    shape = (256, 256)
    n_samples = 200
    iter_per_point = 2

    image = rng.random(shape)

    param_dist_map = [rng.random(shape + (32,)) for _ in range(3)]

    for i in range(3):
        param_dist_map[i] = normalize(param_dist_map[1], axis=-1, keepdims=True)

    mappings = [
        ValueMapping(32, 0, 32),
        ValueMapping(32, 0, 1),
        ValueMapping(32, 0, np.pi, is_cyclic=True),
    ]

    n_points = 50

    centers = rng.integers((0, 0), (shape[0] - 1, shape[1] - 1), size=(n_points, 2))
    parameters = np.array([[8, 0.5, 0] for _ in range(n_points)])

    labels = {'centers': centers, 'parameters': parameters}

    gt_config = labels_to_rectangles(labels, param_names=Rectangle.PARAMETERS)

    image_data = ImageWMaps(
        name='0000',
        shape=shape,
        image=image, detection_map=image,
        param_dist_maps=param_dist_map,
        mappings=mappings,
        param_names=Rectangle.PARAMETERS,
        labels=labels, gt_config=gt_config
    )

    calibration = EnergiesCalibration(
        detection_threshold=0.5, param_dist_remap_coefs=[1.0, 1.0, 1.0],
        param_dist_remap_intercepts=[0.0, 0.0, 0.0], min_area=4, max_area=30
    )

    uec, pec = make_energies(
        image_data=image_data, energy_cal=calibration
    )
    points = EPointsSet(
        points=image_data.gt_config,
        support_shape=image_data.shape,
        unit_energies_constructors=uec, pair_energies_constructors=pec
    )

    kernels, p_kernels = make_kernels(
        image_data, intensity=1.0, rng=rng
    )

    results = []
    perts = []
    for _ in tqdm(range(n_samples)):
        new_points, perturbations = sample_kernel_perturbations(
            kernels=kernels, p_kernels=p_kernels, points=points, rng=rng, iter_per_point=iter_per_point,
            aggregate_pert=True
        )
        results.append(new_points)
        perts.append(perturbations)

        for added in perturbations.addition:
            assert added not in points
        for removed in perturbations.removal:
            assert removed in points

        delta = points.energy_delta(perturbations)

        e_0 = points.total_energy()

        points_new = points.apply_perturbation(perturbations,inplace=False)
        e_1 = points_new.total_energy()

        assert abs(delta - (e_1 - e_0)) < 1e-8


def test_aggregate_perturbations_2():
    rng = np.random.default_rng(0)

    shape = (256, 256)
    n_samples = 200
    iter_per_point = 2

    image = rng.random(shape)

    param_dist_map = [rng.random(shape + (32,)) for _ in range(3)]

    for i in range(3):
        param_dist_map[i] = normalize(param_dist_map[1], axis=-1, keepdims=True)

    mappings = [
        ValueMapping(32, 0, 32),
        ValueMapping(32, 0, 1),
        ValueMapping(32, 0, np.pi, is_cyclic=True),
    ]

    n_points = 50

    centers = rng.integers((0, 0), (shape[0] - 1, shape[1] - 1), size=(n_points, 2))
    parameters = np.array([[8, 0.5, 0] for _ in range(n_points)])

    labels = {'centers': centers, 'parameters': parameters}

    gt_config = labels_to_rectangles(labels, param_names=Rectangle.PARAMETERS)

    image_data = ImageWMaps(
        name='0000',
        shape=shape,
        image=image, detection_map=image,
        param_dist_maps=param_dist_map,
        mappings=mappings,
        param_names=Rectangle.PARAMETERS,
        labels=labels, gt_config=gt_config
    )

    calibration = EnergiesCalibration(
        detection_threshold=0.5, param_dist_remap_coefs=[1.0, 1.0, 1.0],
        param_dist_remap_intercepts=[0.0, 0.0, 0.0], min_area=4, max_area=30
    )


    kernels, p_kernels = make_kernels(
        image_data, intensity=1.0, rng=rng
    )

    uec, pec = make_energies(image_data=image_data, energy_cal=calibration)

    points = EPointsSet(
        points=image_data.gt_config,
        support_shape=image_data.shape[:2],
        unit_energies_constructors=uec,
        pair_energies_constructors=pec,
    )

    image_data.gt_config_set = points

    partial_func = partial(
        sample_multiple_kernel_perturbations,
        rng=rng,
        iter_per_point=iter_per_point,
        calibration=calibration,
        n_samples=n_samples,
        return_perturbations=True,
        aggregate_pert=True
    )

    image_data = [image_data]

    perturbations_per_image = _map_to_images(partial_func, image_data, False)

    for d, perturbations_list in zip(image_data, perturbations_per_image):

        for pert in perturbations_list:

            for added in pert.addition:
                assert added not in points
            for removed in pert.removal:
                assert removed in points

            delta = points.energy_delta(pert)

            e_0 = points.total_energy()

            points_new = points.apply_perturbation(pert,inplace=False)
            e_1 = points_new.total_energy()

            assert abs(delta - (e_1 - e_0)) < 1e-8