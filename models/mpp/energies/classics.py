from dataclasses import dataclass
from typing import Callable

import numpy as np
from skimage.draw import draw

from base.shapes.base_shapes import Point
from base.shapes.rectangle import Rectangle
from models.mpp.energies.base_energies import UnitEnergyConstructor
from utils.morpho import dilate


def contrast_measure_lafarge2010(pixels_in: np.array, pixels_out: np.array) -> float:
    """
    see [1]F. Lafarge, G. Gimel’farb, and X. Descombes, “Geometric feature extraction by a multimarked point process,”

    """

    mean_in = np.mean(pixels_in)
    mean_out = np.mean(pixels_out)
    var_in = np.var(pixels_in)
    var_out = np.var(pixels_out)
    area = pixels_out.size + pixels_in.size

    eps = 1e-8

    return np.sqrt((var_out + var_in) / (area * np.square(mean_in - mean_out) + eps))


def contrast_measure_craciun2015(pixels_in: np.array, pixels_out: np.array) -> float:
    """
    see Craciun

    """

    mean_in = np.mean(pixels_in)
    mean_out = np.mean(pixels_out)
    var_in = np.var(pixels_in)
    var_out = np.var(pixels_out)

    eps = 1e-8

    part_1 = ((mean_in - mean_out) ** 2) / (4 * np.sqrt(var_in + var_out))
    part_2 = - 0.5 * np.log(
        (2 * np.sqrt(var_in * var_out)) /
        (var_in + var_out)
    )

    return part_1 + part_2
    # return part_1
    # return part_2

    # return ((mean_in - mean_out) ** 2) / (4 * np.sqrt(var_in + var_out) + eps)


def contrast_measure_debug(pixels_in: np.array, pixels_out: np.array) -> float:
    mean_in = np.mean(pixels_in)
    mean_out = np.mean(pixels_out)
    var_in = np.var(pixels_in)
    var_out = np.var(pixels_out)

    return np.abs(mean_in - mean_out)
    # return -part_2

def contrast_measure_craciunsimple(pixels_in: np.array, pixels_out: np.array) -> float:
    """
    see Craciun

    """

    mean_in = np.mean(pixels_in)
    mean_out = np.mean(pixels_out)
    var_in = np.var(pixels_in)
    var_out = np.var(pixels_out)

    eps = 1e-8

    return ((mean_in - mean_out) ** 2) / (4 * np.sqrt(var_in + var_out) + eps)

    # return ((mean_in - mean_out) ** 2) / (4 * np.sqrt(var_in + var_out) + eps)


def contrast_ttest(pixels_in: np.array, pixels_out: np.array) -> float:
    mean_in = np.mean(pixels_in)
    mean_out = np.mean(pixels_out)
    var_in = np.var(pixels_in)
    var_out = np.var(pixels_out)
    eps = 1e-8

    return np.abs(mean_in - mean_out) / np.sqrt((var_in / pixels_in.size) + (var_out / pixels_out.size) + eps)


def contrast_simple(pixels_in: np.array, pixels_out: np.array) -> float:
    mean_in = np.mean(pixels_in)
    mean_out = np.mean(pixels_out)
    return np.square(mean_in - mean_out)


@dataclass
class ContrastEnergy(UnitEnergyConstructor):
    image: np.ndarray
    dilation: int
    contrast_measure_type: str
    gap: int = 0
    rgb: bool = False
    thresh: float = 0.0
    erode: int = 0
    normalize: bool = False
    q_fun: Callable[[float], float] = None

    def __post_init__(self):
        if not self.rgb:
            self.image_gs = np.mean(self.image, axis=-1)
        else:
            self.image_gs = None
        self.img_shape = self.image.shape[:2]
        if self.contrast_measure_type == 'lafarge':
            self.contrast_measure = contrast_measure_lafarge2010
            self.default_value = 1e1
            self.fac = 1.0
        elif self.contrast_measure_type == 'craciun':
            self.contrast_measure = contrast_measure_craciun2015
            self.fac = -1.0
            self.default_value = 0
        elif self.contrast_measure_type == 'craciun2':
            self.contrast_measure = contrast_measure_craciunsimple
            self.fac = -1.0
            self.default_value = 0
        elif self.contrast_measure_type == 'mean':
            self.contrast_measure = contrast_simple
            self.fac = -1.0
            self.default_value = 0
        elif self.contrast_measure_type == 't-test':
            self.contrast_measure = contrast_ttest
            self.fac = -1.0
            self.default_value = 0
        elif self.contrast_measure_type == 'debug':
            self.contrast_measure = contrast_measure_debug
            self.fac = 1.0
            self.default_value = -1
        else:
            raise ValueError

        if self.normalize:
            self.image = self.image - np.mean(self.image, axis=(0, 1))
            self.image = self.image / np.mean(np.abs(self.image), axis=(0, 1))

    def compute(self, u: Rectangle) -> float:
        fill_mask, rim_mask = self.compute_masks(u)
        if fill_mask.shape[1] == 0:  # no points
            return self.default_value

        assert len(rim_mask[0]) > 0
        if self.rgb:
            val = sum([self.fac * self.contrast_measure(self.image[fill_mask[0], fill_mask[1], c],
                                                        self.image[rim_mask[0], rim_mask[1], c]) for c in
                       range(3)]) - self.thresh
        else:
            val = self.fac * self.contrast_measure(self.image_gs[fill_mask[0], fill_mask[1]],
                                                   self.image_gs[rim_mask[0], rim_mask[1]]) - self.thresh

        if self.q_fun is not None:
            return self.q_fun(val)
        else:
            return val

    def compute_masks(self, u: Rectangle):
        poly = u.poly_coord
        fill_mask = np.array(draw.polygon(poly[:, 0], poly[:, 1], shape=self.img_shape))

        if len(fill_mask[0]) == 0:
            return np.array([[], []]), np.array([[], []])

        if self.erode > 0:
            dilated_mask = dilate(fill_mask.T, self.img_shape, n_iter=2).T
            rim_mask = np.array(list(set(map(tuple, dilated_mask.T)) - set(map(tuple, fill_mask.T)))).T
            rim_dil = dilate(rim_mask.T, self.img_shape, n_iter=self.erode).T
            fill_mask = np.array(list(set(map(tuple, fill_mask.T)) - set(map(tuple, rim_dil.T)))).T

        if len(fill_mask) == 0:
            return np.array([[], []]), np.array([[], []])

        if self.gap > 0:
            dilated_mask_1 = dilate(fill_mask.T, self.img_shape, n_iter=self.gap).T
            dilated_mask_2 = dilate(dilated_mask_1.T, self.img_shape, n_iter=self.dilation).T
            rim_mask = np.array(list(set(map(tuple, dilated_mask_2.T)) - set(map(tuple, dilated_mask_1.T)))).T
        else:
            dilated_mask = dilate(fill_mask.T, self.img_shape, n_iter=self.dilation).T
            rim_mask = np.array(list(set(map(tuple, dilated_mask.T)) - set(map(tuple, fill_mask.T)))).T
        return fill_mask, rim_mask

    def __hash__(self):
        return id(self)


@dataclass
class GradientEnergy(UnitEnergyConstructor):
    image: np.ndarray
    dilation: int = 1
    eps: float = 1e-8
    thresh: float = 0.0
    rgb: bool = False

    def __post_init__(self):
        if not self.rgb:
            image = np.mean(self.image, axis=-1)
        else:
            image = self.image
        self.grad_image = np.array(np.gradient(image, axis=(0, 1)))
        self.grad_image = np.moveaxis(self.grad_image, 0, -1)
        self.img_shape = self.image.shape[:2]

    def compute(self, u: Rectangle) -> float:
        perimeter, normals = self.compute_outline_and_normal(u)

        grad_values = self.grad_image[perimeter[0], perimeter[1]]
        grad_norm = grad_values
        # grad_norm = grad_values / (np.linalg.norm(grad_values, axis=-1, keepdims=True) + self.eps)
        if self.rgb:
            normals = normals.reshape((-1, 1, 2))
        return -abs(float(np.mean(grad_norm * normals))) - self.thresh

    def compute_outline_and_normal(self, u: Rectangle):
        poly = u.poly_coord
        perimeter = np.array(draw.polygon_perimeter(poly[:, 0], poly[:, 1], shape=self.img_shape))
        tangent_1 = np.diff(perimeter.T, axis=0, append=perimeter.T[:1])
        perimeter_flip = np.flip(perimeter, axis=1)
        tangent_2 = np.diff(perimeter_flip.T, axis=0, append=perimeter_flip.T[:1])
        tangent_2 = np.flip(tangent_2, axis=0)
        normals_1 = np.flip(tangent_1, axis=-1) * np.array([[-1, 1]])
        normals_1 = normals_1 / (np.linalg.norm(normals_1, axis=-1, keepdims=True) + self.eps)
        normals_2 = - np.flip(tangent_2, axis=-1) * np.array([[-1, 1]])
        normals_2 = normals_2 / (np.linalg.norm(normals_2, axis=-1, keepdims=True) + self.eps)
        normals_3 = 0.5 * (normals_1 + normals_2)
        return perimeter, normals_3

    def __hash__(self):
        return id(self)
