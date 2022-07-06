from typing import Tuple
import numpy as np


def sample_point_2d(img_shape: Tuple[int, int], size: int = 1, density: np.ndarray = None,
                    skip_normalization: bool = False,
                    rng: np.random.Generator = None,
                    mask: np.ndarray = None) -> np.ndarray:
    """
    Samples point(s) in a rectangular image according to a specified density. If None is specified then points are drawn
    uniformly
    Parameters
    ----------
    img_shape : shape of rectangle to draw coordinates in
    size : number of samples to draw
    density : density map
    skip_normalization : skip normalization step for the provided density, in that case , must have np.sum(density) == 1

    Returns
    -------
    return coordinates array of shape (size,2)
    """

    if rng is None:
        rng = np.random

    if density is None:
        if mask is None:
            coor_x = rng.choice(np.arange(0, img_shape[0]), size=size)
            coor_y = rng.choice(np.arange(0, img_shape[1]), size=size)
        else:
            density = mask / np.sum(mask)
            ind = rng.choice(np.arange(img_shape[0] * img_shape[1]), p=density.reshape(-1), size=size,
                             replace=False)
            coor_x = ind // img_shape[1]
            coor_y = ind % img_shape[1]
    else:
        if not skip_normalization:
            density = density / np.sum(density)
        if mask is not None:
            density[mask] = 0
            density = density / np.sum(density)
        ind = rng.choice(np.arange(img_shape[0] * img_shape[1]), p=density.reshape(-1), size=size,
                         replace=False)
        coor_x = ind // img_shape[1]
        coor_y = ind % img_shape[1]

    return np.array([coor_x, coor_y]).T
