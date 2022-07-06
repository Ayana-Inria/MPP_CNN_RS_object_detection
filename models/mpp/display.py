import logging
from typing import List, Callable

import numpy as np
from matplotlib import pyplot as plt

from base.shapes.rectangle import Rectangle, show_rectangles
from models.mpp.custom_types.image_w_maps import ImageWMaps


def show_energy_iso(ax: plt.Axes, i, j, function, n_dim, return_contourf=False, **contour_kwargs):
    """
    Displays a heatmap/contour of a decision function that take n_dim-dimentional vectors as input,
    for a i,j show a heatmap of the deicision function in function of i and j with other parameters fixed
    Parameters
    ----------
    ax : the pyplot axis to display the contour
    i : id of first param
    j : id of second param
    function : decision function (for instance a smv.decision_function)
    n_dim : vector dimension of the input of the decision function
    """
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()
    x = np.linspace(min_x, max_x, 100)
    y = np.linspace(min_y, max_y, 100)
    xx, yy = np.meshgrid(x, y)
    arr = np.zeros(xx.shape + (n_dim,))
    arr[:, :, j] = xx
    arr[:, :, i] = yy
    shape = arr.shape[:2]
    Z = -function(arr.reshape((-1, n_dim))).reshape(shape)
    im = ax.contourf(xx, yy, -Z, levels=10, alpha=0.6, cmap='coolwarm', **contour_kwargs)
    if return_contourf:
        return im


def cross_plot(vectors: np.ndarray, dim_names: List[str], labels, label_names: List[str], colors,
               decision_function: Callable = None):
    assert len(vectors) == len(labels)
    n_dims = vectors.shape[1]
    n_rows, n_cols = n_dims, n_dims
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3), squeeze=False)

    color_per_point = [colors[l] for l in labels]

    for i in range(n_dims):
        for j in range(n_dims):
            ax: plt.Axes = axs[i, j]
            if j != i:
                ax.scatter(vectors[:, j], vectors[:, i], color=color_per_point, zorder=10, s=1.0)
                if decision_function is not None:
                    show_energy_iso(ax, i, j, decision_function, n_dims, zorder=0)
            else:
                try:
                    ax.hist([vectors[labels == c, i] for c in np.sort(np.unique(labels))],
                            density=True, label=label_names,
                            color=colors)
                    ax.legend()
                except ValueError as e:
                    logging.warning(f"unable to display hist because {e}")

            if i == (n_dims - 1):
                ax.set_xlabel(dim_names[j])
            if j == 0:
                ax.set_ylabel(dim_names[i])


def show_some_configurations(image_data: List[ImageWMaps],
                             configurations: List[List[Rectangle]]):
    n_cols = min(4, len(image_data))
    n_rows = int(np.ceil(len(image_data) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False)
    axs = axs.ravel()

    for i in range(0, len(image_data)):
        axs[i].imshow(image_data[i].image)
        show_rectangles(axs[i], image_data[i].gt_config, cmap=None, color='tab:blue')
        show_rectangles(axs[i], configurations[i], cmap=None, color='tab:red')
        axs[i].set_title(f"image {image_data[i].name}")
        axs[i].axis('off')
    fig.tight_layout()
