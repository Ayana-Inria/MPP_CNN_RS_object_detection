from typing import List, Union

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def divergence(f, sp=None, indexing="xy"):
    """
    Computes divergence of vector field
    f: array -> vector field components [Fx,Fy,Fz,...]
    sp: array -> spacing between points in respecitve directions [spx, spy,spz,...]
    indexing: "xy" or "ij", see np.meshgrid indexing

    """
    num_dims = len(f)
    if sp is None:
        sp = [1.0] * num_dims

    if indexing == "xy":
        return np.ufunc.reduce(np.add, [np.gradient(f[num_dims - i - 1], sp[i], axis=i) for i in range(num_dims)])
    if indexing == "ij":
        return np.ufunc.reduce(np.add, [np.gradient(f[i], sp[i], axis=i) for i in range(num_dims)])


def divergence_map_from_vector_field(vector_field, normalize=True):
    size = vector_field.shape[0]
    xx = np.linspace(0, size, size)
    yy = np.linspace(0, size, size)
    points = [xx, yy]
    sp = [np.diff(p)[0] for p in points]
    if normalize:
        norm = np.linalg.norm(vector_field, axis=-1)  # todo remove this !!
        new_vec = vector_field / np.stack((norm, norm), axis=-1)
        new_vec[np.isnan(new_vec)] = 0
    else:
        new_vec = vector_field
    div = divergence(np.moveaxis(new_vec, 2, 0), sp, 'ij')

    return div


def normalize(array: Union[List, np.ndarray], l: int = 1, axis=None, **kwargs):
    if type(array) is list:
        array = np.array(array)

    return array / np.linalg.norm(array, ord=l, axis=axis, **kwargs)


def f_beta(p, r, beta):
    div = ((beta ** 2 * p) + r)
    return (1 + beta ** 2) * p * r / div if div > 0 else 0