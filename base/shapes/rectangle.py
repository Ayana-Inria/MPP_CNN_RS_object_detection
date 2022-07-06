from dataclasses import dataclass
from typing import Iterable, Union, Tuple

import numpy as np
from matplotlib import pyplot as plt, patches
from shapely import geometry

from base.shapes.base_shapes import Point


@dataclass(eq=False)
class Rectangle(Point):
    __slots__ = ['size', 'ratio', 'angle']
    size: float
    ratio: float
    angle: float
    PARAMETERS = ['size', 'ratio', 'angle']

    @property
    def length(self) -> float:
        return (2 * self.size) / (1 + self.ratio)

    @property
    def width(self) -> float:
        return self.ratio * self.length

    @property
    def poly_coord(self) -> np.ndarray:
        return rect_to_poly((self.x, self.y), short=self.length, long=self.width,
                            angle=self.angle + np.pi / 2)  # todo WTF ??

    @property
    def poly(self) -> geometry.Polygon:
        return geometry.Polygon(self.poly_coord)

    def __hash__(self):
        return id(self)


def show_rectangles(ax: plt.Axes, points: Iterable[Rectangle], cmap: Union[None, str] = 'jet',
                    fill: bool = False, scores=None, max_score=1.0, min_score=0.0, color=None, **polykwargs):
    if color is not None:
        cmap = None
    for i, p in enumerate(points):
        if cmap is not None:
            if scores is not None:
                v = (scores[i] - min_score) / (max_score - min_score)
            else:
                v = np.random.random()
            c = plt.get_cmap(cmap)(v)
        elif type(color) is not str:
            c = color[i]
        else:
            c = color
        poly = p.poly_coord[:, [1, 0]]
        patch = patches.Polygon(poly, fill=fill, color=c, **polykwargs)
        ax.add_patch(patch)
        if scores is not None:
            tl = np.min(poly, axis=0)
            ax.text(tl[0], tl[1], f"{scores[i]:.2f}", verticalalignment='bottom', color=c, fontsize=10,
                    clip_on=True)


def rotation_matrix(alpha) -> np.ndarray:
    cos, sin = np.cos(alpha), np.sin(alpha)
    return np.array([[cos, -sin], [sin, cos]])


def rect_to_poly(center: Union[Tuple[int, int], np.ndarray], short: float, long: float, angle: float,
                 dilation: int = 0) -> np.ndarray:
    """
    converts rectangle parameters to polygon point coordinates
    Parameters
    ----------
    center : cneter coodinates
    short : length
    long : width
    angle : angle
    dilation : dilation of the shape

    Returns
    -------
    array of coordinates of shape (4,2)

    """
    # centered non rotated coordinates
    poly_coord = np.array([[short / 2 + dilation, long / 2 + dilation],
                           [short / 2 + dilation, - long / 2 - dilation],
                           [-short / 2 - dilation, - long / 2 - dilation],
                           [- short / 2 - dilation, long / 2 + dilation]])
    rot_matrix = rotation_matrix(angle).T
    try:
        rotated = np.matmul(poly_coord, rot_matrix)
        return rotated + center
    except Exception as e:
        print(f"rect_to_poly failed with {e}")
        print(f"{center=},{short=},{long=},{angle=}")
        print(f"{poly_coord=}")
        print(f"{rot_matrix=}")
        raise e


def wla_to_sra(a, b, angle):  # remap to size,ratio,angle
    return (a + b) / 2, a / b, angle


def sra_to_wla(s, r, angle):
    b = (2 * s) / (1 + r)
    return b * r, b, angle


def polygon_to_abw(poly: np.ndarray):
    assert poly.shape == (4, 2)
    norm_axis_1 = np.mean([np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3])])
    norm_axis_2 = np.mean([np.linalg.norm(poly[1] - poly[2]), np.linalg.norm(poly[3] - poly[0])])

    if norm_axis_1 < norm_axis_2:
        a, b = norm_axis_1, norm_axis_2
        axis_vector = np.mean([poly[2], poly[1]], axis=0) - np.mean([poly[0], poly[3]], axis=0)
    else:
        a, b = norm_axis_2, norm_axis_1
        axis_vector = np.mean([poly[1], poly[0]], axis=0) - np.mean([poly[3], poly[2]], axis=0)

    angle = np.arctan2(axis_vector[1], axis_vector[0]) % np.pi

    return a, b, angle
