from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from skimage import draw


@dataclass(eq=False)  # 3.10 has slots=True
class Point(object):
    __slots__ = ['x', 'y']
    x: int
    y: int

    def __hash__(self):
        return id(self)

    def get_coord(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def draw(self, support_shape: Tuple[int, int], dilation: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a coordinates tuple such that (for instance) image[point.draw()]=1 sets the pixels belonging to that
        point to 1
        Parameters
        ----------
        dilation : dilation of the mask
        support_shape : shape of the supporting image
        """
        if dilation != 0:
            return draw.disk((self.x, self.y), radius=dilation, shape=support_shape)
        else:
            coor = self.get_coord()
            return coor[0], coor[1]


@dataclass(eq=False)
class Circle(Point):
    __slots__ = ['radius']
    radius: float

    def draw(self, support_shape: Tuple[int, int], dilation: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a coordinates tuple such that (for instance) image[point.draw()]=1 sets the pixels belonging to that
        point to 1
        Parameters
        ----------
        dilation : dilation of the mask
        support_shape : shape of the supporting image
        """
        return draw.disk((self.x, self.y), radius=self.radius + dilation, shape=support_shape)
