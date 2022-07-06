from __future__ import annotations

from typing import Tuple, List, Set

import numpy as np

from base.shapes.base_shapes import Point

MIN_SPATIAL_RES = 32  # set a minimum for spatial sets to avoid one set perpixel


class PointsSetIterator:
    def __init__(self, points_set: PointsSet):
        self._sets_list = points_set._local_sets
        self._n_sets = len(self._sets_list)
        self._sets_index = 0

        self._set_iterator = None
        self._set_iterator_len = 0
        self._set_iterator_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        # init first
        while self._sets_index < self._n_sets:  # if false, we reached the end
            if self._set_iterator is None:  # initialise new iterator
                local_set = self._sets_list[self._sets_index]
                self._set_iterator = local_set.__iter__()
                self._set_iterator_index = 0
                self._set_iterator_len = len(local_set)

            if self._set_iterator_index < self._set_iterator_len:
                self._set_iterator_index += 1
                return next(self._set_iterator)
            else:  # reached the end of the local set (or it is empty)
                self._set_iterator = None  # throw away iterator
                self._sets_index += 1  # got to next set

        raise StopIteration


# todo rename to PointSet
class PointsSet:
    """
    Set of points, implemented as a spatial hash table
    """

    def __init__(self, support_shape: Tuple[int, int], maximum_interaction_radius: int):
        """

        Parameters
        ----------
        support_shape : shape of the support (usually image.shape)
        maximum_interaction_radius : size of the grid of the spatial hash
        """
        self._spatial_resolution = max(maximum_interaction_radius, MIN_SPATIAL_RES)
        self.support_shape = support_shape
        self._n_x = int(np.ceil(support_shape[0] / self._spatial_resolution))
        self._n_y = int(np.ceil(support_shape[1] / self._spatial_resolution))

        self._local_sets: List[Set[Point]] = [set() for _ in range(self._n_y * self._n_x)]

    def __iter__(self):
        return PointsSetIterator(self)

    def __len__(self):
        n = 0
        for local_set in self._local_sets:
            n += len(local_set)
        return n

    def __copy__(self):
        new = PointsSet(support_shape=self.support_shape,
                        maximum_interaction_radius=self._spatial_resolution)

        # _local_sets is a list of Sets of points, we need to copy the list and the sets, the points don't need to
        # be copied as they don't mutate
        new._local_sets = [s.copy() for s in self._local_sets]
        return new

    def __contains__(self, u: Point):
        return u in self._find_local_point_set(u)

    def _find_local_point_set(self, u: Point) -> Set[Point]:
        """
        Returns the point set the point u belongs to
        Parameters
        ----------
        u : the point you are looking for

        Returns
        -------
        the point set the point u belongs to
        """
        i = u.x // self._spatial_resolution
        j = u.y // self._spatial_resolution
        assert i < self._n_x and j < self._n_y  # Point out of bounds
        return self._local_sets[j + i * self._n_y]

    def add(self, u: Point):
        self._find_local_point_set(u).add(u)

    def remove(self, u: Point):
        self._find_local_point_set(u).remove(u)

    def copy(self) -> PointsSet:
        return self.__copy__()

    def get_potential_neighbors(self, u: Point, radius: float, exclude_itself=True, suppress_warnings=False) -> Set[
        Point]:  # Todo
        """
        uses the spatial hash table structure to return the points that can interact with u
        all points return are within a distance of 0 to 2*radius on each axis (0 to 2*sqrt(2)*radius in terms of
        euclidean norm)
        Parameters
        ----------
        u : a point
        exclude_itself: if True, u will not be returned in the set
        radius : the radius of the neighboring interaction
        suppress_warnings: don't

        Returns
        -------
        The set of points that can interact with u, excluding itself if exclude_itself is true
        """

        max_offset = int(np.ceil(radius / self._spatial_resolution))
        if max_offset > 1:
            if not suppress_warnings:
                print(f"[PointsSet] getting neighbors further than the specified maximum interaction radius")
        interacting_points: Set[Point] = set()
        i_u = u.x // self._spatial_resolution
        j_u = u.y // self._spatial_resolution
        for offset_i in np.arange(-max_offset, max_offset + 1):
            for offset_j in np.arange(-max_offset, max_offset + 1):
                i = i_u + offset_i
                j = j_u + offset_j
                if 0 <= i < self._n_x and 0 <= j < self._n_y:  # check if within bounds
                    interacting_points |= self._local_sets[j + i * self._n_y]
        if exclude_itself:
            return interacting_points - {u}
        else:
            return interacting_points

    def get_neighbors(self, u: Point, radius: float, exclude_itself=True, suppress_warnings=False):
        potential_neighbors = self.get_potential_neighbors(u, radius, exclude_itself, suppress_warnings)
        return {p for p in potential_neighbors if (np.linalg.norm(u.get_coord() - p.get_coord()) <= radius)}

    def _get_i_th_point(self, i: int):
        """
        return the i th point in the set by iterating through all the sets
        #todo make it faster by holding a memory of sets lengths
        Parameters
        ----------
        i : index of the point to return

        Returns
        -------
        the i th point in the set
        """
        k = 0
        for local_set in self._local_sets:
            n = len(local_set)
            if (i - k) >= n:  # the target is further than the current set
                k += n
                continue
            for p in local_set:
                if k == i:
                    return p
                else:
                    k += 1
        raise IndexError

    def random_choice(self, rng: np.random.Generator):
        """

        Returns
        -------
        A point chosen at random in the set
        """
        n = self.__len__()
        chosen = rng.integers(0, n)
        return self._get_i_th_point(i=chosen)

    def get_subsets(self):
        return self._local_sets
