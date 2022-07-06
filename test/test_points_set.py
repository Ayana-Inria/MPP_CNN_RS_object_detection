import numpy as np

from models.mpp.point_set.point_set import PointsSet
from base.shapes.base_shapes import Point

BUNCH_OF_POINTS = [
    Point(0, 0),
    Point(1, 0),
    Point(1, 1),
    Point(3, 0),
    Point(5, 5),
    Point(5, 6),
    Point(8, 0),
]

BUNCH_OF_POINTS_2 = [
    Point(0, 0),
    Point(10, 156),
    Point(89, 56),
    Point(25, 200),
    Point(250, 205),
    Point(189, 4),
    Point(8, 20),
]

R = 3


def test___init__():
    points = PointsSet(
        support_shape=(200, 516),
        maximum_interaction_radius=32
    )
    for u in BUNCH_OF_POINTS:
        points.add(u)

    assert points._n_x == 7
    assert points._n_y == 17

    for p in BUNCH_OF_POINTS:
        assert p in points


def test__find_local_point_set():
    points = PointsSet(
        support_shape=(10, 10),
        maximum_interaction_radius=R
    )
    for u in BUNCH_OF_POINTS:
        points.add(u)

    for u in BUNCH_OF_POINTS:
        s = points._find_local_point_set(u)
        assert u in s


def test_add():
    points = PointsSet(
        support_shape=(10, 10),
        maximum_interaction_radius=3
    )
    for u in BUNCH_OF_POINTS:
        points.add(u)

    u = Point(9, 8)

    points.add(u)

    s = points._find_local_point_set(u)
    assert u in s
    assert u in points


def test_remove():
    points = PointsSet(
        support_shape=(10, 10),
        maximum_interaction_radius=3
    )
    for u in BUNCH_OF_POINTS:
        points.add(u)

    u = BUNCH_OF_POINTS[0]

    points.remove(u)

    s = points._find_local_point_set(u)
    assert u not in s
    assert u not in points


def test_copy():
    points = PointsSet(
        support_shape=(10, 10),
        maximum_interaction_radius=3
    )
    for u in BUNCH_OF_POINTS:
        points.add(u)

    points_2 = points.copy()

    assert points is not points_2
    assert points._local_sets is not points_2._local_sets

    u = Point(9, 8)

    points.add(u)

    assert u in points
    assert u not in points_2


def test___iter__():
    points = PointsSet(
        support_shape=(10, 10),
        maximum_interaction_radius=3
    )
    for u in BUNCH_OF_POINTS:
        points.add(u)

    it = iter(points)

    for _ in BUNCH_OF_POINTS:
        p = next(it)
        assert p in BUNCH_OF_POINTS

    checks = [False for _ in BUNCH_OF_POINTS]
    for p in points:
        assert p in BUNCH_OF_POINTS
        i = BUNCH_OF_POINTS.index(p)
        # print(p)
        checks[i] = True
    assert all(checks)


def test_get_potential_neighbors():
    r = 32
    points = PointsSet(
        support_shape=(256, 256),
        maximum_interaction_radius=r
    )
    rng = np.random.default_rng(0)
    # for u in BUNCH_OF_POINTS:
    for _ in range(1000):
        points.add(Point(rng.integers(0, 255), rng.integers(0, 255)))
        # points.add(u)

    for p1 in points:
        neigh = points.get_potential_neighbors(p1, radius=r, exclude_itself=True)

        for p2 in points:
            if p1 is not p2:
                # dist = np.linalg.norm(p1.get_coord()-p2.get_coord())
                max_dist = np.max(np.abs(p1.get_coord() - p2.get_coord()))
                if p2 in neigh:
                    assert max_dist <= (2 * r)
                else:
                    assert max_dist > r


def test_get_potential_neighbors_2():
    r = 32
    search_radius = [32, 16]
    shape = (147, 222)
    points = PointsSet(
        support_shape=shape,
        maximum_interaction_radius=r
    )
    rng = np.random.default_rng(2)

    points.add(Point(45, 173))

    for _ in range(200):
        coordinates = rng.integers(0, shape)
        points.add(Point(coordinates[0], coordinates[1]))
    for _ in range(100000):
        if rng.random() > 0.5 or len(points) == 0:
            points.add(Point(rng.integers(0, shape[0]), rng.integers(0, shape[0])))
        else:
            p = points.random_choice(rng)
            s_r = rng.choice(search_radius)
            n = list(points.get_potential_neighbors(p, s_r))
            if len(n) > 0:
                points.remove(n[0])
            else:
                points.remove(p)


def test_get_neighbors():
    r = 16
    shape = (147, 222)
    points = PointsSet(
        support_shape=shape,
        maximum_interaction_radius=r
    )
    rng = np.random.default_rng(2)

    points_list = [
        Point(10, 20),
        Point(10, 24),
        Point(14, 21),
        Point(20, 20),
        Point(20, 200)
    ]
    for p in points_list:
        points.add(p)

    neigh = points.get_neighbors(points_list[0], radius=5)
    assert points_list[0] not in neigh
    assert points_list[1] in neigh
    assert points_list[2] in neigh
    assert points_list[3] not in neigh
    assert points_list[4] not in neigh

    neigh = points.get_neighbors(points_list[0], radius=12)
    assert points_list[0] not in neigh
    assert points_list[1] in neigh
    assert points_list[2] in neigh
    assert points_list[3] in neigh
    assert points_list[4] not in neigh


def test_get_neighbors_2():
    from scipy.spatial import distance_matrix
    r_list = [8, 64]
    shape = (128, 128)
    rng = np.random.default_rng(0)
    for r in r_list:
        points = PointsSet(
            support_shape=shape,
            maximum_interaction_radius=r
        )
        for _ in range(200):
            points.add(Point(rng.integers(0, shape[0] - 1), rng.integers(0, shape[1] - 1)))

        for p1 in points:
            neigh = points.get_neighbors(p1, radius=r, exclude_itself=True)

            for p2 in points:
                if p1 is not p2:
                    dist = np.linalg.norm(p1.get_coord() - p2.get_coord())
                    if p2 in neigh:
                        assert dist <= r
                    else:
                        assert dist > r
