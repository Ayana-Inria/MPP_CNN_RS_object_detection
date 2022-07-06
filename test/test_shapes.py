import numpy as np

from base.shapes.base_shapes import Point, Circle


def test_point_get_coord():
    p = Point(2, 8)
    assert np.all(p.get_coord() == np.array([2, 8]))


def test_point_draw():
    p = Point(2, 8)
    c1 = p.draw(support_shape=(10, 10))
    assert np.all(c1 == (2, 8))
    for r in [0, 1, 2, 3, 4]:
        image = np.zeros((20, 20))
        c2 = p.draw(dilation=r, support_shape=image.shape)
        image[c2] = 1
        for x, y in np.argwhere(image > 0):
            assert np.linalg.norm(np.array([2, 8] - np.array([x, y]))) <= r


def test_circle_draw():
    for r in [1, 2, 3, 4, 5, 6]:
        p = Circle(2, 8, r)
        for d in [0, 2, 4]:
            image = np.zeros((20, 20))
            c2 = p.draw(support_shape=image.shape, dilation=d)
            image[c2] = 1
            for x, y in np.argwhere(image > 0):
                assert np.linalg.norm(np.array([2, 8] - np.array([x, y]))) <= (r + d)
