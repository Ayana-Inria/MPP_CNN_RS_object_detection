import matplotlib.pyplot as plt
import numpy as np

CONNECT9 = np.array([[[i, j] for i in [-1, 0, 1]] for j in [-1, 0, 1]]).reshape((-1, 2))
CONNECT5 = np.array([[-1, 0], [1, 0], [0, 0], [0, -1], [0, 1]])


def dilate(coordinates: np.ndarray, image_shape, n_iter: int = 1):
    assert coordinates.shape[1] == 2

    shifted_points = coordinates.reshape((coordinates.shape[0], 1, 2)) + CONNECT5
    new_pos = np.array(list(set().union(*[set(map(tuple, shifted_points[:, i])) for i in range(5)])))
    new_pos = new_pos[np.all(new_pos >= 0, axis=1) & np.all(new_pos < np.array(image_shape), axis=1)]

    if n_iter == 1:
        return new_pos
    else:
        return dilate(new_pos, image_shape, n_iter - 1)


def verif():
    rng = np.random.default_rng(0)
    image = rng.random(size=(32, 32)) > 0.75
    positions = np.array(np.where(image)).T
    image_shape = image.shape

    n_neighbors = 4

    shifted_points = positions.reshape((positions.shape[0], 1, 2)) + CONNECT5

    new_pos = np.array(list(set().union(*[set(map(tuple, shifted_points[:, i])) for i in range(n_neighbors + 1)])))
    new_pos = new_pos[np.all(new_pos >= 0, axis=1) & np.all(new_pos < np.array(image_shape), axis=1)]

    new_image = np.zeros(image.shape)
    new_image[new_pos[:, 0], new_pos[:, 1]] = 1

    fig, axs = plt.subplots(1, 3)
    axs[0].set_title("input")
    axs[0].imshow(image)
    axs[1].set_title("set dilation")
    axs[1].imshow(new_image)
    axs[2].set_title("reference method")
    from scipy.ndimage import binary_dilation
    axs[2].imshow(binary_dilation(image))

    plt.show()


if __name__ == '__main__':
    verif()
