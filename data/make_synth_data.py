import json
import os
import pickle
from typing import List

import numpy as np
from numpy.random import Generator
from skimage.draw import draw
from tqdm import tqdm
import matplotlib.pyplot as plt
from base.shapes.rectangle import Rectangle, polygon_to_abw
from utils.data import get_dataset_base_path
from utils.files import make_if_not_exist, NumpyEncoder


def make_synth(rng: Generator, shape, n_rect, noise):
    random_rectangles = [Rectangle(
        x=rng.integers(0, shape[0]),
        y=rng.integers(0, shape[1]),
        size=rng.normal(8, 1.0),
        ratio=float(np.clip(rng.normal(0.5, 0.1), 0.1, 1)),
        angle=rng.uniform(0, np.pi)
    ) for _ in range(n_rect)]
    valid_rect = []
    for i, r in enumerate(random_rectangles):
        r_poly = r.poly
        intersects = [r_poly.intersection(r2.poly).area for r2 in valid_rect]
        if sum(intersects) == 0:
            valid_rect.append(r)

    random_rectangles = valid_rect

    synth_image = np.ones(shape + (3,)) * 0.5

    for r in random_rectangles:
        poly = r.poly_coord
        poly_draw = draw.polygon(poly[:, 0], poly[:, 1], shape=shape)
        # synth_image[poly_draw] = rng.choice([0, 1.0])
        synth_image[poly_draw] = rng.choice([0, 1.0]) + rng.normal(0, 0.1)
        # synth_image[poly_draw] = rng.choice([0, 1.0]) * np.ones(3) + rng.normal(0, 0.1, size=3)
        # synth_image[poly_draw] = rng.choice([0, 1.0], size=3) + rng.normal(0, 0.4, size=3)
        # synth_image[poly_draw] = rng.uniform(0, 1, size=3)
    synth_image = np.clip(synth_image, 0, 1)
    synth_image = synth_image + rng.normal(0, noise, size=synth_image.shape)
    synth_image = np.clip(synth_image, 0, 1)

    return synth_image, random_rectangles


def make_dataset(subset, save_dir, n_items):
    rng = np.random.default_rng()
    for image_id in tqdm(range(n_items)):
        shape = (256, 256)
        n_rect = 230
        image, rectangles = make_synth(rng, shape, n_rect, noise=0.02)
        rectangles: List[Rectangle]
        n_rect_true = len(rectangles)

        centers = np.array([r.get_coord() for r in rectangles])
        parameters = np.array([polygon_to_abw(r.poly_coord) for r in rectangles])
        categories = ['vehicle' for _ in centers]
        difficult = [False] * len(centers)

        plt.imsave(os.path.join(save_dir, 'images', f"{image_id:04}.png"), image)
        with open(os.path.join(save_dir, 'annotations', f"{image_id:04}.pkl"), 'wb') as f:
            pickle.dump(
                {'centers': centers, 'parameters': parameters, 'categories': categories, 'difficult': difficult},
                f)
        with open(os.path.join(save_dir, 'metadata', f"{image_id:04}.json"), 'w') as f:
            json.dump({
                'shape': list(image.shape),
                'n_objects': n_rect_true
            }, f, cls=NumpyEncoder, indent=1)


def make_synth_dataset():
    dest_base = get_dataset_base_path()
    name = 'synth_01'

    save_dir = os.path.join(dest_base, name)

    make_if_not_exist(save_dir)

    subsets = ['train', 'val']

    sub_folders = ['raw_images', 'images', 'raw_annotations', 'annotations', 'metadata', 'images_w_annotations']

    for ss in subsets:
        subset_save_dir = os.path.join(save_dir, ss)
        make_if_not_exist(subset_save_dir)
        make_if_not_exist([os.path.join(subset_save_dir, s) for s in sub_folders])

        make_dataset(
            subset=ss,
            save_dir=subset_save_dir,
            n_items=32
        )


if __name__ == '__main__':
    make_synth_dataset()
