import os
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor


from utils.display.light_display.image_stack import make_image_from_bunch, add_top_legend, add_left_legend, add_header
from utils.display.light_display.plot import plot_stuff


def remap(x):
    return 0.5 * x / max(abs(x.max()), abs(x.min())) + 0.5


def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


def display_pos_inference(epoch: int,model, images: Tensor, output: Tensor, labels: Dict, loss: Dict,
                          results_path: str, subset=None, learn_mask=False, thickness=1, metrics=None, colors=None):
    images = images.permute((0, 2, 3, 1)).numpy()

    image_size = images.shape[1]

    if images.shape[3] == 1:
        images = np.stack([images[:, :, :, 0] for _ in range(3)], axis=-1)

    target = labels['pointing_map'].detach().permute((0, 2, 3, 1)).cpu().numpy()

    target_color = np.full(images.shape, 0.5)

    target_color[:, :, :, 0] = remap(target[:, :, :, 0])
    target_color[:, :, :, 2] = remap(target[:, :, :, 1])

    output = output.detach().permute((0, 2, 3, 1)).cpu().numpy()
    output_vec = output[:, :, :, :2]
    output_vec_col = np.full(images.shape, 0.5)
    output_vec_col[:, :, :, 0] = remap(output_vec[:, :, :, 0])
    output_vec_col[:, :, :, 2] = remap(output_vec[:, :, :, 1])

    target_center_bin_map = labels['center_binary_map_dil'].detach().cpu().numpy()
    # if thickness != 1:
    #     target_center_bin_map = np.stack(
    #         [morphology.dilation(cbm, morphology.disk(radius=thickness - 1)) for cbm in target_center_bin_map],
    #         axis=0
    #     )
    target_center_bin_map = plt.get_cmap('magma')(target_center_bin_map.astype(float))[:, :, :, :3]

    if learn_mask:
        output_mask = output[:, :, :, 2]
        output_mask = sigmoid(output_mask)
        output_mask_col = plt.get_cmap('gray')(output_mask)[:, :, :, :3]
        target_mask_col = plt.get_cmap('gray')(labels['mask'].detach().cpu().numpy())[:, :, :, :3]

    all_d_maps = []
    for i in range(8):
        detection_energy = model.vec2detection_map(
            vector_map=output_vec[i],
            mask=output_mask[i]
        )
        density_map_col = plt.get_cmap('plasma')(detection_energy)[:, :, :3]

        # coor_map = np.stack(np.mgrid[:image_size, :image_size], axis=-1)
        # shifted: np.ndarray = coor_map + output_vec[i]
        #
        # shifted_int = shifted.astype(int)
        # shifted_int = shifted_int.reshape((-1, 2))
        # unique_coord, counts = np.unique(shifted_int, return_counts=True, axis=0)
        # mask = np.min(np.logical_and(unique_coord < image_size, unique_coord >= 0), axis=-1)
        # unique_coord = unique_coord[mask]
        # counts = counts[mask]
        # density_map = np.zeros((image_size, image_size))
        # density_map[unique_coord[:, 0], unique_coord[:, 1]] = counts
        # density_map = density_map / np.max(density_map)
        # density_map_col = plt.get_cmap('viridis')(density_map)[:, :, :3]
        # if learn_mask:
        #     mask = output_mask[i] < 0.5
        #     density_map_col[mask] = 0.5

        all_d_maps.append(density_map_col)
    all_d_maps = np.stack(all_d_maps, axis=0)

    if learn_mask:
        all_figs = (
            images, target_color, target_mask_col, output_vec_col, output_mask_col, all_d_maps, target_center_bin_map)
        legend = ['input', 'target', 'target_mask', 'output', 'output_mask', 'est. energy', 'target_centers']
    else:
        all_figs = (images, target_color, output_vec_col, all_d_maps, target_center_bin_map)
        legend = ['input', 'target', 'output', 'est. energy', 'target_centers']
    n_figs = len(all_figs)
    images = np.concatenate(all_figs, axis=0)

    if subset is not None:
        images = np.concatenate([images[[j * 8 + i for j in range(n_figs)]] for i in subset])

        big_image = make_image_from_bunch(images, nrow=n_figs)

        big_image, _ = add_top_legend(
            big_image,
            texts=legend)
    else:

        big_image = make_image_from_bunch(images)
        big_image, _ = add_left_legend(
            big_image, header_size=0,
            texts=legend)

        big_image, header_size = add_header(big_image, f"[{epoch:03}] l:{loss['loss']:.2f}")

        if metrics is not None:
            loss_plot = plot_stuff(metrics, h=image_size, w=big_image.shape[1], color=colors)
            big_image = np.concatenate((loss_plot, big_image), axis=0)

    if subset is not None:
        plt.imsave(os.path.join(results_path, f'summary_{epoch:04}.png'), big_image)
    else:
        plt.imsave(os.path.join(results_path, f'res_{epoch:04}.png'), big_image)
