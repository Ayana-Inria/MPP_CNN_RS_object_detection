import os
from typing import List, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import morphology
from skimage.draw import draw
from torch import Tensor

from utils.display.light_display.image_stack import multi_hist_image, make_image_from_bunch, add_left_legend, \
    add_header, \
    add_top_legend
from utils.display.light_display.plot import plot_stuff
from models.shape_net.mappings import ValueMapping, output_vector_to_value, class_id_to_value
from base.shapes.rectangle import sra_to_wla, rect_to_poly


def _pred_to_image(pred, centers, image, color, mix_factor, thickness: int = 1):
    if type(centers) is torch.Tensor:
        centers = centers.cpu().numpy()
    if type(pred) is torch.Tensor:
        pred = pred.cpu().numpy()
    true_poly_mask = np.zeros(image.shape[:2], dtype=bool)
    for c in centers:
        a, b, angle = sra_to_wla(pred[0][c[0], c[1]], pred[1][c[0], c[1]], pred[2][c[0], c[1]])
        mask = rect_to_poly(center=(c[0], c[1]), long=b, short=a, angle=angle)
        true_poly_mask[draw.polygon_perimeter(mask[:, 0], mask[:, 1], shape=image.shape[:2])] = 1
    if thickness > 1:
        true_poly_mask = morphology.dilation(true_poly_mask, morphology.disk(radius=thickness - 1))

    image[true_poly_mask] = (1 - mix_factor) * image[true_poly_mask] + mix_factor * color
    return image


def _pred_to_image2(centers, params, scores, image, color, max_score: float = 1.0):
    image = image.copy()
    cmap = None
    if color is None:
        color = (0, 1.0, 0)
    elif type(color) == str:
        cmap = plt.get_cmap(color)

    for i, c in enumerate(centers):
        if cmap is not None:
            if scores is not None:
                color = cmap(np.clip(scores[i] / max_score, 0, max_score))
        a, b, angle = params[i]
        pts = rect_to_poly(center=(c[0], c[1]), long=b, short=a, angle=angle)
        pts.reshape((-1, 1, 2))
        pts = pts.astype(np.int32)
        pts = np.flip(pts, axis=-1)
        cv2.polylines(image, [pts], True, color)
        if scores is not None:
            cv2.putText(image, "{:.2f}".format(scores[i]), (int(np.min(pts[..., 0])), int(np.min(pts[..., 1])) - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, thickness=1)

    return image


def display_shape_inference(epoch: int, images: Tensor, output: List[Tensor], labels: Dict, loss: Dict,
                            mappings: List[ValueMapping], results_path: str, thickness: int = 1, subset=None,
                            metrics=None, colors=None):
    loss_mask = labels['loss_mask'].numpy()
    for i in range(loss_mask.shape[0]):
        if np.max(loss_mask[i]) > 0:
            loss_mask[i] = loss_mask[i] / np.max(loss_mask[i])
    loss_mask_color = plt.get_cmap('gray')(loss_mask)[:, :, :, :3]

    output = [t.detach().cpu().numpy() for t in output]  # list of B,C,W,H

    distributions = np.moveaxis(np.array(output), (0, 1, 2, 3, 4), (1, 0, 4, 2, 3))  # F,B,C,W,H to B,F,W,H,C

    values_map = output_vector_to_value(output, mappings)
    images = images.permute((0, 2, 3, 1)).numpy()

    image_size = images.shape[1]

    if images.shape[3] == 1:
        images = np.stack([images[:, :, :, 0] for _ in range(3)], axis=-1)
    images_w_shapes = images.copy()

    plot_w, plot_1, plot_2 = [], [], []
    plot_cmap = plt.get_cmap('plasma')
    target_values_map = []
    min_plot_size = 10
    for i in range(8):
        centers = np.array(np.where(labels['center_binary_map'][i].numpy())).T
        feature_value_classes = [labels['value_class_map'][j][i].numpy() for j in range(3)]
        feature_values = class_id_to_value(feature_value_classes, mappings=mappings)
        target_values_map.append(feature_values)

        images_w_shapes[i] = _pred_to_image(feature_values, centers=centers,
                                            image=images_w_shapes[i],
                                            color=np.array([0.2, 1, 0.2]),
                                            mix_factor=0.5, thickness=thickness)

        images_w_shapes[i] = _pred_to_image([values_map[j][i] for j in range(3)], centers=centers,
                                            image=images_w_shapes[i],
                                            color=np.array([0.2, 0.2, 1]),
                                            mix_factor=0.5, thickness=thickness)

        if len(centers) > 0:
            plot_w.append(
                multi_hist_image(size=image_size, distribution=distributions[i, 2, centers[:, 0], centers[:, 1]],
                                 vmin=0,
                                 vmax='auto',
                                 gt=feature_value_classes[2][centers[:, 0], centers[:, 1]],
                                 plot_cmap=plot_cmap, min_plot_size=min_plot_size)
            )
            plot_1.append(
                multi_hist_image(size=image_size, distribution=distributions[i, 0, centers[:, 0], centers[:, 1]],
                                 vmin=0,
                                 vmax='auto',
                                 gt=feature_value_classes[0][centers[:, 0], centers[:, 1]],
                                 plot_cmap=plot_cmap, min_plot_size=min_plot_size)
            )
            plot_2.append(
                multi_hist_image(size=image_size, distribution=distributions[i, 1, centers[:, 0], centers[:, 1]],
                                 vmin=0,
                                 vmax='auto',
                                 gt=feature_value_classes[1][centers[:, 0], centers[:, 1]],
                                 plot_cmap=plot_cmap, min_plot_size=min_plot_size)
            )
        else:
            empty = np.zeros((image_size, image_size, 3))
            plot_w.append(empty)
            plot_1.append(empty)
            plot_2.append(empty)

    target_values_map = np.array(target_values_map).swapaxes(0, 1)

    plot_w = np.array(plot_w)
    plot_1 = np.array(plot_1)
    plot_2 = np.array(plot_2)

    fig_w = values_map[2] / mappings[2].v_max
    fig_w = np.stack([plt.get_cmap('twilight')(arr)[:, :, :3] for arr in fig_w], axis=0)

    fig_w_t = target_values_map[2] / mappings[2].v_max
    fig_w_t = np.stack([plt.get_cmap('twilight')(arr)[:, :, :3] for arr in fig_w_t], axis=0)

    fig_a = values_map[0] / mappings[0].v_max
    fig_a = np.stack([plt.get_cmap('plasma')(arr)[:, :, :3] for arr in fig_a], axis=0)

    fig_b = values_map[1] / mappings[1].v_max
    fig_b = np.stack([plt.get_cmap('plasma')(arr)[:, :, :3] for arr in fig_b], axis=0)

    fig_loss = np.sum([loss[f'pixel_loss_feat{i}'].detach().cpu().numpy() for i in range(3)], axis=0)
    fig_loss = fig_loss / np.max(fig_loss)
    fig_loss = np.stack([plt.get_cmap('magma')(arr)[:, :, :3] for arr in fig_loss], axis=0)

    all_figs = (
        images, images_w_shapes, loss_mask_color, fig_w, plot_w, fig_a, plot_1, fig_b, plot_2, fig_loss)
    n_figs = len(all_figs)
    images = np.concatenate(all_figs, axis=0)

    if subset is not None:
        images = np.concatenate([images[[j * 8 + i for j in range(n_figs)]] for i in subset])

        big_image = make_image_from_bunch(images, nrow=n_figs)

        big_image, _ = add_top_legend(
            big_image,
            texts=["input", "gt+pred", "mask", "m_angle", 'h_angle', 'm_size', 'h_size',
                   'm_ratio',
                   'h_ratio', "losses"])
    else:

        big_image = make_image_from_bunch(images)
        big_image, _ = add_left_legend(
            big_image, header_size=0,
            texts=["input", "gt+pred", "mask", "m_angle", 'h_angle', 'm_size', 'h_size',
                   'm_ratio',
                   'h_ratio', 'losses'])

        big_image, header_size = add_header(big_image, f"[{epoch:03}] l:{loss['loss']:.2f}")

        if metrics is not None:
            loss_plot = plot_stuff(metrics, h=image_size, w=big_image.shape[1], color=colors)
            big_image = np.concatenate((loss_plot, big_image), axis=0)

    if subset is not None:
        plt.imsave(os.path.join(results_path, f'summary_{epoch:04}.png'), big_image)
    else:
        plt.imsave(os.path.join(results_path, f'res_{epoch:04}.png'), big_image)
