import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

from utils.display.boxes import bboxes_over_image_cv2
from utils.display.light_display.image_stack import make_image_from_bunch, add_header, add_left_legend
from utils.display.light_display.plot import plot_stuff


def display_rcnn_inference(epoch, images: Tensor, gt_boxes, boxes: List, scores: List, train_loss, val_loss, colors,
                           results_path: str):
    n_images = len(images)
    all_raw_images = []
    for i in range(n_images):
        raw_img = images[i].cpu().permute((1, 2, 0)).numpy()
        all_raw_images.append(raw_img)
    all_gt_boxes_images = bboxes_over_image_cv2(all_raw_images, gt_boxes, None)
    all_boxes_images = bboxes_over_image_cv2(all_raw_images, boxes, scores)
    images = np.concatenate([all_raw_images, all_gt_boxes_images, all_boxes_images], axis=0)

    big_image = make_image_from_bunch(images, nrow=n_images)

    big_image, _ = add_left_legend(big_image, header_size=0, texts=["input", "gt", "pred"])

    big_image, header_size = add_header(big_image, f"[{epoch:03}] train:{train_loss[-1]:.2f} | val:{val_loss[-1]:.2f}")

    image_size = images.shape[1]

    loss_plot = plot_stuff(np.array([train_loss, val_loss]), h=image_size, w=big_image.shape[1], color=colors)
    big_image = np.concatenate((loss_plot, big_image), axis=0)

    plt.imsave(os.path.join(results_path, f'res_{epoch:04}.png'), big_image)
