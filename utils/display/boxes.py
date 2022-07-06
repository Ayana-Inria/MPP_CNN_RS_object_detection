import cv2
import numpy as np
from matplotlib import pyplot as plt, patches
from torch import Tensor


def bboxes_over_image_cv2(images, boxes, color=None, scores=None):
    all_img = []
    cmap = None
    if color is None:
        color = (0, 1.0, 0)
    elif type(color) == str:
        cmap = plt.get_cmap(color)

    batch_mode = True
    if type(images) == np.ndarray and len(images.shape) == 3:  # one single image
        batch_mode = False
        images = [images]
        boxes = [boxes]
        if scores is not None:
            scores = [scores]
    for i in range(len(images)):
        if type(images[i]) == Tensor:
            img = images[i].permute(1, 2, 0).cpu().numpy()
        elif type(images[i]) == np.ndarray:
            img = images[i].copy()
        else:
            raise TypeError

        l_boxes = boxes[i]
        l_scores = scores[i] if scores is not None else None

        for j in range(len(l_boxes)):
            if cmap is not None:
                if scores is not None:
                    color = cmap(l_scores[j])
            cv2.rectangle(img, (int(l_boxes[j][0]), int(l_boxes[j][1])),
                          (int(l_boxes[j][2]), int(l_boxes[j][3])), color, 1)
            if l_scores is not None:
                cv2.putText(img, "{:.2f}".format(l_scores[j]), (int(l_boxes[j][0]), int(l_boxes[j][1]) - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, thickness=1)

        all_img.append(img)
    if batch_mode:
        return all_img
    else:
        return all_img[0]


def bboxes_pyplot(boxes, scores, ax: plt.Axes, cmap=None, color='tab:orange', score_thresh: float = -1):
    for b, s in zip(boxes, scores):
        if s >= score_thresh:
            if cmap is not None:
                color = cmap(s)
            ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, edgecolor=color, lw=2))
            ax.text(b[0], b[1], f"{s:.2f}", verticalalignment='bottom', color=color, fontsize=10)
