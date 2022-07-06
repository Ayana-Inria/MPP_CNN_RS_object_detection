from typing import Dict, List, Union, Iterable

import numpy as np
from scipy.ndimage import binary_dilation
from tqdm import tqdm


def precision_recall_curve_on_detection_map(detection_map: Union[np.ndarray, List[np.ndarray]],
                                            labels: Union[Dict, List[Dict]], num_thresholds: int = None,
                                            dilation: int = 1,
                                            thresholds: Iterable[float] = None):
    if thresholds is None:
        assert num_thresholds is not None
        thresholds = np.linspace(0, 1, num_thresholds)

    if not type(detection_map) is list:
        detection_map = [detection_map]
        labels = [labels]

    x = []
    y = []

    for k in range(len(detection_map)):
        shape = detection_map[k].shape[:2]
        assert len(shape) == 2
        bin_label_map = np.zeros(shape, dtype=bool)
        centers = labels[k]['centers']
        if len(centers) > 0:
            bin_label_map[centers[:, 0], centers[:, 1]] = True
            bin_label_map = binary_dilation(bin_label_map, iterations=dilation)

        x.append(detection_map[k].ravel())
        y.append(bin_label_map.ravel())

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    precision, recall = compute_precision_recall(x, y, thresholds)
    precision = np.array(precision)
    recall = np.array(recall)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': (precision * recall) / (precision + recall)
    }

    return thresholds, metrics


def compute_precision_recall(scores, labels, thresholds):
    precision = []
    recall = []
    for t in tqdm(thresholds):
        pos = scores > t
        tp = np.sum(np.logical_and(pos, labels))
        fp = np.sum(np.logical_and(pos, ~labels))

        precision.append(tp / (tp + fp))
        recall.append(tp / np.sum(labels))

    return precision, recall
