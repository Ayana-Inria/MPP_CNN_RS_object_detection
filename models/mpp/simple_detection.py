from typing import List

import numpy as np
from scipy.ndimage import filters

from models.mpp.data_loaders import PARAM_NAMES
from models.shape_net.mappings import ValueMapping
from base.shapes.rectangle import Rectangle


def local_max_detection(detection_map, threshold, neighboring_distance):
    data_max = filters.maximum_filter(detection_map, neighboring_distance)
    maxima = (detection_map == data_max)
    data_min = filters.minimum_filter(detection_map, neighboring_distance)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    return np.where(maxima)


def local_maximum_detection(detection_map, threshold, neighboring_distance, parameters_dist_maps,
                            mappings: List[ValueMapping]) -> List[Rectangle]:
    nms_points_x, nms_points_y = local_max_detection(detection_map, threshold, neighboring_distance)

    points = []
    for x, y in zip(nms_points_x, nms_points_y):
        params = {
            p: mappings[i].class_to_value(np.argmax(parameters_dist_maps[i][x, y])) for i, p in enumerate(PARAM_NAMES)
        }
        points.append(Rectangle(x, y, **params))
    return points