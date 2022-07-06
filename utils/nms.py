import numpy as np
from scipy.spatial import distance_matrix
from sklearn.metrics import pairwise_distances


def nms(bounding_boxes, confidence_score, threshold, return_index=False):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        if return_index:
            return [], [], []
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_index = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_index.append(index)
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    if return_index:
        return picked_boxes, picked_score, picked_index
    return picked_boxes, picked_score


def nms_distance(centers, confidence_score, threshold, return_index=False):
    # If no bounding boxes, return empty list
    if len(centers) == 0:
        if return_index:
            return [], [], []
        return [], []

    # Bounding boxes
    centers = np.array(centers)



    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_centers = []
    picked_score = []
    picked_index = []

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_index.append(index)
        picked_centers.append(centers[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)

        dist_to_picked = np.linalg.norm(centers[index] - centers[order[:-1]],axis=-1)

        left = np.where(dist_to_picked > threshold)
        order = order[left]

    if return_index:
        return picked_centers, picked_score, picked_index
    return picked_centers, picked_score