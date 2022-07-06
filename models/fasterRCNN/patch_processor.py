import numpy as np
import torch
from skimage.draw import polygon2mask

from data.image_dataset import LabelProcessor
from base.shapes.rectangle import rect_to_poly


class FasterRCNNPatchProcessor(LabelProcessor):

    def process(self, patch: np.ndarray, centers: np.ndarray, params: np.ndarray, idx):

        boxes = []
        num_objs = 0
        mask_image = np.zeros(patch.shape[:2], dtype=bool)
        for c, p in zip(centers, params):
            a, b, w = p
            poly = rect_to_poly(c, short=a, long=b, angle=w)
            mask = polygon2mask(patch.shape[:2], poly)
            mask_image = np.logical_or(mask_image, mask)

            x_min = np.min(poly[:, 0])
            x_max = np.max(poly[:, 0])
            y_min = np.min(poly[:, 1])
            y_max = np.max(poly[:, 1])

            # old
            # if(x_min >= patch[0] - self.mask_size // 2 and x_max < patch[0] + self.mask_size // 2 and y_min >= patch[1] - self.mask_size // 2  and y_max < patch[1] + self.mask_size // 2):
            if (x_min >= 0 and x_max < patch.shape[0] and y_min >= 0 and y_max < patch.shape[1]):
                boxes.append([y_min, x_min, y_max, x_max])
                num_objs += 1

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape((-1, 4))
        mask_image = torch.tensor(mask_image, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        labels = torch.ones((num_objs,), dtype=torch.int64)

        if (len(boxes) > 0):
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.tensor([0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        tensor_image = torch.tensor(patch).permute((2, 0, 1))

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = mask_image
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return tensor_image, target
