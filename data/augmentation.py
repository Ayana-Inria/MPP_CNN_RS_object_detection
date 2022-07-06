import glob
import os
from dataclasses import dataclass

import albumentations as alb
import numpy as np

from utils.data import get_dataset_base_path


def rot_90_coor(coor: np.ndarray, size: int, k: int):
    if k == 0 or len(coor) == 0:
        return coor
    else:
        r_arr = coor.copy()
        r_arr[..., 0] = size - 1 - coor[..., 1]
        r_arr[..., 1] = coor[..., 0]
        return rot_90_coor(r_arr, size, k - 1)


def medium_aug(hist_match_images_paths=None):
    return [
        alb.FromFloat(dtype='uint8', always_apply=True),
        alb.RandomRotate90(),
        alb.Flip(),
        (
            alb.HistogramMatching(reference_images=hist_match_images_paths, blend_ratio=(0.1, 0.75))
            if hist_match_images_paths is not None else alb.NoOp()
        ),
        alb.OneOf([
            alb.CLAHE(),
            alb.RGBShift()
        ]),
        alb.OneOf([
            alb.MedianBlur(blur_limit=3, p=0.1),
            alb.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        alb.GaussNoise(),
        alb.ToFloat(always_apply=True)
    ]


def strong_aug(hist_match_images_paths=None):
    return [
        alb.FromFloat(dtype='uint8', always_apply=True),
        alb.RandomRotate90(),
        alb.Flip(),
        (
            alb.HistogramMatching(reference_images=hist_match_images_paths, blend_ratio=(0.1, 0.75))
            if hist_match_images_paths is not None else alb.NoOp()
        ),
        alb.RandomShadow(),
        alb.RandomFog(),
        alb.OneOf([
            alb.ChannelShuffle(),
            alb.ChannelDropout()
        ]),
        alb.RandomBrightnessContrast(),
        alb.OneOf([
            alb.CLAHE(),
            alb.RGBShift(),
            alb.ToGray(p=0.1)
        ]),
        alb.Downscale(scale_min=0.9, scale_max=0.9),
        alb.OneOf([
            alb.MedianBlur(blur_limit=3, p=0.1),
            alb.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        alb.GaussNoise(),
        alb.ToFloat(always_apply=True)
    ]


@dataclass
class DataAugment:
    rng: np.random.Generator
    dataset: str
    subset: str
    hist_match_images: bool = False
    aug_level: str = "medium"

    def __post_init__(self):
        self.hist_match_images_paths = None
        if self.hist_match_images:
            self.hist_match_images_paths = glob.glob(
                os.path.join(get_dataset_base_path(), self.dataset, self.subset, 'images/*.png'))
            assert len(self.hist_match_images_paths) > 0

        if self.aug_level == 'medium':
            augs = medium_aug(self.hist_match_images_paths)
        elif self.aug_level == 'strong':
            augs = strong_aug(self.hist_match_images_paths)
        else:
            raise ValueError

        self.transformer = alb.Compose(
            augs,
            keypoint_params=alb.KeypointParams(format='xya', remove_invisible=False, angle_in_degrees=False)
        )

    def transform(self, patch, centers, params):
        n_points = len(centers)

        if n_points == 0:
            keypoints = []
        else:
            angles = params[:, 2]
            keypoints = np.stack([centers[:, 1], centers[:, 0], angles], axis=-1)

        transformed = self.transformer(image=patch, keypoints=keypoints)
        transformed_image = transformed['image']
        transformed_keypoints = np.array(transformed['keypoints'])

        if n_points == 0:
            new_centers = np.array([])
            new_params = np.array([])
        else:
            new_angles = transformed_keypoints[:, 2] % np.pi
            new_centers = np.stack([transformed_keypoints[:, 1], transformed_keypoints[:, 0]], axis=-1).astype(int)
            new_params = np.stack([params[:, 0], params[:, 1], new_angles], axis=-1)

        return transformed_image, new_centers, new_params, None
