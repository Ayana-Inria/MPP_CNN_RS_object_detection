import numpy as np


def extract_patch(image: np.ndarray, center_anchor: np.ndarray, patch_size: int):
    assert center_anchor.shape == (2,)
    tl_anchor = center_anchor - patch_size // 2
    shape = np.array(image.shape[:2])
    centers_offset = np.zeros((2,),dtype=int)

    if tl_anchor[0] < 0 or tl_anchor[0] + patch_size >= shape[0]:
        image = np.pad(image, ((patch_size // 2, patch_size // 2), (0, 0), (0, 0)), 'constant')

        centers_offset[0] = patch_size // 2

        tl_anchor[0] = tl_anchor[0] + patch_size // 2
    if tl_anchor[1] < 0 or tl_anchor[1] + patch_size >= shape[1]:
        image = np.pad(image, ((0, 0), (patch_size // 2, patch_size // 2), (0, 0)), 'constant')
        centers_offset[1] = patch_size // 2
        tl_anchor[1] = tl_anchor[1] + patch_size // 2

    s = np.s_[tl_anchor[0]:tl_anchor[0] + patch_size, tl_anchor[1]:tl_anchor[1] + patch_size]
    patch = image[s]
    return patch, tl_anchor, centers_offset