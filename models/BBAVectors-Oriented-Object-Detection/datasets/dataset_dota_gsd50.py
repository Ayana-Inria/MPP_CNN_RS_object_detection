import os
import pickle

import cv2
import numpy as np

from base.shapes.rectangle import rect_to_poly
from data.DOTA_devkit.ResultMerge import mergebypoly
from datasets.base import BaseDataset
from utils.data import get_dataset_base_path, fetch_data_paths


class CustomDOTA(BaseDataset):
    def __init__(self, data_dir, phase, input_h=608, input_w=608, down_ratio=4):
        super(CustomDOTA, self).__init__(data_dir, phase, input_h, input_w, down_ratio)
        if self.phase == 'eval' or self.phase == 'test':
            self.subset = 'val'
        else:
            self.subset = self.phase

        self.category = ['vehicle']

        self.input_h = input_h
        self.input_w = input_w

        self.base_data_dir = get_dataset_base_path()
        self.img_ids = self.load_img_ids()
        self.num_classes = len(self.category)
        self.cat_ids = {cat: i for i, cat in enumerate(self.category)}
        self.image_path = os.path.join(self.base_data_dir, 'images')
        self.label_path = os.path.join(self.base_data_dir, 'labelTxt')  # todo ??

    def load_img_ids(self):
        """
        Definition: generate self.img_ids
        Usage: index the image properties (e.g. image name) for training, testing and evaluation
        Format: self.img_ids = [list]
        Return: self.img_ids
        """

        paths_dict = fetch_data_paths(self.data_dir, self.subset)
        img_ids = [os.path.split(p)[1].split('.')[0] for p in paths_dict['images']]
        return img_ids

    def load_image(self, index):
        """
        Definition: read images online
        Input: index, the index of the image in self.img_ids
        Return: image with H x W x 3 format
        """
        img_id = self.img_ids[index]
        img_path = os.path.join(self.base_data_dir, self.data_dir, self.subset, 'images', f"{img_id}.png")
        img = cv2.imread(img_path)
        # img = cv2.resize(img, dsize=(img.shape[0] * 2, img.shape[1] * 2))
        if img is None:
            print(f"did not find : {img_path}")
            raise FileNotFoundError
        return img

    def load_annoFolder(self, img_id):
        """
        Return: the path of annotation
        Note: You may not need this function
        """
        annot_path = os.path.join(self.base_data_dir, self.data_dir, self.subset, 'annotations', f"{img_id}.pkl")
        return annot_path

    def load_annotation(self, index):
        """
        Return: dictionary of {'pts': float np array of [bl, tl, tr, br],
                                'cat': int np array of class_index}
        Explaination:
                bl: bottom left point of the bounding box, format [x, y]
                tl: top left point of the bounding box, format [x, y]
                tr: top right point of the bounding box, format [x, y]
                br: bottom right point of the bounding box, format [x, y]
                class_index: the category index in self.category
                    example: self.category = ['ship]
                             class_index of ship = 0
        """
        img_id = self.img_ids[index]
        annot_path = os.path.join(self.base_data_dir, self.data_dir, self.subset, 'annotations', f"{img_id}.pkl")
        with open(annot_path, 'rb') as f:
            labels_dict = pickle.load(f)

        centers = labels_dict['centers']
        params = labels_dict['parameters']

        polys = np.array([rect_to_poly(c, p[0], p[1], p[2]) for p, c in zip(params, centers)])
        polys = np.flip(polys, axis=-1)
        cat = np.array([0 for _ in polys])
        if len(polys) == 0:
            polys = np.array([])
            cat = np.array([])

        annotation = {}
        annotation['pts'] = polys
        annotation['cat'] = cat
        annotation['dif'] = labels_dict['difficult']

        return annotation

    def merge_crop_image_results(self, result_path, merge_path):
        mergebypoly(result_path, merge_path)
