import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from utils.data import get_inference_path
from utils.files import NumpyEncoder


# task 1 : https://captain-whu.github.io/DOTA/tasks.html , det with OBB
# code adapted from https://github.com/CAPTAIN-WHU/DOTA_devkit/blob/master/dota-v1.5_evaluation_task1.py


def dota_eval(model_dir: str, dataset: str, subset: str, det_type: str, postfix=''):
    assert det_type in ['obb', 'hbb']
    model_name = os.path.split(model_dir)[1]
    dota_files_path = get_inference_path(model_name=model_name, dataset=dataset, subset=subset)
    dota_files_path = os.path.join(dota_files_path, 'dota' + postfix)

    det_path = os.path.join(dota_files_path, 'det', r'{:s}.txt')
    annot_path = os.path.join(dota_files_path, 'gt', r'{:s}.txt')
    image_set_file = os.path.join(dota_files_path, 'imageSet.txt')

    classnames = ['vehicle']

    for iou_t in [0.05, 0.1, 0.25, 0.5, 0.75]:
        print(f"IOU thresh = {iou_t}")
        results = {}
        classaps = []
        mean_ap = 0

        for classname in classnames:
            print(f'doing class {classname}')
            if det_type == 'obb':
                sys.path.append(os.path.join(os.getcwd(), 'data/DOTA_devkit'))
                from data.DOTA_devkit.dota_evaluation_task1 import voc_eval
                print('using OBB')
                rec, prec, ap = voc_eval(
                    detpath=det_path,
                    annopath=annot_path,
                    imagesetfile=image_set_file,
                    classname=classname,
                    ovthresh=iou_t,
                    use_07_metric=False
                )
            else:  # call task 2 for non oriented BB
                # todo this one does not work since the gt is not handeled correcly, should either use task 1 or fix
                sys.path.append(os.path.join(os.getcwd(), 'data/DOTA_devkit'))
                from data.DOTA_devkit.dota_evaluation_task2 import voc_eval
                print('using HBB')
                rec, prec, ap = voc_eval(
                    detpath=det_path,
                    annopath=annot_path,
                    imagesetfile=image_set_file,
                    classname=classname,
                    ovthresh=iou_t,
                    use_07_metric=False
                )
            mean_ap = mean_ap + ap
            classaps.append(ap)
            print(f"ap : {ap}")

            results[classname] = {
                'ap': ap,
                'precision': prec,
                'recall': rec
            }
            try:
                plt.figure(figsize=(8, 4))
                plt.xlabel('recall')
                plt.ylabel('precision')
                plt.plot(rec, prec)
                plt.savefig(os.path.join(dota_files_path, f'prec_rec_curve_{iou_t:.2f}.png'))
                plt.close('all')
            except Exception as e:
                print("error ocured while displaying figures")
                print(e)

        mean_ap = mean_ap / len(classnames)
        print('map:', mean_ap)
        classaps = np.array(classaps)
        print('classaps: ', classaps)

        with open(os.path.join(dota_files_path, f'metrics{iou_t:.2f}.json'), 'w') as f:
            json.dump(results, f, cls=NumpyEncoder, indent=1)


def main():
    frcnn_model = '/workspaces/nef/home/jmabon/models/fasterrcnn/fasterRCNN_dota_20/'
    shapenet_model = '/workspaces/nef/home/jmabon/models/shapenet/shape_dota_22/'

    dota_eval(
        model_dir=shapenet_model,
        dataset='DOTA_gsd50',
        subset='val',
        det_type='obb'
    )


if __name__ == '__main__':
    main()
