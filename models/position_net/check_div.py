# temporary experiment please remove
import json
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import xgboost
from scipy.ndimage import binary_dilation
from tqdm import tqdm

from metrics.detection import compute_precision_recall
from models.position_net.pos_net_model import PosNetModel
from utils.data import fetch_data_paths, get_model_config_by_name
from utils.files import make_if_not_exist
from utils.math_utils import divergence_map_from_vector_field

DILATION = 1


def make_data(id_list, paths_dict, pos_model):
    id_re = re.compile(r'([0-9]+).*.png')

    vec_samples = []
    labels = []
    legacy_samples = []
    mask_samples = []

    for i in tqdm(id_list):
        pf = paths_dict['images'][i]
        af = paths_dict['annotations'][i]
        patch_id = int(id_re.match(os.path.split(pf)[1]).group(1))
        print(f"opening {patch_id}")

        img = plt.imread(pf)[..., :3]
        with open(af, 'rb') as f:
            labels_dict = pickle.load(f)
        centers, params = labels_dict['centers'], labels_dict['parameters']
        try:
            output_mask, output_vec, _ = pos_model.infer_on_image(img, centers, params)
        except Exception as e:
            print(f"inference failed with")
            print(e)
            print("skipping")
            continue

        div = divergence_map_from_vector_field(output_vec, normalize=False)

        legacy_score = pos_model.vec2detection_map(output_vec, output_mask)

        bin_mask = np.zeros(img.shape[:2], dtype=bool)
        bin_mask[centers[:, 0], centers[:, 1]] = True
        bin_mask = binary_dilation(bin_mask, iterations=DILATION)

        vec_samples.append(np.stack([div.ravel(), output_mask.ravel()], axis=-1))
        labels.append(bin_mask.ravel())
        legacy_samples.append(legacy_score.ravel())
        mask_samples.append(output_mask.ravel())

    vec_samples = np.concatenate(vec_samples, axis=0)
    labels = np.concatenate(labels, axis=0)
    legacy_samples = np.concatenate(legacy_samples, axis=0)
    mask_samples = np.concatenate(mask_samples, axis=0)
    samples = {
        'vec': vec_samples,
        'legacy': legacy_samples,
        'div un-norm': np.clip(-vec_samples[:, 0] / 2, 0, 1) * mask_samples,
    }
    return samples, labels


def check_div(dataset: str):
    config_file = get_model_config_by_name('posvec_dota_22')
    print(config_file)

    save_dir = os.path.join(os.path.split(config_file)[0], 'div_check')
    make_if_not_exist(save_dir)

    subset = 'train'
    paths_dict = fetch_data_paths(dataset=dataset, subset=subset)

    n_train = 64
    n_val = 64

    rng = np.random.default_rng(0)
    sampled_images_id = rng.choice(range(len(paths_dict['images'])), size=n_train + n_val, replace=False)
    id_train = sampled_images_id[:n_train]
    id_val = sampled_images_id[n_train:]

    with open(config_file, 'r') as f:
        pos_config = json.load(f)
    pos_model = PosNetModel(config=pos_config, train=False, load=True, dataset=dataset)

    train_data, train_label = make_data(id_train, paths_dict, pos_model)
    print(f"extracted {len(train_data['vec'])} samples with {np.sum(train_label)}")
    val_data, val_label = make_data(id_val, paths_dict, pos_model)
    print(f"extracted {len(val_data['vec'])} samples with {np.sum(val_label)}")

    dtrain = xgboost.DMatrix(train_data['vec'], label=train_label)
    dval = xgboost.DMatrix(val_data['vec'], label=val_label)

    param = {'max_depth': 3, 'eta': 1.0, 'objective': 'binary:logistic'}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'
    evallist = [(dval, 'eval'), (dtrain, 'train')]

    num_round = 10
    bst = xgboost.train(param, dtrain, num_round, evallist)

    bst.save_model(os.path.join(save_dir, 'xgboost.model'))

    xg_score_val = bst.predict(dval, iteration_range=(0, bst.best_iteration))

    thresholds = np.linspace(0, 1, 100)
    results = {}
    for name, score in [("xgboost", xg_score_val), ("legacy", val_data['legacy']), ('div un-norm', val_data['div un-norm'])]:
        precision, recall = compute_precision_recall(score, val_label, thresholds)
        results[name] = {
            "precision": precision,
            "recall": recall
        }

    fig, axs = plt.subplots(1, 1, figsize=(8, 4))
    axs: plt.Axes
    axs.set_xlim(0, 1)
    axs.set_ylim(0, 1)
    axs.set_xlabel('recall')
    axs.set_ylabel('precision')

    X = np.linspace(0, 1, 100)
    Y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(X, Y)
    Z = 2 * (X * Y) / (X + Y)
    cs = axs.contour(X, Y, Z, levels=[0.25, 0.5, 0.6, 0.7, 0.8, 0.9])
    axs.clabel(cs, inline=1, fontsize=10, fmt='F1=%.2f')

    for k, v in results.items():
        axs.plot(v['recall'], v['precision'], label=k)

    axs.legend()
    plt.savefig(os.path.join(save_dir, 'precision_recall.png'))
    plt.close('all')
