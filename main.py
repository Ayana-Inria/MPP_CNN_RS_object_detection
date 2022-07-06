import argparse
import json
import os
import sys

from base.base_model import BaseModel
from utils.data import resolve_model_config_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='model to use')
    parser.add_argument('-d', '--dataset', help='dataset to use, defaults to the one specified in config')
    parser.add_argument('-p', '--procedure', help='procedure to execute')
    parser.add_argument('-c', '--config',
                        help='model config file, can pass model name if model already is in bba_models')
    parser.add_argument('-o', '--overwrite', action='store_true', help='if set then overwrites existing model')
    parser.add_argument('-r', '--resume', action='store_true', help='resumes training from checkpoint')
    args = parser.parse_args()

    model_type = args.model
    procedure = args.procedure
    dataset = args.dataset
    overwrite_model = args.overwrite and procedure == 'train'
    overwrite_results = args.overwrite and procedure != 'train'
    train_flag = args.procedure == 'train'
    load_flag = args.resume or args.procedure not in ['train', 'data_preview']

    if procedure == "check_div":
        from models.position_net.check_div import check_div
        check_div(dataset=dataset)
        print("done !")
        return

    config_file = resolve_model_config_path(args.config)

    with open(config_file, 'r') as f:
        config = json.load(f)

    if procedure == 'translate_dota':
        from data.translation.translate_DOTA import translate_dota
        translate_dota(config)
        print('done !')
        return
    elif procedure == 'translate_cowc':
        from data.translation.translate_COWC import translate_cowc
        translate_cowc(config)
        print('done !')
        return
    model: BaseModel

    if model_type == 'posnet':
        from models.position_net.pos_net_model import PosNetModel
        model = PosNetModel(config, overwrite=overwrite_model, load=load_flag, train=train_flag, dataset=dataset)
    elif model_type == 'shapenet':
        from models.shape_net.shape_net_model import ShapeNetModel
        model = ShapeNetModel(config, overwrite=overwrite_model, load=load_flag, train=train_flag, dataset=dataset)
    elif model_type == 'mpp':
        from models.mpp.mpp_model import MPPModel
        model = MPPModel(config, overwrite=overwrite_model, load=load_flag, phase='train' if train_flag else 'val',
                         dataset=dataset)
    elif model_type == 'fasterrcnn':
        from models.fasterRCNN.faster_rcnn_model import FasterRCNNModel
        model = FasterRCNNModel(config, overwrite=overwrite_model, load=load_flag, train=train_flag, dataset=dataset)
    elif model_type == 'oracle':
        from models.oracle.oracle_model import OracleModel
        model = OracleModel(config, dataset=dataset)
    elif model_type == 'bbavec':
        sys.path.append(os.path.join(sys.path[0], 'models/BBAVectors-Oriented-Object-Detection'))
        sys.path.append(os.path.join(sys.path[0], 'data/DOTA_devkit'))
        from bbavec_model import BBAVec
        model = BBAVec(config, load=load_flag, overwrite=overwrite_model, dataset=dataset)
    else:
        raise ValueError
    if procedure == 'train':
        print("training starts now")
        model.train()
    elif procedure == 'data_preview':
        print("previewing data")
        model.data_preview()
    elif procedure == 'infer':
        print('infering on dataset')
        model.infer(subset='val', min_confidence=0.2, display_min_confidence=0.5, overwrite=overwrite_results)
    elif procedure == 'eval':
        print('evaluating metrics')
        model.eval()
    elif procedure == 'infereval':
        print('infering on dataset')
        model.infer(subset='val', min_confidence=0.2, display_min_confidence=0.5, overwrite=overwrite_results)
        print('evaluating metrics')
        model.eval()
    else:
        raise ValueError

    # if procedure == 'train':
    #     print("training starts now")
    #     model.train()
    # elif procedure == 'data_preview':
    #     print("previewing data")
    #     model.data_preview()
    # elif procedure == 'infer':
    #     print('infering on dataset')
    #     model.infer(min_confidence=0.25, display_min_confidence=0.5)

    print('done !')


if __name__ == '__main__':
    main()
