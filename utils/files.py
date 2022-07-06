import json
import os
from typing import Union, List

import numpy as np


def make_gif(base_path: str, input_files: str, output_file: str):
    files = os.path.join(base_path, input_files)
    target = os.path.join(base_path, output_file)
    os.system(f'convert -loop 0 -delay 20 {files} {target}')


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def make_if_not_exist(path: Union[str, List[str]], recursive=False):
    if not type(path) is list:
        path = [path]
    for p in path:
        if not os.path.exists(p):
            if recursive:
                sub = os.path.split(p)[0]
                make_if_not_exist(sub, recursive=True)
            os.mkdir(p)


def find_existing_path(possible_base_paths: List[str]) -> str:
    for p in possible_base_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError
