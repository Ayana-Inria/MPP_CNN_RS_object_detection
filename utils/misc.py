from datetime import datetime
from typing import Dict, List, Union
import numpy as np

DictOfLists = Dict[str, List]


def append_lists_in_dict(dict_to_update: DictOfLists, new_values: Dict[str, Union[float, str]], add_keys=True):
    for k, v in new_values.items():
        if k in dict_to_update:
            dict_to_update[k].append(v)
        elif add_keys:
            dict_to_update[k] = [v]
        else:
            raise KeyError


def reduce_dict(input_dict):
    new_dict = {}
    for k, v in input_dict.items():
        new_dict[k] = np.mean(v)
    return new_dict


def timestamp() -> str:
    return datetime.now().strftime("%y%m%d-%H%M%S")
