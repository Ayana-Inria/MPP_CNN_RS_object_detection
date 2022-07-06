import logging
from dataclasses import dataclass
from typing import Union, Iterable, List

import numpy as np
import torch


@dataclass
class ValueMapping:
    n_classes: int
    v_min: float
    v_max: float
    is_cyclic: bool = False

    def __post_init__(self):
        self.feature_mapping = np.linspace(self.v_min, self.v_max, num=self.n_classes + 1)[:-1]

    def offset_mapping(self, offset_value):
        """
        for when you scrww up
        :param offset_value:
        :return: None
        """
        self.v_min += offset_value
        self.v_max += offset_value
        self.__post_init__()

    def get_step(self) -> float:
        return float(np.mean(np.diff(self.feature_mapping)))

    def get_range(self) -> float:
        return self.v_max - self.v_min

    @property
    def range(self) -> float:
        return self.v_max - self.v_min

    def clip(self, value: float) -> float:
        if not self.is_cyclic:
            return float(np.clip(value, self.v_min, self.v_max))
        else:
            return ((value - self.v_min) % self.range) + self.v_min

    def value_to_class(self, value: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """
        converts the value (or 1D array of values) into the corresponding integer class
        :param value: input values to convert
        :return: integer value(s) of the corresponding class/bin
        """
        if not (np.all(self.v_min <= value) and np.all(value <= self.v_max)):
            logging.warning(f"feature value {value} out of range [{self.v_min:.2f},{self.v_max:.2f}]")

        def f_mapping(x):
            return np.max(np.argwhere(np.greater_equal(x, self.feature_mapping)))

        if type(value) is np.ndarray:
            f_class = np.array(list(map(f_mapping, value)))
        else:
            f_class = f_mapping(value)
        return f_class

    def class_to_value(self, class_id: Union[int, Iterable[int], torch.Tensor]) -> Union[float, np.ndarray]:
        """
        Converts a class/bin integer index to the associated value of the class/bin
        :param class_id: integer class/bin index
        :return: values mapped to the classes
        """
        if type(class_id) is np.ndarray:
            return self.feature_mapping[class_id]
        elif type(class_id) is torch.Tensor:
            return self.feature_mapping[class_id.cpu().detach().numpy()]
        else:
            return self.feature_mapping[class_id]

    def value_to_one_hot(self, value: Union[float, np.ndarray], interpolation=None):
        """
        converts the value (or 1D array of values) into the corresponding integer class as a one hot vector
        :param interpolation: if 'linear' will assign a weight to the 2 closest classes proportional to the distance
        to each.
        :param value: input values to convert
        :return: one_hot vector of the corresponding class/bin
        """
        closest_class = self.value_to_class(value)
        remainder = np.remainder(value, self.get_step()) / self.get_step()
        if type(value) is np.ndarray:
            n_values = value.shape[0]
            h = np.zeros((n_values, self.n_classes))
            if interpolation == 'linear':
                h[np.arange(n_values), closest_class] = 1 - remainder
                h[np.arange(n_values), np.clip(closest_class + 1, 0, self.n_classes - 1)] = remainder + h[
                    np.arange(n_values), np.clip(closest_class + 1, 0, self.n_classes - 1)]
            elif interpolation is None:
                h[np.arange(n_values), closest_class] = 1
            else:
                raise ValueError
        else:
            h = np.zeros(self.n_classes)
            if interpolation == 'linear':
                if closest_class == self.n_classes - 1:
                    h[closest_class] = 1
                else:
                    h[closest_class] = 1 - remainder
                    h[closest_class + 1] = remainder
            elif interpolation is None:
                h[closest_class] = 1
            else:
                raise ValueError

        return h


def values_to_class_id(values: List[float], mappings: List[ValueMapping], as_tensor=False):
    result = []
    if len(values) == 0:
        return []
    if type(values[0]) in [list, tuple]:  # list of tuples
        itt = zip(np.array(values).swapaxes(0, 1), mappings)
    else:
        itt = zip(values, mappings)

    for v, mapping in itt:

        f_class = mapping.value_to_class(v)
        if as_tensor:
            result.append(torch.tensor(f_class))
        else:
            result.append(f_class)
    return result


def class_id_to_value(class_ids: List[int], mappings: List[ValueMapping]):
    results = []
    if type(class_ids[0]) in [list, tuple]:  # list of tuples
        itt = zip(np.array(class_ids).swapaxes(0, 1), mappings)
    elif type(class_ids[0]) is torch.Tensor:
        itt = zip(torch.stack(class_ids), mappings)
    else:
        itt = zip(class_ids, mappings)
    for c, mapping in itt:
        results.append(mapping.class_to_value(c))
    return results


def output_vector_to_value(output_vector, mappings: List[ValueMapping]):
    results = []
    for arr, mapping in zip(output_vector, mappings):
        if len(arr.shape) == 2:  # (B,C)
            value = mapping.class_to_value(np.argmax(arr, axis=1))
            results.append(value)
        elif len(arr.shape) == 4:  # (B,C,H,W)
            value = mapping.class_to_value(np.argmax(arr, axis=1))
            results.append(value)
        else:
            raise ValueError

    return results
