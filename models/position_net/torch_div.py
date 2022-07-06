from typing import List

import torch
from torch import Tensor
from torch.nn import Module, functional


def torch_divergence(f: Tensor, indexing='xy'):
    """
    :param f: vecotr field compunents of shape (B,C,H,W) where C is the number of dims
    :param indexing: "xy" or "ij", see np.meshgrid indexing
    """
    b = f.shape[0]
    if b == 1:
        f = torch.concat([f, f], dim=0)
    num_dims = f.shape[1]
    spacing = 1.0
    if indexing == "xy":
        stack = [torch.gradient(f[:, num_dims - i - 1], dim=i + 1)[0] for i in range(num_dims)]
    elif indexing == "ij":
        stack = [torch.gradient(f[:, i], dim=i + 1)[0] for i in range(num_dims)]
    else:
        raise ValueError
    stack = torch.stack(stack, dim=1)
    res = torch.sum(stack, dim=1, keepdim=True)
    if b == 1:
        return res[[0]]
    return res


class Divergence(Module):

    def __init__(self, div_channels: List[int], mask_channel: int):
        super(Divergence, self).__init__()
        self.div_channels = div_channels
        self.mask_channel = mask_channel

    def forward(self, x: Tensor):
        div_c = torch_divergence(x[:, self.div_channels], indexing='ij')

        mask = x[:, [self.mask_channel]]

        return div_c * mask
