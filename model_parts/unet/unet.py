from typing import List

import torch.nn.functional as F
from torch import nn, Tensor

from model_parts.unet.unet_parts import DoubleConv, Down, Up


def pad_before_infer(image: Tensor, depth):
    div = 2 ** depth
    shape = image.shape[1:]

    pad = [0, 0]
    for i in range(2):
        if shape[i] % div != 0:
            pad[i] = div - (shape[i] % div)

    if pad[0] != 0 or pad[1] != 0:
        pad_image = F.pad(image, pad=(0, pad[1], 0, pad[0]))
        return pad_image, pad
    return image, pad


class Unet(nn.Module):

    def __init__(self, hidden_dims: List[int], in_channels: int, device):
        super(Unet, self).__init__()

        self.device = device

        self.descending_path = nn.ModuleList()
        # if hidden_dims is None:
        #     hidden_dims = [32, 64, 128, 256, 512]

        for i, out_channels in enumerate(hidden_dims):
            if i == 0:
                op = DoubleConv(in_channels, out_channels)
            else:
                op = Down(in_channels, out_channels)
            self.descending_path.append(op)
            in_channels = out_channels

        self.ascending_path = nn.ModuleList()
        for out_channels in hidden_dims[::-1][1:]:
            self.ascending_path.append(Up(in_channels, out_channels))
            in_channels = out_channels

        self.out_channels = in_channels

    def forward(self, x: Tensor):
        last_tensor = x.float()
        x_down = []
        for down in self.descending_path:
            last_tensor = down(last_tensor)
            x_down.append(last_tensor)

        for up, concat_tensor in zip(self.ascending_path, x_down[::-1][1:]):
            last_tensor = up(last_tensor, concat_tensor)

        return last_tensor
