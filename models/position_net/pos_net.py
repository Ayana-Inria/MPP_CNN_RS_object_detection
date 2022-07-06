from typing import List

import torch
from torch import nn, Tensor

from model_parts.unet.unet import Unet


class PosNet(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, device, hidden_dims: List[int] = None):
        super(PosNet, self).__init__()
        self.device = device

        self.backbone = Unet(
            hidden_dims=hidden_dims,
            in_channels=in_channels,
            device=device
        ).to(device)

        self.final_layer = nn.Conv2d(
            in_channels=self.backbone.out_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            device=self.device
        )

    def forward(self, x: Tensor) -> Tensor:
        last_tensor = self.backbone.forward(x)
        return self.final_layer(last_tensor)

    def forward_with_softmax(self, x: Tensor, temperature: float = 1.0) -> Tensor:
        return torch.softmax(self.forward(x) / temperature, dim=1)

    def forward_class(self, x: Tensor) -> Tensor:
        return torch.argmax(self.forward(x), dim=1)
