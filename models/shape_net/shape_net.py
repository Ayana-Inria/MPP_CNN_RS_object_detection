import math
from typing import List, Union

from torch import nn, Tensor
import torch

from model_parts.unet.unet import Unet

PI = torch.tensor(math.pi)


class ShapeNet(nn.Module):

    def __init__(self, in_channels, out_features: int, out_feat_size: Union[int, List[int]], device,
                 hidden_dims: List = None):
        super(ShapeNet, self).__init__()

        self.device = device

        self.backbone = Unet(
            hidden_dims=hidden_dims,
            in_channels=in_channels,
            device=device
        )

        self.final_layers = nn.ModuleList()
        if type(out_feat_size) is int:
            out_feat_size = [out_feat_size for _ in range(out_features)]
        for feat_size in out_feat_size:
            self.final_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.backbone.out_channels,
                        out_channels=feat_size,
                        kernel_size=(1, 1), device=device
                    )
                )
            )

    def forward(self, x: Tensor, **kwargs) -> List[Tensor]:
        last_tensor = self.backbone.forward(x)

        features = []
        for fl in self.final_layers:
            features.append(fl(last_tensor))
        return features

    def forward_with_softmax(self, x: Tensor, temperature: float = 1.0) -> List[Tensor]:
        last_tensor = self.backbone.forward(x)

        features = []
        for fl in self.final_layers:
            features.append(torch.softmax(fl(last_tensor) / temperature, dim=1))
        return features
