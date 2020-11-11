import torch
import torch.nn as nn
import torch.nn.init


class BiasInitialization:
    def _init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)


class Dummy(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
