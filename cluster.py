"""
Sonata Lite - Cluster (Head) layer for self-distillation
Author: Andrei-Iosif Ioja (iojaandrei11@gmail.com)

Adaptation from Sonata v1m1 Base
    Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
    Link: https://github.com/Pointcept/Pointcept/blob/main/pointcept/models/sonata/sonata_v1m1_base.py
"""

from packaging import version
import torch
import torch.nn as nn
from registry import MODELS

@MODELS.register_module()
class OnlineCluster(nn.Module):
    """
    The implementation of online clustering layer for self-distillation.
    """
    def __init__(self, in_ch: int, hidd_ch: int = 4096, em_ch: int = 512, num_prototypes: int = 4096):
        """
        :param in_ch: The number of input channels.
        :param hidd_ch: The number of hidden channels.
        :param em_ch: The number of embedding channels.
        :param num_prototypes: The number of prototypes.
        """
        super().__init__()
        # The first two linear layers of the head (+L2 normalization, that is applied in the forward method)
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidd_ch),
            nn.GELU(),
            nn.Linear(hidd_ch, em_ch)
        )
        self.apply(self._init_weights)
        # Prototype layer
        if version.parse(torch.__version__) >= version.parse("2.1.0"):
            self.prototype = nn.utils.parametrizations.weight_norm(
                nn.Linear(em_ch, num_prototypes, bias=False)
            )
            # Set the magnitude (length) of each vector to 1 and non-trainable
            self.prototype.parametrizations.weight.original0.data.fill_(1)
            self.prototype.parametrizations.weight.original0.requires_grad = False
        else:
            self.prototype = nn.utils.weight_norm(
                nn.Linear(em_ch, num_prototypes, bias=False)
            )
            # Set the magnitude (length) of each vector to 1 and non-trainable
            self.prototype.weight_g.data.fill_(1)
            self.prototype.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        # Apply L2 normalization before prototyping layer
        x = nn.functional.normalize(x, dim=-1, p=2, eps=(1e-6 if x.dtype == torch.float16 else 1e-12))
        similarity = self.prototype(x)
        return similarity

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
