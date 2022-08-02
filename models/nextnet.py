import math

import torch
from torch import nn

from models.modules import ConvNextBlock, SinusoidalPosEmb


class NextNet(nn.Module):
    """
    A backbone model comprised of a chain of ConvNext blocks, with skip connections.
    The skip connections are connected similar to a "U-Net" structure (first to last, middle to middle, etc).
    """
    def __init__(self, in_channels=3, out_channels=3, depth=9, filters_per_layer=64):
        super().__init__()

        if isinstance(filters_per_layer, list) or isinstance(filters_per_layer, tuple):
            dims = filters_per_layer
        else:
            dims = [filters_per_layer] * depth

        #time_dim = dim = filters_per_layer
        time_dim = filters_per_layer[0]
        self.depth = depth
        self.layers = nn.ModuleList([])

        # First block doesn't have a normalization layer
        self.layers.append(ConvNextBlock(in_channels, dims[0], time_emb_dim=time_dim, norm=False))

        for i in range(1, math.ceil(self.depth / 2)):
            self.layers.append(ConvNextBlock(dims[i - 1], dims[i], time_emb_dim=time_dim, norm=True))
        for i in range(math.ceil(self.depth / 2), depth):
            self.layers.append(ConvNextBlock(2 * dims[i - 1], dims[i], time_emb_dim=time_dim, norm=True))

        # After all blocks, do a 1x1 conv to get the required amount of output channels
        self.final_conv = nn.Conv2d(dims[depth - 1], out_channels, 1)

        # Encoder for positional embedding of timestep
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )

    def forward(self, x, t):
        time_embedding = self.time_encoder(t)

        residuals = []
        for layer in self.layers[0: math.ceil(self.depth / 2)]:
            x = layer(x, time_embedding)
            residuals.append(x)

        for layer in self.layers[math.ceil(self.depth / 2): self.depth]:
            x = torch.cat((x, residuals.pop()), dim=1)
            x = layer(x, time_embedding)

        return self.final_conv(x)
