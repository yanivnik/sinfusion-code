import torch
import torch.nn as nn

from models.modules import SinusoidalPosEmb
from einops import rearrange


class ZSSRNet(nn.Module):
    """
    The base noise model is built exactly like the fully convolutional model from the ZSSR work
    For more information and code, see - https://github.com/assafshocher/ZSSR).
    """

    def __init__(self, depth=8, filters_per_layer=64, kernel_size=3):
        super().__init__()

        layers = []

        # First layer
        layers.append(nn.Conv2d(3, filters_per_layer, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode='replicate', bias=False))
        layers.append(nn.ReLU())

        # Mid layers
        for layer_idx in range(1, depth - 1):
            layers.append(nn.Conv2d(filters_per_layer, filters_per_layer, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode='replicate', bias=False))
            layers.append(nn.ReLU())

        # Last layer (no activation)
        layers.append(nn.Conv2d(filters_per_layer, 3, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode='replicate', bias=False))

        #self.noise_model = nn.Sequential(*layers)
        self.layers = layers
        for i in range(len(layers)):
            self.register_module(f"layer{i}", layers[i])

        # TODO FINISH HANDLING T POSITIONAL EMBEDDING
        # Encode for positional embedding of timestep
        time_dim = filters_per_layer
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )

    def forward(self, x, t):
        time_embedding = self.time_encoder(t)

        y = x
        y = self.layers[0](y)
        y = self.layers[1](y)
        for layer in self.layers[2:-1]:
            if isinstance(layer, nn.Conv2d):
                y = layer(y) + rearrange(time_embedding, 'n d -> n d 1 1')
            else:
                y = layer(y)
        y = self.layers[-1](y)
        return y

        # return self.noise_model(x)
