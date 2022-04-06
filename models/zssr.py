import torch
import torch.nn as nn

from models.modules import SinusoidalPosEmb


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

        self.noise_model = nn.Sequential(*layers)

        # TODO FINISH HANDLING T POSITIONAL EMBEDDING
        # Encode for positional embedding of timestep
        #time_dim = filters_per_layer
        #self.time_encoder = nn.Sequential(
        #    SinusoidalPosEmb(time_dim),
        #    nn.Linear(time_dim, time_dim * 4),
        #    nn.GELU(),
        #    nn.Linear(time_dim * 4, time_dim)
        #)

    def forward(self, x, t):
        #time_embedding = self.time_encoder(t)
        #print(time_embedding.shape)
        return self.noise_model(x)
