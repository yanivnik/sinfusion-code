import torch
import torch.nn as nn


class ZSSRNet(nn.Module):
    def __init__(self, depth=8):
        super().__init__()

        self.depth = depth

        # create conv-relu network
        layers = []
        for i in range(self.depth - 1):
            inc, outc = 64, 64
            if i == 0: inc = 3
            layers.append(nn.Conv2d(inc, outc, 3, padding=1, padding_mode='replicate', bias=False))
            layers.append(nn.ReLU())
        # add last layer
        layers.append(nn.Conv2d(64, 3, 3, padding=1, padding_mode='replicate', bias=False))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        inputs = x.clone()
        x = self.body(x)
        x = torch.add(x, inputs)

        return x



