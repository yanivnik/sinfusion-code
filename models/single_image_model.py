from models.base_diffusion_model import BaseDiffusionModel
import torch
# from common_utils.fold2d import unfold2d
import torch.nn as nn


class SingleImageDiffusionModel(BaseDiffusionModel):
    """
    A diffusion model meant to be trained on a single image as input.
    """

    def __init__(self, in_size, t_range, img_channels=3):
        super().__init__(in_size, t_range)

        kernel_size = 3
        self.conv_channels = 64

        self.conv0 = nn.Conv2d(img_channels, self.conv_channels, kernel_size, padding=kernel_size // 2)
        self.conv1 = nn.Conv2d(self.conv_channels, self.conv_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(self.conv_channels, self.conv_channels, kernel_size, padding=kernel_size // 2)
        self.conv3 = nn.Conv2d(self.conv_channels, self.conv_channels, kernel_size, padding=kernel_size // 2)
        self.conv4 = nn.Conv2d(self.conv_channels, self.conv_channels, kernel_size, padding=kernel_size // 2)
        self.conv5 = nn.Conv2d(self.conv_channels, self.conv_channels, kernel_size, padding=kernel_size // 2)
        self.conv6 = nn.Conv2d(self.conv_channels, self.conv_channels, kernel_size, padding=kernel_size // 2)
        self.conv7 = nn.Conv2d(self.conv_channels, img_channels, kernel_size, padding=kernel_size // 2)
        self.gelu = nn.GELU()

    def forward(self, x, t):
        _, C, H, W = x.shape
        y = self.conv0(x)
        y = self.gelu(y)
        y = self.conv1(y) + self.pos_encoding(t, self.self.conv_channels, H)
        y = self.gelu(y)
        y = self.conv2(y) + self.pos_encoding(t, self.self.conv_channels, H)
        y = self.gelu(y)
        y = self.conv3(y) + self.pos_encoding(t, self.self.conv_channels, H)
        y = self.gelu(y)
        y = self.conv4(y) + self.pos_encoding(t, self.self.conv_channels, H)
        y = self.gelu(y)
        y = self.conv5(y) + self.pos_encoding(t, self.self.conv_channels, H)
        y = self.gelu(y)
        y = self.conv6(y) + self.pos_encoding(t, self.self.conv_channels, H)
        y = self.gelu(y)
        y = self.conv7(y)
        return y


class UnfoldedSingleImageDiffusionModel(BaseDiffusionModel):
    def __init__(self, in_size, t_range, img_channels=3):
        super().__init__(in_size, t_range)


    def forward(self, x, t):
        pass