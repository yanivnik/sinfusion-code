import torch

from base_diffusion_model import BaseDiffusionModel
from modules import Up, Down, DoubleConv, SAWrapper, OutConv


# TODO: REFACTOR AND DOCUMENT THESE CLASSES AND THEIR USAGES
#       OR SIMPLY REMOVE THEM IF WE DECIDE WE DON'T NEED THE CODE

class DiffusionModel(BaseDiffusionModel):
    """
    Basic Diffusion model, used for datasets with 32x32 inputs (padded MNIST, CIFAR)
    """
    def __init__(self, in_size, t_range, img_channels=3):
        super().__init__(in_size, t_range)

        bilinear = True
        self.inc = DoubleConv(img_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, img_channels)
        self.sa1 = SAWrapper(256, 8)
        self.sa2 = SAWrapper(256, 4)
        self.sa3 = SAWrapper(128, 8)

    def forward(self, x, t):
        """
        Model is U-Net with added positional encodings and self-attention layers.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1) + self.pos_encoding(t, 128, 16)
        x3 = self.down2(x2) + self.pos_encoding(t, 256, 8)
        x3 = self.sa1(x3)
        x4 = self.down3(x3) + self.pos_encoding(t, 256, 4)
        x4 = self.sa2(x4)
        x = self.up1(x4, x3) + self.pos_encoding(t, 128, 8)
        x = self.sa3(x)
        x = self.up2(x, x2) + self.pos_encoding(t, 64, 16)
        x = self.up3(x, x1) + self.pos_encoding(t, 64, 32)
        output = self.outc(x)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer


# TODO REFACTOR INTO A SINGLE CLASS FOR ALL SIZES OF MODELS
class LargeDiffusionModel(BaseDiffusionModel):
    """
    Diffusion model used for 256x256 inputs
    """
    def __init__(self, in_size, t_range, img_channels=3):
        super().__init__(in_size, t_range)

        self.inc = DoubleConv(img_channels, 64)
        self.down1 = Down(64, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 128)
        self.down4 = Down(128, 256)
        self.down5 = Down(256, 256)

        self.up1 = Up(512, 128, bilinear=True)
        self.up2 = Up(256, 128, bilinear=True)
        self.up3 = Up(256, 64, bilinear=True)
        self.up4 = Up(128, 64, bilinear=True)
        self.up5 = Up(128, 64, bilinear=True)
        self.outc = OutConv(64, img_channels)

        self.sa1 = SAWrapper(256, 16)
        self.sa2 = SAWrapper(256, 8)
        self.sa3 = SAWrapper(128, 16)

    def forward(self, x, t):
        """
        Model is U-Net with added positional encodings and self-attention layers.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1) + self.pos_encoding(t, 64, 128)
        x3 = self.down2(x2) + self.pos_encoding(t, 128, 64)
        x4 = self.down3(x3) + self.pos_encoding(t, 128, 32)
        x5 = self.down4(x4) + self.pos_encoding(t, 256, 16)
        x5 = self.sa1(x5)
        x6 = self.down5(x5) + self.pos_encoding(t, 256, 8)

        x6 = self.sa2(x6)

        x = self.up1(x6, x5) + self.pos_encoding(t, 128, 16)
        x = self.sa3(x)
        x = self.up2(x, x4) + self.pos_encoding(t, 128, 32)
        x = self.up3(x, x3) + self.pos_encoding(t, 64, 64)
        x = self.up4(x, x2) + self.pos_encoding(t, 64, 128)
        x = self.up5(x, x1) + self.pos_encoding(t, 64, 256)
        output = self.outc(x)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer
