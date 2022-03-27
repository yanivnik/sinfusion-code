import torch
import torch.nn as nn
import numpy as np


class SeparableConv3D(nn.Module):
    def __init__(self, in_channels, kernels, padding_mode='zeros'):
        super(SeparableConv3D, self).__init__()
        c = in_channels
        k = len(kernels[0])
        pad = k // 2

        self.c1 = torch.nn.Conv1d(in_channels=c, groups=c, out_channels=c, kernel_size=k, padding=pad, padding_mode=padding_mode, bias=False)
        self.c2 = torch.nn.Conv1d(in_channels=c, groups=c, out_channels=c, kernel_size=k, padding=pad, padding_mode=padding_mode, bias=False)
        self.c3 = torch.nn.Conv1d(in_channels=c, groups=c, out_channels=c, kernel_size=k, padding=pad, padding_mode=padding_mode, bias=False)

        fix_kernel = lambda k: torch.tensor(k).float().unsqueeze(0).unsqueeze(0).expand(in_channels, -1, -1)
        w1 = fix_kernel(kernels[0])
        w2 = fix_kernel(kernels[1])
        w3 = fix_kernel(kernels[2])

        self.c1.weight.data = w1
        self.c2.weight.data = w2
        self.c3.weight.data = w3

    def forward(self, x):
        assert len(x.shape) == 5, "input needs to be of size: [N, C, T, H, W] (won't work for multiple channels at the moment)"
        # assert x.shape[0] == 1, "won't work for batches?"
        N, C, T, H, W = x.shape
        #             N  T  H  W  C
        x = x.permute(0, 2, 3, 4, 1)
        # conv1d on W         N  T  H  C  W                                                  #         N  T  H  W  C
        x = self.c1(x.permute(0, 1, 2, 4, 3).reshape(N * T * H, C, W)).reshape(N, T, H, C, W)  # .permute(0, 1, 2, 4, 3)
        # conv1d on H         N  T  W  C  H                                                  #
        x = self.c2(x.permute(0, 1, 4, 3, 2).reshape(N * T * W, C, H)).reshape(N, T, W, C, H)  # .permute(0, 1, 2, 4,3)
        # conv1d on T         N  H  W  C  T                                                  #
        x = self.c3(x.permute(0, 4, 2, 3, 1).reshape(N * H * W, C, T)).reshape(N, H, W, C, T)  # .permute(0, 1, 4, 2, 3)
        # permute back to original
        #             N, C, T, H, W
        x = x.permute(0, 3, 4, 1, 2)
        return x


def gauss1d(k=3, sigma=0.5):
    n = (k - 1.) / 2.
    x = np.ogrid[-n:n + 1]
    h = np.exp(-x * x / (2. * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


@torch.no_grad()
def generalized_sobel(vid, k=5, device='cpu'):
    dk = np.diff(gauss1d(k + 1, sigma=0.5)) * 2
    gk = gauss1d(k, sigma=0.5)

    vid = vid.mean(dim=1, keepdim=True)
    in_channels = vid.shape[1]

    sepconv3d = SeparableConv3D(in_channels=in_channels, padding_mode='replicate', kernels=[dk, gk, gk]).to(device)
    out_x = sepconv3d(vid)
    sepconv3d = SeparableConv3D(in_channels=in_channels, padding_mode='replicate', kernels=[gk, dk, gk]).to(device)
    out_y = sepconv3d(vid)
    sepconv3d = SeparableConv3D(in_channels=in_channels, padding_mode='replicate', kernels=[gk, gk, dk]).to(device)
    out_t = sepconv3d(vid)

    return torch.cat([out_x, out_y, out_t], dim=1)


@torch.no_grad()
def grad_uv(vid):
    out_x, out_y, out_t = generalized_sobel(vid)

    u_x = out_t / (1 + out_x)
    u_y = out_t / (1 + out_y)

    uv = torch.cat([u_x, u_y], dim=1)
    uv_norm = uv.norm(dim=1, keepdim=True)

    return torch.cat([uv, uv_norm], dim=1)
