import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numbers


class GaussianSmoothing(nn.Module):
    """
    from: https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10?u=nivniv
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).

    Example:
        smoothing = GaussianSmoothing(3, 5, 1)
        input = torch.rand(1, 3, 100, 100)
        input = F.pad(input, (2, 2, 2, 2), mode='reflect')
        output = smoothing(input)
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input, padding=0):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=padding)


class PatchExtractor2d(nn.Module):
    def __init__(self, in_channels, k, padding=0, device='cuda'):
        super().__init__()
        self.padding = padding
        self.weight = torch.zeros(k ** 2 * in_channels, in_channels, k, k).to(device)
        for cin in range(in_channels):
            for j in range(k):
                for l in range(k):
                    cout = k ** 2 * cin + k * j + l
                    self.weight[cout, cin, j, l] = 1

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=self.padding)


class PatchExtractor3d(nn.Module):
    def __init__(self, in_channels, k, padding=0, device='cuda'):
        super().__init__()
        self.padding = padding
        self.weight = torch.zeros(k ** 3 * in_channels, in_channels, k, k, k).to(device)
        for cin in range(in_channels):
            for i in range(k):
                for j in range(k):
                    for l in range(k):
                        cout = k ** 3 * cin + k ** 2 * i + k * j + l
                        self.weight[cout, cin, i, j, l] = 1

    def forward(self, x):
        return F.conv3d(x, self.weight, padding=self.padding)
