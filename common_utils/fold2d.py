import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

__author__ = "Ben Feinstein (ben.feinstein@weizmann.ac.il)"

__all__ = ['unfold2d', 'fold2d']


def unfold2d(input, kernel_size, stride=1, use_padding=True):
    # input dimensions (4D): n, c, h, w
    # kernel: kh, kw
    # output dimensions (6D): n, c, kh, kw, h', w'
    # output can be viewed as: n, c * kh * kw, h' * w'
    if input.dim() != 4:
        raise ValueError('expects a 4D tensor as input')
    n, c, h, w = input.size()
    kh, kw = kernel_size = _pair(kernel_size)
    sh, sw = stride = _pair(stride)
    if use_padding:
        ph, pw = padding = (kh - sh, kw - sw)
    else:
        ph, pw = padding = (0, 0)
    oh, ow = (h + 2 * ph - kh) // sh + 1, (w + 2 * pw - kw) // sw + 1
    # input = F.pad(input, pad=(pw, pw, ph, ph), mode='replicate')  # XXX: diff
    # output = F.unfold(input, kernel_size, stride=stride, padding=0)
    output = F.unfold(input, kernel_size, stride=stride, padding=padding)
    output = output.view(n, c, kh, kw, oh, ow)
    return output


def fold2d(input, stride=1, use_padding=True, *, reduce='sum', std=1.7):  # noqa
    # input dimensions (6D): n, c, kh, kw, h', w'
    # output dimensions (4D): n, c, h, w
    if input.dim() != 6:
        raise ValueError('expects a 6D tensor as input')
    n, c, kh, kw, h, w = input.shape
    sh, sw = stride = _pair(stride)
    if reduce == 'sum':
        output = _fold2d_sum(input, stride, use_padding)
    elif reduce == 'median':
        return _fold2d_median(input, stride, use_padding)
    elif reduce == 'mean':
        weights = _get_weights_fold2d_mean(input, kh, kw)
        output = _fold2d_sum(input, stride, use_padding)
        if use_padding:
            norm = weights[:, :, ::sh, ::sw, :, :].sum()
        else:
            weights = weights.expand(1, 1, kh, kw, h, w)
            norm = _fold2d_sum(weights, stride, use_padding)
            norm[norm == 0] = 1
        output = output / norm
    elif reduce == 'weighted_mean':
        weights = _get_weights_fold2d_weighted_mean(input, kh, kw, sh, sw, std)
        output = _fold2d_sum(input * weights, stride, use_padding)
        if use_padding and sh == 1 and sw == 1:
            norm = weights.sum()
        else:
            weights = weights.expand(1, 1, kh, kw, h, w)
            norm = _fold2d_sum(weights, stride, use_padding)
            norm[norm == 0] = 1
        output = output / norm
    else:
        raise ValueError(f'unknown reduction: {reduce}')
    return output


def _fold2d_sum(input, stride, use_padding):
    if input.dim() != 6:
        raise ValueError('expects a 6D tensor as input')
    n, c, kh, kw, h, w = input.shape
    sh, sw = stride = _pair(stride)
    if use_padding:
        ph, pw = padding = (kh - sh, kw - sh)
    else:
        ph, pw = padding = (0, 0)
    oh, ow = output_size = (sh * (h - 1) + kh - 2 * ph, sw * (w - 1) + kw - 2 * pw)  # noqa
    kernel_size = (kh, kw)
    input = input.reshape(n, c * kh * kw, h * w)
    output = F.fold(input, output_size, kernel_size, stride=stride, padding=padding)  # noqa
    return output


def _fold2d_median(input, stride, use_padding):
    if input.dim() != 6:
        raise ValueError('expects a 6D tensor as input')
    n, c, kh, kw, h, w = input.shape
    sh, sw = stride = _pair(stride)
    dh, dw = kh // sh, kw // sw
    if kh % sh != 0:
        raise ValueError('kh should be divisible by sh')
    if kw % sw != 0:
        raise ValueError('kw should be divisible by sw')

    if use_padding:
        ph, pw = (kh - sh, kw - sw)
        oh, ow = (sh * (h - 1) + kh - 2 * ph, sw * (w - 1) + kw - 2 * pw)
        # output = input.new_full(size=(dh * dw, n, c, oh, ow), fill_value=float('nan'))  # XXX: debug  # noqa
        output = input.new_zeros(size=(dh * dw, n, c, oh, ow))
        for i in range(kh):
            for j in range(kw):
                ii, jj = i // sh, j // sw
                sii, sjj = (kh - i - 1) // sh, (kw - j - 1) // sw
                output[ii * dw + jj, :, :, i % sh::sh, j % sw::sw] = input[:, :, i, j, sii:h - ii, sjj:w - jj]  # noqa
        output = torch.median(output, dim=0)[0]

    else:
        if not hasattr(torch, 'nanmedian'):
            raise RuntimeError('fold2d_median with use_padding==False depends on torch.nanmedian()')  # noqa
            # pass  # XXX: debug
        if sh != 1 or sw != 1:
            raise NotImplementedError('fold2d_median with use_padding==False and stride!=1 is not implemented')  # noqa
        oh, ow = (sh * (h - 1) + kh, sw * (w - 1) + kw)
        output = input.new_full(size=(dh * dw, n, c, oh, ow), fill_value=float('nan'))  # noqa
        # output = input.new_zeros(size=(dh * dw, n, c, oh, ow))  # XXX: debug
        for i in range(kh):
            for j in range(kw):
                ii, jj = i // sh, j // sw
                output[ii * dw + jj, :, :, i:h + i:sh, j:w + j:sw] = input[:, :, i, j, :, :]  # noqa
        output = torch.nanmedian(output, dim=0)[0]
        # output = torch.median(output, dim=0)[0]  # XXX: debug

    return output


def _get_weights_fold2d_mean(input, kh, kw):
    weights = input.new_ones(size=(kh, kw))
    return weights.view(1, 1, kh, kw, 1, 1)


def _get_weights_fold2d_weighted_mean(input, kh, kw, sh, sw, std):
    to = {'device': input.device, 'dtype': input.dtype}
    gh = sh * torch.linspace(-1, 1, kh, **to)
    gw = sw * torch.linspace(-1, 1, kw, **to)
    nh = torch.exp(-0.5 * (gh / std).pow(2))
    nw = torch.exp(-0.5 * (gw / std).pow(2))
    weights = torch.einsum('i,j->ij', nh, nw)
    return weights.view(1, 1, kh, kw, 1, 1)
