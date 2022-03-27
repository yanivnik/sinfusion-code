from math import pi

import torch


def support_sz(sz):
    def wrapper(f):
        f.support_sz = sz
        return f

    return wrapper


@support_sz(4)
def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return ((1.5 * absx3 - 2.5 * absx2 + 1.) * (absx <= 1.).type_as(x) +
            (-0.5 * absx3 + 2.5 * absx2 - 4. * absx + 2.) *
            ((1. < absx) & (absx <= 2.)).type_as(x))


@support_sz(4)
def lanczos2(x):
    eps = torch.finfo(x.dtype).eps
    return (((torch.sin(pi * x) * torch.sin(pi * x / 2) + eps) /
             ((pi**2 * x**2 / 2) + eps)) * (abs(x) < 2).type_as(x))


@support_sz(6)
def lanczos3(x):
    eps = torch.finfo(x.dtype).eps
    return (((torch.sin(pi * x) * torch.sin(pi * x / 3) + eps) /
             ((pi**2 * x**2 / 3) + eps)) * (abs(x) < 3).type_as(x))


@support_sz(2)
def linear(x):
    return ((x + 1) * ((-1 <= x) & (x < 0)).type_as(x) + (1 - x) *
            ((0 <= x) & (x <= 1)).type_as(x))


@support_sz(1)
def box(x):
    return ((-0.5 < x) & (x <= 0.5)).type_as(x)
