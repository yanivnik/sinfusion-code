import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .image import tensor2npimg


def plot_colorbar(x, figsize=(15,15)):
    plt.figure(figsize=figsize)
    ax = plt.gca()
    im = ax.imshow(x)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


def show_img(x, figsize=(15,15)):
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(tensor2npimg(x))
    plt.show()


def norm_affine_minmax(x, image=True):
    """
    Linear transform x: [min, max] --> [0,1]  (image=True)
    Linear transform x: [min, max] --> [-1,1] (image=False)
    """
    maxmin = x.max() - x.min()
    if maxmin == 0:
        return torch.ones_like(x)
    if image:
        return x.add(-x.min()).div(maxmin)
    else:
        return x.add(-x.min()).div(maxmin).mul(2).add(-1)
