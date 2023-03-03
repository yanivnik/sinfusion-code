import os
import tempfile

import numpy as np
import torch
from PIL import Image

try:
    import ipywidgets as widgets
    from IPython.display import display

    __IMSHOW_ENABLED = True
except ImportError:
    __IMSHOW_ENABLED = False


__all__ = ['to_numpy', 'np2pt', 'pt2np', 'imread', 'imwrite', 'imshow']

_BOUNDS = (0.0, 1.0)


def to_numpy(tensor, clone=True):
    tensor = tensor.detach()
    tensor = tensor.clone() if clone else tensor
    return tensor.cpu().numpy()


def np2pt(arr):
    return torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def pt2np(tensor):
    return to_numpy(tensor.squeeze(0).permute(1, 2, 0))


def _img_to_float32(image, bounds):
    vmin, vmax = bounds
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = np.clip((vmax - vmin) * image + vmin, vmin, vmax)
    return image


def _torch_to_np(image):
    image = to_numpy(image)
    if image.ndim == 3:
        image = image.transpose((1, 2, 0))
    elif image.ndim == 4:
        image = image.transpose((0, 2, 3, 1))
    else:
        raise ValueError()
    return image


def _img_to_uint8(image, bounds):
    if isinstance(image, torch.Tensor):
        image = _torch_to_np(image)

    if image.dtype != np.uint8:
        vmin, vmax = bounds
        image = (image.astype(np.float32) - vmin) / (vmax - vmin)
        image = (image * 255.0).round().clip(0, 255).astype(np.uint8)
    return image


def _check_path(path):
    path = os.path.abspath(path)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    return path


## Image
def imread(fname, bounds=_BOUNDS, mode='RGB', pt=True, **kwargs):
    # image = image_to_array(load_img(fname, **kwargs))
    image = Image.open(fname, **kwargs).convert(mode=mode)
    image = _img_to_float32(image, bounds)
    if pt:
        image = np2pt(image)
    return image


def imwrite(fname, image, bounds=_BOUNDS, **kwargs):
    fname = _check_path(fname)
    image = _img_to_uint8(image, bounds)
    image = Image.fromarray(image)
    image.save(fname, **kwargs)


if __IMSHOW_ENABLED:
    def _imshow(image, bounds, *, fname=None):
        fd = None
        if fname is None:
            fd, fname = tempfile.mkstemp(suffix='.png')
        imwrite(fname, image, bounds)
        if fd is not None:
            os.close(fd)

        img_output = widgets.Image.from_file(fname)
        display(img_output)

else:
    def _imshow(image, bounds, *, fname=None):
        raise NotImplementedError()


def imshow(img, bounds=_BOUNDS, **kwargs):
    if len(img.shape) == 4 and img.shape[0] == 1:
        img = img.squeeze(0)
    if len(img.shape) == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)

    return _imshow(img, bounds, **kwargs)


def tensor2npimg(x, vmin=-1, vmax=1, normmaxmin=False, to_numpy=True):
    """tensor in [-1,1] (1x3xHxW) --> numpy image ready to plt.imshow"""
    if normmaxmin:
        vmin = x.min().item()
        vmax = x.max().item()
    final = x[0].add(-vmin).div(vmax-vmin).mul(255).add(0.5).clamp(0, 255)

    if to_numpy:
        final = final.permute(1, 2, 0)
        # if input has 1-channel, pass grayscale to numpy
        if final.shape[-1] == 1:
            final = final[:,:,0]
        return final.to('cpu', torch.uint8).numpy()
    else:
        return final.to('cpu', torch.uint8)


torch255tonpimg = lambda x: x[0].add(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
