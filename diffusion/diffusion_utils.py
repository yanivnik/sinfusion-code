import os
from collections.abc import Iterable

import numpy as np
import torch
from PIL import Image


def cosine_noise_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


def extract(a, t, x_shape):
    """
    Get the values from a in location t, broadcasted to x_shape.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    y = out.reshape(b, *((1,) * (len(x_shape) - 1)))
    return y


def save_diffusion_sample(sample, output_path=None, wandb_logger=None):
    """
    Normalizes the image which was sampled from a diffusion model and saves it to an output file.

    Args:
        sample (Torch.tensor): A tensor containing the sample (or a batch of samples) to be saved.
        output_path (string): The output path to save the image in.
    """
    assert output_path is not None or wandb_logger is not None, 'You must either supply an output path to save the images'
    sample = (sample.clamp(-1, 1) + 1) / 2
    sample = (sample * 255).type(torch.uint8).moveaxis(1, 3).cpu().numpy()

    if output_path:
        if len(sample.shape) == 4:
            # Handle batch of samples
            for i in range(sample.shape[0]):
                dirname, fpath = os.path.split(output_path)
                current_sample_output_path = os.path.join(dirname, f'{i}_{fpath}')
                Image.fromarray(sample[i]).save(current_sample_output_path)
        else:
            Image.fromarray(sample).save(output_path)

    if wandb_logger is not None:
        wandb_logger.log_image(key="samples", images=list(sample))


def get_pyramid_parameter_as_list(param, pyramid_levels):
    """
    Convert param to be a list of length pyramid_levels.
    If param is already a list (or any sequence) then make sure it is in the correct length.
    """
    if isinstance(param, Iterable):
        assert len(param) == pyramid_levels
        return param
    else:
        # Use the same parameter for all levels
        return [param] * pyramid_levels
