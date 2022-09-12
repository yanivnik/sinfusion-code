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


def linear_noise_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)


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
        if sample.shape[0] > 1:
            # Handle batch of samples
            for i in range(sample.shape[0]):
                dirname, fpath = os.path.split(output_path)
                current_sample_output_path = os.path.join(dirname, f'{i}_{fpath}')
                Image.fromarray(sample[i]).save(current_sample_output_path)
        else:
            Image.fromarray(sample[0]).save(output_path)

    if wandb_logger is not None:
        wandb_logger.log_image(key="samples", images=list(sample))


def to_torch(tensor):
    return torch.tensor(tensor, dtype=torch.float32)
