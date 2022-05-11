import os
import numpy as np
import torch
from PIL import Image


# TODO DOCUMENT AND REFACTOR


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
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def save_diffusion_sample(sample, output_path):
    """
    Normalizes the image which was sampled from a diffusion model and saves it to an output file.

    Args:
        sample (Torch.tensor): A tensor containing the sample (or a batch of samples) to be saved.
        output_path (string): The output path to save the image in.
    """
    sample = (sample.clamp(-1, 1) + 1) / 2
    sample = (sample * 255).type(torch.uint8).moveaxis(1, 3).cpu().numpy()

    if len(sample.shape) == 4:
        # Handle batch of samples
        for i in range(sample.shape[0]):
            current_sample_output_path = os.path.splitext(output_path)[0] + str(i) + os.path.splitext(output_path)[1]
            Image.fromarray(sample[i]).save(current_sample_output_path)
    else:
        Image.fromarray(sample).save(output_path)
