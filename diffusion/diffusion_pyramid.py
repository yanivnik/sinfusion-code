from collections.abc import Iterable

import pytorch_lightning as pl
import torch
from PIL import Image

from common_utils.resize_right import resize
from datasets.cropset import CropSet
from diffusion.diffusion import GaussianDiffusion
from models.zssr import ZSSRNet
from diffusion.diffusion_utils import save_diffusion_sample


class GaussianDiffusionPyramid(object):
    """
    A class for a diffusion pyramid. Each level in the pyramid is a diffusion model
    which handled a different scale of the single input image.
    The coarsest level in the pyramid (level 0) handles the lowest resolution, and the last
    level in the pyramid handles the original image (highest) resolution.
    """
    def __init__(self, image_path, levels, size_ratios, timesteps, models=None, logger=None):
        """
        Args:
            image_path (str):
                The path to the image which the pyramid is trained upon.
            levels (int):
                The amount of levels in the pyramid.
            size_ratios (list(float) or float):
                A list of floats, representing the scale ratio between each level with its previous level.
                 If a single float is given, all the ratios are equal to it.
            timesteps (list(int) or int):
                A list of integers, representing T (diffusion timesteps) for each level.
                If a single int is given, all the timesteps are equal to it.
            models (list(model)):
                A list of configurable backbone models for each level.
        """
        self.levels = levels

        if isinstance(size_ratios, Iterable):
            assert len(size_ratios) == (levels - 1)
            self.size_ratios = size_ratios
        else:
            # Use the same size ratio for all levels
            self.size_ratios = [size_ratios] * (levels - 1)

        # Generate image pyramid
        self.images = [Image.open(image_path)]
        for ratio in self.size_ratios:
            image_size = (int(self.images[0].size[0] * ratio), int(self.images[0].size[1] * ratio))
            self.images.insert(0, self.images[0].resize(image_size))

        if isinstance(timesteps, Iterable):
            assert len(timesteps) == levels
            self.timesteps = timesteps
        else:
            # Use the same timestep parameter for the training and sampling in each level
            self.timesteps = [timesteps] * levels

        if models is None:
            # Default itentical backbone networks
            models = []
            for i in range(levels):
                models.append(ZSSRNet(filters_per_layer=64, kernel_size=3))

        self.diffusion_models = []
        for i in range(0, self.levels):
            self.diffusion_models.append(GaussianDiffusion(model=models[i], timesteps=self.timesteps[0],
                                                           noising_timesteps_ratio=self.timesteps[i] / self.timesteps[0], # TODO WRITE THIS CODE BETTER
                                                           auto_sample=False))

        self.logger = logger or False

    def train(self, training_steps):
        """
        Train the models in the pyramid, level after level.

        Args:
            training_steps (list(int)): The amount of training examples per level.
        """
        for level in range(self.levels):
            dataset = CropSet(image=self.images[level], crop_size=(32, 32))
            loader = torch.utils.data.DataLoader(dataset, batch_size=1)
            trainer = pl.Trainer(logger=self.logger, max_steps=training_steps[level], gpus=1, auto_select_gpus=True)
            trainer.fit(self.diffusion_models[level], loader)

    def sample(self, sample_size, batch_size):
        """
        Sample a batch of images from the pyramid.

        First, the coarsest level samples an LR sample from pure noise.
        Then, for each other level, the previous level samples are upsampled (bicubic), partially noised,
        and then denoised by the current level to generate a higher resolution sample.

        Args:
            sample_size (tuple(int, int) or int): The spatial dimensions of the final sample output.
            batch_size (int): The size of the batch to sample.
        """
        sample_size = sample_size if isinstance(sample_size, tuple) else (sample_size, sample_size)
        sample_size_per_level = [sample_size]
        for ratio in reversed(self.size_ratios):
            sample_size_per_level.insert(0, (int(sample_size_per_level[0][0] * ratio), int(sample_size_per_level[0][1] * ratio)))

        # In the coarsest level, pure noise is sampled
        sample = self.diffusion_models[0].sample(image_size=sample_size_per_level[0], batch_size=batch_size)
        save_diffusion_sample(sample, f"{self.logger.log_dir}/level_0_sample.png")  # For debugging

        for level in range(1, self.levels):
            # Upsample the lower level output
            sample = resize(sample, out_shape=sample_size_per_level[level])

            # Generate and add noise to the upsampled output
            noise = torch.randn_like(sample)
            timestep_tensor = torch.tensor((self.timesteps[level] - 1,))
            sample = self.diffusion_models[level].q_sample(sample, t=timestep_tensor, noise=noise)

            sample = self.diffusion_models[level].sample(image_size=sample_size_per_level[level], batch_size=batch_size,
                                                         custom_initial_img=sample)
            save_diffusion_sample(sample, f"{self.logger.log_dir}/level_{level}_sample.png")  # For debugging

        return sample
