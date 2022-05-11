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
    TODO DOCUMENT CLASS AND ALL METHODS
    """
    def __init__(self, image_path, levels, size_ratios, timesteps, models=None, logger=None):
        """
        0 - Coarsest level
        TODO DOCUMENT
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

        self.diffusion_models = []
        if models is None:
            # Default itentical backbone networks
            models = []
            for i in range(levels):
                models.append(ZSSRNet(filters_per_layer=64, kernel_size=3))

        for i in reversed(range(0, self.levels)):
            self.diffusion_models.insert(0, GaussianDiffusion(model=models[i], timesteps=1000,
                                                              noising_timesteps_ratio=self.timesteps[i] / 1000, # TODO WRITE THIS CODE BETTER
                                                              auto_sample=False))

        self.logger = logger or False

    def train(self, training_steps):
        """
        Train the models in the pyramid, one by one.

        Args:
            training_steps ():
        """
        for level in range(self.levels):
            # TODO TRAIN ON UPSAMPLED IMAGES SO THAT TARGET WILL BE CORRECT LARGER IMAGE
            dataset = CropSet(image=self.images[level], crop_size=(32, 32))
            loader = torch.utils.data.DataLoader(dataset, batch_size=1)
            trainer = pl.Trainer(logger=self.logger, max_steps=training_steps[level], gpus=1, auto_select_gpus=True)
            trainer.fit(self.diffusion_models[level], loader)

    def sample(self, sample_size, batch_size):
        """

        """
        sample_size = sample_size if isinstance(sample_size, tuple) else (sample_size, sample_size)

        sample_size_per_level = [sample_size]
        for ratio in reversed(self.size_ratios):
            sample_size_per_level.insert(0, (int(sample_size_per_level[0][0] * ratio), int(sample_size_per_level[0][1] * ratio)))

        # In the coarsest level, pure noise is sampled
        sample = self.diffusion_models[0].sample(image_size=sample_size_per_level[0], batch_size=batch_size)
        save_diffusion_sample(sample, f"{self.logger.log_dir}/level_0_sample.png")

        for level in range(1, self.levels):
            # Upsample the lower level output
            sample = resize(sample, out_shape=sample_size_per_level[level])
            save_diffusion_sample(sample, f"{self.logger.log_dir}/level_{level-1}_upscaled_sample.png")

            # Generate and add noise to the upsampled output
            noise = torch.randn_like(sample)
            timestep_tensor = torch.tensor((self.timesteps[level] - 1,))
            sample = self.diffusion_models[level].q_sample(sample, t=timestep_tensor, noise=noise)
            save_diffusion_sample(sample, f"{self.logger.log_dir}/level_{level-1}_noised_sample.png")

            sample = self.diffusion_models[level].sample(image_size=sample_size_per_level[level], batch_size=batch_size,
                                                         custom_initial_img=sample)
            save_diffusion_sample(sample, f"{self.logger.log_dir}/level_{level}_sample.png")

        return sample
