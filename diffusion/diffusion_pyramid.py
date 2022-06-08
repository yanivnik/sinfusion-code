import os

import pytorch_lightning as pl
import torch
from PIL import Image

from common_utils.resize_right import resize
from datasets.cropset import CropSet
from diffusion.diffusion import Diffusion
from diffusion.diffusion_utils import get_pyramid_parameter_as_list
from models.zssr import ZSSRNet


class DiffusionPyramid(object):
    """
    A class for a diffusion pyramid. Each level in the pyramid is a diffusion model
    which handled a different scale of the single input image.
    The coarsest level in the pyramid (level 0) handles the lowest resolution, and the last
    level in the pyramid handles the original image (highest) resolution.
    """
    def __init__(self, image_path, levels, size_ratios, timesteps, crop_size, network_filters, logger=False):
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
            crop_size (int):
                The size of the crops to take from each level in the pyramid. Notice that the same size of crop is
                used across all levels.
            network_filters (list(int) or int):
                The amount of filters to be used in each convolutional layer of the backbone models.
        """
        self.levels = levels
        self.size_ratios = get_pyramid_parameter_as_list(size_ratios, levels - 1)
        self.timesteps = get_pyramid_parameter_as_list(timesteps, levels)
        self.network_filters = get_pyramid_parameter_as_list(network_filters, levels)
        self.crop_size = (crop_size, crop_size)

        # Generate image pyramid
        self.images = [Image.open(image_path)]
        for ratio in self.size_ratios:
            self.images.insert(0, resize(self.images[0], scale_factors=ratio))

        self.logger = logger
        self.diffusion_models = []
        self.initialize_diffusion_models()
        assert len(self.diffusion_models) == levels

        self.datasets = []
        self.initialize_datasets()
        assert len(self.datasets) == levels

    def initialize_diffusion_models(self):
        """
        Initialize the diffusion models for all levels in the pyramid.
        Override this method to supply different models.
        """
        # Use default identical backbone networks
        for level in range(self.levels):
            self.diffusion_models.append(Diffusion(model=ZSSRNet(filters_per_layer=self.network_filters[level]),
                                                   timesteps=self.timesteps[0],
                                                   noising_timesteps_ratio=self.timesteps[level] / self.timesteps[0],  # TODO WRITE THIS CODE BETTER
                                                   auto_sample=False))

    def initialize_datasets(self):
        """
        Initialize the datasets for all levels in the pyramid.
        Override this method to supply different datasets.
        """
        for level in range(self.levels):
            self.datasets.append(CropSet(image=self.images[level], crop_size=self.crop_size))

    def train(self, training_steps, log_progress):
        """
        Train the models in the pyramid, level after level.

        Args:
            training_steps (list(int)): The amount of training examples per level.
        """
        training_steps = get_pyramid_parameter_as_list(training_steps, self.levels)
        for level in range(self.levels):
            loader = torch.utils.data.DataLoader(self.datasets[level], batch_size=1)
            model_checkpoint_callback = pl.callbacks.ModelCheckpoint(filename=f'level={level}-' + '{step}')
            trainer = pl.Trainer(logger=self.logger,
                                 max_steps=training_steps[level],
                                 gpus=1,
                                 auto_select_gpus=True,
                                 callbacks=[model_checkpoint_callback],
                                 enable_progress_bar=log_progress)
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
            sample_size_per_level.insert(0, (int(sample_size_per_level[0][0] * ratio),
                                             int(sample_size_per_level[0][1] * ratio)))

        # In the coarsest level, pure noise is sampled
        sample = self.diffusion_models[0].sample(image_size=sample_size_per_level[0], batch_size=batch_size)

        for level in range(1, self.levels):
            # Upsample the lower level output
            sample = resize(sample, out_shape=sample_size_per_level[level])

            # Generate and add noise to the upsampled output
            noise = torch.randn_like(sample)
            timestep_tensor = torch.tensor((self.timesteps[level] - 1,))
            sample = self.diffusion_models[level].q_sample(sample, t=timestep_tensor, noise=noise)
            sample = self.diffusion_models[level].sample(image_size=sample_size_per_level[level], batch_size=batch_size,
                                                         custom_initial_img=sample)

        return sample

    @classmethod
    def load_from_checkpoint(cls, checkpoint_dir_path, image_path, levels, size_ratios, timesteps):
        """
        Load a new diffusion pyramid from existing pre-trained checkpoints.

        Args:
            checkpoint_dir_path (str): The path to the checkpoint directory which should contain a checkpoint file
                per level in the pyramid. Each filename in the directory should start with a "level={i}" prefix.
                For example -
                    level=0-step=99.ckpt, level=1-step=999.ckpt, etc.
        """
        checkpoint_files = os.listdir(checkpoint_dir_path)
        assert len(checkpoint_files) == levels, 'The checkpoint directory must include a checkpoint file for each level'
        new_pyramid = cls(image_path=image_path, levels=levels, size_ratios=size_ratios, timesteps=timesteps)

        # Override the inner diffusion models of the pyramid with the checkpoints
        checkpoint_models = []
        for level in range(levels):
            assert checkpoint_files[level].startswith(f'level={level}'), 'Unexpected order of checkpoint files or ' \
                                                                         'missing files'

            # Dynamically find the type of the model implementation
            dm_impl_class = new_pyramid.diffusion_models[level].__class__

            # Load the checkpoint for the current pyramid level from its checkpoint
            checkpoint_path = os.path.join(checkpoint_dir_path, checkpoint_files[level])
            checkpoint_models.append(dm_impl_class.load_from_checkpoint(checkpoint_path))

        new_pyramid.diffusion_models = checkpoint_models
        return new_pyramid
