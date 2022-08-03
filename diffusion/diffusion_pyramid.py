import os

import pytorch_lightning as pl
import torch

from common_utils.ben_image import imread
from common_utils.resize_right import resize
from datasets.cropset import CropSet
from datasets.sr_cropset import SRCropSet
from diffusion.diffusion import Diffusion
from diffusion.diffusion_utils import get_pyramid_parameter_as_list
from diffusion.diffusion_utils import save_diffusion_sample
from diffusion.sr_diffusion import TheirsSRDiffusion
from models.nextnet import NextNet


class DiffusionPyramid(object):
    """
    A class for a diffusion pyramid, based on the SR3 model (https://arxiv.org/abs/2104.07636).
    Each level in the pyramid is a diffusion model which handles a different scale of the single input image.

    The coarsest level in the pyramid (level 0) generates a low resolution image from pure noise,
    while the rest of the layers use a conditional diffusion model to add high frequencies to the generated images,
    thus increasing the resolution of the generated images.
    """
    def __init__(self, image_path, levels, size_ratios, timesteps, crop_size, network_filters, network_depth,
                 logger=False):
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
        self.network_depth = get_pyramid_parameter_as_list(network_depth, levels)
        self.crop_size = (crop_size, crop_size)

        # Generate image pyramid
        self.images = [imread(image_path)[0]]
        for ratio in self.size_ratios[::-1]:
            self.images.insert(0, resize(self.images[0], scale_factors=ratio))

        self.logger = logger
        self.diffusion_models = []
        self.initialize_diffusion_models()
        assert len(self.diffusion_models) == levels

        self.datasets = []
        self.initialize_datasets()
        assert len(self.datasets) == levels

    def initialize_diffusion_models(self):
        # Coarsest layer backbone is a NextNet
        models = [NextNet(filters_per_layer=self.network_filters[0], depth=self.network_depth[0])]

        for i in range(1, self.levels):
            # models.append(ZSSRNet(in_channels=6, filters_per_layer=self.network_filters[i],
            #                      depth=self.network_depth[i]))
            models.append(NextNet(in_channels=6, filters_per_layer=self.network_filters[i],
                                  depth=self.network_depth[i]))

        self.diffusion_models.append(Diffusion(model=models[0], timesteps=self.timesteps[0], auto_sample=False,
                                               recon_loss_factor=0, recon_image=self.images[
                0]))  # Currently disabled recon loss for faster training
        for i in range(1, self.levels):
            # self.diffusion_models.append(SRDiffusion(model=models[i], timesteps=self.timesteps[i]))

            # This uses the hacky model which implements the continous sampling trick from WaveGrad.
            # TODO: Delete the hacky model and implement the sampling trick normally
            self.diffusion_models.append(TheirsSRDiffusion(model=models[i], timesteps=self.timesteps[i],
                                                           recon_loss_factor=0,
                                                           # Currently disabled recon loss for faster training
                                                           recon_image=self.images[i].unsqueeze(0),
                                                           recon_image_lr=resize(self.images[i - 1],
                                                                                 out_shape=self.images[
                                                                                     i].shape).unsqueeze(0)))

    def initialize_datasets(self):
        laplace = [self.images[0]]
        for level in range(1, self.levels):
            laplace.append(self.images[level] - resize(self.images[level - 1],
                                                       scale_factors=1 / self.size_ratios[level - 1],
                                                       out_shape=self.images[level].shape))

        self.datasets.append(CropSet(image=self.images[0], crop_size=self.crop_size))
        for level in range(1, self.levels):
            self.datasets.append(SRCropSet(hr=self.images[level],
                                           # self.datasets.append(SRCropSet(hr=laplace[level], # UNCOMMENT THIS WHEN USING LAPLACE (TODO REMOVE LINE)
                                           lr=self.images[level - 1],
                                           crop_size=self.crop_size))

    def train(self, training_steps, log_progress):
        """
        Train the models in the pyramid, level after level.

        Args:
            training_steps (list(int)): The amount of training examples per level.
        """
        training_steps = get_pyramid_parameter_as_list(training_steps, self.levels)
        for level in range(self.levels):
            loader = torch.utils.data.DataLoader(self.datasets[level], batch_size=1)
            callbacks = [pl.callbacks.ModelCheckpoint(filename=f'level={level}-' + '{step}'),
                         pl.callbacks.ModelSummary(max_depth=-1)]
            trainer = pl.Trainer(logger=self.logger,
                                 max_steps=training_steps[level],
                                 gpus=1,
                                 auto_select_gpus=True,
                                 callbacks=callbacks,
                                 enable_progress_bar=log_progress)
            trainer.fit(self.diffusion_models[level], loader)

    def sample(self, sample_size, batch_size, debug=False):
        """
        Sample a batch of images from the pyramid.

        First, the coarsest level samples an LR sample from pure noise.
        Then, for each level, the samples are upsampled (bicubic) and used to condition the diffusion model
        of the next level generate a sample with higher resolution and higher frequencies.

        Args:
            sample_size (tuple(int, int) or int): The spatial dimensions of the final sample output.
            batch_size (int): The size of the batch to sample.
            debug (boolean): Weather or not the sampling should be performed in debug mode (saves mid-level results).
        """
        sample_size = sample_size if isinstance(sample_size, tuple) else (sample_size, sample_size)

        sample_size_per_level = [sample_size]
        for ratio in reversed(self.size_ratios):
            sample_size_per_level.insert(0, (
            int(sample_size_per_level[0][0] * ratio), int(sample_size_per_level[0][1] * ratio)))

        # In the coarsest level, pure noise is sampled without conditioning
        sample = self.diffusion_models[0].sample(image_size=sample_size_per_level[0], batch_size=batch_size)

        if debug:
            save_diffusion_sample(sample, f"{self.logger.log_dir}/level_0_sample.png")

        for level in range(1, self.levels):
            lr_sample = sample.clone().detach()

            # Upsample the lower level output to condition the higher level
            resized_lr_sample = resize(lr_sample, out_shape=sample_size_per_level[level])

            # In any of the other levels, noise is sampled and is conditioned on the sample from the previous layer
            # laplace_sample = self.diffusion_models[level].sample(lr=resized_lr_sample) # UNCOMMENT THIS WHEN USING LAPLACE (TODO REMOVE LINE)
            # sample = laplace_sample + resized_lr_sample  # UNCOMMENT THIS WHEN USING LAPLACE (TODO REMOVE LINE)
            sample = self.diffusion_models[level].sample(lr=resized_lr_sample)

            if debug:
                # save_diffusion_sample(laplace_sample, f"{self.logger.log_dir}/level_{level}_laplace.png") # UNCOMMENT THIS WHEN USING LAPLACE (TODO REMOVE LINE)
                save_diffusion_sample(sample, f"{self.logger.log_dir}/level_{level}_sample.png")

        return sample

    @classmethod
    def load_from_checkpoint(cls, checkpoint_dir_path, image_path, levels, size_ratios, timesteps, crop_size, network_filters):
        """
        Load a new diffusion pyramid from existing pre-trained checkpoints.

        Args:
            checkpoint_dir_path (str): The path to the checkpoint directory which should contain a checkpoint file
                per level in the pyramid. Each filename in the directory should start with a "level={i}" prefix.
                For example -
                    level=0-step=99.ckpt, level=1-step=999.ckpt, etc.
        """
        checkpoint_files = sorted(os.listdir(checkpoint_dir_path))
        assert len(checkpoint_files) == levels, 'The checkpoint directory must include a checkpoint file for each level'
        new_pyramid = cls(image_path=image_path, levels=levels, size_ratios=size_ratios, timesteps=timesteps,
                          crop_size=crop_size, network_filters=network_filters)

        # Override the inner diffusion models of the pyramid with the checkpoints
        checkpoint_models = []
        for level in range(levels):
            assert checkpoint_files[level].startswith(f'level={level}'), f'Unexpected order of checkpoint files or ' \
                                                                         f'missing files. Expected "level={level}...", ' \
                                                                         f'found {checkpoint_files[level]}'

            # Dynamically find the type of the model implementation
            dm_impl_class = new_pyramid.diffusion_models[level].__class__

            # Load the checkpoint for the current pyramid level from its checkpoint
            checkpoint_path = os.path.join(checkpoint_dir_path, checkpoint_files[level])
            checkpoint_models.append(dm_impl_class.load_from_checkpoint(checkpoint_path))

        new_pyramid.diffusion_models = checkpoint_models
        return new_pyramid
