from common_utils.resize_right import resize
from datasets.cropset import CropSet
from datasets.sr_cropset import SRCropSet
from diffusion.diffusion import Diffusion
from diffusion.diffusion_pyramid import DiffusionPyramid
from diffusion.diffusion_utils import save_diffusion_sample
from diffusion.sr_diffusion import SRDiffusion
from models.nextnet import NextNet
from models.zssr import ZSSRNet


class SRDiffusionPyramid(DiffusionPyramid):
    """
    A class for a diffusion pyramid, based on the SR3 model (https://arxiv.org/abs/2104.07636).
    Each level in the pyramid is a diffusion model which handles a different scale of the single input image.

    The coarsest level in the pyramid (level 0) generates a low resolution image from pure noise,
    while the rest of the layers use a conditional diffusion model to add high frequencies to the generated images,
    thus increasing the resolution of the generated images.
    """
    def __init__(self, image_path, levels, size_ratios, timesteps, crop_size, network_filters, network_depth, logger=None):
        super().__init__(image_path, levels, size_ratios, timesteps, crop_size, network_filters, network_depth, logger)

    def initialize_diffusion_models(self):
        # Coarsest layer backbone is a NextNet
        models = [NextNet(filters_per_layer=self.network_filters[0], depth=self.network_depth[0])]

        # The rest of the layer backbones are simpler ZSSRNet
        for i in range(1, self.levels):
            models.append(ZSSRNet(in_channels=6, filters_per_layer=self.network_filters[i],
                                  depth=self.network_depth[i]))

        self.diffusion_models.append(Diffusion(model=models[0], timesteps=self.timesteps[0], auto_sample=False))
        for i in range(1, self.levels):
            self.diffusion_models.append(SRDiffusion(model=models[i], timesteps=self.timesteps[i]))

    def initialize_datasets(self):
        laplace = [self.images[0]]
        for level in range(1, self.levels):
            laplace.append(self.images[level] - resize(self.images[level - 1],
                                                       scale_factors=1 / self.size_ratios[level - 1],
                                                       out_shape=self.images[level].shape))

        self.datasets.append(CropSet(image=self.images[0], crop_size=self.crop_size))
        for level in range(1, self.levels):
            self.datasets.append(SRCropSet(hr=self.images[level],
            #self.datasets.append(SRCropSet(hr=laplace[level], # UNCOMMENT THIS WHEN USING LAPLACE (TODO REMOVE LINE)
                                           lr=self.images[level - 1],
                                           crop_size=self.crop_size))

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
            sample_size_per_level.insert(0, (int(sample_size_per_level[0][0] * ratio), int(sample_size_per_level[0][1] * ratio)))

        # In the coarsest level, pure noise is sampled without conditioning
        sample = self.diffusion_models[0].sample(image_size=sample_size_per_level[0], batch_size=batch_size)

        if debug:
            save_diffusion_sample(sample, f"{self.logger.log_dir}/level_0_sample.png")

        for level in range(1, self.levels):
            lr_sample = sample.clone().detach()

            # Upsample the lower level output to condition the higher level
            resized_lr_sample = resize(lr_sample, out_shape=sample_size_per_level[level])

            # In any of the other levels, noise is sampled and is conditioned on the sample from the previous layer
            #laplace_sample = self.diffusion_models[level].sample(lr=resized_lr_sample, image_size=sample_size_per_level[level]) # UNCOMMENT THIS WHEN USING LAPLACE (TODO REMOVE LINE)
            sample = self.diffusion_models[level].sample(lr=resized_lr_sample, image_size=sample_size_per_level[level])

            #sample = laplace_sample + resized_lr_sample # UNCOMMENT THIS WHEN USING LAPLACE (TODO REMOVE LINE)

            if debug:
                # save_diffusion_sample(laplace_sample, f"{self.logger.log_dir}/level_{level}_laplace.png") # UNCOMMENT THIS WHEN USING LAPLACE (TODO REMOVE LINE)
                save_diffusion_sample(sample, f"{self.logger.log_dir}/level_{level}_sample.png")

        return sample
