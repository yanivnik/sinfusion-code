import os

import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader

from datasets.cropset import CropSet
from diffusion.diffusion import GaussianDiffusion
from diffusion.diffusion_pyramid import GaussianDiffusionPyramid
from diffusion.diffusion_utils import save_diffusion_sample
from models.zssr import ZSSRNet


def train_single_diffusion():
    # Training hyperparameters
    diffusion_timesteps = 1000
    training_steps = 100_000
    batch_size = 1  # Each batch contains a single crop, since the batch is actually made of the patches within the crop
    image_name = 'balloons.png'

    # Create datasets and data loaders
    train_dataset = CropSet(image=Image.open(f'./images/{image_name}'), crop_size=(32, 32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    # Create model and trainer
    model = ZSSRNet(filters_per_layer=64, kernel_size=3)
    diffusion = GaussianDiffusion(model, channels=3, timesteps=diffusion_timesteps, sample_size=(64, 64),
                                  sample_every_n_steps=1000)
    tb_logger = pl.loggers.TensorBoardLogger("lightning_logs/", name=image_name)
    trainer = pl.Trainer(max_steps=training_steps, log_every_n_steps=10, gpus=1, auto_select_gpus=True,
                         logger=tb_logger)

    # Train model (samples are generated during training)
    trainer.fit(diffusion, train_loader)


def train_pyramid_diffusion():
    image_name = 'balloons.png'
    levels = 5
    coarsest_size_ratio = 0.25
    timesteps_per_level = [1000, 500, 500, 500, 500]
    training_steps_per_level = [100_000] + [100_000] * (levels - 1)
    sample_batch_size = 8
    tb_logger = pl.loggers.TensorBoardLogger("lightning_logs/pyramid/", name=image_name)

    print('Training generation pyramid')
    pyramid = GaussianDiffusionPyramid(f'./images/{image_name}', levels=levels,
                                         size_ratios=coarsest_size_ratio ** (1.0 / (levels - 1)),
                                         timesteps=timesteps_per_level,
                                         models=None, logger=tb_logger)
    pyramid.train(training_steps_per_level)

    print('Sampling generated images from pyramid')
    samples = pyramid.sample((128, 128), sample_batch_size)
    save_diffusion_sample(samples, f"{tb_logger.log_dir}/sample_.png")


def main():
    # TODO ADD EXTERNAL CONFIGURATION SUPPORT
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    train_pyramid_diffusion()


if __name__ == '__main__':
    main()
