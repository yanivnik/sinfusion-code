import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from common_utils.ben_image import imread
from config import *
from datasets.cropset import CropSet
from diffusion.diffusion import Diffusion
from diffusion.diffusion_pyramid_sr import SRDiffusionPyramid
from diffusion.diffusion_utils import save_diffusion_sample
from metrics.sifid_score import get_sifid_scores
from models.nextnet import NextNet


def train_single_diffusion(cfg):
    # Training hyperparameters
    training_steps = 30_000

    # Create datasets and data loaders
    train_dataset = CropSet(image=imread(f'./images/{cfg.image_name}')[0], crop_size=(cfg.crop_size, cfg.crop_size))
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)

    # Create model and trainer
    model = NextNet(filters_per_layer=cfg.network_filters, depth=cfg.network_depth)
    diffusion = Diffusion(model, channels=3, timesteps=cfg.diffusion_timesteps,
                          sample_size=(186, 248), sample_every_n_steps=1000, auto_sample=True)

    model_callbacks = [pl.callbacks.ModelCheckpoint(filename=f'single-level-' + '{step}'),
                       pl.callbacks.ModelSummary(max_depth=-1)]
    wandb_logger = pl.loggers.WandbLogger(project="single-image-diffusion")
    tb_logger = pl.loggers.TensorBoardLogger("lightning_logs/", name=cfg.image_name)
    trainer = pl.Trainer(max_steps=training_steps, log_every_n_steps=10, gpus=1, auto_select_gpus=True,
                         logger=tb_logger, callbacks=model_callbacks)

    # Train model (samples are generated during training)
    trainer.fit(diffusion, train_loader)


def train_pyramid_diffusion(cfg):
    training_steps_per_level = [30_000] + [1] * (cfg.pyramid_levels-1)
    sample_batch_size = 16
    image_path = f'./images/{cfg.image_name}'

    wandb_logger = pl.loggers.WandbLogger(project="single-image-diffusion")
    tb_logger = pl.loggers.TensorBoardLogger("lightning_logs/pyramid/", name=cfg.image_name)

    print('Training generation pyramid')
    pyramid = SRDiffusionPyramid(image_path,
                                 levels=cfg.pyramid_levels,
                                 size_ratios=cfg.pyramid_coarsest_ratio ** (1.0 / (cfg.pyramid_levels - 1)),
                                 timesteps=cfg.diffusion_timesteps,
                                 crop_size=cfg.crop_size,
                                 network_filters=cfg.network_filters,
                                 network_depth=cfg.network_depth,
                                 logger=tb_logger)  # TODO HANDLE WANDB
    pyramid.train(training_steps_per_level, log_progress=cfg.log_progress)

    print('Sampling generated images from pyramid')
    samples = pyramid.sample((186, 248), sample_batch_size, debug=True)
    save_diffusion_sample(samples, f"{tb_logger.log_dir}/sample_.png", wandb_logger)
    wandb_logger.log_metrics({'SIFID': get_sifid_scores(imread(image_path), samples).mean()})


def main():
    cfg = BALLOONS_PYRAMID_CONFIG
    cfg = parse_cmdline_args_to_config(cfg)

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.available_gpus

    log_config(cfg)
    train_pyramid_diffusion(cfg)
    #train_single_diffusion(cfg)


if __name__ == '__main__':
    main()
