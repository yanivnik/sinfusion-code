import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from common_utils.ben_image import imread
from config import *
from datasets.ccg_half_noisy_cropset import CCGSemiNoisyCropSet
from datasets.cropset import CropSet
from datasets.frameset import FrameSet
from diffusion.conditional_diffusion import ConditionalDiffusion
from diffusion.diffusion import Diffusion
from models.nextnet import NextNet


def train_simple_diffusion(cfg):
    # Training hyperparameters
    training_steps = 200_000

    images = torch.cat([imread(f'./images/{image_name}') for image_name in cfg.image_name.split('-')], dim=0)

    # Create training datasets and data loaders
    crop_size = cfg.crop_size if isinstance(cfg.crop_size, tuple) else (cfg.crop_size, cfg.crop_size)
    train_dataset = CropSet(image=images, crop_size=crop_size, use_flip=False)
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)

    # Create evaluation datasets and data loaders
    val_loader = None
    if cfg.eval_during_training:
        val_dataset = CropSet(image=imread(f'./images/{cfg.image_name}')[0], crop_size=(cfg.crop_size, cfg.crop_size), dataset_size=10)
        val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4)

    # Create model
    model = NextNet(in_channels=3, filters_per_layer=cfg.network_filters, depth=cfg.network_depth)
    diffusion = Diffusion(model, channels=3, timesteps=cfg.diffusion_timesteps, auto_sample=False)

    model_callbacks = [pl.callbacks.ModelSummary(max_depth=-1)]
    if cfg.eval_during_training:
        model_callbacks.append(pl.callbacks.ModelCheckpoint(filename='single-level-{step}-{val_loss:.2f}', save_last=True,
                                                            save_top_k=3, monitor='val_loss', mode='min'))
    else:
        model_callbacks.append(pl.callbacks.ModelCheckpoint(filename='single-level-{step}', save_last=True,
                               save_top_k=10, monitor='train_loss', mode='min'))

    tb_logger = pl.loggers.TensorBoardLogger("lightning_logs/", name=cfg.image_name)
    trainer = pl.Trainer(max_steps=training_steps,
                         val_check_interval=0.2 if cfg.eval_during_training else 1.0,
                         gpus=1, auto_select_gpus=True,
                         logger=tb_logger, log_every_n_steps=10,
                         callbacks=model_callbacks)

    # Train model
    trainer.fit(diffusion, train_loader, val_loader)


def train_ccg_diffusion(cfg):
    # Training hyperparameters
    training_steps = 200_000

    # Create training datasets and data loaders
    train_dataset = CCGSemiNoisyCropSet(image=imread(f'./images/{cfg.image_name}')[0],
                                        crop_size=(cfg.crop_size, cfg.crop_size))
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)

    # Create evaluation datasets and data loaders
    val_loader = None
    if cfg.eval_during_training:
        val_dataset = CCGSemiNoisyCropSet(image=imread(f'./images/{cfg.image_name}')[0],
                                          crop_size=(cfg.crop_size, cfg.crop_size), dataset_size=10)
        val_loader = DataLoader(val_dataset, batch_size=val_dataset.dataset_size, num_workers=4)

    # Create model
    model = NextNet(in_channels=6, filters_per_layer=cfg.network_filters, depth=cfg.network_depth)
    diffusion = ConditionalDiffusion(model, channels=3, timesteps=cfg.diffusion_timesteps, auto_sample=False)

    model_callbacks = [pl.callbacks.ModelSummary(max_depth=-1)]
    if cfg.eval_during_training:
        model_callbacks.append(pl.callbacks.ModelCheckpoint(filename=f'single-level-' + '{step}-{val_loss:.2f}',
                                                            save_top_k=3, monitor='val_loss', mode='min',
                                                            save_last=True))
    else:
        model_callbacks.append(pl.callbacks.ModelCheckpoint(filename=f'single-level-' + '{step}', save_last=True))

    tb_logger = pl.loggers.TensorBoardLogger("lightning_logs/", name=cfg.image_name)
    trainer = pl.Trainer(max_steps=training_steps,
                         val_check_interval=0.2 if cfg.eval_during_training else 1.0,
                         gpus=1, auto_select_gpus=True,
                         logger=tb_logger, log_every_n_steps=10,
                         callbacks=model_callbacks)

    # Train model
    trainer.fit(diffusion, train_loader, val_loader)


def train_video_diffusion(cfg):
    # Training hyperparameters
    training_steps = 300_000

    # Create training datasets and data loaders
    images = []
    for frame in range(1, len(os.listdir(f'./images/video/{cfg.image_name}')) + 1):
        images.append(imread(f'./images/video/{cfg.image_name}/{frame}.png'))
    images = torch.cat(images, dim=0)

    crop_size = cfg.crop_size if isinstance(cfg.crop_size, tuple) else (cfg.crop_size, cfg.crop_size)
    train_dataset = FrameSet(frames=images, crop_size=crop_size)
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)

    # Create model
    model = NextNet(in_channels=6, filters_per_layer=cfg.network_filters, depth=cfg.network_depth,
                    frame_conditioned=True)
    diffusion = ConditionalDiffusion(model, channels=3, timesteps=cfg.diffusion_timesteps, auto_sample=False)

    model_callbacks = [pl.callbacks.ModelSummary(max_depth=-1),
                       pl.callbacks.ModelCheckpoint(filename='single-level-{step}', save_last=True,
                                                    save_top_k=10, monitor='train_loss', mode='min')]

    tb_logger = pl.loggers.TensorBoardLogger("lightning_logs/", name=cfg.image_name)
    trainer = pl.Trainer(max_steps=training_steps,
                         val_check_interval=1.0,
                         gpus=1, auto_select_gpus=True,
                         logger=tb_logger, log_every_n_steps=10,
                         callbacks=model_callbacks)

    # Train model
    trainer.fit(diffusion, train_loader)


def main():
    cfg = BALLOONS2_VIDEO_CONFIG
    cfg = parse_cmdline_args_to_config(cfg)

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.available_gpus

    log_config(cfg)

    if cfg.training_method == 'ccg':
        train_ccg_diffusion(cfg)
    elif cfg.training_method == 'video':
        train_video_diffusion(cfg)
    else:
        train_simple_diffusion(cfg)


if __name__ == '__main__':
    main()
