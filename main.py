import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from common_utils.image import imread
from common_utils.video import read_frames_from_dir
from config import *
from datasets.cropset import CropSet
from datasets.frameset import FrameSet
from datasets.interpolation_frameset import TemporalInterpolationFrameSet
from diffusion.conditional_diffusion import ConditionalDiffusion
from diffusion.diffusion import Diffusion
from models.nextnet import NextNet


def train_image_diffusion(cfg):
    """
    Train a diffusion model on a single image.
    Args:
        cfg (Config): Configuration object.
    """
    # Training hyperparameters
    training_steps = 50_000

    image = imread(f'./images/{cfg.image_name}')

    # Create training datasets and data loaders
    crop_size = int(min(image[0].shape[-2:]) * 0.95)
    train_dataset = CropSet(image=image, crop_size=crop_size, use_flip=False)
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)

    # Create model
    model = NextNet(in_channels=3, filters_per_layer=cfg.network_filters, depth=cfg.network_depth)
    diffusion = Diffusion(model, training_target='x0', timesteps=cfg.diffusion_timesteps,
                          auto_sample=True, sample_size=image[0].shape[-2:])

    model_callbacks = [pl.callbacks.ModelSummary(max_depth=-1)]
    model_callbacks.append(pl.callbacks.ModelCheckpoint(filename='single-level-{step}', save_last=True,
                           save_top_k=3, monitor='train_loss', mode='min'))

    tb_logger = pl.loggers.TensorBoardLogger("lightning_logs/", name=cfg.image_name, version=cfg.run_name)
    trainer = pl.Trainer(max_steps=training_steps,
                         gpus=1, auto_select_gpus=True,
                         logger=tb_logger, log_every_n_steps=10,
                         callbacks=model_callbacks)

    # Train model
    trainer.fit(diffusion, train_loader)


def train_video_predictor(cfg):
    """
    Train a DDPM frame Predictor model on a single video.
    Args:
        cfg (Config): Configuration object.
    """
    # Training hyperparameters
    training_steps = 200_000

    # Create training datasets and data loaders
    frames = read_frames_from_dir(f'./images/video/{cfg.image_name}')
    crop_size = (int(frames[0].shape[-2] * 0.95), int(frames[0].shape[-1:] * 0.95))
    train_dataset = FrameSet(frames=frames, crop_size=crop_size)
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)

    # Create model
    model = NextNet(in_channels=6, filters_per_layer=cfg.network_filters, depth=cfg.network_depth, frame_conditioned=True)
    diffusion = ConditionalDiffusion(model, training_target='noise', noise_schedule='cosine',
                                     timesteps=cfg.diffusion_timesteps)

    model_callbacks = [pl.callbacks.ModelSummary(max_depth=-1),
                       pl.callbacks.ModelCheckpoint(filename='single-level-{step}', save_last=True,
                                                    save_top_k=3, monitor='train_loss', mode='min')]

    tb_logger = pl.loggers.TensorBoardLogger("lightning_logs/", name=cfg.image_name, version=cfg.run_name + '_predictor')
    trainer = pl.Trainer(max_steps=training_steps,
                         gpus=1, auto_select_gpus=True,
                         logger=tb_logger, log_every_n_steps=10,
                         callbacks=model_callbacks)

    # Train model
    trainer.fit(diffusion, train_loader)


def train_video_projector(cfg):
    """
    Train a DDPM frame Projector model on a single video.
    Args:
        cfg (Config): Configuration object.
    """
    # Training hyperparameters
    training_steps = 100_000

    # Create training datasets and data loaders
    frames = read_frames_from_dir(f'./images/video/{cfg.image_name}')
    crop_size = int(min(frames[0].shape[-2:]) * 0.95)
    train_dataset = CropSet(image=frames, crop_size=crop_size, use_flip=False)
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)

    # Create model
    model = NextNet(in_channels=3, filters_per_layer=cfg.network_filters, depth=cfg.network_depth)
    diffusion = Diffusion(model, training_target='noise', noise_schedule='cosine', timesteps=cfg.diffusion_timesteps)

    model_callbacks = [pl.callbacks.ModelSummary(max_depth=-1),
                       pl.callbacks.ModelCheckpoint(filename='single-level-{step}', save_last=True,
                                                    save_top_k=3, monitor='train_loss', mode='min')]

    tb_logger = pl.loggers.TensorBoardLogger("lightning_logs/", name=cfg.image_name, version=cfg.run_name + '_projector')
    trainer = pl.Trainer(max_steps=training_steps,
                         gpus=1, auto_select_gpus=True,
                         logger=tb_logger, log_every_n_steps=10,
                         callbacks=model_callbacks)

    # Train model
    trainer.fit(diffusion, train_loader)


def train_video_interpolator(cfg):
    """
    Train a DDPM frame interpolator model on a single video.
    Args:
        cfg (Config): Configuration object.
    """
    # Training hyperparameters
    training_steps = 50_000

    # Create training datasets and data loaders
    frames = read_frames_from_dir(f'./images/video/{cfg.image_name}')
    crop_size = int(min(frames[0].shape[-2:]) * 0.95)
    train_dataset = TemporalInterpolationFrameSet(frames=frames, crop_size=crop_size)
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)

    # Create model
    model = NextNet(in_channels=9, filters_per_layer=cfg.network_filters, depth=cfg.network_depth)
    diffusion = ConditionalDiffusion(model, training_target='x0', timesteps=cfg.diffusion_timesteps)

    model_callbacks = [pl.callbacks.ModelSummary(max_depth=-1),
                       pl.callbacks.ModelCheckpoint(filename='single-level-{step}', save_last=True,
                                                    save_top_k=3, monitor='train_loss', mode='min')]

    tb_logger = pl.loggers.TensorBoardLogger("lightning_logs/", name=cfg.image_name, version=cfg.run_name + '_interpolator')
    trainer = pl.Trainer(max_steps=training_steps,
                         gpus=1, auto_select_gpus=True,
                         logger=tb_logger, log_every_n_steps=10,
                         callbacks=model_callbacks)

    # Train model
    trainer.fit(diffusion, train_loader)


def main():
    cfg = BALLOONS_IMAGE_CONFIG
    cfg = parse_cmdline_args_to_config(cfg)

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.available_gpus

    log_config(cfg)

    if cfg.task == 'video':
        train_video_predictor(cfg)
        train_video_projector(cfg)
    elif cfg.task == 'video_interp':
        train_video_interpolator(cfg)
        train_video_projector(cfg)
    elif cfg.task == 'image':
        train_image_diffusion(cfg)
    else:
        raise Exception(f'Unknown task: {cfg.task}')

if __name__ == '__main__':
    main()
