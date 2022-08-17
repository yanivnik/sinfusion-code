import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from common_utils.ben_image import imread
from config import *
from datasets.ccg_half_noisy_cropset import CCGSemiNoisyCropSet
from datasets.cropset import CropSet
from diffusion.conditional_diffusion import ConditionalDiffusion
from diffusion.diffusion import Diffusion
from diffusion.diffusion_pyramid import DiffusionPyramid
from diffusion.diffusion_utils import save_diffusion_sample
from metrics.sifid_score import get_sifid_scores
from models.nextnet import NextNet


def train_simple_diffusion(cfg):
    # Training hyperparameters
    training_steps = 200_000

    # Create training datasets and data loaders
    train_dataset = CropSet(image=imread(f'./images/{cfg.image_name}')[0], crop_size=(cfg.crop_size, cfg.crop_size))
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)

    # Create evaluation datasets and data loaders
    val_loader = None
    if cfg.eval_during_training:
        val_dataset = CropSet(image=imread(f'./images/{cfg.image_name}')[0], crop_size=(cfg.crop_size, cfg.crop_size), dataset_size=10)
        val_loader = DataLoader(val_dataset, batch_size=val_dataset.dataset_size, num_workers=4)

    # Create model
    model = NextNet(in_channels=3, filters_per_layer=cfg.network_filters, depth=cfg.network_depth)
    diffusion = Diffusion(model, channels=3, timesteps=cfg.diffusion_timesteps, auto_sample=False)

    model_callbacks = [pl.callbacks.ModelSummary(max_depth=-1)]
    if cfg.eval_during_training:
        model_callbacks.append(pl.callbacks.ModelCheckpoint(filename=f'single-level-' + '{step}-{val_loss:.2f}',
                                                            save_top_k=3, monitor='val_loss', mode='min'))
    else:
        model_callbacks.append(pl.callbacks.ModelCheckpoint(filename=f'single-level-' + '{step}'))

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
    training_steps = 100_000

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
                                                            save_top_k=3, monitor='val_loss', mode='min'))
    else:
        model_callbacks.append(pl.callbacks.ModelCheckpoint(filename=f'single-level-' + '{step}'))

    tb_logger = pl.loggers.TensorBoardLogger("lightning_logs/", name=cfg.image_name)
    trainer = pl.Trainer(max_steps=training_steps,
                         val_check_interval=0.2 if cfg.eval_during_training else 1.0,
                         gpus=1, auto_select_gpus=True,
                         logger=tb_logger, log_every_n_steps=10,
                         callbacks=model_callbacks)

    # Train model
    trainer.fit(diffusion, train_loader, val_loader)


def train_pyramid_diffusion(cfg):
    training_steps_per_level = [50_000] * cfg.pyramid_levels
    sample_batch_size = 16
    image_path = f'./images/{cfg.image_name}'

    wandb_logger = pl.loggers.WandbLogger(project=cfg.project_name)
    tb_logger = pl.loggers.TensorBoardLogger("lightning_logs/pyramid/", name=cfg.image_name)

    print('Training generation pyramid')
    pyramid = DiffusionPyramid(image_path,
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
    cfg = LIGHTNING_SIMPLE_CONFIG
    cfg = parse_cmdline_args_to_config(cfg)

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.available_gpus

    log_config(cfg)

    if cfg.pyramid_levels is not None and cfg.training_method == 'pyramid':
        train_pyramid_diffusion(cfg)
    elif cfg.training_method == 'ccg':
        train_ccg_diffusion(cfg)
    else:
        train_simple_diffusion(cfg)


if __name__ == '__main__':
    main()
