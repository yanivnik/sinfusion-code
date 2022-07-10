import argparse
import sys


class Config:
    # Constant configuration options
    project_name = 'single-image-diffusion'
    image_name = 'balloons.png'

    # Deployment configuration options
    log_progress = True     # Should loggers log training progress bar
    available_gpus = '0'    # list of available gpus (in CUDA_VISIBLE_DEVICES format)

    # Diffusion configuration options
    diffusion_timesteps = 500

    # Backbone model and dataset configuration options
    network_filters = 64    # Amount of filters in backbone network conv layers
    network_depth = 9
    crop_size = 19

    # Optimization configuration options
    initial_lr = 0.0002
    lr_schedule = 'single'  # Learning rate scheduling strategy. TODO IMPLEMENT USAGE.

    # Generation pyramid configuration options
    pyramid_levels = 5
    pyramid_coarsest_ratio = 0.135  # The size ratio between the image in the coarsest level and the original image

    def __init__(self):
        pass


def log_config(cfg):
    for k in dir(cfg):
        if k[:1] != '_':
            print(f'{k}={getattr(cfg, k)}')


def parse_cmdline_args_to_config(cfg):
    parser = argparse.ArgumentParser(description='Command line configuration')
    parser.add_argument('--pyramid_levels', type=int, help='Amount of levels in the generation pyramid')
    parser.add_argument('--pyramid_coarsest_ratio', type=float, help='Size ratio between the coarsest image in the pyramid and the original image')
    parser.add_argument('--diffusion_timesteps', type=int, help='Amount of diffusion timesteps to perform per level')
    parser.add_argument('--initial_lr', type=float, help='Initial value of LR')
    # parser.add_argument('--lr_schedule', type=str, choices=['single', 'multi-level', 'logarithmic'])
    parser.add_argument('--network_filters', type=int, help='Amount of filters per convolutional level in the backbone networks')
    parser.add_argument('--crop_size', type=int, help='Size of crops to train the backbone models on')
    parser.add_argument('--no_progress', dest='log_progress', action='store_false')

    args = parser.parse_args(sys.argv[1:])

    # Override cfg attribute values with supplied cmdline args
    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)

    return cfg


# Pre-made configurations
BALLOONS_PYRAMID_CONFIG = Config()
BALLOONS_PYRAMID_CONFIG.image_name = 'balloons.png'
BALLOONS_PYRAMID_CONFIG.pyramid_levels = 5
BALLOONS_PYRAMID_CONFIG.diffusion_timesteps = 500
BALLOONS_PYRAMID_CONFIG.crop_size = 19
BALLOONS_PYRAMID_CONFIG.pyramid_coarsest_ratio = 0.135

BALLOONS_SINGLE_CONFIG = Config()
BALLOONS_SINGLE_CONFIG.image_name = 'balloons.png'
BALLOONS_SINGLE_CONFIG.pyramid_levels = None
BALLOONS_SINGLE_CONFIG.pyramid_coarsest_ratio = None
BALLOONS_SINGLE_CONFIG.diffusion_timesteps = 500
BALLOONS_SINGLE_CONFIG.crop_size = 120
BALLOONS_SINGLE_CONFIG.network_depth = 9

MOUNTAINS3_PYRAMID_CONFIG = Config()
MOUNTAINS3_PYRAMID_CONFIG.image_name = 'mountains3.png'
MOUNTAINS3_PYRAMID_CONFIG.pyramid_levels = 5
MOUNTAINS3_PYRAMID_CONFIG.diffusion_timesteps = 500
MOUNTAINS3_PYRAMID_CONFIG.crop_size = 19
MOUNTAINS3_PYRAMID_CONFIG.pyramid_coarsest_ratio = 0.141

LIGHTNING_PYRAMID_CONFIG = Config()
LIGHTNING_PYRAMID_CONFIG.image_name = 'lightning1.png'
LIGHTNING_PYRAMID_CONFIG.pyramid_levels = 5
LIGHTNING_PYRAMID_CONFIG.diffusion_timesteps = 500
LIGHTNING_PYRAMID_CONFIG.crop_size = 19
LIGHTNING_PYRAMID_CONFIG.pyramid_coarsest_ratio = 0.141

STARRYNIGHT_PYRAMID_CONFIG = Config()
STARRYNIGHT_PYRAMID_CONFIG.image_name = 'starry_night.png'
STARRYNIGHT_PYRAMID_CONFIG.pyramid_levels = 5
STARRYNIGHT_PYRAMID_CONFIG.diffusion_timesteps = 500
STARRYNIGHT_PYRAMID_CONFIG.crop_size = 19
STARRYNIGHT_PYRAMID_CONFIG.pyramid_coarsest_ratio = 0.135