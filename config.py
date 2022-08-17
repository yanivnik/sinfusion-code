import argparse
import sys


class Config:
    # Constant configuration options
    project_name = 'single-image-diffusion'
    image_name = 'balloons.png'

    # Deployment configuration options
    log_progress = True     # Should loggers log training progress bar
    available_gpus = '1'    # list of available gpus (in CUDA_VISIBLE_DEVICES format)
    eval_during_training = False  # Should perform valuation loop during training

    # Diffusion configuration options
    diffusion_timesteps = 500
    training_method = 'simple'

    # Backbone model and dataset configuration options
    network_filters = 64    # Amount of filters in backbone network conv layers
    network_depth = 9
    crop_size = 128

    # Optimization configuration options
    initial_lr = 0.0002
    lr_schedule = 'single'  # Learning rate scheduling strategy. TODO IMPLEMENT USAGE.

    # Generation pyramid configuration options
    pyramid_levels = 5
    pyramid_coarsest_ratio = 0.135  # The size ratio between the image in the coarsest level and the original image

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def log_config(cfg):
    for k in dir(cfg):
        if k[:1] != '_':
            print(f'{k}={getattr(cfg, k)}')


def parse_cmdline_args_to_config(cfg):
    parser = argparse.ArgumentParser(description='Command line configuration')
    parser.add_argument('--image_name', type=str, help='The image to train the model on')
    parser.add_argument('--training_method', type=str, choices=['simple', 'ccg', 'pyramid'], help='The type of training mechanism')
    parser.add_argument('--pyramid_levels', type=int, help='Amount of levels in the generation pyramid')
    parser.add_argument('--pyramid_coarsest_ratio', type=float, help='Size ratio between the coarsest image in the pyramid and the original image')
    parser.add_argument('--diffusion_timesteps', type=int, help='Amount of diffusion timesteps to perform per level')
    parser.add_argument('--initial_lr', type=float, help='Initial value of LR')
    # parser.add_argument('--lr_schedule', type=str, choices=['single', 'multi-level', 'logarithmic'])
    parser.add_argument('--network_depth', type=int, help='Depth of the backbone network (amount of blocks)')
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
BALLOONS_SIMPLE_CONFIG = Config(image_name='balloons.png',
                                pyramid_levels=None,
                                diffusion_timesteps=500,
                                crop_size=185,
                                network_depth=16,
                                network_filters=64)

LIGHTNING_SIMPLE_CONFIG = Config(image_name='lightning1.png',
                                 pyramid_levels=None,
                                 diffusion_timesteps=500,
                                 crop_size=160,
                                 network_depth=16,
                                 network_filters=64)

STARRY_NIGHT_SIMPLE_CONFIG = Config(image_name='starry_night.png',
                                    pyramid_levels=None,
                                    diffusion_timesteps=500,
                                    crop_size=195,
                                    network_depth=16,
                                    network_filters=64)

BIRDS_CCG_CONFIG = Config(image_name='birds.png',
                          training_method='ccg',
                          pyramid_levels=None,
                          diffusion_timesteps=500,
                          crop_size=128,
                          network_depth=9,
                          network_filters=[32, 64, 64, 128, 256, 128, 64, 64, 32]) # TODO YANIV: LOOK INTO CHANGING NUMBER OF FILTERS PER CONVNEXT BLOCK

STONE_CCG_CONFIG = Config(image_name='stone.png',
                          training_method='ccg',
                          pyramid_levels=None,
                          diffusion_timesteps=500,
                          crop_size=128,
                          network_depth=9,
                          network_filters=[32, 64, 64, 128, 256, 128, 64, 64, 32]) # TODO YANIV: LOOK INTO CHANGING NUMBER OF FILTERS PER CONVNEXT BLOCK

SEASCAPE_CCG_CONFIG = Config(image_name='seascape.png',
                             training_method='ccg',
                             pyramid_levels=None,
                             diffusion_timesteps=500,
                             crop_size=128,
                             network_depth=9,
                             network_filters=[32, 64, 64, 128, 256, 128, 64, 64, 32]) # TODO YANIV: LOOK INTO CHANGING NUMBER OF FILTERS PER CONVNEXT BLOCK

BALLOONS_PYRAMID_CONFIG = Config(image_name='balloons.png',
                                 training_method='pyramid',
                                 pyramid_levels=5,
                                 pyramid_coarsest_ratio=0.135,
                                 diffusion_timesteps=500,
                                 crop_size=19,
                                 network_depth=9)

MOUNTAINS3_PYRAMID_CONFIG = Config(image_name='mountains3.png',
                                   training_method='pyramid',
                                   pyramid_levels=5,
                                   pyramid_coarsest_ratio=0.141,
                                   diffusion_timesteps=500,
                                   crop_size=19,
                                   network_depth=9)
