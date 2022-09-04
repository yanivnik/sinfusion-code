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
    parser.add_argument('--training_method', type=str, choices=['simple', 'ccg'], help='The type of training mechanism')
    parser.add_argument('--diffusion_timesteps', type=int, help='Amount of diffusion timesteps to perform per level')
    parser.add_argument('--initial_lr', type=float, help='Initial value of LR')
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

### Configurations for diverse generation, retargeting, from sketch, etc ###
BALLOONS_SIMPLE_CONFIG = Config(image_name='balloons.png',
                                crop_size=185,
                                network_depth=16,
                                network_filters=64)

BALLOONS_SIMPLE_VS_PRETRAIN_CONFIG = Config(image_name='balloons_scale=0.729.png',
                                            crop_size=128,
                                            network_depth=16,
                                            network_filters=64)

LIGHTNING_SIMPLE_CONFIG = Config(image_name='lightning1.png',
                                 crop_size=160,
                                 network_depth=16,
                                 network_filters=64)

STARRY_NIGHT_SIMPLE_CONFIG = Config(image_name='starry_night.png',
                                    crop_size=195,
                                    network_depth=16,
                                    network_filters=64)

MOUNTAINS3_SIMPLE_CONFIG = Config(image_name='mountains3.png',
                                  crop_size=168,
                                  network_depth=16,
                                  network_filters=64)

PENGUINS_SIMPLE_CONFIG = Config(image_name='penguins.png',
                                crop_size=114,
                                network_depth=16,
                                network_filters=64)

DOLPHINS_SIMPLE_CONFIG = Config(image_name='Dolphins.jpg',
                                crop_size=165,
                                network_depth=16,
                                network_filters=64)

BIRDS3_SIMPLE_CONFIG = Config(image_name='Birds3.jpg',
                              crop_size=165,
                              network_depth=16,
                              network_filters=64)


### Configurations for visual summary ###
BALLOONS_SIMPLE_SMALL_CROPS_CONFIG = Config(image_name='balloons.png',
                                            crop_size=64,
                                            network_depth=11,
                                            network_filters=[32, 64, 64, 128, 128, 256, 128, 128, 64, 64, 32])

MOUNTAINS1_SIMPLE_SMALL_CROPS_CONFIG = Config(image_name='mountains1.jpg',
                                              crop_size=64,
                                              network_depth=11,
                                              network_filters=[32, 64, 64, 128, 128, 256, 128, 128, 64, 64, 32])

MOUNTAINS2_SIMPLE_SMALL_CROPS_CONFIG = Config(image_name='mountains2.png',
                                              crop_size=64,
                                              network_depth=11,
                                              network_filters=[32, 64, 64, 128, 128, 256, 128, 128, 64, 64, 32])

### Configurations for inpainting ###

BIRDS_CCG_CONFIG = Config(image_name='birds.png',
                          training_method='ccg',
                          crop_size=128,
                          network_depth=9,
                          network_filters=[32, 64, 64, 128, 256, 128, 64, 64, 32]) # TODO YANIV: LOOK INTO CHANGING NUMBER OF FILTERS PER CONVNEXT BLOCK

STONE_CCG_CONFIG = Config(image_name='stone.png',
                          training_method='ccg',
                          crop_size=128,
                          network_depth=9,
                          network_filters=[32, 64, 64, 128, 256, 128, 64, 64, 32]) # TODO YANIV: LOOK INTO CHANGING NUMBER OF FILTERS PER CONVNEXT BLOCK

SEASCAPE_CCG_CONFIG = Config(image_name='seascape.png',
                             training_method='ccg',
                             crop_size=128,
                             network_depth=9,
                             network_filters=[32, 64, 64, 128, 256, 128, 64, 64, 32]) # TODO YANIV: LOOK INTO CHANGING NUMBER OF FILTERS PER CONVNEXT BLOCK