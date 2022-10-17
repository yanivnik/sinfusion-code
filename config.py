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
    parser.add_argument('--run_name', type=str, help='A name for the current experiment')
    parser.add_argument('--image_name', type=str, help='The image to train the model on')
    parser.add_argument('--training_method', type=str, choices=['simple', 'ccg', 'video'], help='The type of training mechanism')
    parser.add_argument('--diffusion_timesteps', type=int, help='Amount of diffusion timesteps to perform per level')
    parser.add_argument('--initial_lr', type=float, help='Initial value of LR')
    parser.add_argument('--network_depth', type=int, help='Depth of the backbone network (amount of blocks)')
    parser.add_argument('--network_filters', type=int, help='Amount of filters per convolutional level in the backbone networks')
    parser.add_argument('--crop_size', type=int, help='Size of crops to train the backbone models on')
    parser.add_argument('--no_progress', dest='log_progress', action='store_false')
    parser.add_argument('--available_gpus', type=str, help='The gpu indexes to run on, in CUDA format (0,1,2...)')

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

COWS_SIMPLE_CONFIG = Config(image_name='cows.png',
                            crop_size=140,
                            network_depth=16,
                            network_filters=64)

BALLOONS_COLLOSEUM_COMBINED_CONFIG = Config(image_name='balloons.png-colusseum_changed.png',
                                crop_size=178,
                                network_depth=16,
                                network_filters=64)

TAPESTRY_COMBINED_CONFIG = Config(image_name='tap1.png-tap2.png-tap3.png',
                                crop_size=100,
                                network_depth=16,
                                network_filters=64)

BALLOONS_SIMPLE_VS_PRETRAIN_CONFIG = Config(image_name='balloons_scale=0.729.png',
                                            crop_size=128,
                                            network_depth=16,
                                            network_filters=64)

MOUNTAINS3_SIMPLE_VS_PRETRAIN_CONFIG = Config(image_name='mountains3_scale=0.91_0.52.png',
                                              crop_size=135,
                                              network_depth=16,
                                              network_filters=64)

TREE_SIMPLE_VS_PRETRAIN_CONFIG = Config(image_name='tree_scale=0.52_0.52.png',
                                        crop_size=125,
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

COLUSSEUM_SIMPLE_CONFIG = Config(image_name='colusseum_changed.png',
                                 crop_size=178,
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

DOLPHINS_SIMPLE_CONFIG = Config(image_name='dolphins.jpg',
                                crop_size=165,
                                network_depth=16,
                                network_filters=64)

BIRDS3_SIMPLE_CONFIG = Config(image_name='birds_3.jpg',
                              crop_size=(165, 230),
                              network_depth=16,
                              network_filters=64)

BIRDS_SIMPLE_CONFIG = Config(image_name='birds.png',
                             crop_size=160,
                             network_depth=16,
                             network_filters=64)

STONE_SIMPLE_CONFIG = Config(image_name='stone.png',
                             crop_size=(160, 230),
                             network_depth=16,
                             network_filters=64)

MOUNTAINS_SIMPLE_CONFIG = Config(image_name='mountains.jpg',
                                 crop_size=(360, 400),
                                 network_depth=16,
                                 network_filters=64)

BIRDS3_SIMPLE_VS_PRETRAIN_CONFIG = Config(image_name='birds_3_scale=0.96_0.64.png',
                                        crop_size=158,
                                        network_depth=16,
                                        network_filters=64)

NIEGHBOURHOOD_SIMPLE_CONFIG = Config(image_name='neighbourhood_small.png',
                                     crop_size=175,
                                     network_depth=16,
                                     network_filters=64)


### Configurations for video generation ###
BALLOONS2_VIDEO_CONFIG = Config(image_name='air_balloons',
                                training_method='video',
                                crop_size=135,
                                network_depth=16,
                                network_filters=64)

BALLOONS2_256_VIDEO_CONFIG = Config(image_name='air_balloons_256',
                                    training_method='video',
                                    crop_size=256,
                                    network_depth=16,
                                    network_filters=64)

TORNADO_VIDEO_CONFIG = Config(image_name='tornado',
                              training_method='video',
                              crop_size=131,
                              network_depth=16,
                              network_filters=64)

DUTCH_VIDEO_CONFIG = Config(image_name='dutch2',
                            training_method='video',
                            crop_size=(131, 180),
                            network_depth=16,
                            network_filters=64)

SKI_VIDEO_CONFIG = Config(image_name='ski_slope',
                          training_method='video',
                          crop_size=180,
                          network_depth=16,
                          network_filters=64)

FISH_VIDEO_CONFIG = Config(image_name='fish',
                          training_method='video',
                          crop_size=(135, 220),
                          network_depth=16,
                          network_filters=64)

BIRDS4_VIDEO_CONFIG = Config(image_name='birds4',
                             training_method='video',
                             crop_size=256,
                             network_depth=16,
                             network_filters=64)

JUNCTION_VIDEO_CONFIG = Config(image_name='junction',
                               training_method='video',
                               crop_size=180,
                               network_depth=16,
                               network_filters=64)

LIZARD_VIDEO_CONFIG = Config(image_name='lizard',
                                training_method='video',
                                crop_size=180,
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

MOUNTAINS3_SIMPLE_SMALL_CROPS_CONFIG = Config(image_name='mountains3.png',
                                            crop_size=64,
                                            network_depth=11,
                                            network_filters=[32, 64, 64, 128, 128, 256, 128, 128, 64, 64, 32])

TREE_SIMPLE_SMALL_CROPS_CONFIG = Config(image_name='tree.png',
                                            crop_size=64,
                                            network_depth=11,
                                            network_filters=[32, 64, 64, 128, 128, 256, 128, 128, 64, 64, 32])

BIRDS3_SIMPLE_SMALL_CROPS_CONFIG = Config(image_name='birds_3.jpg',
                                          crop_size=64,
                                          network_depth=11,
                                          network_filters=[32, 64, 64, 128, 128, 256, 128, 128, 64, 64, 32])

### Configurations for inpainting ###

FRUIT_CCG_CONFIG = Config(image_name='fruit.png',
                          training_method='ccg',
                          crop_size=128,
                          network_depth=8,
                          network_filters=64)

STONE_CCG_CONFIG = Config(image_name='stone.png',
                          training_method='ccg',
                          crop_size=128,
                          network_depth=8,
                          network_filters=64)

STARRY_NIGHT_CCG_CONFIG = Config(image_name='starry_night.png',
                                 training_method='ccg',
                                 crop_size=128,
                                 network_depth=8,
                                 network_filters=64)

SEASCAPE_CCG_CONFIG = Config(image_name='seascape.png',
                             training_method='ccg',
                             crop_size=128,
                             network_depth=8,
                             network_filters=64)
