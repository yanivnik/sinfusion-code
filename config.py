import argparse
import sys


class Config:
    # Constant configuration options
    project_name = 'SinFusion'

    # Deployment configuration options
    available_gpus = '0'    # list of available gpus (in CUDA_VISIBLE_DEVICES format)
    run_name = None

    # Diffusion configuration options
    diffusion_timesteps = 50
    task = 'image'

    # Backbone model and dataset configuration options
    network_filters = 64    # Amount of filters in backbone network conv layers
    network_depth = 16

    # Data related configuration
    image_name = 'balloons.png'

    # Optimization configuration options
    initial_lr = 0.0002

    # Sampling configuration options
    output_dir = 'outputs'
    frame_diff = 1
    output_video_len = 100
    interpolation_rate = 4
    start_frame_index = None
    sample_count = 1
    sample_size = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def log_config(cfg):
    for k in dir(cfg):
        if k[:1] != '_':
            print(f'{k}={getattr(cfg, k)}')


def _tuple_of_ints(s):
    s = s.replace('(', '').replace(')', '')
    mapped_int = map(int, s.split(','))
    return tuple(mapped_int)


def parse_cmdline_args_to_config(cfg):
    parser = argparse.ArgumentParser(description='Command line configuration')

    # General and training-only configuration arguments
    parser.add_argument('--run_name', type=str, help='A name for the current training run')
    parser.add_argument('--image_name', type=str, help='The image to train the model on')
    parser.add_argument('--task', type=str, choices=['image', 'video', 'video_interp'],
                        help='The type of task for training / sampling')
    parser.add_argument('--diffusion_timesteps', type=int, help='Amount of diffusion timesteps to perform per level')
    parser.add_argument('--network_depth', type=int, help='Depth of the backbone network (amount of blocks)')
    parser.add_argument('--network_filters', type=int,
                        help='Amount of filters per convolutional level in the backbone networks')
    parser.add_argument('--available_gpus', type=str, help='The gpu indexes to run on, in CUDA format (0,1,2...)')
    parser.add_argument('--initial_lr', type=float, help='Initial value of LR')

    # Sampling arguments
    parser.add_argument('--output_dir', type=str, help='The directory to save the generated images/videos to')
    parser.add_argument('--output_video_len', type=int, help='Number of frames to generate in output video')
    parser.add_argument('--interpolation_rate', type=int,
                        help='Factor by which the video length will be increased (e.g. 4 -> 4x temporal upsampling).')
    parser.add_argument('--frame_diff', type=int,
                        help='The frame difference (k) between each two generated frames (e.g. 1 means simple '
                             'forward generation, 2 means faster movements, -1 means backward generation, etc).')
    parser.add_argument('--start_frame_index', type=int, help='Index of the frame to start generation from. '
                                                              'If not supplied, the first frame is generated using '
                                                              'the DDPM frame Projector.')
    parser.add_argument('--sample_count', type=int, help='Amount of samples to generate')
    parser.add_argument('--sample_size', type=_tuple_of_ints, help='Spatial size of samples to generate. '
                                                                   'Defines the frame size in case of video, '
                                                                   'or image size in case of image.')

    args = parser.parse_known_args(sys.argv[1:])

    # Override cfg attribute values with supplied cmdline args
    for k, v in vars(args[0]).items():
        if v is not None:
            setattr(cfg, k, v)

    return cfg


# Pre-made configurations

### Configurations for diverse generation, retargeting, from sketch/edit, etc ###
BALLOONS_IMAGE_CONFIG = Config(image_name='balloons.png')
COWS_IMAGE_CONFIG = Config(image_name='cows.png')
LIGHTNING_IMAGE_CONFIG = Config(image_name='lightning1.png')
STARRY_NIGHT_IMAGE_CONFIG = Config(image_name='starry_night.png')
MOUNTAINS3_IMAGE_CONFIG = Config(image_name='mountains3.png')
PENGUINS_IMAGE_CONFIG = Config(image_name='penguins.png')
DOLPHINS_IMAGE_CONFIG = Config(image_name='dolphins.jpg')
BIRDS3_IMAGE_CONFIG = Config(image_name='birds_3.jpg')
BIRDS_IMAGE_CONFIG = Config(image_name='birds.png')
STONE_IMAGE_CONFIG = Config(image_name='stone.png')
MOUNTAINS_IMAGE_CONFIG = Config(image_name='mountains.jpg')
NIEGHBOURHOOD_IMAGE_CONFIG = Config(image_name='neighbourhood_small.png')

### Configurations for video generation ###
WALKING_SCENE_VIDEO_CONFIG = Config(image_name='walking_scene', task='video')
AIR_BALLOONS_VIDEO_CONFIG = Config(image_name='air_balloons', task='video')
SKI_VIDEO_CONFIG = Config(image_name='ski_slope', task='video')
FISH_VIDEO_CONFIG = Config(image_name='fish', task='video')
BIRDS4_VIDEO_CONFIG = Config(image_name='birds4', task='video')
TORNADO_VIDEO_CONFIG = Config(image_name='tornado', task='video')
ANTS_VIDEO_CONFIG = Config(image_name='ants', task='video')
ANTS2_VIDEO_CONFIG = Config(image_name='ants2', task='video')
BOAT_RACE_VIDEO_CONFIG = Config(image_name='boat_race', task='video')
POOL_VIDEO_CONFIG = Config(image_name='pool', task='video')
BASE_FLIGHT_VIDEO_CONFIG = Config(image_name='base_flight', task='video')
BALLET_VIDEO_CONFIG = Config(image_name='ballet', task='video')
SAIL_VIDEO_CONFIG = Config(image_name='sail_amsterdam', task='video')
DUTCH_VIDEO_CONFIG = Config(image_name='dutch2', task='video')
SHEEP_VIDEO_CONFIG = Config(image_name='sheep', task='video')
BIRDS_VIDEO_CONFIG = Config(image_name='birds', task='video')
FACE5_VIDEO_CONFIG = Config(image_name='mead_face_005', task='video')
FACE9_VIDEO_CONFIG = Config(image_name='mead_face_009', task='video')
FACE11_VIDEO_CONFIG = Config(image_name='mead_face_011', task='video')
FACE27_VIDEO_CONFIG = Config(image_name='mead_face_027', task='video')
FACE39_VIDEO_CONFIG = Config(image_name='mead_face_039', task='video')

### Configurations for video interpolation (temporal upsampling) ###
FAN_VIDEO_CONFIG = Config(image_name='star_fan', task='video_interp')
HULAHUOOP_VIDEO_CONFIG = Config(image_name='hula_hoop', task='video_interp')
BILLIARD_VIDEO_CONFIG = Config(image_name='billiard', task='video_interp')
