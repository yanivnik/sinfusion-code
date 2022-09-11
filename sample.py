import os
import random
import string

import torch

from common_utils.ben_image import imread
from common_utils.resize_right import resize
from common_utils.video import torchvid2mp4
from config import *
from diffusion.conditional_diffusion import ConditionalDiffusion
from diffusion.diffusion_utils import save_diffusion_sample
from models.nextnet import NextNet


def random_name():
    vowels = "aeiou"
    consonants = "".join(set(string.ascii_lowercase) - set(vowels))
    return ''.join([f'{random.choice(consonants)}{random.choice(vowels)}' for i in range(4)])


def generate_video(cfg, save_frames=True):
    """
    Generates and saves a video (in mp4 format) based on the configuration parameters.

    Args:
         save_frames: Should the single frames also be saved.
    """
    # Create sample directory
    sample_directory = os.path.join(cfg.sample_directory, 'Video Generation', f'{cfg.image_name}', random_name())
    os.makedirs(sample_directory, exist_ok=True)
    print(f'Sample directory: {sample_directory}')

    # Load model
    path = f'/home/yanivni/data/remote_projects/single-image-diffusion/lightning_logs/{cfg.image_name}/{cfg.version_name}/checkpoints/last.ckpt'
    model = ConditionalDiffusion.load_from_checkpoint(path,
                                                      model=NextNet(in_channels=6, depth=cfg.network_depth,
                                                                    filters_per_layer=cfg.network_filters,
                                                                    frame_conditioned=True),
                                                      timesteps=cfg.diffusion_timesteps).cuda()

    total_frame_count = len(os.listdir(f'./images/video/{cfg.image_name}'))

    # Choose starting frame - TODO YANIV SOLVE THE STARTING FRAME ISSUE
    start_frame_idx = random.randint(1, total_frame_count)
    start_frame = imread(f'./images/video/{cfg.image_name}/{start_frame_idx}.png').cuda() * 2 - 1
    if save_frames:
        save_diffusion_sample(start_frame, os.path.join(sample_directory, '1.png'))
    samples = [start_frame]

    # Sample frames
    for frame in range(2, total_frame_count + 1):
        s = model.sample(condition=samples[-1], frame=frame)
        samples.append(s)
        if save_frames:
            save_diffusion_sample(s, os.path.join(sample_directory, f'{frame}.png'))

    # Save video
    ordered_samples = torch.cat(samples, dim=0)
    resized_samples = resize(ordered_samples, out_shape=(3, (start_frame.shape[-2] // 2) * 2, (start_frame.shape[-1] // 2) * 2))
    torchvid2mp4(resized_samples.permute((1, 0, 2, 3)), os.path.join(sample_directory, 'video.mp4'))


def generate_diverse_samples(cfg, sizes=None):
    pass


def inpaint(cfg):
    pass


def visual_summarize(cfg):
    pass


def harmonize(cfg):
    pass


def generate_from_edit(cfg):
    pass


def generate_from_sketch(cfg):
    pass


def main():
    cfg = BALLOONS2_MEDIUM_VIDEO_CONFIG
    cfg = parse_cmdline_args_to_config(cfg)

    cfg.version_name = '1-all-frames-crop170'
    cfg.sample_directory = '/home/yanivni/data/tmp/organized-outputs/'

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.available_gpus

    log_config(cfg)

    if cfg.training_method == 'video':
        generate_video(cfg)


if __name__ == '__main__':
    main()
