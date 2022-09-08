import os
from random import random

import torch

from common_utils.ben_image import imread
from config import *
from diffusion.conditional_diffusion import ConditionalDiffusion
from diffusion.diffusion_utils import save_diffusion_sample
from models.nextnet import NextNet


def sample_video(cfg):
    # Create sample directory
    sample_directory = os.path.join(f'/home/yanivni/data/tmp/organized-outputs/', 'Video Generation', f'{cfg.video_name}')
    if os.path.exists(sample_directory) and len(os.listdir(sample_directory)) > 0:
        sample_index = int(sorted(os.listdir(sample_directory))[-1]) + 1
    else:
        sample_index = 1
    sample_directory = os.path.join(sample_directory, str(sample_index))
    os.makedirs(sample_directory, exist_ok=True)

    # Load model
    path = f'/home/yanivni/data/remote_projects/single-image-diffusion/lightning_logs/{cfg.video_name}/{cfg.version_name}/checkpoints/last.ckpt'
    model = ConditionalDiffusion.load_from_checkpoint(path,
                                                      model=NextNet(in_channels=6, depth=cfg.network_depth,
                                                                    filters_per_layer=cfg.network_filters,
                                                                    frame_conditioned=True),
                                                      timesteps=cfg.diffusion_timesteps).to(device='cuda:0')

    total_frame_count = len(os.listdir(f'./images/video/{cfg.video_name}'))

    # Choose starting frame - TODO YANIV SOLVE THE STARTING FRAME ISSUE
    start_frame_idx = random.randint(1, total_frame_count)
    start_frame = imread(f'./images/video/{cfg.video_name}/{start_frame_idx}.png') * 2 - 1
    samples = [start_frame]

    # Sample frames
    for frame in range(2, total_frame_count + 1):
        s = model.sample(condition=samples[-1], frame=frame)
        samples.append(s)
        save_diffusion_sample(s, os.path.join(sample_directory, '{frame}.png'))

    # Save video
    ordered_samples = torch.cat(samples, dim=0)
    torchvid2mp4(resize(ordered_samples, out_shape=(3, 134, 134)).permute((1, 0, 2, 3)),
                 '/home/yanivni/data/tmp/video_generation/sample3.mp4')


def main():
    cfg = BALLOONS2_VIDEO_CONFIG
    cfg = parse_cmdline_args_to_config(cfg)

    cfg.version_name = '3-all-frames-full-crop'

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.available_gpus

    log_config(cfg)

    if cfg.training_method == 'video':
        sample_video(cfg)


if __name__ == '__main__':
    main()
