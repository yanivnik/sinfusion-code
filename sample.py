import os
import random
import string

import numpy as np
import torch

from common_utils.ben_image import imread
from common_utils.resize_right import resize
from common_utils.video import torchvid2mp4
from config import *
from diffusion.conditional_diffusion import ConditionalDiffusion
from diffusion.diffusion import Diffusion
from diffusion.diffusion_utils import save_diffusion_sample
from models.nextnet import NextNet
from settings import results_dir, datasets_dir
import argparse


def hold_gpus():
    # catch gpus so gpu-manager would know not to touch them
    bla = []
    for i in range(torch.cuda.device_count()):
        bla.append(torch.tensor(1.0).to(f'cuda:{i}'))
    return bla


def random_name():
    vowels = "aeiou"
    consonants = "".join(set(string.ascii_lowercase) - set(vowels))
    return ''.join([f'{random.choice(consonants)}{random.choice(vowels)}' for i in range(4)])


def get_model_path(image_name, version_name):
    return f'/home/yanivni/data/remote_projects/single-image-diffusion/lightning_logs/{image_name}/{version_name}/checkpoints/last.ckpt'


def noise_img(img, model, t):
    batch_size = img.shape[0]
    if isinstance(model, Diffusion):
        noisy_img = model.q_sample(img, t)
    elif isinstance(model, ConditionalDiffusion):
        continuous_sqrt_alpha_hat = torch.FloatTensor(np.random.uniform(model.sqrt_alphas_hat_prev[t - 1], model.sqrt_alphas_hat_prev[t], size=batch_size)).to(img.device).view(batch_size, -1)
        noisy_img = model.q_sample(img, continuous_sqrt_alpha_hat.view(-1, 1, 1, 1))
    else:
        raise Exception

    return noisy_img


def generate_video(cfg, save_frames=True):
    """
    Generates and saves a video (in mp4 format) based on the configuration parameters.

    Args:
         save_frames: Should the single frames also be saved.
    """
    run_id = cfg.run_name or random_name()

    # Create sample directory
    sample_directory = os.path.join(cfg.sample_directory, 'Video Generation', f'{cfg.image_name}', cfg.experiment_name, run_id)
    video_dir = os.path.join(datasets_dir, 'images', 'video', f'{cfg.image_name}')
    os.makedirs(sample_directory, exist_ok=True)
    print(f'Sample directory: {sample_directory}')

    # Load model
    path = get_model_path(cfg.image_name, cfg.version_name)
    model = ConditionalDiffusion.load_from_checkpoint(path,
                                                      model=NextNet(in_channels=6, depth=cfg.network_depth,
                                                                    filters_per_layer=cfg.network_filters,
                                                                    frame_conditioned=True),
                                                      timesteps=cfg.diffusion_timesteps).cuda()

    total_frame_count = len(os.listdir(video_dir))

    # Choose starting frame
    start_frame_idx = 1  #random.randint(1, total_frame_count)
    start_frame = imread(os.path.join(video_dir, f'{start_frame_idx}.png')).cuda() * 2 - 1
    if save_frames:
        save_diffusion_sample(start_frame, os.path.join(sample_directory, '0.png'))
    samples = [start_frame]

    # Sample frames
    for frame in range(1, total_frame_count + 1):
        s = model.sample(condition=samples[-1], frame=1)#frame)
        samples.append(s)
        if save_frames:
            save_diffusion_sample(s, os.path.join(sample_directory, f'{frame}.png'))

    # Save video
    ordered_samples = torch.cat(samples, dim=0)
    resized_samples = resize(ordered_samples, out_shape=(3, (start_frame.shape[-2] // 2) * 2, (start_frame.shape[-1] // 2) * 2))
    torchvid2mp4(resized_samples.permute((1, 0, 2, 3)), os.path.join(sample_directory, '..', f'video_{run_id}.mp4'))


def generate_diverse_samples(cfg, sizes=None):
    # Create sample directory
    sample_directory = os.path.join(cfg.sample_directory, 'Diverse Generation', f'{cfg.image_name}', random_name())
    print(f'Sample directory: {sample_directory}')

    # Load model
    path = get_model_path(cfg.image_name, cfg.version_name)
    model = Diffusion.load_from_checkpoint(path, model=NextNet(depth=16), timesteps=500, strict=False).cuda()

    if sizes is None:
        sizes = [tuple(imread(f'./images/{cfg.image_name}').shape[-2:])]
        sizes += [(sizes[0][0] * 2, sizes[0][1]), (sizes[0][0], sizes[0][1] * 2), (sizes[0][0] * 2, sizes[0][1] * 2)]

    # Sample and save images
    batch = 9
    for size in sizes:
        samples = model.sample(image_size=size, batch_size=batch)
        out_dir = os.path.join(sample_directory, f'{size[0]}x{size[1]}')
        os.makedirs(out_dir, exist_ok=True)
        save_diffusion_sample(samples, os.path.join(out_dir, 'sample.png'))



def visual_summarize(cfg):
    pass


def harmonize(cfg):
    pass


def generate_from_edit(cfg):
    pass


def generate_from_sketch(cfg):
    pass


def main():
    gpu_holder = hold_gpus()

    cfg = BALLOONS2_VIDEO_CONFIG
    cfg = parse_cmdline_args_to_config(cfg)

    cfg.version_name = '17-1-2-3-framediff-with-100k-200k-curriculum'
    cfg.experiment_name = 'runname_test'
    cfg.sample_directory = os.path.join(results_dir, 'organized-outputs')

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.available_gpus

    log_config(cfg)

    if cfg.training_method == 'video':
        generate_video(cfg)


if __name__ == '__main__':
    main()
