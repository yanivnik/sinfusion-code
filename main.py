import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
from torch.utils.data import DataLoader
from PIL import Image
import pytorch_lightning as pl

from models.zssr import ZSSRNet
from diffusion.diffusion import GaussianDiffusion
from datasets.cropset import CropSet


# Training hyperparameters
diffusion_timesteps = 1000
training_steps = 5000
batch_size = 1  # Each batch contains a single crop, since the batch is actually made of the patches within the crop
image_name = 'balloons.png'

# Create datasets and data loaders
train_dataset = CropSet(image_path=f'./images/{image_name}')
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

# Create model and trainer
model = ZSSRNet()
diffusion = GaussianDiffusion(model, channels=3, timesteps=diffusion_timesteps)

tb_logger = pl.loggers.TensorBoardLogger(
    "lightning_logs/",
    name=image_name
)

trainer = pl.Trainer(
    max_steps=training_steps,
    log_every_n_steps=10,
    gpus=1,
    auto_select_gpus=True,
    logger=tb_logger
)

# Train model
trainer.fit(diffusion, train_loader)

### Sample from model
sample_grid_size = 5
samples = diffusion.sample(image_size=(32,32), batch_size=sample_grid_size * sample_grid_size)

s = (samples.clamp(-1, 1) + 1) / 2
s = (s * 255).type(torch.uint8).moveaxis(1, 3)
s = s.reshape(-1, sample_grid_size, sample_grid_size, 32, 32, 3)

def stack_samples(samples, stack_dim):
    samples = list(torch.split(samples, 1, dim=1))
    for i in range(len(samples)):
        samples[i] = samples[i].squeeze(1)
    return torch.cat(samples, dim=stack_dim)

s = stack_samples(s, 2)
s = stack_samples(s, 2)

im = Image.fromarray(s.cpu().numpy()[0])
im.save(f"{trainer.logger.log_dir}/samples.png")