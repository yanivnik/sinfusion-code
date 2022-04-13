import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets.cropset import CropSet
from diffusion.diffusion import GaussianDiffusion
from models.zssr import ZSSRNet


def main():
    # TODO ADD EXTERNAL CONFIGURATION SUPPORT
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    # Training hyperparameters
    diffusion_timesteps = 1000
    training_steps = 5000
    batch_size = 1  # Each batch contains a single crop, since the batch is actually made of the patches within the crop
    image_name = 'balloons.png'

    # Create datasets and data loaders
    train_dataset = CropSet(image_path=f'./images/{image_name}', crop_size=(32, 32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    # Create model and trainer
    model = ZSSRNet()
    diffusion = GaussianDiffusion(model, channels=3, timesteps=diffusion_timesteps, sample_size=(32, 32),
                                  sample_every_n_steps=1000)
    tb_logger = pl.loggers.TensorBoardLogger("lightning_logs/", name=image_name)
    trainer = pl.Trainer(max_steps=training_steps, log_every_n_steps=10, gpus=1, auto_select_gpus=True, logger=tb_logger)

    # Train model (samples are generated during training)
    trainer.fit(diffusion, train_loader)


if __name__ == '__main__':
    main()
