
import pytorch_lightning as pl  # TODO REPLACE WITH NORMAL TRAINING
from torch.utils.data import DataLoader
import torch
import imageio
from datasets.singleset import SingleSet
from models.single_image_model import SingleImageDiffusionModel


def train_patch_diffusion(image_name, tb_logger=None, diffusion_steps=1000, max_epochs=10000):
    dataset = SingleSet(image_path=f'./images/{image_name}')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SingleImageDiffusionModel(dataset.size * dataset.size, diffusion_steps, dataset.depth)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=1,
        auto_select_gpus=True,
        logger=tb_logger
    )

    trainer.fit(model, loader)
    return model


def sample_from_model(model, tb_logger):
    n_hold_final = 25

    # Generate samples from denoising process
    gen_samples = []
    x = torch.randn((1, 3, 256, 256))
    sample_steps = torch.arange(model.t_range - 1, 0, -1)
    for t in sample_steps:
        x = model.denoise_sample(x, t)
        if t % 50 == 0:
            gen_samples.append(x)
    for _ in range(n_hold_final):
        gen_samples.append(x)
    gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1)
    gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2

    # Process samples and save as gif
    gen_samples = (gen_samples * 255).type(torch.uint8)
    gen_samples = gen_samples.reshape(-1, 256, 256, 3)

    def stack_samples(gen_samples, stack_dim):
        gen_samples = list(torch.split(gen_samples, 1, dim=1))
        for i in range(len(gen_samples)):
            gen_samples[i] = gen_samples[i].squeeze(1)
        return torch.cat(gen_samples, dim=stack_dim)

    #gen_samples = stack_samples(gen_samples, 2)
    #gen_samples = stack_samples(gen_samples, 2)

    imageio.mimsave(
        f"{tb_logger.log_dir}/pred.gif",
        list(gen_samples),
        fps=5,
    )


def main():
    image_name = 'balloons.png'
    tb_logger = pl.loggers.TensorBoardLogger("lightning_logs/", name=image_name)
    model = train_patch_diffusion(image_name, tb_logger)
    sample_from_model(model, tb_logger)


if __name__ == '__main__':
    main()
