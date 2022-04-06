import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from tqdm import tqdm

from diffusion.diffusion_utils import extract, cosine_noise_schedule


class GaussianDiffusion(LightningModule):
    def __init__(self, model, *, channels=3, timesteps=1000):
        super().__init__()
        self.channels = channels
        self.model = model

        betas = cosine_noise_schedule(timesteps)

        alphas = 1. - betas
        alphas_hat = np.cumprod(alphas, axis=0)
        alphas_hat_prev = np.append(1., alphas_hat[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32))
        self.register_buffer('alphas_hat', torch.tensor(alphas_hat, dtype=torch.float32))
        self.register_buffer('alphas_hat_prev', torch.tensor(alphas_hat_prev, dtype=torch.float32))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_hat', torch.tensor(np.sqrt(alphas_hat), dtype=torch.float32))
        self.register_buffer('sqrt_one_minus_alphas_hat', torch.tensor(np.sqrt(1. - alphas_hat), dtype=torch.float32))
        self.register_buffer('log_one_minus_alphas_hat', torch.tensor(np.log(1. - alphas_hat), dtype=torch.float32))
        self.register_buffer('sqrt_recip_alphas_hat', torch.tensor(np.sqrt(1. / alphas_hat), dtype=torch.float32))
        self.register_buffer('sqrt_recipm1_alphas_hat', torch.tensor(np.sqrt(1. / alphas_hat - 1), dtype=torch.float32))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_hat_prev) / (1. - alphas_hat)

        self.register_buffer('posterior_variance', torch.tensor(posterior_variance, dtype=torch.float32))

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.tensor(np.log(np.maximum(posterior_variance, 1e-20)), dtype=torch.float32))
        self.register_buffer('posterior_mean_coef1', torch.tensor(betas * np.sqrt(alphas_hat_prev) / (1. - alphas_hat), dtype=torch.float32))
        self.register_buffer('posterior_mean_coef2', torch.tensor((1. - alphas_hat_prev) * np.sqrt(alphas) / (1. - alphas_hat), dtype=torch.float32))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_hat, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_hat, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_hat, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_hat, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_hat, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True):
        b, *_ = x.shape
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def sample(self, image_size=(32, 32), batch_size=16):
        sample_shape = (batch_size, self.channels, image_size[0], image_size[1])

        img = torch.randn(sample_shape).cuda()
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t_tensor = torch.full((batch_size, ), t, dtype=torch.long)
            img = self.p_sample(img, t_tensor)
        return img

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_hat, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_hat, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(x_noisy, t)

        return F.mse_loss(noise, x_recon)

    def forward(self, x, *args, **kwargs):
        b, c, h, w = x.shape
        t = torch.randint(0, self.num_timesteps, (b,)).long()
        return self.p_losses(x, t, *args, **kwargs)

    def training_step(self, batch):
        loss = self.forward(batch)
        self.log('train/loss', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optim
