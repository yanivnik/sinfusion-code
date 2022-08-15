import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from diffusion.diffusion_utils import cosine_noise_schedule, save_diffusion_sample, to_torch


class SRDiffusion(LightningModule):
    def __init__(self, model, channels=3, timesteps=1000, initial_lr=2e-4, auto_sample=True):
        super().__init__()

        self.step_counter = 0  # Overall step counter used to sample every n global steps
        self.sample_every_n_steps = 1000
        self.auto_sample = auto_sample

        self.channels = channels
        self.model = model
        self.initial_lr = initial_lr

        betas = cosine_noise_schedule(timesteps)
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
                             (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).view((batch_size,)).to(x.device)
        x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.model(torch.cat([condition_x, x], dim=1), noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def sample(self, lr):
        hr_img = torch.randn_like(lr) # LR should be a low resolution image after upsampling
        for i in reversed(range(0, self.num_timesteps)):
            hr_img = self.p_sample(hr_img, i, condition_x=lr)
        return hr_img

    def sample_ddim(self, lr, x_T=None, sampling_step_size=100):
        """
        Sample from the model, using the DDIM sampling process.
        The DDIM implicit sampling process is determinstic, and will always generate the same output
        if given the same input.

        Args:
            lr (torch.tensor): The low resolution image used to condition the sampling process.
            x_T (torch.tensor): The initial noise to start the sampling process from. Can be None.
            sampling_step_size (int): The step size between each t in the sampling process. The higher this value is,
                                      the faster the sampling process (as well as lower image quality).
        """
        batch_size = lr.shape[0]
        seq = range(0, self.num_timesteps, sampling_step_size)
        seq_next = [-1] + list(seq[:-1])

        if x_T is None:
            x_t = torch.randn(lr.shape, device=self.device)
        else:
            x_t = x_T

        zipped_reversed_seq = list(zip(reversed(seq), reversed(seq_next)))[:-1]
        for t, t_next in zipped_reversed_seq:
            noise_level = torch.FloatTensor(
                [self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).view((batch_size,)).to(x_t.device)

            e_t = self.model(torch.cat([lr, x_t], dim=1), noise_level)
            predicted_x0 = (x_t - self.sqrt_one_minus_alphas_cumprod[t] * e_t) / self.sqrt_alphas_cumprod[t]
            direction_to_x_t = self.sqrt_one_minus_alphas_cumprod[t_next] * e_t
            x_t = self.sqrt_alphas_cumprod[t_next] * predicted_x0 + direction_to_x_t

        t_tensor = torch.full((batch_size,), 0, dtype=torch.int64, device=self.device)
        e_t = self.model(torch.cat([lr, x_t], dim=1), t_tensor)
        x_0 = (x_t - self.sqrt_one_minus_alphas_cumprod[0] * e_t) / self.sqrt_alphas_cumprod[0]
        return x_0

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def forward(self, x_in, noise=None):
        self.i += 1

        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        x_recon = self.model(torch.cat([x_in['LR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod.view(-1))

        return F.mse_loss(noise, x_recon)

    def training_step(self, batch, batch_idx):
        if self.auto_sample and self.step_counter % self.sample_every_n_steps == 0:
            self.sample_and_save_output(batch['LR'], f"{self.logger.log_dir}/sample_{self.step_counter}.png")

        loss = self.forward(batch)
        self.log('train/loss', loss)
        self.step_counter += 1
        return loss

    @torch.no_grad()
    def sample_and_save_output(self, lr, output_path):
        """
        Sample a single image, normalize it, and save into an output file.

        Args:
            output_path (String): The path to save the image in.
            sample_size (tuple or int): The spatial dimensions of the image. If an int is passed, it is used for
                                        both spatial dimensions.
        """
        sample = self.sample(lr=lr)
        save_diffusion_sample(lr, f"{self.logger.log_dir}/sample_{self.step_counter}_conditioning.png")
        save_diffusion_sample(sample, output_path)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        return optim
