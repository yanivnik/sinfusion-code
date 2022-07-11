import numpy as np
import torch
import torch.nn.functional as F

from diffusion.diffusion import Diffusion


# TODO IF THIS SR WORKS WELL THAN REFACTOR IT INTO THE DIFFUSION CLASS
class SRDiffusion(Diffusion):
    """
    A basic implementation of SR3 diffusion.
    """
    def __init__(self, model, timesteps):
        super().__init__(model, timesteps=timesteps, auto_sample=False)

        self.sqrt_alphas_hat_prev = np.sqrt(self.alphas_hat_prev.cpu().numpy())

    def forward(self, x, *args, **kwargs):
        x_hr, x_lr = x['HR'], x['LR']
        batch_size = x_hr.shape[0]

        t = torch.randint(0, self.num_timesteps, (batch_size,), dtype=torch.int64, device=self.device)
        #t = np.random.randint(1, self.num_timesteps)
        #continuous_sqrt_alphas_hat = torch.FloatTensor(
        #    np.random.uniform(self.sqrt_alphas_hat_prev[t - 1],
        #                      self.sqrt_alphas_hat_prev[t],
        #                      size=batch_size)
        #).to(device=x_hr.device).view(batch_size)

        noise = torch.randn_like(x_hr)
        # x_hr_noisy = self.q_sample(x_start=x_hr, continuous_sqrt_alphas_hat=continuous_sqrt_alphas_hat, noise=noise)
        x_hr_noisy = self.q_sample(x_start=x_hr, t=t, noise=noise)
        noise_recon = self.model(torch.cat([x_lr, x_hr_noisy], dim=1), t)
        return F.mse_loss(noise, noise_recon)

    # def q_sample(self, x_start, continuous_sqrt_alphas_hat, noise=None):
    #     if noise is None:
    #         noise = torch.randn_like(x_start)
    #     return continuous_sqrt_alphas_hat * x_start + (1 - continuous_sqrt_alphas_hat ** 2).sqrt() * noise

    @torch.no_grad()
    def sample(self, lr, image_size):
        batch_size = lr.shape[0]
        image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        sample_shape = (batch_size, self.channels, image_size[0], image_size[1])

        hr_img = torch.randn(sample_shape, device=self.device)
        for t in reversed(range(0, self.num_timesteps)):
            t_tensor = torch.full((batch_size, ), t, dtype=torch.int64, device=self.device)
            hr_img = self.p_sample_conditioned(hr_img, lr, t_tensor)
        return hr_img


    @torch.no_grad()
    def p_sample_conditioned(self, x_hr, x_lr, t, clip_denoised=True):
        b, *_ = x_hr.shape
        model_mean, _, model_log_variance = self.p_mean_variance_conditioned(x_hr=x_hr, x_lr=x_lr, t=t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x_hr)
        # no noise when t == 0 # TODO REWRITE BETTER
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_hr.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_mean_variance_conditioned(self, x_hr, x_lr, t, clip_denoised):
        x_recon = self.predict_start_from_noise(x_hr, t=t, noise=self.model(torch.cat([x_lr, x_hr], dim=1), t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x_hr, t=t)
        return model_mean, posterior_variance, posterior_log_variance







from inspect import isfunction
def exists(x):
    return x is not None
from functools import partial

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
from diffusion.diffusion_utils import cosine_noise_schedule
from pytorch_lightning import LightningModule
class TheirsSRDiffusion(LightningModule):
    def __init__(self, model, channels=3, timesteps=1000, recon_loss_factor=0.0, recon_image=None, recon_image_lr=None, initial_lr=2e-4):
        super().__init__()

        self.step_counter = 0  # Overall step counter used to sample every n global steps

        self.channels = channels
        self.model = model
        self.initial_lr = initial_lr

        self.recon_loss_factor = recon_loss_factor
        self.i = 0
        if self.recon_loss_factor > 0:
            assert recon_image is not None
            self.register_buffer('recon_image', recon_image)
            self.register_buffer('recon_image_lr', recon_image_lr)
            self.register_buffer('recon_noise', torch.randn_like(recon_image))

        to_torch = partial(torch.tensor, dtype=torch.float32)

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
    def sample(self, lr, image_size):
        batch_size = lr.shape[0]
        image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        sample_shape = (batch_size, self.channels, image_size[0], image_size[1])

        hr_img = torch.randn(sample_shape, device=self.device)
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
        noise = default(noise, lambda: torch.randn_like(x_start))

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

        x_recon = self.model(
            torch.cat([x_in['LR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod.view(-1))

        loss = F.mse_loss(noise, x_recon)

        if self.recon_loss_factor > 0 and self.i % 5 == 0:
            # Add a reconstruction loss between the original image and the DDIM
            # sampling result of the constant reconstruction noise.
            generated_image = self.sample_ddim(lr=self.recon_image_lr,
                                               image_size=self.recon_image.shape[-2:],
                                               sampling_step_size=self.num_timesteps // 10,
                                               custom_initial_img=self.recon_noise)
            loss = loss + F.mse_loss(generated_image, self.recon_image)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log('train/loss', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        return optim
