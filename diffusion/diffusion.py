import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from common_utils.common import two_tuple
from diffusion.diffusion_utils import save_diffusion_sample, to_torch, linear_noise_schedule, cosine_noise_schedule


class Diffusion(LightningModule):
    def __init__(self, model, channels=3, timesteps=1000,
                 initial_lr=2e-4, training_target='x0', noise_schedule='cosine',
                 auto_sample=False, sample_every_n_steps=1000, sample_size=(32, 32)):
        """
        Args:
            model (torch.nn.Module):
                The model used to predict noise for reverse diffusion.
            channels (int):
                The amount of input channels in each image.
            timesteps (int):
                The amount of timesteps used to generate the noising schedule.
            initial_lr (float):
                The initial learning rate for the diffusion training.
            training_target (str):
                The type of parameterization to train the backbone model on.
                Can be either 'x0' or 'noise'.
            noise_schedule (str):
                The type of noise schedule to be used.
                Can be either 'linear' or 'cosine'.
            auto_sample (bool):
                Should the model perform sampling during training.
                If False, the following sampling parameters are ignored.
            sample_every_n_steps (int):
                The amount of global steps (step == training batch) after which the model is
                sampled from.
            sample_size (tuple):
                The spatial dimensions of the sample during auto sampling.
        """
        super().__init__()

        self.step_counter = 0  # Overall step counter used to sample every n global steps
        self.auto_sample = auto_sample
        self.sample_every_n_steps = sample_every_n_steps
        self.sample_size = sample_size

        self.channels = channels
        self.model = model
        self.initial_lr = initial_lr
        self.training_target = training_target.lower()
        assert self.training_target in ['x0', 'noise']

        assert noise_schedule in ['linear', 'cosine']
        if noise_schedule == 'linear':
            betas = linear_noise_schedule(timesteps)
        else:
            betas = cosine_noise_schedule(timesteps)

        self.num_timesteps = int(betas.shape[0])

        alphas = 1. - betas
        alphas_hat = np.cumprod(alphas, axis=0)
        alphas_hat_prev = np.append(1., alphas_hat[:-1])

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_hat', to_torch(alphas_hat))
        self.register_buffer('alphas_hat_prev', to_torch(alphas_hat_prev))
        self.register_buffer('sqrt_alphas_hat', to_torch(np.sqrt(alphas_hat)))
        self.register_buffer('sqrt_one_minus_alphas_hat', to_torch(np.sqrt(1. - alphas_hat)))
        self.register_buffer('log_one_minus_alphas_hat', to_torch(np.log(1. - alphas_hat)))
        self.register_buffer('sqrt_recip_alphas_hat', to_torch(np.sqrt(1. / alphas_hat)))
        self.register_buffer('sqrt_recipm1_alphas_hat', to_torch(np.sqrt(1. / alphas_hat - 1)))
        posterior_variance = betas * (1. - alphas_hat_prev) / (1. - alphas_hat)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_hat_prev) / (1. - alphas_hat)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1. - alphas_hat_prev) * np.sqrt(alphas) / (1. - alphas_hat)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_hat[t] * x_t - self.sqrt_recipm1_alphas_hat[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_variance = self.posterior_variance[t]
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised):
        batch_size = x.shape[0]
        t_tensor = torch.full((batch_size,), t, dtype=torch.int64, device=self.device)

        if self.training_target == 'x0':
            x_recon = self.model(x, t_tensor)
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t_tensor))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True):
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)  # no noise when t == 0
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def sample(self, image_size=(32, 32), batch_size=16, custom_initial_img=None, custom_timesteps=None):
        """
        Sample an image from noise via the reverse diffusion process.
        Args:
            image_size (int or tuple(int, int)):
                Spatial size of image to sample.
            batch_size (int):
                Amount of images to sample.
            custom_initial_img (torch.tensor):
                A non-default image to start the reverse process from. If this parameter is specified, both image_size
                and batch_size parameters are ignored.
                This can be used for denoising of partially noised images.
            custom_timesteps (int):
                A non-default diffusion timesteps parameter. If this parameter is specified, the reverse process is
                iterated for this given number of steps. Otherwise, the timestep parameter configured in the constructor
                is used.
                This can be used for denoising of partially noised images.
        """
        image_size = two_tuple(image_size)
        sample_shape = (batch_size, self.channels, image_size[0], image_size[1])

        timesteps = custom_timesteps or self.num_timesteps
        img = custom_initial_img if custom_initial_img is not None else torch.randn(sample_shape, device=self.device)
        for t in reversed(range(0, timesteps)):
            img = self.p_sample(img, t)
        return img

    @torch.no_grad()
    def sample_ddim(self, x_T=None, image_size=(32, 32), batch_size=16, sampling_step_size=100):
        """
        Sample from the model, using the DDIM sampling process.
        The DDIM implicit sampling process is determinstic, and will always generate the same output
        if given the same input.

        Args:
            x_T (torch.tensor): The initial noise to start the sampling process from. Can be None.
            image_size (int or tuple(int)): Used as the sample spatial dimensions in case x_T is None.
            batch_size (int): Used as the sample batch size in case x_T is None.
            sampling_step_size (int): The step size between each t in the sampling process. The higher this value is,
                                      the faster the sampling process (as well as lower image quality).
        """
        seq = range(0, self.num_timesteps, sampling_step_size)
        seq_next = [-1] + list(seq[:-1])

        if x_T is None:
            image_size = two_tuple(image_size)
            sample_shape = (batch_size, self.channels, image_size[0], image_size[1])
            x_t = torch.randn(sample_shape, device=self.device)
        else:
            batch_size = x_T.shape[0] if len(x_T.shape) == 4 else 1
            x_t = x_T

        zipped_reversed_seq = list(zip(reversed(seq), reversed(seq_next)))
        for t, t_next in zipped_reversed_seq:
            t_tensor = torch.full((batch_size,), t, dtype=torch.int64, device=self.device)
            e_t = self.model(x_t, t_tensor)

            predicted_x0 = (x_t - self.sqrt_one_minus_alphas_hat[t] * e_t) / self.sqrt_alphas_hat[t]
            if t > 0:
                direction_to_x_t = self.sqrt_one_minus_alphas_hat[t_next] * e_t
                x_t = self.sqrt_alphas_hat[t_next] * predicted_x0 + direction_to_x_t
            else:
                x_t = predicted_x0

        return x_t

    def q_sample(self, x_start, t, noise=None):
        """
        Perform forward diffusion (noising) in a single step.
        This method returns x_t, which is x_0 noised for t timesteps.

        Args:
            x_start (torch.Tensor): Represents the original image (x_0).
            t (int): The timestep that measures the amount of noise to add.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        return self.sqrt_alphas_hat[t] * x_start + self.sqrt_one_minus_alphas_hat[t] * noise

    def forward(self, x, *args, **kwargs):
        x = x.get('IMG')
        batch_size = x.shape[0]

        # Sample t uniformly
        t = np.random.randint(0, self.num_timesteps)

        # Generate white noise
        noise = torch.randn_like(x)

        # Produce x_t (noisy version of input after t noising steps)
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)

        # Attempt to reconstruct white noise that was used in forward process
        t_tensor = torch.full((batch_size, ), t, dtype=torch.int64, device=self.device)
        if self.training_target == 'x0':
            x0_recon = self.model(x_noisy, t_tensor)
            return F.mse_loss(x, x0_recon)
        else:
            noise_recon = self.model(x_noisy, t_tensor)
            return F.mse_loss(noise, noise_recon)

    def training_step(self, batch, batch_idx):
        if self.auto_sample and self.step_counter % self.sample_every_n_steps == 0:
            sample = self.sample(image_size=self.sample_size, batch_size=1)
            save_diffusion_sample(sample, f'{self.logger.log_dir}/sample_{self.step_counter}.png')

        loss = self.forward(batch)
        self.log('train_loss', loss)
        self.step_counter += 1
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[20], gamma=0.1, verbose=True)
        return [optim], [scheduler]
