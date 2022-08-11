import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from diffusion.diffusion_utils import extract, cosine_noise_schedule, save_diffusion_sample, to_torch
from metrics.swd_score import get_swd_scores


class Diffusion(LightningModule):
    def __init__(self, model, channels=3, timesteps=1000,
                 initial_lr=2e-4, recon_loss_factor=0.0, recon_image=None,
                 auto_sample=True, sample_every_n_steps=1000, sample_size=(32, 32)):
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
            recon_loss_factor (float):
                The weight to apply to the reconstruction loss during training. If this is 0,
                reconstruction loss is not used. NOTICE - using reconstruction loss might severly decrease
                performance.
            recon_image (torch.tensor):
                The image to use during reconstruction loss. The loss is an MSE between this given image and
                the result of deterministic sampling (AKA reconstruction) from a constant noise.
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

        self.i = 0
        self.recon_loss_factor = recon_loss_factor
        if self.recon_loss_factor > 0:
            assert recon_image is not None
            self.register_buffer('recon_image', (recon_image * 2) - 1)
            self.register_buffer('recon_noise', torch.randn_like(recon_image))

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

    def p_mean_variance(self, x, t, clip_denoised):
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
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))) # no noise when t == 0
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def sample(self, image_size=(32, 32), batch_size=16, custom_initial_img=None, custom_timesteps=None):
        image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        sample_shape = (batch_size, self.channels, image_size[0], image_size[1])

        timesteps = custom_timesteps or self.num_timesteps
        img = custom_initial_img if custom_initial_img is not None else torch.randn(sample_shape, device=self.device)
        for t in reversed(range(0, timesteps)):
            t_tensor = torch.full((batch_size, ), t, dtype=torch.int64, device=self.device)
            img = self.p_sample(img, t_tensor)
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
            image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
            sample_shape = (batch_size, self.channels, image_size[0], image_size[1])
            x_t = torch.randn(sample_shape, device=self.device)
        else:
            batch_size = x_T.shape[0] if len(x_T.shape) == 4 else 1
            image_size = x_T.shape[-2:]
            x_t = x_T

        zipped_reversed_seq = list(zip(reversed(seq), reversed(seq_next)))
        for t, t_next in zipped_reversed_seq:
            t_tensor = torch.full((batch_size,), t, dtype=torch.int64, device=self.device)
            t_next_tensor = torch.full((batch_size,), t_next, dtype=torch.int64, device=self.device)

            e_t = self.model(x_t, t_tensor)

            # TODO REPLACE WITH - predicted_x0 = self.predict_start_from_noise(x_t, t_tensor, e_t)
            predicted_x0 = (x_t - extract(self.sqrt_one_minus_alphas_hat, t_tensor, x_t.shape) * e_t) / \
                           extract(self.sqrt_alphas_hat, t_tensor, x_t.shape)
            if t > 0:
                direction_to_x_t = extract(self.sqrt_one_minus_alphas_hat, t_next_tensor, x_t.shape) * e_t
                x_t = extract(self.sqrt_alphas_hat, t_next_tensor, x_t.shape) * predicted_x0 + direction_to_x_t
            else:
                x_t = predicted_x0

        return x_t

    @torch.no_grad()
    def sample_and_save_output(self, output_path, sample_size):
        """
        Sample a single image, normalize it, and save into an output file.

        Args:
            output_path (String): The path to save the image in.
            sample_size (tuple or int): The spatial dimensions of the image. If an int is passed, it is used for
                                        both spatial dimensions.
        """
        sample = self.sample(image_size=sample_size, batch_size=1)
        save_diffusion_sample(sample, output_path)

    def q_sample(self, x_start, t, noise=None):
        """
        Perform forward diffusion (noising) in a single step.
        This method returns x_t, which is x_0 noised for t timesteps.

        Args:
            x_start (torch.Tensor): Represents the original image (x_0).
            t (torch.Tensor): The timestep that measures
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_hat, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_hat, t, x_start.shape) * noise
        )

    def forward(self, x, *args, **kwargs):
        self.i += 1
        batch_size = x.shape[0]

        # Sample t uniformly
        t = torch.randint(0, self.num_timesteps, (batch_size,), dtype=torch.int64, device=self.device)

        # Generate white noise
        noise = torch.randn_like(x)

        # Produce x_t (noisy version of input after t noising steps)
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)

        # Attempt to reconstruct white noise that was used in forward process
        noise_recon = self.model(x_noisy, t)

        loss = F.mse_loss(noise, noise_recon)

        if self.recon_loss_factor > 0 and self.i % 5 == 0:
            # Add a reconstruction loss between the original image and the DDIM
            # sampling result of the constant reconstruction noise.
            generated_image = self.sample_ddim(x_T=self.recon_noise, sampling_step_size=self.num_timesteps // 10)
            loss = loss + F.mse_loss(generated_image, self.recon_image.unsqueeze(0))

        return loss

    def training_step(self, batch, batch_idx):
        if self.auto_sample and self.step_counter % self.sample_every_n_steps == 0:
            self.sample_and_save_output(f"{self.logger.log_dir}/sample_{self.step_counter}.png",
                                        sample_size=self.sample_size)

        loss = self.forward(batch)
        self.log('train_loss', loss)
        self.step_counter += 1
        return loss

    def validation_step2(self, batch, batch_idx):
        samples = self.sample(image_size=batch.shape[-2:], batch_size=1)
        val_loss = get_swd_scores(batch, samples)
        self.log('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        # TODO SUPPORT SCHEDULER
        return optim
