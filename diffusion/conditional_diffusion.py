import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule

from common_utils.resize_right import resize
from diffusion.diffusion_utils import cosine_noise_schedule, save_diffusion_sample, to_torch


class ConditionalDiffusion(LightningModule):
    def __init__(self, model, channels=3, timesteps=1000,
                 initial_lr=2e-4, recon_loss_factor=0.0, recon_image=None, recon_condition_image=None,
                 auto_sample=True, sample_every_n_steps=1000):
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
            recon_condition_image (torch.tensor):
                The image used as input to the deterministic sampling as conditioning when calculating the recon loss.
            auto_sample (bool):
                Should the model perform sampling during training.
                If False, the following sampling parameters are ignored.
            sample_every_n_steps (int):
                The amount of global steps (step == training batch) after which the model is
                sampled from.
        """
        super().__init__()

        self.step_counter = 0  # Overall step counter used to sample every n global steps
        self.sample_every_n_steps = sample_every_n_steps
        self.auto_sample = auto_sample

        self.channels = channels
        self.model = model
        self.initial_lr = initial_lr

        self.recon_loss_factor = recon_loss_factor
        self.i = 0
        if self.recon_loss_factor > 0:
            assert recon_image is not None
            self.register_buffer('recon_image', (recon_image * 2) - 1)
            self.register_buffer('recon_condition_image', (recon_condition_image * 2) - 1)
            self.register_buffer('recon_noise', torch.randn_like(recon_image))

        betas = cosine_noise_schedule(timesteps)
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_hat = np.cumprod(alphas, axis=0)
        alphas_hat_prev = np.append(1., alphas_hat[:-1])
        self.sqrt_alphas_hat_prev = np.sqrt(np.append(1., alphas_hat))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
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
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_hat_prev[t + 1]]).repeat(batch_size, 1).view((batch_size,)).to(x.device)
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
    def sample(self, condition):
        img = torch.randn_like(condition)
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, i, condition_x=condition)
        return img

    @torch.no_grad()
    def sample_ddim(self, condition, x_T=None, sampling_step_size=100):
        """
        Sample from the model, using the DDIM sampling process.
        The DDIM implicit sampling process is determinstic, and will always generate the same output
        if given the same input.

        Args:
            condition (torch.tensor): The image used to condition the sampling process.
            x_T (torch.tensor): The initial noise to start the sampling process from. Can be None.
            sampling_step_size (int): The step size between each t in the sampling process. The higher this value is,
                                      the faster the sampling process (as well as lower image quality).
        """
        batch_size = condition.shape[0]
        seq = range(0, self.num_timesteps, sampling_step_size)
        seq_next = [-1] + list(seq[:-1])

        if x_T is None:
            x_t = torch.randn(condition.shape, device=self.device)
        else:
            x_t = x_T

        zipped_reversed_seq = list(zip(reversed(seq), reversed(seq_next)))[:-1]
        for t, t_next in zipped_reversed_seq:
            noise_level = torch.FloatTensor(
                [self.sqrt_alphas_hat_prev[t + 1]]).repeat(batch_size, 1).view((batch_size,)).to(x_t.device)

            e_t = self.model(torch.cat([condition, x_t], dim=1), noise_level)
            predicted_x0 = (x_t - self.sqrt_one_minus_alphas_hat[t] * e_t) / self.sqrt_alphas_hat[t]
            direction_to_x_t = self.sqrt_one_minus_alphas_hat[t_next] * e_t
            x_t = self.sqrt_alphas_hat[t_next] * predicted_x0 + direction_to_x_t

        t_tensor = torch.full((batch_size,), 0, dtype=torch.int64, device=self.device)
        e_t = self.model(torch.cat([condition, x_t], dim=1), t_tensor)
        x_0 = (x_t - self.sqrt_one_minus_alphas_hat[0] * e_t) / self.sqrt_alphas_hat[0]
        return x_0

    @torch.no_grad()
    def sample_ccg(self, sample_size, batch_size, window_size, stride, padding_size, use_ddim=False):
        """
        Generate a sample using the conditional-crop-generation method.
        This generates a crop from pure noise in one of the corners of the image and then performs
        an autoregressive "sliding-window" method to generate further crops that overlap the
        already-generated parts of the image.

        Args:
            sample_size (tuple(int)): The spatial dimensions of the sample during auto sampling.
            batch_size (int): The batch size to sample.
            window_size (tuple(int)):
            stride (tuple(int)):
            padding_size (tuple(int)): The spatial dimensions of the zero padding to add.
            use_ddim (bool): If True, DDIM is used for the sampling process (which generates faster samples).
                Else, DDPM sampling is used.
        """

        # Initialize image with frame
        sample_shape = (batch_size, self.channels, sample_size[0], sample_size[1])
        img = torch.randn(sample_shape).to(device=self.device)
        img = torchvision.transforms.Pad(padding_size)(img)

        # Move sliding window across image to conditionally generate image
        for h in range(0, img.shape[-2], stride[0]):
            for w in range(0, img.shape[-1], stride[1]):
                if use_ddim:
                    img[:, :, h: h + window_size[0], w: w + window_size[1]] = \
                        self.sample_ddim(condition=img[:, :, h: h + window_size[0], w: w + window_size[1]],
                                         sampling_step_size=50)
                else:
                    img[:, :, h: h + window_size[0], w: w + window_size[1]] = \
                        self.sample(condition=img[:, :, h: h + window_size[0], w: w + window_size[1]])

        return img[:, :, padding_size[0]:-padding_size[0], padding_size[1]:-padding_size[1]]

    def q_sample(self, x_start, continuous_sqrt_alpha_hat, noise=None):
        noise = noise if noise is not None else torch.randn_like(x_start)
        return continuous_sqrt_alpha_hat * x_start + (1 - continuous_sqrt_alpha_hat ** 2).sqrt() * noise

    def forward(self, x_in, noise=None):
        self.i += 1

        x_start = x_in['IMG']
        b = x_start.shape[0]
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_hat = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_hat_prev[t - 1],
                self.sqrt_alphas_hat_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_hat = continuous_sqrt_alpha_hat.view(
            b, -1)

        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start,
                                continuous_sqrt_alpha_hat=continuous_sqrt_alpha_hat.view(-1, 1, 1, 1),
                                noise=noise)

        x_recon = self.model(
            torch.cat([x_in['CONDITION_IMG'], x_noisy], dim=1), continuous_sqrt_alpha_hat.view(-1))

        loss = F.mse_loss(noise, x_recon)

        if self.recon_loss_factor > 0 and self.i % 5 == 0:
            # Add a reconstruction loss between the original image and the DDIM
            # sampling result of the constant reconstruction noise.
            generated_image = self.sample_ddim(condition=self.recon_condition_image,
                                               x_T=self.recon_noise,
                                               sampling_step_size=self.num_timesteps // 10)
            loss = loss + F.mse_loss(generated_image, self.recon_image.unsqueeze(0))

        return loss

    def training_step(self, batch, batch_idx):
        if self.auto_sample and self.step_counter % self.sample_every_n_steps == 0:
            self.sample_and_save_output(batch['CONDITION_IMG'], f"{self.logger.log_dir}/sample_{self.step_counter}.png")

        loss = self.forward(batch)
        self.log('train/loss', loss)
        self.step_counter += 1
        return loss

    @torch.no_grad()
    def sample_and_save_output(self, condition, output_path):
        """
        Sample a single image, normalize it, and save into an output file.

        Args:
            output_path (String): The path to save the image in.
            sample_size (tuple or int): The spatial dimensions of the image. If an int is passed, it is used for
                                        both spatial dimensions.
        """
        sample = self.sample(condition=condition)
        save_diffusion_sample(condition, f"{self.logger.log_dir}/sample_{self.step_counter}_conditioning.png")
        save_diffusion_sample(sample, output_path)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        return optim
