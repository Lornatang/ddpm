# Copyright (c) Lornatang. All rights reserved.
from collections import namedtuple
from typing import Any, Optional, Tuple, Union

import torch
from einops import rearrange, reduce
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F_torch

from ddpm.utils.ops import extract
from ddpm.utils.scheduler import linear_beta_schedule

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])

__all__ = [
    "GaussianDiffusion",
]


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            model: Any,
            image_size: Union[int, Tuple[int, int]],
            time_steps: int = 1000,
            sampling_time_steps: int = 250,
            ddim_sampling_eta: float = 0.,
            offset_noise_strength: float = 0.,
            min_snr_loss_weight: bool = False,
            min_snr_gamma: int = 5,
            immiscible: bool = False
    ) -> None:
        """Initializes the GaussianDiffusion model.

        Args:
            model (Any): The neural network model used for diffusion.
            image_size (Union[int, Tuple[int, int]]): The size of the input images.
            time_steps (int, optional): The number of time steps for the diffusion process. Defaults to 1000.
            sampling_time_steps (int, optional): The number of time steps for sampling. Defaults to 250.
            ddim_sampling_eta (float, optional): The eta value for ddim sampling. Defaults to 0.
            offset_noise_strength (float, optional): The strength of the offset noise. Defaults to 0.
            min_snr_loss_weight (bool, optional): Whether to use minimum SNR loss weight. Defaults to False.
            min_snr_gamma (int, optional): The gamma value for minimum SNR. Defaults to 5.
            immiscible (bool, optional): Whether to use immiscible diffusion. Defaults to False.
        """
        super().__init__()
        self.model = model
        self.in_channels = self.model.in_channels

        # Ensure image_size is a tuple of two integers
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(image_size) == 2, "image size must be an integer or a tuple/list of two integers"
        self.image_size = image_size

        # Generate beta schedule using sigmoid function
        betas = linear_beta_schedule(time_steps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = F_torch.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        time_steps, = betas.shape
        self.num_time_steps = int(time_steps)

        # Sampling related parameters
        self.sampling_time_steps = time_steps if sampling_time_steps is None else sampling_time_steps
        assert self.sampling_time_steps <= time_steps, "sampling time steps must be less than or equal to total time steps"
        self.is_ddim_sampling = self.sampling_time_steps < time_steps
        self.ddim_sampling_eta = ddim_sampling_eta

        # Helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1. - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1. / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1. / alphas_cumprod - 1))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer("posterior_variance", posterior_variance)
        register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer("posterior_mean_coef2", (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Immiscible diffusion
        self.immiscible = immiscible

        # Offset noise strength
        self.offset_noise_strength = offset_noise_strength

        # Derive loss weight
        snr = alphas_cumprod / (1 - alphas_cumprod)
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)
        register_buffer("loss_weight", maybe_clipped_snr / (snr + 1))

    def forward(self, image: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass for the Gaussian Diffusion model.

        Args:
            image (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The computed loss for the input batch of images.
        """
        # Get the batch size, number of channels, height, and width of the input image
        batch_size, channels, height, width = image.shape

        # Ensure the height and width of the input image match the expected image size
        assert height == self.image_size[0] and width == self.image_size[1], f"height and width of image must be {self.image_size}"

        # Randomly sample time steps for each image in the batch
        sample_time = torch.randint(0, self.num_time_steps, (batch_size,), device=image.device).long()

        # Normalize the output to the range [-1, 1]
        image = (image * 2.) - 1.

        # Compute the loss for the input batch of images
        out = self.p_losses(image, sample_time, *args, **kwargs)

        return out

    def q_posterior(
            self,
            x_start: torch.Tensor,
            x_time_step: torch.Tensor,
            time_step: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the posterior mean and variance for the diffusion process.

        Args:
            x_start (torch.Tensor): The starting image tensor.
            x_time_step (torch.Tensor): The noisy image tensor at time step `t`.
            time_step (torch.Tensor): The time steps tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The posterior mean, variance, and log variance clipped tensors.
        """
        posterior_mean = (
                extract(self.posterior_mean_coef1, time_step, x_time_step.shape) * x_start +
                extract(self.posterior_mean_coef2, time_step, x_time_step.shape) * x_time_step
        )
        posterior_variance = extract(self.posterior_variance, time_step, x_time_step.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, time_step, x_time_step.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self,
            x: torch.Tensor,
            time_step: torch.Tensor,
            clip_denoised: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the mean and variance for the diffusion process.

        Args:
            x (torch.Tensor): The input tensor.
            time_step (torch.Tensor): The time steps tensor.
            clip_denoised (bool, optional): Whether to clip the predicted x_start. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The model mean, posterior variance,
            posterior log variance, and predicted x_start tensors.
        """
        preds = self.model_predictions(x, time_step)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_time_step=x, time_step=time_step)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_sample(
            self,
            x: torch.Tensor,
            time_step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from the Gaussian diffusion process at a given time step.

        Args:
            x (torch.Tensor): The input tensor.
            time_step (int): The current time step.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The predicted image tensor and the predicted starting image tensor.
        """
        batched_times = torch.full((x.shape[0],), time_step, device=self.betas.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x, batched_times, clip_denoised=True)
        noise = torch.randn_like(x) if time_step > 0 else 0.  # no noise if t == 0
        pred_image = model_mean + (0.5 * model_log_variance).exp() * noise

        return pred_image, x_start

    def p_sample_loop(self, shape: Tuple[int, ...], return_all_time_steps: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Perform the sampling loop for the Gaussian diffusion process.

        Args:
            shape (Tuple[int, ...]): The shape of the tensor to be sampled.
            return_all_time_steps (bool, optional): Whether to return the samples from all time steps. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, ...]]: The final sampled tensor, or a tuple of tensors from all time steps if `return_all_time_steps` is True.
        """
        # Initialize the image with random noise
        image = torch.randn(shape, device=self.betas.device)
        images = [image]

        # Initialize x_start to None
        x_start = None

        # Sampling loop over all time steps in reverse order
        for time_step in reversed(range(0, self.num_time_steps)):
            # Perform a single step of sampling
            image, x_start = self.p_sample(image, time_step)

            # Append the current image to the list of images
            images.append(image)

        # Return the final image or all images depending on the flag
        out = image if not return_all_time_steps else torch.stack(images, dim=1)

        # Normalize the output to the range [0, 1]
        out = (out + 1.) * 0.5

        return out

    @torch.inference_mode()
    def ddim_sample(self, shape: Tuple[int, ...], return_all_time_steps: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Perform the sampling loop using the Denoising Diffusion Implicit Model (DDIM) algorithm.

        Args:
            shape (Tuple[int, ...]): The shape of the tensor to be sampled.
            return_all_time_steps (bool, optional): Whether to return the samples from all time steps. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, ...]]: The final sampled tensor, or a tuple of tensors from all time steps if `return_all_time_steps` is True.
        """
        # Define the time steps for DDIM sampling
        time_step = torch.linspace(-1, self.num_time_steps - 1, steps=self.sampling_time_steps + 1)
        time_step = list(reversed(time_step.int().tolist()))
        time_pairs = list(zip(time_step[:-1], time_step[1:]))

        # Initialize the image with random noise
        image = torch.randn(shape, device=self.betas.device)
        images = [image]

        # Initialize x_start to None
        x_start = None

        # Sampling loop over all time pairs
        for current_time, next_time in time_pairs:
            # Create a tensor with the current time step for all samples
            time_step = torch.full((shape[0],), current_time, device=self.betas.device, dtype=torch.long)
            # Predict the noise and reconstruct x_start
            pred_noise, x_start, *_ = self.model_predictions(image, time_step, clip_x_start=True)

            # If the next time step is the final step, set the image to x_start
            if next_time < 0:
                image = x_start
                images.append(image)
                continue

            # Calculate the alpha values for the current and next time steps
            alpha = self.alphas_cumprod[current_time]
            alpha_next = self.alphas_cumprod[next_time]

            # Calculate the sigma value for DDIM sampling
            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            # Generate random noise
            noise = torch.randn_like(image)

            # Update the image using the DDIM sampling formula
            image = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            # Append the current image to the list of images
            images.append(image)

        # Return the final image or all images depending on the flag
        out = image if not return_all_time_steps else torch.stack(images, dim=1)
        # Unnormalize the output
        out = (out + 1) * 0.5

        return out

    def sample(self, batch_size: int = 16, return_all_time_steps: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Generate samples from the Gaussian diffusion process.

        Args:
            batch_size (int, optional): The number of samples to generate. Defaults to 16.
            return_all_time_steps (bool, optional): Whether to return the samples from all time steps. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, ...]]: The final sampled tensor, or a tuple of tensors from all time steps if `return_all_time_steps` is True.
        """
        (height, width), channels = self.image_size, self.in_channels
        sample_func = self.ddim_sample if self.is_ddim_sampling else self.p_sample_loop
        sample = sample_func((batch_size, channels, height, width), return_all_time_steps)

        return sample

    @staticmethod
    def noise_assignment(x_start: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Assign noise to the starting image tensor using the linear sum assignment method.

        Args:
            x_start (torch.Tensor): The starting image tensor.
            noise (torch.Tensor): The noise tensor.

        Returns:
            torch.Tensor: The assigned noise tensor.
        """
        x_start, noise = tuple(rearrange(t, "b ... -> b (...)") for t in (x_start, noise))
        dist = torch.cdist(x_start, noise)
        _, assign = linear_sum_assignment(dist.cpu())
        out = torch.from_numpy(assign).to(dist.device)

        return out

    def q_sample(self, x_start: torch.Tensor, time_step: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from the Gaussian diffusion process at a given time step.

        Args:
            x_start (torch.Tensor): The starting image tensor.
            time_step (torch.Tensor): The time steps tensor.
            noise (Optional[torch.Tensor], optional): The noise tensor. Defaults to None.

        Returns:
            torch.Tensor: The noisy image tensor at time step `t`.
        """
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)

        if self.immiscible:
            assign = self.noise_assignment(x_start, noise)
            noise = noise[assign]

        return (
                extract(self.sqrt_alphas_cumprod, time_step, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, time_step, x_start.shape) * noise
        )

    def prediction_latent_var(self, x_start: torch.Tensor, latent_var: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Prediction the latent variable from the starting image tensor and noise.

        Args:
            x_start (torch.Tensor): The starting image tensor.
            latent_var (torch.Tensor): The latent variable tensor.
            noise (torch.Tensor): The noise tensor.

        Returns:
            torch.Tensor: The predicted latent variable tensor.
        """
        return (
                extract(self.sqrt_alphas_cumprod, latent_var, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, latent_var, x_start.shape) * x_start
        )

    def prediction_start_from_latent_var(self, x_time_step: torch.Tensor, time_step: torch.Tensor, latent_var: torch.Tensor) -> torch.Tensor:
        """Prediction the starting image tensor from the latent variable `v`.

        Args:
            x_time_step (torch.Tensor): The noisy image tensor at time step `t`.
            time_step (torch.Tensor): The time steps tensor.
            latent_var (torch.Tensor): The latent variable tensor.

        Returns:
            torch.Tensor: The predicted starting image tensor.
        """
        return (
                extract(self.sqrt_alphas_cumprod, time_step, x_time_step.shape) * x_time_step -
                extract(self.sqrt_one_minus_alphas_cumprod, time_step, x_time_step.shape) * latent_var
        )

    def prediction_noise_from_start(self, x_time_step: torch.Tensor, time_step: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Prediction the noise from the starting image tensor.

        Args:
            x_time_step (torch.Tensor): The noisy image tensor at time step `t`.
            time_step (torch.Tensor): The time steps tensor.
            x0 (torch.Tensor): The starting image tensor.

        Returns:
            torch.Tensor: The predicted noise tensor.
        """
        return (
                (extract(self.sqrt_recip_alphas_cumprod, time_step, x_time_step.shape) * x_time_step - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, time_step, x_time_step.shape)
        )

    def model_predictions(
            self,
            x: torch.Tensor,
            time_step: torch.Tensor,
            clip_x_start: bool = False
    ) -> ModelPrediction:
        """Generate model predictions for the Gaussian Diffusion process.

        Args:
            x (torch.Tensor): The input tensor.
            time_step (torch.Tensor): The time steps tensor.
            clip_x_start (bool, optional): Whether to clip the predicted x_start. Defaults to False.

        Returns:
            ModelPrediction: The predicted noise and x_start tensors.
        """
        model_output = self.model(x, time_step)

        x_start = self.prediction_start_from_latent_var(x, time_step, model_output)
        if clip_x_start:
            x_start.clamp_(-1., 1.)

        pred_noise = self.prediction_noise_from_start(x, time_step, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_losses(
            self,
            x_start: torch.Tensor,
            time_step: torch.Tensor,
            noise: Optional[torch.Tensor] = None,
            offset_noise_strength: Optional[float] = None
    ) -> torch.Tensor:
        """Compute the loss for the Gaussian Diffusion model.

        Args:
            x_start (torch.Tensor): The starting image tensor.
            time_step (torch.Tensor): The time steps tensor.
            noise (Optional[torch.Tensor], optional): The noise tensor. Defaults to None.
            offset_noise_strength (Optional[float], optional): The strength of the offset noise. Defaults to None.

        Returns:
            torch.Tensor: The computed loss.
        """
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)

        offset_noise_strength = self.offset_noise_strength if offset_noise_strength is None else offset_noise_strength
        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device=self.betas.device)
            noise += offset_noise_strength * rearrange(offset_noise, "b c -> b c 1 1")

        # Noise sample
        x_noisy = self.q_sample(x_start=x_start, time_step=time_step, noise=noise)

        # Predict and take gradient step
        model_output = self.model(x_noisy, time_step)
        predicted_v = self.prediction_latent_var(x_start, time_step, noise)

        loss = F_torch.mse_loss(model_output, predicted_v, reduction="none")
        loss = reduce(loss, "b ... -> b", "mean")
        loss_weight = extract(self.loss_weight, time_step, loss.shape)
        loss = (loss * loss_weight).mean()

        return loss

    @torch.inference_mode()
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, time_step: Optional[int] = None, lambda_ratio: float = 0.5) -> torch.Tensor:
        """Interpolate between two images using the diffusion process.

        Args:
            x1 (torch.Tensor): The first image tensor.
            x2 (torch.Tensor): The second image tensor.
            time_step (Optional[int], optional): The time step to start the interpolation. Defaults to the last time step.
            lambda_ratio (float, optional): The interpolation ratio between x1 and x2. Defaults to 0.5.

        Returns:
            torch.Tensor: The interpolated image tensor.
        """
        assert x1.shape == x2.shape, "Input images must have the same shape"

        # Set the time step to the last time step if not provided
        time_step = self.num_timesteps - 1 if time_step is None else time_step

        # Create a tensor filled with the time step value
        t_batched = torch.full((x1.shape[0],), time_step, device=x1.device)

        # Sample noisy versions of x1 and x2 at the given time step
        xt1, xt2 = map(lambda x: self.q_sample(x, t_batched), (x1, x2))

        # Interpolate between the noisy versions of x1 and x2
        image = (1 - lambda_ratio) * xt1 + lambda_ratio * xt2

        x_start = None
        # Reverse diffusion process to generate the interpolated image
        for i in reversed(range(0, time_step)):
            image, x_start = self.p_sample(image, i)

        return image
