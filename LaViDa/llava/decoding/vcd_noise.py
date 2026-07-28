"""Official VCD (Leng et al., CVPR 2024) diffusion-style image noise.

Ported from https://github.com/DAMO-NLP-SG/VCD ``vcd_utils/vcd_add_noise.py``.
Applies the closed-form forward process q(x_t | x_0) at a fixed timestep.
"""
from __future__ import annotations

from typing import Union

import torch


def add_diffusion_noise(
    image_tensor: torch.Tensor,
    noise_step: int = 500,
    *,
    num_steps: int = 1000,
) -> torch.Tensor:
    """Return a noisy copy of ``image_tensor`` at diffusion step ``noise_step``.

    Args:
        image_tensor: Image tensor of any shape (noise is applied elementwise).
        noise_step: Timestep in ``[0, num_steps)``. Paper default is 500.
        num_steps: Total diffusion steps used to build the beta schedule.
    """
    if not 0 <= int(noise_step) < int(num_steps):
        raise ValueError(
            f"noise_step must be in [0, {num_steps}), got {noise_step}"
        )

    betas = torch.linspace(-6, 6, int(num_steps))
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    t = int(noise_step)
    noise = torch.randn_like(image_tensor)
    alphas_t = alphas_bar_sqrt[t].to(
        device=image_tensor.device, dtype=image_tensor.dtype
    )
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t].to(
        device=image_tensor.device, dtype=image_tensor.dtype
    )
    return alphas_t * image_tensor + alphas_1_m_t * noise


def noise_images(
    images: Union[torch.Tensor, list],
    noise_step: int = 500,
) -> Union[torch.Tensor, list]:
    """Apply ``add_diffusion_noise`` to a tensor or a list of image tensors."""
    if isinstance(images, torch.Tensor):
        return add_diffusion_noise(images, noise_step)
    if isinstance(images, (list, tuple)):
        return [noise_images(img, noise_step) for img in images]
    raise TypeError(f"Unsupported images type: {type(images)!r}")
