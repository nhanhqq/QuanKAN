from __future__ import annotations

import torch


class RobustEEGAugment:
    def __init__(
        self,
        noise_std: float = 0.035,
        gain: float = 0.10,
        channel_drop: float = 0.05,
        time_mask: float = 0.10,
        band_drop_probability: float = 0.15,
        apply_probability: float = 0.85,
    ) -> None:
        self.noise_std = float(noise_std)
        self.gain = float(gain)
        self.channel_drop = float(channel_drop)
        self.time_mask = float(time_mask)
        self.band_drop_probability = float(band_drop_probability)
        self.apply_probability = float(apply_probability)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        if torch.rand(()) > self.apply_probability:
            return x
        valid_steps = x.abs().sum(dim=(1, 2)) > 0
        if not valid_steps.any():
            return x
        valid = valid_steps[:, None, None]
        scale = x[valid_steps].std(unbiased=False).clamp_min(1e-3)
        gain = 1.0 + (2.0 * torch.rand(()) - 1.0) * self.gain
        noise = torch.randn_like(x) * (self.noise_std * scale)
        x = torch.where(valid, x * gain + noise, x)

        if self.channel_drop > 0:
            keep_channels = torch.rand(x.size(1), device=x.device) >= self.channel_drop
            if not keep_channels.any():
                keep_channels[torch.randint(x.size(1), (1,))] = True
            x[:, ~keep_channels, :] = 0.0

        valid_length = int(valid_steps.sum().item())
        max_mask = round(valid_length * self.time_mask)
        if max_mask > 0:
            width = int(torch.randint(1, max_mask + 1, ()).item())
            start = int(torch.randint(0, max(valid_length - width + 1, 1), ()).item())
            x[start : start + width] = 0.0

        if x.size(2) > 1 and torch.rand(()) < self.band_drop_probability:
            band = int(torch.randint(x.size(2), ()).item())
            x[:, :, band] = 0.0
        return x
