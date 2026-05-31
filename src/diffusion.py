"""
Phase 3: DDPM/DDIM Conditional Diffusion — from scratch in raw PyTorch.

Components:
  - SinusoidalPositionEmbedding: timestep → sinusoidal → MLP → time_features
  - NoiseScheduler: cosine β schedule, ᾱ precompute, forward diffuse, DDPM/DDIM sampling
  - ResBlock + TimeEmbed: GroupNorm → SiLU → Conv with time modulation
  - ConditionalUNet: U-Net that predicts noise given (photo, noisy_sketch, t)

Training:
  For each (photo, sketch) pair:
    1. t ~ Uniform(0, T)
    2. ε ~ N(0, I)
    3. x_t = √ᾱ_t * sketch + √(1-ᾱ_t) * ε
    4. ε_pred = model(concat(photo, x_t), t)
    5. loss = MSE(ε_pred, ε)

Inference: DDIM, 50 steps, deterministic.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# SINUSOIDAL TIME EMBEDDING
# ═══════════════════════════════════════════════════════════════

class SinusoidalPositionEmbedding(nn.Module):
    """Transformer-style sinusoidal positional encoding for diffusion timesteps.

    PE(t, 2i)   = sin(t / 10000^(2i/dim))
    PE(t, 2i+1) = cos(t / 10000^(2i/dim))

    Followed by MLP: Linear(dim) → SiLU → Linear(dim) → SiLU → Linear(dim)
    """

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) or (B, 1) integer timesteps.

        Returns: (B, dim) embedded time features.
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B, 1)
        t = t.float()

        half_dim = self.dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t * emb.unsqueeze(0)               # (B, half_dim)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))             # pad to odd dim

        return self.mlp(emb)


# ═══════════════════════════════════════════════════════════════
# NOISE SCHEDULER
# ═══════════════════════════════════════════════════════════════

class NoiseScheduler:
    """DDPM noise schedule with cosine β schedule (Nichol & Dhariwal 2021).

    Precomputes:
      β_t, α_t = 1 - β_t, ᾱ_t = cumprod(α_t)
      √ᾱ_t, √(1 - ᾱ_t), √(1 / α_t), √ᾱ_{t-1}

    Supports forward diffuse (jump to any t in O(1)) and
    DDPM/DDIM reverse sampling steps.
    """

    def __init__(self, T: int = 1000, schedule: str = "cosine", s: float = 0.008):
        self.T = T
        self.schedule = schedule
        self.s = s

        # ── β schedule ──
        if schedule == "cosine":
            betas = self._cosine_beta_schedule(T, s)
        elif schedule == "linear":
            betas = self._linear_beta_schedule(T)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # ── Register as buffers ──
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # Precomputed coefficients for forward/reverse
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

        # For DDPM reverse: x_{t-1} = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_pred) + σ_t * z
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.coeff_beta_noise = betas / self.sqrt_one_minus_alpha_bars

        # Posterior variance (for DDPM sampling, optional)
        self.posterior_variance = betas * (1.0 - self.alpha_bars.roll(1)) / (1.0 - self.alpha_bars)
        self.posterior_variance[0] = betas[0]  # first step: no prior alpha_bar

    @staticmethod
    def _linear_beta_schedule(T: int) -> torch.Tensor:
        """Linear schedule: β_t from 1e-4 to 0.02 (DDPM paper)."""
        return torch.linspace(1e-4, 0.02, T)

    @staticmethod
    def _cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule (Nichol & Dhariwal 2021).

        ᾱ_t = f(t) / f(0) where f(t) = cos((t/T + s)/(1+s) * π/2)²
        β_t = 1 - ᾱ_t / ᾱ_{t-1}, clamped to ≤ 0.999
        """
        steps = torch.arange(T + 1, dtype=torch.float32)
        x = (steps / T + s) / (1 + s) * math.pi / 2
        alpha_bars = torch.cos(x) ** 2
        alpha_bars = alpha_bars / alpha_bars[0]  # normalize so ᾱ_0 = 1

        betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
        betas = torch.clamp(betas, max=0.999)
        return betas

    def to(self, device: torch.device):
        """Move all precomputed tensors to device."""
        for name in ['betas', 'alphas', 'alpha_bars',
                      'sqrt_alpha_bars', 'sqrt_one_minus_alpha_bars',
                      'sqrt_recip_alphas', 'coeff_beta_noise',
                      'posterior_variance']:
            tensor = getattr(self, name)
            setattr(self, name, tensor.to(device))
        return self

    def forward_diffuse(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: x_t = √ᾱ_t * x₀ + √(1-ᾱ_t) * ε.

        Args:
            x0: clean image, shape (B, C, H, W)
            t: timesteps, shape (B,) or (B, 1), int in [0, T-1]
            noise: optional pre-generated noise; if None, sample N(0, I)

        Returns:
            x_t: noised image at timestep t
            noise: the noise that was added (for loss computation)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.flatten()

        # Index precomputed values for each timestep in batch
        # Expand to (B, 1, 1, 1) for broadcasting
        a = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)

        x_t = a * x0 + b * noise
        return x_t, noise

    def ddim_step(self, model, x_t: torch.Tensor, condition: torch.Tensor,
                   t: int, t_prev: int, eta: float = 0.0) -> torch.Tensor:
        """Single DDIM reverse step.

        Args:
            model: noise predictor ε_θ(concat(condition, x_t), t)
            x_t: current noisy image (B, C, H, W)
            condition: photo (B, 3, H, W) — always clean
            t: current timestep (int, same for whole batch)
            t_prev: previous timestep (int, smaller)
            eta: 0 = deterministic DDIM, 1 = stochastic (DDPM-like)

        Returns:
            x_{t_prev}: slightly denoised image
        """
        # Predict noise
        t_batch = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        model_input = torch.cat([condition, x_t], dim=1)
        eps_pred = model(model_input, t_batch)

        # ᾱ_t and ᾱ_{t-1}
        alpha_bar_t = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)

        # Predicted x0
        x0_pred = (x_t - self.sqrt_one_minus_alpha_bars[t] * eps_pred) / self.sqrt_alpha_bars[t]

        # Direction pointing to x_t
        dir_xt = self.sqrt_one_minus_alpha_bars[t_prev] * eps_pred

        # Optional noise for stochastic sampling
        sigma = 0.0
        if eta > 0:
            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) *
                (1 - alpha_bar_t / alpha_bar_prev)
            )
            noise = torch.randn_like(x_t)
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps_pred

        x_prev = self.sqrt_alpha_bars[t_prev] * x0_pred + dir_xt

        if eta > 0:
            x_prev = x_prev + sigma * noise

        return x_prev

    @torch.no_grad()
    def sample_ddim(self, model, condition: torch.Tensor,
                     num_steps: int = 50, eta: float = 0.0) -> torch.Tensor:
        """DDIM sampling: start from pure noise, denoise in num_steps.

        Args:
            model: noise predictor
            condition: photo (B, 3, H, W)
            num_steps: number of DDIM steps (fewer = faster, 50 is standard)
            eta: 0 = deterministic, 1 = stochastic

        Returns:
            Generated sketch (B, 3, H, W) in pixel space [-1, 1]
        """
        model.eval()
        B, _, H, W = condition.shape
        device = condition.device

        # Start from pure noise
        x_t = torch.randn(B, 3, H, W, device=device)

        # Uniformly spaced timesteps (descending)
        step_indices = torch.linspace(self.T - 1, 0, num_steps, device=device).long()

        for i in range(num_steps):
            t = step_indices[i].item()
            t_prev = step_indices[i + 1].item() if i + 1 < num_steps else -1
            x_t = self.ddim_step(model, x_t, condition, t, t_prev, eta=eta)

        return x_t

    @torch.no_grad()
    def sample_ddpm(self, model, condition: torch.Tensor, eta: float = 1.0) -> torch.Tensor:
        """DDPM sampling: full T reverse steps. Slow but highest quality."""
        return self.sample_ddim(model, condition, num_steps=self.T, eta=eta)


# ═══════════════════════════════════════════════════════════════
# RESIDUAL BLOCK WITH TIME EMBEDDING
# ═══════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """Diffusion residual block: GroupNorm → SiLU → Conv2d with time modulation.

    Time embedding is injected via FiLM-style scale + shift after each GroupNorm:
      h = GroupNorm(h)
      h = h * (1 + scale) + shift   ← time modulation
      h = SiLU(h)
      h = Conv2d(h)
    """

    def __init__(self, in_channels: int, out_channels: int, time_dim: int,
                 groups: int = 32, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(min(groups, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Time embedding → scale + shift for each norm
        # norm1 sees in_channels, norm2 sees out_channels
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels * 2 + in_channels * 2)
        )

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Residual connection: 1x1 conv if channel size changes
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W), t_emb: (B, time_dim)"""
        # ── Time modulation params ──
        in_ch = self.in_channels
        out_ch = self.out_channels
        t_out = self.time_mlp(t_emb)  # (B, 2*in_ch + 2*out_ch)
        scale1, shift1, scale2, shift2 = t_out.split([in_ch, in_ch, out_ch, out_ch], dim=1)
        scale1 = scale1.unsqueeze(-1).unsqueeze(-1)
        shift1 = shift1.unsqueeze(-1).unsqueeze(-1)
        scale2 = scale2.unsqueeze(-1).unsqueeze(-1)
        shift2 = shift2.unsqueeze(-1).unsqueeze(-1)

        # ── Block 1 ──
        h = self.norm1(x)
        h = h * (1.0 + scale1) + shift1
        h = F.silu(h)
        h = self.conv1(h)

        # ── Block 2 ──
        h = self.norm2(h)
        h = h * (1.0 + scale2) + shift2
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


# ═══════════════════════════════════════════════════════════════
# CONDITIONAL U-NET (Noise Predictor)
# ═══════════════════════════════════════════════════════════════

class ConditionalUNet(nn.Module):
    """U-Net for diffusion noise prediction.

    Input: concat(photo, noisy_sketch) → 6 channels
    Extra: timestep t → sinusoidal embedding → injected at every ResBlock
    Output: predicted noise → 3 channels

    Architecture: same encoder-decoder structure as Phase 1 UNet,
    but with GroupNorm, SiLU, and time conditioning.

    Channel progression (base_ch=64):
        Encoder: 64 → 128 → 256 → 512 → 512 → 512 (bottleneck)
        Decoder: 512 → 512 → 256 → 128 → 64 → 3 (output)
    """

    def __init__(self, in_channels: int = 6, out_channels: int = 3,
                 base_ch: int = 64, num_levels: int = 5,
                 time_dim: int = 256, groups: int = 32, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_ch = base_ch
        self.num_levels = num_levels
        self.time_dim = time_dim

        # ── Time embedding ──
        self.time_embed = SinusoidalPositionEmbedding(dim=time_dim)

        # ── Initial conv (no normalization yet) ──
        self.initial = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)

        # ── Encoder ──
        enc_chs = []
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = base_ch
        for i in range(num_levels):
            multiplier = min(2 ** i, 8)
            out_ch = base_ch * multiplier
            enc_chs.append(out_ch)
            self.enc_blocks.append(
                ResBlock(in_ch, out_ch, time_dim, groups, dropout)
            )
            self.pools.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
            in_ch = out_ch

        # ── Bottleneck ──
        bottleneck_ch = in_ch  # matches last encoder output
        self.bottleneck1 = ResBlock(bottleneck_ch, bottleneck_ch, time_dim, groups, dropout)
        self.bottleneck2 = ResBlock(bottleneck_ch, bottleneck_ch, time_dim, groups, dropout)

        # ── Decoder ──
        self.dec_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in range(num_levels - 1, -1, -1):
            dec_out = enc_chs[i]
            # Upsample: convert in_ch → dec_out, then 2× upsample
            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(in_ch, dec_out, kernel_size=3, padding=1),
                )
            )
            # After concat with skip: dec_out + dec_out = 2*dec_out
            self.dec_blocks.append(
                ResBlock(dec_out * 2, dec_out, time_dim, groups, dropout)
            )
            in_ch = dec_out

        # ── Output conv ──
        self.output_norm = nn.GroupNorm(min(groups, base_ch), base_ch)
        self.output_conv = nn.Conv2d(base_ch, out_channels, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # Zero-initialize the last conv of output for stability
        nn.init.constant_(self.output_conv.weight, 0.0)
        nn.init.constant_(self.output_conv.bias, 0.0)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (B, 6, H, W) = concat(photo, noisy_sketch), t: (B,) timesteps"""
        # ── Time embedding ──
        t_emb = self.time_embed(t)     # (B, time_dim)

        # ── Initial conv ──
        h = self.initial(x)            # (B, base_ch, H, W)

        # ── Encoder ──
        skips = []
        for i in range(self.num_levels):
            h = self.enc_blocks[i](h, t_emb)
            skips.append(h)
            h = self.pools[i](h)       # stride-2 conv for downsampling

        # ── Bottleneck ──
        h = self.bottleneck1(h, t_emb)
        h = self.bottleneck2(h, t_emb)

        # ── Decoder ──
        for i in range(self.num_levels):
            skip = skips[self.num_levels - 1 - i]
            h = self.ups[i](h)
            h = torch.cat([h, skip], dim=1)
            h = self.dec_blocks[i](h, t_emb)

        # ── Output ──
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)
        return h


# ═══════════════════════════════════════════════════════════════
# DIFFUSION WRAPPER — training + sampling convenience
# ═══════════════════════════════════════════════════════════════

class DiffusionModel:
    """Convenience wrapper bundling scheduler + model.

    Usage:
        model = DiffusionModel(T=1000, device='cuda')
        loss = model.training_loss(photo, sketch)       # train step
        sketch = model.sample(photo, num_steps=50)      # inference
    """

    def __init__(self, T: int = 1000, schedule: str = "cosine",
                 base_ch: int = 64, time_dim: int = 256,
                 device: str = "cpu"):
        self.T = T
        self.device = torch.device(device)

        self.scheduler = NoiseScheduler(T=T, schedule=schedule)
        self.scheduler.to(self.device)

        self.model = ConditionalUNet(
            in_channels=6, out_channels=3,
            base_ch=base_ch, time_dim=time_dim,
        ).to(self.device)

        self.time_embed = self.model.time_embed

    def forward_diffuse(self, x0: torch.Tensor, t: torch.Tensor,
                        noise: Optional[torch.Tensor] = None):
        return self.scheduler.forward_diffuse(x0, t, noise)

    def training_loss(self, photo: torch.Tensor, sketch: torch.Tensor) -> torch.Tensor:
        """Single training step loss.

        Args:
            photo: (B, 3, H, W) clean photo
            sketch: (B, 3, H, W) clean sketch (target)

        Returns:
            MSE loss between predicted and actual noise
        """
        B = photo.shape[0]

        # Random timesteps
        t = torch.randint(0, self.T, (B,), device=photo.device)

        # Forward diffuse: add noise to sketch
        x_t, noise = self.scheduler.forward_diffuse(sketch, t)

        # Predict noise
        model_input = torch.cat([photo, x_t], dim=1)
        eps_pred = self.model(model_input, t)

        # Simple MSE
        loss = F.mse_loss(eps_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, photo: torch.Tensor, num_steps: int = 50,
               eta: float = 0.0) -> torch.Tensor:
        """Generate sketch from photo using DDIM sampling."""
        return self.scheduler.sample_ddim(
            self.model, photo, num_steps=num_steps, eta=eta
        )
