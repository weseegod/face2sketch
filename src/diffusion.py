"""
Phase 3: DDPM/DDIM Diffusion — noise scheduler, forward diffusion,
training loop, and sampling. REUSES the U-Net from Phase 1-2.

THIS FILE IS FOR PHASE 3 ONLY — not needed for Phase 1-2 (GAN).
The GAN training loop is in train.py. The GAN generator is in unet.py.

═══════════════════════════════════════════════════════════════
WHEN YOU GET HERE (after Phase 2):
═══════════════════════════════════════════════════════════════

Your U-Net from Phase 1 already does photo → sketch in one pass.
Diffusion works differently:

  GAN:         photo ──► U-Net ──► sketch  (1 forward pass, deterministic)
  
  Diffusion:   photo + random_noise ──► U-Net ──► predicted_noise
               Repeat 50-1000 times, each step denoising slightly.
               Result: sketch emerges gradually from noise.
               
The U-Net ARCHITECTURE is the same, but:
  - Input: 6 channels (photo concat noisy_sketch) instead of 3
  - Extra input: timestep t → sinusoidal embedding → injected at each level
  - GroupNorm instead of BatchNorm
  - SiLU instead of ReLU
  - Predicts NOISE ε, not the sketch directly

References:
  - DDPM (Ho et al. 2020) — the original diffusion paper
  - DDIM (Song et al. 2021) — fast deterministic sampling
  - Improved DDPM (Nichol & Dhariwal 2021) — cosine schedule, learned variance
  - ControlNet (Zhang et al. 2023) — conditional control for diffusion
  - The Annotated Diffusion Model (HuggingFace blog) — line-by-line code

═══════════════════════════════════════════════════════════════
TWO APPROACHES (pick one when you get here)
═══════════════════════════════════════════════════════════════

PATH A: DDPM from scratch (educational, needs more data)
  - Build NoiseScheduler, SinusoidalEmbedding, ConditionalUNet
  - Train on your 322+184 pairs (or augmented to ~10K)
  - Deep understanding of every component
  - ~20h training on L4, moderate quality

PATH B: Stable Diffusion + ControlNet (production, works with your data)
  - Load pretrained SD 1.5
  - Add ControlNet conditioned on your face photo
  - Train only ControlNet + LoRA (~20M params, ~4h)
  - Production quality, works with 184 pairs
  - Less "from scratch" learning, but this is how industry does it

═══════════════════════════════════════════════════════════════
MODULE OVERVIEW (Path A — from scratch)
═══════════════════════════════════════════════════════════════

Classes/Functions to implement:
  1. SinusoidalPositionEmbedding — timestep → sinusoidal vector → MLP
  2. NoiseScheduler              — β schedule, ᾱ_t precomputation
  3. ConditionalUNet             — modified Phase 1 UNet + time conditioning
  4. forward_diffuse             — x_t = √ᾱ_t * x₀ + √(1-ᾱ_t) * ε
  5. train_step                  — one training iteration
  6. sample_ddpm                 — ancestral sampling (slow, high quality)
  7. sample_ddim                 — deterministic sampling (fast, standard)
"""

# ═══════════════════════════════════════════════════════════════
# 1. NOISE SCHEDULER
# ═══════════════════════════════════════════════════════════════
#
# NoiseScheduler(T=1000, schedule="cosine"):
#   Manages β_t — how much noise is added at each timestep.
#
#   Precompute:
#     β_t    = variance schedule (linear or cosine)
#     α_t    = 1 - β_t
#     ᾱ_t    = cumprod(α_t)
#     √ᾱ_t   = sqrt(ᾱ_t)        ← scale for clean image
#     √(1-ᾱ) = sqrt(1 - ᾱ_t)    ← scale for noise
#
#   Cosine schedule (recommended):
#     ᾱ_t = f(t) / f(0)
#     f(t) = cos((t/T + s) / (1 + s) * π/2)²
#     s = 0.008 (small offset to prevent β_t near 0 at t=0)
#
#   Forward diffusion (O(1) — jump to any timestep):
#     x_t = √ᾱ_t * x₀ + √(1-ᾱ_t) * ε
#     This is the magic of DDPM — no iteration needed!


# ═══════════════════════════════════════════════════════════════
# 2. TIME EMBEDDING
# ═══════════════════════════════════════════════════════════════
#
# SinusoidalPositionEmbedding(dim=256):
#   Same as positional encoding in Transformers, but for timesteps.
#
#   PE(t, 2i)   = sin(t / 10000^(2i/dim))
#   PE(t, 2i+1) = cos(t / 10000^(2i/dim))
#
#   Followed by MLP: Linear(dim) → SiLU → Linear(dim) → SiLU
#   Output injected at each ResBlock via FiLM or simple addition.


# ═══════════════════════════════════════════════════════════════
# 3. TRAINING (conditional DDPM)
# ═══════════════════════════════════════════════════════════════
#
# For each batch (photo, sketch):
#   1. t ~ Uniform(0, T)                    # random timestep per image
#   2. ε ~ N(0, I)                          # random Gaussian noise
#   3. x_t = √ᾱ_t * sketch + √(1-ᾱ_t) * ε  # add noise to sketch
#   4. input = concat(photo, x_t)           # condition on photo
#   5. ε_pred = model(input, t)             # predict the noise
#   6. loss = MSE(ε_pred, ε)                # simple!
#
# Key insight: the condition (photo) NEVER changes during diffusion.
# Only the target (sketch) gets noised. The photo is always clean,
# concatenated as a constant guide.


# ═══════════════════════════════════════════════════════════════
# 4. SAMPLING (DDIM, 50 steps)
# ═══════════════════════════════════════════════════════════════
#
# Start: x_T ~ N(0, I)  (pure noise, same shape as sketch)
# Photo: constant condition throughout
#
# For t in [T, T-1, ..., 1] (50 evenly-spaced steps):
#   ε_pred = model(concat(photo, x_t), t)
#   x_{t-1} = denoise(x_t, ε_pred, t)
#
# After T steps: x_0 ≈ sketch
#
# DDIM is deterministic: same noise → same output.
# η=0 → deterministic, η=1 → stochastic (equivalent to DDPM).


# ═══════════════════════════════════════════════════════════════
# 5. DIFFERENCES FROM GAN (Phase 1-2):
# ═══════════════════════════════════════════════════════════════
#
# |                     | GAN (Phase 1-2)        | Diffusion (Phase 3) |
# |---------------------|------------------------|---------------------|
# | Inference steps     | 1 forward pass         | 50-1000 forward passes |
# | Inference speed     | ~10ms                  | ~1-40s              |
# | Training loss       | L1 + adversarial       | Simple MSE          |
# | Stability           | Tricky (adversarial)   | Very stable          |
# | Data requirement    | 100-1000 pairs         | 5K-50K+ pairs       |
# | Output diversity    | Deterministic (1→1)    | Stochastic (1→many) |
# | U-Net input         | 3 channels (photo)     | 6 channels (photo+noisy) |
# | U-Net extra input   | None                   | Time embedding       |
# | Normalization       | BatchNorm, ReLU        | GroupNorm, SiLU      |
