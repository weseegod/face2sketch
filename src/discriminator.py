"""
Phase 1: PatchGAN Discriminator for pix2pix GAN.

Architecture: Fully-convolutional discriminator that classifies
70×70 image patches as real or fake, rather than the whole image.

References:
  - pix2pix (Isola et al. 2017) — introduced PatchGAN
  - LSGAN (Mao et al. 2017) — alternative: least-squares loss
  - pix2pixHD (Wang et al. 2018) — multi-scale discriminator extension

═══════════════════════════════════════════════════════════════
WHY PATCHGAN?
═══════════════════════════════════════════════════════════════

A standard discriminator outputs ONE number per image: P(real).
  → Easy to fool, focuses on global structure, ignores local details

PatchGAN outputs an N×N grid of predictions — each cell judges
whether a 70×70 receptor field in the input is real.
  → Forces the discriminator to evaluate texture, sharpness, local consistency
  → 900 "critics" per image instead of 1
  → Much better at detecting GAN artifacts (blurry patches, noise patterns)

The discriminator is ONLY used during training. At inference, you
only need the generator (single forward pass, O(1)).

═══════════════════════════════════════════════════════════════
MODULE OVERVIEW
═══════════════════════════════════════════════════════════════

Classes to implement:
  1. DiscriminatorBlock — Conv2d → BatchNorm → LeakyReLU (building block)
  2. PatchGANDiscriminator — stack of blocks ending in N×N prediction map
"""

# ═══════════════════════════════════════════════════════════════
# IMPORTS (will need these)
# ═══════════════════════════════════════════════════════════════
# import torch
# import torch.nn as nn


# ═══════════════════════════════════════════════════════════════
# 1. DISCRIMINATOR BLOCK
# ═══════════════════════════════════════════════════════════════
#
# DiscriminatorBlock(in_channels, out_channels, stride, use_batchnorm):
#   One layer of the PatchGAN discriminator.
#
#   Forward pass:
#     x → Conv2d(in_ch, out_ch, kernel=4, stride, padding=1)
#       → [BatchNorm2d(out_ch) if use_batchnorm]
#       → LeakyReLU(negative_slope=0.2, inplace=True)
#
#   Key design choices:
#     - kernel=4, stride with padding=1: standard for PatchGAN
#     - LeakyReLU(0.2): prevents dead neurons when D is winning
#       (ReLU would output 0 for negative activations → gradient dies)
#     - No BatchNorm on the FIRST layer (use_batchnorm=False there)
#       Standard GAN practice — noise in first BN hurts feature learning
#     - BatchNorm on all other layers for training stability


# ═══════════════════════════════════════════════════════════════
# 2. PATCHGAN DISCRIMINATOR
# ═══════════════════════════════════════════════════════════════
#
# PatchGANDiscriminator(in_channels=6, ndf=64, n_layers=3):
#   Input: concatenated(photo, sketch_or_fake) → 6 channels
#   Output: N×N patch prediction map
#
#   Full architecture (with n_layers=3, ndf=64, input=6×256×256):
#
#     Input: (B, 6, 256, 256)
#
#     Layer 0: Conv(6→64, k4, s2, p1)     → (B, 64, 128, 128)
#              LeakyReLU(0.2)              (no BatchNorm on first layer!)
#
#     Layer 1: Conv(64→128, k4, s2, p1)   → (B, 128, 64, 64)
#              BatchNorm → LeakyReLU(0.2)
#
#     Layer 2: Conv(128→256, k4, s2, p1)  → (B, 256, 32, 32)
#              BatchNorm → LeakyReLU(0.2)
#
#     Layer 3: Conv(256→512, k4, s1, p1)  → (B, 512, 31, 31)
#              BatchNorm → LeakyReLU(0.2)   (stride=1 — stop downsampling)
#
#     Output:  Conv(512→1, k4, s1, p1)    → (B, 1, 30, 30)
#              Sigmoid
#
#     Each output cell corresponds to a 70×70 patch in the input image.
#
#   Why n_layers=3?
#     Each stride-2 conv halves the spatial size. After 3 of them:
#       256 → 128 → 64 → 32
#     With k4 convs (receptive field grows by 3 per layer over 2× downsampling),
#     the receptive field at the output layer is ~70×70.
#     This 70×70 patch size was found empirically optimal:
#       - Small enough to detect local artifacts (blurry patches)
#       - Large enough to see facial features (eye, nose, mouth)
#
#   Channel progression:
#     ndf → ndf*2 → ndf*4 → ndf*8 → 1
#     64  → 128   → 256   → 512   → 1
#
#   Total params: ~2.8M (very lightweight — 15× smaller than generator)
#
#   The discriminator processes CONCATENATED (photo, sketch) pairs:
#     in_channels = 6 (3 for photo + 3 for sketch)
#     This way, the discriminator sees BOTH the input condition AND the
#     output, and judges whether they belong together (real pair) or not
#     (generated sketch with this photo). This is called "conditional GAN"
#     — the discriminator conditions its judgment on the input photo.


# ═══════════════════════════════════════════════════════════════
# 3. LOSS COMPUTATION (used in training loop, not inside discriminator)
# ═══════════════════════════════════════════════════════════════
#
# Discriminator Loss:
#   For real pairs:   BCE(D(photo, real_sketch), target=real_smooth)
#   For fake pairs:   BCE(D(photo, fake_sketch.detach()), target=fake_smooth)
#   Total:            (real_loss + fake_loss) / 2
#
#   .detach() on fake_sketch is CRITICAL:
#     The discriminator loss should ONLY update the discriminator.
#     Without .detach(), gradients flow back to the generator through
#     the discriminator loss path, creating a messy combined gradient.
#     The generator should ONLY be updated through its own loss.
#
# Generator Adversarial Loss:
#   BCE(D(photo, fake_sketch), target=1.0)
#   "How well did I fool the discriminator?"
#   The generator wants D to think its fakes are real.
#   Target = 1.0 (not smoothed) — we want the generator to AIM HIGH.
#
# Label Smoothing (stability trick):
#   real_smooth  = 0.9  (instead of 1.0 — prevents D overconfidence)
#   fake_smooth  = 0.0  (no smoothing for fake — don't encourage ambiguity)
#
#   Why one-sided? Smoothing fake labels (to 0.1) would tell the
#   discriminator "some fakes might be real" → D becomes less certain
#   about rejecting fakes → G gets a weaker training signal.
#   Smoothing only real labels keeps D cautious but not confused.


# ═══════════════════════════════════════════════════════════════
# 4. MULTI-SCALE EXTENSION (Phase 2, optional)
# ═══════════════════════════════════════════════════════════════
#
# pix2pixHD uses THREE discriminators at different scales:
#   D1: original resolution (256×256 → 30×30 output)
#   D2: 2× downsampled (128×128 → 14×14 output)
#   D3: 4× downsampled (64×64  → 6×6 output)
#
# Each D is identical in architecture, just operating at different scales.
# D1 focuses on fine details (70×70 patch at full res)
# D2 focuses on mid-level structure (140×140 receptive field at half res)
# D3 focuses on global structure (280×280 receptive field at quarter res)
#
# Loss = mean(D1_loss + D2_loss + D3_loss)
#
# This is overkill for our data size but worth knowing as an industry
# pattern. For 322 pairs, a single PatchGAN is sufficient.
