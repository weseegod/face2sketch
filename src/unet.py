"""
Phase 1: U-Net Generator for pix2pix GAN.

Architecture: Encoder-Decoder with skip connections.
Used for image-to-image translation: photo → sketch/caricature.

References:
  - pix2pix (Isola et al. 2017) — original paired translation with U-Net
  - U-Net (Ronneberger et al. 2015) — biomedical segmentation, skip connections
  - pix2pixHD (Wang et al. 2018) — multi-scale generator extension

═══════════════════════════════════════════════════════════════
MODULE OVERVIEW
═══════════════════════════════════════════════════════════════

Classes to implement:
  1. ConvBlock       — Conv2d → BatchNorm → ReLU (the U-Net building block)
  2. EncoderBlock    — ConvBlock×2 + MaxPool (produces features + skip)
  3. DecoderBlock    — UpConv + concat(skip) + ConvBlock×2
  4. UNetGenerator   — full U-Net: encoder → bottleneck → decoder → output

THIS U-NET IS REUSED IN PHASE 3 (DIFFUSION) WITH MODIFICATIONS:
  - Phase 1-2 (GAN):     BatchNorm, ReLU, no time embedding
  - Phase 3 (Diffusion):  GroupNorm, SiLU, + time embedding injection
                          Additional input channels (photo + noisy target = 6)
"""

# ═══════════════════════════════════════════════════════════════
# IMPORTS (will need these)
# ═══════════════════════════════════════════════════════════════
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# 1. CONV BLOCK — The U-Net Building Block
# ═══════════════════════════════════════════════════════════════
#
# ConvBlock(in_channels, out_channels, use_dropout=False):
#   A double-convolution block used everywhere in the U-Net.
#
#   Forward pass:
#     x → Conv2d(in_ch, out_ch, kernel=3, padding=1)  ← preserve spatial size
#       → BatchNorm2d(out_ch)
#       → ReLU(inplace=True)
#       → [Dropout2d(0.5) if use_dropout]              ← only in decoder
#       → Conv2d(out_ch, out_ch, kernel=3, padding=1)
#       → BatchNorm2d(out_ch)
#       → ReLU(inplace=True)
#
#   This pattern repeats at EVERY encoder/decoder level.
#   Two 3×3 convs, BatchNorm after each, ReLU after each.
#
#   Dropout is only used in the DECODER blocks (use_dropout=True there).
#   This is a standard U-Net regularization technique — the encoder
#   already compresses information, dropout would lose too much signal.
#
#   The original U-Net paper uses unpadded convolutions (spatial size
#   changes each conv). pix2pix uses padded convolutions (spatial size
#   is preserved). We follow pix2pix: use padding=1.


# ═══════════════════════════════════════════════════════════════
# 2. ENCODER BLOCK — Downsample + Extract Features
# ═══════════════════════════════════════════════════════════════
#
# EncoderBlock(in_channels, out_channels):
#   One level of the contracting path.
#
#   Forward pass:
#     x → ConvBlock(in_ch, out_ch)          ← extract features at this resolution
#       → save as "skip" for decoder
#       → MaxPool2d(kernel=2, stride=2)     ← halve spatial size
#
#   Returns: (downsampled_x, skip_connection)
#     downsampled_x: (B, out_ch, H/2, W/2)  ← feeds into next encoder
#     skip:          (B, out_ch, H, W)      ← saved for decoder's concat
#
#   The MaxPool is what creates the U-Net's hierarchical representation.
#   Each level "sees" the image at a different scale:
#     Level 0 (256×256): fine details — eyes, nose shape, hair texture
#     Level 1 (128×128): facial features — eye placement, mouth position
#     Level 2 (64×64):   face structure — oval shape, proportions
#     Level 3 (32×32):   head position — where the face is, orientation
#     Level 4 (16×16):   global context — lighting, overall composition
#     Bottleneck (8×8):  compressed representation of the entire face


# ═══════════════════════════════════════════════════════════════
# 3. DECODER BLOCK — Upsample + Combine with Skip
# ═══════════════════════════════════════════════════════════════
#
# DecoderBlock(in_channels, out_channels, use_dropout=False):
#   One level of the expanding path.
#
#   Forward pass:
#     x → ConvTranspose2d(in_ch, out_ch, kernel=4, stride=2, padding=1)
#         OR: Upsample(scale_factor=2, mode='bilinear') → Conv2d(in_ch, out_ch, 3)
#       → concat(x, skip_connection) along channel dim
#         (THIS DOUBLES CHANNELS: out_ch + skip_ch = 2×out_ch)
#       → ConvBlock(2×out_ch, out_ch, use_dropout)   ← compress back
#
#   The skip connection is the U-Net's signature. Without it, the decoder
#   only sees bottleneck features → blurry, imprecise outputs.
#   With skip connections, the decoder can reference the encoder's fine
#   details directly. This is why U-Nets produce sharp, spatially-accurate
#   outputs despite the massive compression in the bottleneck.
#
#   Upsampling approach:
#     Option A: ConvTranspose2d — learned upsampling, can produce checkerboard
#     Option B: Upsample + Conv2d — deterministic upsampling + learned conv
#     Option B is preferred. Use bilinear or nearest-neighbor upsampling,
#     then a 3×3 conv to refine. This is what pix2pix uses.


# ═══════════════════════════════════════════════════════════════
# 4. FULL U-NET GENERATOR
# ═══════════════════════════════════════════════════════════════
#
# UNetGenerator(in_channels=3, out_channels=3, ngf=64, num_levels=5):
#
#   Full architecture:
#     1. Initial conv: 3 → ngf (no activation yet — raw pixel input)
#
#     2. Encoder (num_levels downsampling blocks):
#        Level 0: ngf    → ngf      (256→128,   skip saved)
#        Level 1: ngf    → ngf*2    (128→64,    skip saved)
#        Level 2: ngf*2  → ngf*4    (64→32,     skip saved)
#        Level 3: ngf*4  → ngf*8    (32→16,     skip saved)
#        Level 4: ngf*8  → ngf*8    (16→8,      skip saved)
#
#     3. Bottleneck: ConvBlock(ngf*8, ngf*8) at 8×8
#        (No pooling here — just process the most compressed features)
#
#     4. Decoder (num_levels upsampling blocks, REVERSE order):
#        Level 4: ngf*8  → ngf*8    (8→16,   concat skip4)
#        Level 3: ngf*8  → ngf*4    (16→32,  concat skip3)
#        Level 2: ngf*4  → ngf*2    (32→64,  concat skip2)
#        Level 1: ngf*2  → ngf      (64→128, concat skip1)
#        Level 0: ngf    → ngf      (128→256, concat skip0)
#
#        Dropout is ON for decoder levels (use_dropout=True)
#
#     5. Output: Conv2d(ngf, out_channels=3, kernel=1) → tanh
#        The 1×1 conv maps feature space to RGB
#        tanh squashes output to [-1, 1] (matching normalized training data)
#
#   Channel progression (with ngf=64):
#     Encoder: 64 → 128 → 256 → 512 → 512 → 512
#     Decoder: 512 → 512 → 256 → 128 → 64 → 3
#
#   Total params: ~41M at ngf=64 with 5 levels
#   (pix2pix uses ngf=64, num_levels varies by image size)
#
#   Important: The encoder output at EACH level (before MaxPool) is saved
#   as a skip connection. The decoder receives these in REVERSE order.
#   Level 0 skip is used by decoder level 0 (the FINAL upsampling step).
#   Level 4 skip is used by decoder level 4 (the FIRST upsampling step).
#
#   At inference: single forward pass photo → sketch. No loops, no iterations.
#   GAN inference is O(1) — this is the key speed advantage over diffusion.
