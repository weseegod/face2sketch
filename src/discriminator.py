"""
Phase 1: PatchGAN Discriminator for pix2pix GAN.

Classifies 70×70 image patches as real or fake instead of the whole image.
Output: N×N feature map where each cell = P(real) for one patch.
"""

import torch
import torch.nn as nn


class DiscriminatorBlock(nn.Module):
    """One PatchGAN layer: Conv2d → [BatchNorm] → LeakyReLU(0.2)."""

    def __init__(self, in_channels, out_channels, stride, use_batchnorm=True,
                 leaky_slope=0.2):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride,
                      padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(leaky_slope, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator for conditional GAN.

    Input: concatenated (photo, sketch_or_fake) → 6 channels.
    Output: (B, 1, 30, 30) prediction map for 256×256 input.

    Architecture (n_layers=3, ndf=64):
      C64  (k4,s2,p1) → 128×128  [no BN on first layer]
      C128 (k4,s2,p1) → 64×64
      C256 (k4,s2,p1) → 32×32
      C512 (k4,s1,p1) → 31×31
      C1   (k4,s1,p1) → 30×30  [Sigmoid output]
    """

    def __init__(self, in_channels=6, ndf=64, n_layers=3,
                 use_batchnorm=True, leaky_slope=0.2):
        super().__init__()
        self.in_channels = in_channels

        layers = []

        # Layer 0: no BatchNorm (standard GAN practice)
        layers.append(
            DiscriminatorBlock(in_channels, ndf, stride=2, use_batchnorm=False,
                               leaky_slope=leaky_slope)
        )

        # Middle layers: stride-2 downsampling with BatchNorm
        in_ch = ndf
        for i in range(1, n_layers):
            out_ch = ndf * (2 ** i)
            layers.append(
                DiscriminatorBlock(in_ch, out_ch, stride=2, use_batchnorm=use_batchnorm,
                                   leaky_slope=leaky_slope)
            )
            in_ch = out_ch

        # Penultimate layer: stride=1 (don't downsample further)
        out_ch = ndf * (2 ** n_layers)
        layers.append(
            DiscriminatorBlock(in_ch, out_ch, stride=1, use_batchnorm=use_batchnorm,
                               leaky_slope=leaky_slope)
        )

        # Output layer: 1 channel prediction map + Sigmoid
        layers.append(
            nn.Sequential(
                nn.Conv2d(out_ch, 1, kernel_size=4, stride=1, padding=1),
                nn.Sigmoid(),
            )
        )

        self.model = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, photo, sketch):
        """Forward pass.

        Args:
            photo: (B, 3, H, W) condition image
            sketch: (B, 3, H, W) real or generated sketch

        Returns:
            (B, 1, H', W') patch prediction map
        """
        x = torch.cat([photo, sketch], dim=1)  # (B, 6, H, W)
        return self.model(x)
