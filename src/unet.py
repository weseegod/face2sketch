"""
Phase 1: U-Net Generator for pix2pix GAN.

Architecture: Encoder-Decoder with skip connections.
Used for image-to-image translation: photo → sketch/caricature.

References:
  - pix2pix (Isola et al. 2017)
  - U-Net (Ronneberger et al. 2015)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Double convolution block: Conv2d → BN → ReLU → Conv2d → BN → ReLU.
    Optional dropout (only used in decoder).
    """

    def __init__(self, in_channels, out_channels, use_dropout=False, dropout=0.5):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """One encoder level: ConvBlock → MaxPool.
    Returns (downsampled, skip_connection).
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)          # features BEFORE pooling (saved for decoder)
        down = self.pool(skip)       # halved spatial size
        return down, skip


class DecoderBlock(nn.Module):
    """One decoder level: Upsample → Conv → concat(skip) → ConvBlock."""

    def __init__(self, in_channels, out_channels, use_dropout=False, dropout=0.5):
        super().__init__()
        # Upsample by 2×, then refine with a conv
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )
        # After concat with skip, channels double, then compressed back
        self.conv = ConvBlock(out_channels * 2, out_channels, use_dropout=use_dropout, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)     # concat along channel dim
        x = self.conv(x)
        return x


class UNetGenerator(nn.Module):
    """Full U-Net generator for pix2pix.

    Encoder: 5 levels of downsampling + skip connections saved at each level.
    Bottleneck: ConvBlock at lowest resolution.
    Decoder: 5 levels of upsampling, each concatting the matching skip.
    Output: Conv2d → tanh → [-1, 1] sketch.

    Channel progression (ngf=64):
        Encoder: 64 → 128 → 256 → 512 → 512 → 512 (bottleneck)
        Decoder: 512 → 512 → 256 → 128 → 64 → 3 (output)
    """

    def __init__(self, in_channels=3, out_channels=3, ngf=64, num_levels=5,
                 use_dropout=True, dropout=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ngf = ngf
        self.num_levels = num_levels

        # ── Initial conv (no activation, raw pixel input) ──
        self.initial = nn.Conv2d(in_channels, ngf, kernel_size=3, padding=1, bias=False)

        # ── Encoder ──
        # enc_chs[i] = output channels of encoder level i (0..num_levels-1)
        enc_chs = []
        self.encoders = nn.ModuleList()
        in_ch = ngf
        for i in range(num_levels):
            multiplier = min(2 ** i, 8)
            out_ch = ngf * multiplier
            enc_chs.append(out_ch)
            self.encoders.append(EncoderBlock(in_ch, out_ch))
            in_ch = out_ch

        # ── Bottleneck ──
        bottleneck_ch = ngf * 8
        self.bottleneck = ConvBlock(bottleneck_ch, bottleneck_ch)

        # ── Decoder ──
        # Decoder processes deep→shallow: i = num_levels-1, ..., 0
        # Decoder level i concats encoder skip from encoder level i.
        # dec_out = enc_chs[i] (output channels matching the skip source)
        self.decoders = nn.ModuleList()
        for i in range(num_levels - 1, -1, -1):
            dec_in = in_ch                   # channels from previous level
            dec_out = enc_chs[i]             # channels matching encoder level i skip
            self.decoders.append(
                DecoderBlock(dec_in, dec_out, use_dropout=use_dropout, dropout=dropout)
            )
            in_ch = dec_out

        # ── Output conv: feature maps → RGB ──
        self.output_conv = nn.Sequential(
            nn.Conv2d(ngf, out_channels, kernel_size=1),
            nn.Tanh(),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # Initial conv
        x = self.initial(x)

        # Encoder: collect skips
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder: deep→shallow order (already iterates i=4,3,2,1,0)
        # The i-th decoder block (counting from 0) should concat
        # the skip from encoder level (num_levels-1-i).
        # E.g.: decoder0 (deepest) → skips[4], decoder4 (shallowest) → skips[0]
        for idx, decoder in enumerate(self.decoders):
            skip_idx = self.num_levels - 1 - idx
            x = decoder(x, skips[skip_idx])

        # Output
        return self.output_conv(x)
