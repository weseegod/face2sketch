"""
Tests for U-Net Generator and PatchGAN Discriminator.
"""

import unittest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unet import UNetGenerator, ConvBlock, EncoderBlock, DecoderBlock
from discriminator import PatchGANDiscriminator, DiscriminatorBlock


class TestConvBlock(unittest.TestCase):
    def test_forward_shape(self):
        block = ConvBlock(64, 128)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        self.assertEqual(out.shape, (2, 128, 32, 32))

    def test_dropout(self):
        block = ConvBlock(64, 64, use_dropout=True, dropout=0.5)
        x = torch.randn(2, 64, 16, 16)
        block.train()
        out = block(x)
        self.assertEqual(out.shape, (2, 64, 16, 16))


class TestEncoderBlock(unittest.TestCase):
    def test_forward(self):
        enc = EncoderBlock(64, 128)
        x = torch.randn(2, 64, 64, 64)
        down, skip = enc(x)
        self.assertEqual(down.shape, (2, 128, 32, 32))
        self.assertEqual(skip.shape, (2, 128, 64, 64))


class TestDecoderBlock(unittest.TestCase):
    def test_forward(self):
        dec = DecoderBlock(512, 256)
        x = torch.randn(2, 512, 32, 32)
        skip = torch.randn(2, 256, 64, 64)
        out = dec(x, skip)
        self.assertEqual(out.shape, (2, 256, 64, 64))


class TestUNetGenerator(unittest.TestCase):
    def setUp(self):
        self.model = UNetGenerator(in_channels=3, out_channels=3, ngf=64,
                                    num_levels=5)

    def test_param_count(self):
        n = sum(p.numel() for p in self.model.parameters())
        # Should be ~37M for ngf=64, 5 levels
        self.assertGreater(n, 20_000_000)
        self.assertLess(n, 50_000_000)

    def test_forward_shape_256(self):
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (2, 3, 256, 256))

    def test_forward_shape_128(self):
        x = torch.randn(4, 3, 128, 128)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (4, 3, 128, 128))

    def test_output_range(self):
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out = self.model(x)
        # Output should be in [-1, 1] due to tanh
        self.assertTrue(out.min() >= -1.0)
        self.assertTrue(out.max() <= 1.0)

    def test_different_inputs_produce_different_outputs(self):
        """Mode collapse check: different inputs should give different outputs."""
        x1 = torch.randn(2, 3, 256, 256)
        x2 = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out1 = self.model(x1)
            out2 = self.model(x2)
        diff = (out1 - out2).abs().mean()
        self.assertGreater(diff.item(), 0.0,
                           "Different inputs should produce different outputs")

    def test_small_ngf(self):
        model = UNetGenerator(in_channels=3, out_channels=3, ngf=32, num_levels=4)
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (2, 3, 128, 128))


class TestDiscriminatorBlock(unittest.TestCase):
    def test_forward(self):
        block = DiscriminatorBlock(64, 128, stride=2, use_batchnorm=True)
        x = torch.randn(2, 64, 128, 128)
        out = block(x)
        self.assertEqual(out.shape, (2, 128, 64, 64))

    def test_no_batchnorm(self):
        block = DiscriminatorBlock(6, 64, stride=2, use_batchnorm=False)
        x = torch.randn(2, 6, 256, 256)
        out = block(x)
        self.assertEqual(out.shape, (2, 64, 128, 128))


class TestPatchGANDiscriminator(unittest.TestCase):
    def setUp(self):
        self.model = PatchGANDiscriminator(in_channels=6, ndf=64, n_layers=3)

    def test_param_count(self):
        n = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(n, 2_000_000)
        self.assertLess(n, 4_000_000)

    def test_forward_shape_256(self):
        photo = torch.randn(2, 3, 256, 256)
        sketch = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out = self.model(photo, sketch)
        self.assertEqual(out.shape, (2, 1, 30, 30))

    def test_output_range(self):
        photo = torch.randn(2, 3, 256, 256)
        sketch = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out = self.model(photo, sketch)
        self.assertTrue(out.min() >= 0.0)
        self.assertTrue(out.max() <= 1.0)

    def test_batch_size(self):
        for bs in [1, 4, 8]:
            photo = torch.randn(bs, 3, 256, 256)
            sketch = torch.randn(bs, 3, 256, 256)
            with torch.no_grad():
                out = self.model(photo, sketch)
            self.assertEqual(out.shape, (bs, 1, 30, 30))


class TestIntegration(unittest.TestCase):
    """End-to-end: generator + discriminator working together."""

    def test_gan_forward(self):
        gen = UNetGenerator(in_channels=3, out_channels=3, ngf=64, num_levels=5)
        disc = PatchGANDiscriminator(in_channels=6, ndf=64, n_layers=3)

        photo = torch.randn(2, 3, 256, 256)
        real_sketch = torch.randn(2, 3, 256, 256)

        with torch.no_grad():
            fake_sketch = gen(photo)
            d_real = disc(photo, real_sketch)
            d_fake = disc(photo, fake_sketch)

        self.assertEqual(fake_sketch.shape, (2, 3, 256, 256))
        self.assertEqual(d_real.shape, (2, 1, 30, 30))
        self.assertEqual(d_fake.shape, (2, 1, 30, 30))
        self.assertTrue(fake_sketch.min() >= -1.0)
        self.assertTrue(fake_sketch.max() <= 1.0)


if __name__ == "__main__":
    unittest.main()
