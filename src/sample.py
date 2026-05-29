"""
Phase 1: Sampling & Visualization utilities for pix2pix GAN.

GAN inference is a single forward pass — O(1), deterministic.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image as tv_save_image
from PIL import Image
import torchvision.transforms as T

from unet import UNetGenerator
from data_loader import DATASET_MEAN, DATASET_STD


def load_checkpoint(checkpoint_path, device='cpu'):
    """Load generator from a checkpoint file.

    Supports both full checkpoints (dict with 'generator') and
    generator-only state dicts.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and 'generator' in ckpt:
        state = ckpt['generator']
    else:
        state = ckpt

    model = UNetGenerator()
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, size=(256, 256), mean=None, std=None):
    """Load and preprocess a single image for inference."""
    if mean is None:
        mean = DATASET_MEAN
    if std is None:
        std = DATASET_STD

    img = Image.open(image_path).convert('RGB')
    tfs = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return tfs(img).unsqueeze(0)  # (1, 3, H, W)


def postprocess_tensor(tensor, mean=None, std=None):
    """Convert normalized tensor back to [0, 1] range for visualization."""
    if mean is None:
        mean = DATASET_MEAN
    if std is None:
        std = DATASET_STD

    mean_t = torch.tensor(mean, device=tensor.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=tensor.device).view(1, 3, 1, 1)
    return (tensor * std_t + mean_t).clamp(0, 1)


def generate(model, photo_path, device='cpu', size=(256, 256)):
    """Single-image inference: photo → sketch.

    Args:
        model: UNetGenerator (already on device, in eval mode)
        photo_path: path to input photo
        device: torch device
        size: resize to this before feeding to model

    Returns:
        PIL Image of the generated sketch
    """
    img_tensor = preprocess_image(photo_path, size=size).to(device)

    with torch.no_grad():
        sketch = model(img_tensor)                         # [-1, 1]
        sketch = postprocess_tensor(sketch)                 # [0, 1]
        sketch = sketch.squeeze(0).cpu()

    # Convert to PIL
    sketch_pil = T.ToPILImage()(sketch)
    return sketch_pil


def generate_batch(model, dataloader, num_samples=8, device='cpu'):
    """Generate sketches for the first `num_samples` validation pairs.

    Returns:
        photos:        (num_samples, 3, H, W) in [0, 1]
        real_sketches: (num_samples, 3, H, W) in [0, 1]
        fake_sketches: (num_samples, 3, H, W) in [0, 1]
    """
    photos, real_sketches = next(iter(dataloader))
    photos = photos[:num_samples].to(device)
    real_sketches = real_sketches[:num_samples].to(device)

    with torch.no_grad():
        fake_sketches = model(photos)

    # Denormalize all for visualization
    photos = postprocess_tensor(photos)
    real_sketches = postprocess_tensor(real_sketches)
    fake_sketches = postprocess_tensor(fake_sketches)

    return photos.cpu(), real_sketches.cpu(), fake_sketches.cpu()


def save_sample_grid(model, val_loader, epoch, save_dir, device='cpu',
                     num_samples=8):
    """Create and save a visualization grid: Photo | Real Sketch | Fake Sketch.

    Saves: {save_dir}/epoch_{epoch:03d}.png
    """
    photos, real_sketches, fake_sketches = generate_batch(
        model, val_loader, num_samples=num_samples, device=device
    )

    # Arrange: for each sample, stack [photo, real, fake] vertically
    rows = []
    for i in range(num_samples):
        row = torch.stack([photos[i], real_sketches[i], fake_sketches[i]], dim=0)
        rows.append(row)
    grid = torch.cat(rows, dim=3)  # stack horizontally along width

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tv_save_image(grid, save_dir / f'epoch_{epoch:03d}.png')
    print(f"Saved sample grid: {save_dir / f'epoch_{epoch:03d}.png'}")


def create_comparison_grid(model, test_loader, save_path, device='cpu',
                           num_samples=8):
    """Create a side-by-side comparison: Photos | Generated | Ground Truth.

    Layout:
        Row 1: Input Photos
        Row 2: Generated Sketches
        Row 3: Ground Truth Sketches
    """
    photos, real_sketches, fake_sketches = generate_batch(
        model, test_loader, num_samples=num_samples, device=device
    )

    # Stack into three rows
    top = make_grid(photos, nrow=num_samples)
    mid = make_grid(fake_sketches, nrow=num_samples)
    bot = make_grid(real_sketches, nrow=num_samples)

    combined = torch.cat([top, mid, bot], dim=1)  # stack vertically
    tv_save_image(combined, save_path)
    print(f"Saved comparison grid: {save_path}")


def process_directory(model, input_dir, output_dir, device='cpu',
                      size=(256, 256)):
    """Process all images in a directory and save generated sketches."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    files = sorted([f for f in input_dir.iterdir()
                    if f.suffix.lower() in extensions])

    for img_path in files:
        sketch = generate(model, img_path, device=device, size=size)
        out_path = output_dir / f'sketch_{img_path.stem}.png'
        sketch.save(out_path)
        print(f"Saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate sketches with pix2pix')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint .pt file')
    parser.add_argument('--input', help='Single image path')
    parser.add_argument('--input-dir', help='Directory of images')
    parser.add_argument('--output-dir', default='outputs/', help='Output directory')
    parser.add_argument('--device', default='cpu', help='Device: cpu or cuda')
    parser.add_argument('--size', type=int, default=256, help='Resize images to this')

    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint, device=args.device)
    print(f"Loaded model from {args.checkpoint}")

    if args.input:
        sketch = generate(model, args.input, device=args.device,
                          size=(args.size, args.size))
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'sketch_{Path(args.input).stem}.png'
        sketch.save(out_path)
        print(f"Saved: {out_path}")
    elif args.input_dir:
        process_directory(model, args.input_dir, args.output_dir,
                          device=args.device, size=(args.size, args.size))
    else:
        parser.print_help()
